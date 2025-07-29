use std::io::Error;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::{ArrayRef, RecordBatch};
use lopdf::{
    Document, Object, Stream,
    content::{Content, Operation},
    dictionary,
};
use phymes_core::session::common_traits::{BuildableTrait, BuilderTrait};
use phymes_core::table::arrow_table::{ArrowTable, ArrowTableBuilderTrait};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::{Level, event};

type ParsedPage = (String, u32, Vec<String>);

/// Extract text from a PDF document(s) and return it as an ArrowTable
///
/// # Arguments
/// * `docs` - A slice of tuples containing the document id and the Document object
/// * `name` - The name of the resulting ArrowTable
///
/// # Returns
/// * `Result<ArrowTable>` - An ArrowTable containing the page number and text extracted from the PDF documents
///
/// # Notes
/// * The output schema of the ArrowTable matches that used in the document RAG session plans i.e.,
///   `document_id`: The ID of the document, `chunk_id`: The page number, `text`: The text content of the page.
///
/// # Errors
/// * Returns an error if text extraction fails for any page in the document
pub fn extract_pdf_text(docs: &[(&str, &Document)], name: &str) -> Result<ArrowTable> {
    // Extract the page number and text along with any errors from the documents
    let pages: Vec<Result<ParsedPage, Error>> = docs
        .into_par_iter()
        .map(|(id, doc)| {
            doc.get_pages()
                .into_par_iter()
                .map(
                    |(page_num, page_id): (u32, (u32, u16))| -> Result<ParsedPage, Error> {
                        // Extract text from the page
                        let text = doc.extract_text(&[page_num]).map_err(|e| {
                            Error::other(format!(
                                "Failed to extract text from page {page_num} id={page_id:?}: {e:}"
                            ))
                        })?;
                        Ok((
                            id.to_string(),
                            page_num,
                            text.split(|c| ['\n', '\r'].contains(&c))
                                .map(|s| s.trim_end().to_string())
                                .collect::<Vec<String>>(),
                        ))
                    },
                )
                .collect::<Vec<Result<ParsedPage, Error>>>()
        })
        .flatten()
        .collect();

    // Create an ArrowTable to hold the extracted text
    let mut document_id_vec = Vec::new();
    let mut chunk_id_vec = Vec::new(); // the page number
    let mut text_vec = Vec::new();
    for page in pages {
        match page {
            Ok((id, page_num, lines)) => {
                document_id_vec.push(id);
                chunk_id_vec.push(format!("{page_num}"));
                // Join the lines of text into a single string for each page
                text_vec.push(lines.join(" "));
            }
            Err(e) => {
                event!(Level::ERROR, "{e:?}");
            }
        }
    }
    let document_id_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(document_id_vec));
    let chunk_id_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(chunk_id_vec));
    let text_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(text_vec));
    let batch = RecordBatch::try_from_iter(vec![
        ("chunk_id", chunk_id_arr),
        ("document_id", document_id_arr),
        ("text", text_arr),
    ])?;
    ArrowTable::get_builder()
        .with_name(name)
        .with_record_batches(vec![batch])?
        .build()
}

static IGNORE_TYPE_NAMES: &[&[u8]] = &[
    b"Length",
    b"BBox",
    b"FormType",
    b"Matrix",
    b"Type",
    b"XObject",
    b"Subtype",
    b"Filter",
    b"ColorSpace",
    b"Width",
    b"Height",
    b"BitsPerComponent",
    b"Length1",
    b"Length2",
    b"Length3",
    b"PTEX.FileName",
    b"PTEX.PageNumber",
    b"PTEX.InfoDict",
    b"FontDescriptor",
    b"ExtGState",
    b"MediaBox",
    b"Annot",
];

static IGNORE_KEYS: &[&[u8]] = &[
    b"Producer",
    b"ModDate",
    b"Creator",
    b"ProcSet",
    b"XObject",
    b"MediaBox",
    b"Annots",
];

/// Filter a PDF document to remove unwanted objects and keys
pub fn filter_pdf(mut doc: Document) -> Document {
    // Extract the page IDs from the document
    let page_ids: Vec<(u32, u16)> = doc.get_pages().values().cloned().collect::<Vec<_>>();

    // Iterate over each page and filter out unwanted objects
    for page_id in page_ids {
        let _ = doc
            .get_object_mut(page_id)
            .iter_mut()
            .filter_map(|object| {
                if IGNORE_TYPE_NAMES.contains(&object.type_name().unwrap_or_default()) {
                    None
                } else {
                    Some(object)
                }
            })
            .map(|object| {
                // Remove unwanted keys manually
                let keys_to_remove: Vec<Vec<u8>> = match object.as_dict() {
                    Ok(dict) => dict
                        .iter()
                        .filter_map(|(k, _)| {
                            if IGNORE_KEYS.contains(&k.as_slice()) {
                                Some(k.clone())
                            } else {
                                None
                            }
                        })
                        .collect(),
                    Err(_) => Vec::new(),
                };
                for key in keys_to_remove {
                    if let Ok(dict_mut) = object.as_dict_mut() {
                        dict_mut.remove(&key);
                    }
                }
            });
    }
    doc
}

/// Load the PDF document in memory
///
/// # Arguments
/// * `doc` - A byte slice containing the PDF document data
///
/// # Returns
/// * `Result<Document>` - The loaded PDF document
pub fn load_pdf_document(doc: &[u8]) -> Result<Document> {
    match Document::load_mem(doc) {
        Ok(document) => Ok(document),
        Err(e) => Err(anyhow!(format!(
            "Failed to load PDF document in memory: {e}"
        ))),
    }
}

/// Make a PDF document with text content for testing purposes
#[allow(dead_code)]
fn make_pdf_document(contents: &[&str]) -> Document {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Courier",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });
    let mut page_id_vec = Vec::new();
    for content_str in contents {
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 48.into()]),
                Operation::new("Td", vec![100.into(), 600.into()]),
                Operation::new("Tj", vec![Object::string_literal(*content_str)]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
        });
        page_id_vec.push(page_id.into());
    }
    let pages = dictionary! {
        "Type" => "Pages",
        "Count" => page_id_vec.len() as i32,
        "Kids" => page_id_vec,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.compress();
    doc
}

#[cfg(test)]
mod tests {
    use phymes_core::{session::common_traits::MappableTrait, table::arrow_table::ArrowTableTrait};

    use super::*;

    #[test]
    fn test_extract_pdf_text() {
        // Create several PDF document in memory
        let doc_1 = filter_pdf(make_pdf_document(&["1\n2\n3", "4\n5\n6"]));
        let doc_2 = filter_pdf(make_pdf_document(&["1\n2\n3", "4\n5\n6"]));
        let docs = [("doc_1", &doc_1), ("doc_2", &doc_2)];

        // Extract text from the PDF document
        let table = extract_pdf_text(&docs, "test_table").unwrap();

        // Check the results
        assert_eq!(table.get_name(), "test_table");
        assert_eq!(table.count_rows(), 4);
        assert_eq!(
            table.get_column_as_str_vec("document_id"),
            ["doc_1", "doc_1", "doc_2", "doc_2"]
        );
        assert_eq!(
            table.get_column_as_str_vec("chunk_id"),
            ["1", "2", "1", "2"]
        );
        assert_eq!(
            table.get_column_as_str_vec("text"),
            ["123 ", "456 ", "123 ", "456 "]
        );
    }
}
