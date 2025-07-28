use std::sync::Arc;
use std::io::{Error, ErrorKind, Write};

use arrow::array::{ArrayRef, RecordBatch};
use lopdf::Document;
use phymes_core::session::common_traits::{BuildableTrait, BuilderTrait};
use phymes_core::table::arrow_table::{ArrowTable, ArrowTableBuilderTrait};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use anyhow::Result;
use tracing::{event, Level};

/// Extract text from a PDF document(s) and return it as an ArrowTable
/// 
/// # Arguments
/// * `docs` - A slice of tuples containing the document id and the Document object
/// * `name` - The name of the resulting ArrowTable
/// 
/// # Returns
/// * `Result<ArrowTable>` - An ArrowTable containing the page number and text extracted from the PDF documents
/// 
/// # Errors
/// * Returns an error if text extraction fails for any page in the document
pub fn extract_pdf_text(docs: &[(&str, &Document)], name: &str) -> Result<ArrowTable> {
    // Extract the page number and text along with any errors from the documents
    let pages: Vec<Result<(String, u32, Vec<String>), Error>> = docs
        .par_iter()
        .map(|(id, doc)| doc.get_pages()
            .into_par_iter()
            .map(
                |(page_num, page_id): (u32, (u32, u16))| -> Result<(String, u32, Vec<String>), Error> {
                    let text = doc.extract_text(&[page_num]).map_err(|e| {
                        Error::new(
                            ErrorKind::Other,
                            format!("Failed to extract text from page {page_num} id={page_id:?}: {e:}"),
                        )
                    })?;
                    Ok((
                        id.to_string(),
                        page_num,
                        text.split('\n')
                            .map(|s| s.trim_end().to_string())
                            .collect::<Vec<String>>(),
                    ))
                },
            )
            .collect::<Vec<Result<(String, u32, Vec<String>), Error>>>()
        ).flatten()
        .collect();

    // Create an ArrowTable to hold the extracted text
    let mut document_id_vec = Vec::new();
    let mut chunk_id_vec = Vec::new();
    let mut text_vec = Vec::new();
    let mut page_num_vec = Vec::new();
    for page in pages {
        match page {
            Ok((id, page_num, lines)) => {
                document_id_vec.push(id);
                chunk_id_vec.push(format!("{page_num}"));
                page_num_vec.push(page_num);
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
    let page_num_arr: ArrayRef = Arc::new(arrow::array::UInt32Array::from(page_num_vec));
    let batch = RecordBatch::try_from_iter(
        vec![
            ("document_id", document_id_arr),
            ("chunk_id", chunk_id_arr),
            ("page_num", page_num_arr),
            ("text", text_arr),
        ],
    )?;
    ArrowTable::get_builder()
        .with_name(name)
        .with_record_batches(vec![batch])?
        .build()
}