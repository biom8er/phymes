use std::{collections::HashMap, io::Error, iter::zip, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::array::{ArrayRef, ListArray, RecordBatch, StringArray, UInt8Array};
use candle_core::Device;
use lopdf::{
    Document, Object, Stream,
    content::{Content, Operation},
    dictionary,
};

use phymes_core::{BuildableTrait, BuilderTrait, DataEncoding, DataFormat, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait};
use phymes_schemas::{
    AvailableSubjects, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tracing::{Level, event, instrument};

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Chunk documents by splitting a StringArray column in a [RecordBatch] into multiple rows based on a defined criteria
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExtractPDF {
    lhs_pk: String,
    lhs_values: String,
}

impl MappableTrait for ExtractPDF {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for ExtractPDF {
    fn get_description(&self) -> String {
        "Extract text from PDF documents".to_string()
    }
    fn to_json_tool_schema(&self) -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_pk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_values".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::Array),
                description: Some(
                    "A list of value column identifiers for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = Function {
            name: Self::get_static_name().to_string(),
            description: Some(self.get_description()),
            parameters: FunctionParameters {
                schema_type: JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
                ]),
            },
        };
        let tool = Tool {
            r#type: ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

impl DataOperatorTrait for ExtractPDF {
    fn new(config: &DataConfig) -> Result<Self>
    where
        Self: Sized,
    {
        let lhs_pk = config.lhs_pk.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;
        let lhs_values = config
            .lhs_values
            .as_ref()
            .cloned()
            .ok_or(anyhow!(
                "Missing `lhs_values` for `{}`.",
                Self::get_static_name()
            ))?
            .first()
            .cloned()
            .ok_or(anyhow!(
                "`lhs_values` is empty for `{}`.",
                Self::get_static_name()
            ))?;
        Ok(ExtractPDF { lhs_pk, lhs_values })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        let docs = prepare_pdf_documents(&self.lhs_pk, &self.lhs_values, lhs_args);
        extract_pdf(&docs)
    }
}

type ParsedPage = (String, u32, Vec<String>);

/// Extract text from a PDF document(s) and return it as an ArrowTable
///
/// # Arguments
/// * `docs` - A slice of tuples containing the document id and the Document object
///
/// # Returns
/// * `Result<RecordBatch>` - A RecordBatch containing the page number and text extracted from the PDF documents
///
/// # Notes
/// * The output schema of the RecordBatch matches that used in the document RAG session plans i.e.,
///   `document_id`: The ID of the document, `chunk_id`: The page number, `text`: The text content of the page.
///
/// # Errors
/// * Returns an error if text extraction fails for any page in the document
#[instrument(skip(docs))]
pub fn extract_pdf(docs: &[(String, Document)]) -> Result<RecordBatch> {
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
                chunk_id_vec.push(format!("{id}_{page_num}"));
                document_id_vec.push(id);
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
    Ok(batch)
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

/// Prepare PDF documents for processing by extracting the document ID and PDF data
pub fn prepare_pdf_documents(
    lhs_pk: &str,
    lhs_values: &str,
    docs: &[RecordBatch],
) -> Vec<(String, Document)> {
    docs.iter()
        .flat_map(|batch| {
            let document_id = batch
                .column_by_name(lhs_pk)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default().to_string())
                .collect::<Vec<_>>();
            let pdf_data = batch
                .column_by_name(lhs_values)
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap()
                .iter()
                .map(|s| {
                    let bytes = s
                        .unwrap()
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap())
                        .collect::<Vec<u8>>();
                    filter_pdf(load_pdf_document(&bytes).unwrap())
                })
                .collect::<Vec<_>>();
            zip(document_id, pdf_data)
        })
        .collect()
}

/// Make a PDF document with text content for testing purposes
#[allow(dead_code)]
pub fn make_pdf_document(contents: &[&str]) -> Document {
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
    use phymes_core::{BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait};

    use super::*;

    #[test]
    fn test_extract_pdf_text() {
        // Create several PDF document in memory
        let doc_1 = filter_pdf(make_pdf_document(&["1\n2\n3", "4\n5\n6"]));
        let doc_2 = filter_pdf(make_pdf_document(&["1\n2\n3", "4\n5\n6"]));
        let docs = [("doc_1".to_string(), doc_1), ("doc_2".to_string(), doc_2)];

        // Extract text from the PDF document
        let batch = extract_pdf(&docs).unwrap();

        // Check the results
        let table = Subject::get_builder()
            .with_name("")
            .with_record_batches(vec![batch])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(table.count_rows(), 4);
        assert_eq!(
            table.get_column_as_vec_str("document_id"),
            ["doc_1", "doc_1", "doc_2", "doc_2"]
        );
        assert_eq!(
            table.get_column_as_vec_str("chunk_id"),
            ["doc_1_1", "doc_1_2", "doc_2_1", "doc_2_2"]
        );
        assert_eq!(
            table.get_column_as_vec_str("text"),
            ["123 ", "456 ", "123 ", "456 "]
        );
    }

    #[test]
    fn test_make_pdf_document() {
        // Make the PDF documents
        let document_texts = &[
            "Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.\n\nA linear chain of amino acid residues is called a polypeptide. A protein contains at least one long polypeptide. Short polypeptides, containing less than 20–30 residues, are rarely considered to be proteins and are commonly called peptides. The individual amino acid residues are bonded together by peptide bonds and adjacent amino acid residues. The sequence of amino acid residues in a protein is defined by the sequence of a gene, which is encoded in the genetic code. In general, the genetic code specifies 20 standard amino acids; but in certain organisms the genetic code can include selenocysteine and—in certain archaea—pyrrolysine. Shortly after or even during synthesis, the residues in a protein are often chemically modified by post-translational modification, which alters the physical and chemical properties, folding, stability, activity, and ultimately, the function of the proteins. Some proteins have non-peptide groups attached, which can be called prosthetic groups or cofactors. Proteins can work together to achieve a particular function, and they often associate to form stable protein complexes.\n\nOnce formed, proteins only exist for a certain period and are then degraded and recycled by the cell's machinery through the process of protein turnover. A protein's lifespan is measured in terms of its half-life and covers a wide range. They can exist for minutes or years with an average lifespan of 1-2 days in mammalian cells. Abnormal or misfolded proteins are degraded more rapidly either due to being targeted for destruction or due to being unstable.\n\nLike other biological macromolecules such as polysaccharides and nucleic acids, proteins are essential parts of organisms and participate in virtually every process within cells. Many proteins are enzymes that catalyse biochemical reactions and are vital to metabolism. Some proteins have structural or mechanical functions, such as actin and myosin in muscle, and the cytoskeleton's scaffolding proteins that maintain cell shape. Other proteins are important in cell signaling, immune responses, cell adhesion, and the cell cycle. In animals, proteins are needed in the diet to provide the essential amino acids that cannot be synthesized. Digestion breaks the proteins down for metabolic use.",
            "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.\n\nThe two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.\n\nBoth strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a process called translation.\n\nWithin eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed.",
            "Lipids are a broad group of organic compounds which include fats, waxes, sterols, fat-soluble vitamins (such as vitamins A, D, E and K), monoglycerides, diglycerides, phospholipids, and others. The functions of lipids include storing energy, signaling, and acting as structural components of cell membranes.[3][4] Lipids have applications in the cosmetic and food industries, and in nanotechnology.[5]\n\nLipids may be broadly defined as hydrophobic or amphiphilic small molecules; the amphiphilic nature of some lipids allows them to form structures such as vesicles, multilamellar/unilamellar liposomes, or membranes in an aqueous environment. Biological lipids originate entirely or in part from two distinct types of biochemical subunits or building-blocks: ketoacyl and isoprene groups.[3] Using this approach, lipids may be divided into eight categories: fatty acyls, glycerolipids, glycerophospholipids, sphingolipids, saccharolipids, and polyketides (derived from condensation of ketoacyl subunits); and sterol lipids and prenol lipids (derived from condensation of isoprene subunits).[3]\n\nAlthough the term lipid is sometimes used as a synonym for fats, fats are a subgroup of lipids called triglycerides. Lipids also encompass molecules such as fatty acids and their derivatives (including tri-, di-, monoglycerides, and phospholipids), as well as other sterol-containing metabolites such as cholesterol.[6] Although humans and other mammals use various biosynthetic pathways both to break down and to synthesize lipids, some essential lipids cannot be made this way and must be obtained from the diet.\n\n",
            "The cell is the basic structural and functional unit of all forms of life. Every cell consists of cytoplasm enclosed within a membrane; many cells contain organelles, each with a specific function. The term comes from the Latin word cellula meaning 'small room'. Most cells are only visible under a microscope. Cells emerged on Earth about 4 billion years ago. All cells are capable of replication, protein synthesis, and motility.\n\nCells are broadly categorized into two types: eukaryotic cells, which possess a nucleus, and prokaryotic cells, which lack a nucleus but have a nucleoid region. Prokaryotes are single-celled organisms such as bacteria, whereas eukaryotes can be either single-celled, such as amoebae, or multicellular, such as some algae, plants, animals, and fungi. Eukaryotic cells contain organelles including mitochondria, which provide energy for cell functions, chloroplasts, which in plants create sugars by photosynthesis, and ribosomes, which synthesise proteins.\n\nCells were discovered by Robert Hooke in 1665, who named them after their resemblance to cells inhabited by Christian monks in a monastery. Cell theory, developed in 1839 by Matthias Jakob Schleiden and Theodor Schwann, states that all organisms are composed of one or more cells, that cells are the fundamental unit of structure and function in all living organisms, and that all cells come from pre-existing cells.",
        ];
        let mut pdf = make_pdf_document(document_texts);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();

        // Convert from bytes back to the PDF document
        let pdf_test = filter_pdf(load_pdf_document(&bytes).unwrap());

        // Check that the original and test PDF documents are the same
        let batch = extract_pdf(&[("pdf".to_string(), pdf)]).unwrap();
        let batch_test = extract_pdf(&[("pdf".to_string(), pdf_test)]).unwrap();

        assert_eq!(batch, batch_test);
    }
}
