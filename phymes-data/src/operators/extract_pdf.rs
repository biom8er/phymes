use core::num;
use std::{collections::{BTreeMap, HashMap}, io::Error, iter::zip, sync::Arc};

use anyhow::{Ok, Result, anyhow};
use arrow::array::{ArrayRef, ListArray, RecordBatch, StringArray, UInt8Array};
use candle_core::Device;
use lopdf::{
    Document, Encoding, Object, Stream, content::{Content, Operation}, dictionary
};

use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use phymes_subject::MappableTrait;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tracing::{Level, event, instrument};

use crate::{DataConfig, DataOperatorTrait, ToolTrait};

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

/// Tm
#[derive(Debug, Clone, PartialEq)]
struct Tm {
    /// scale param 1
    a: f32,
    /// skew param 1
    b: f32,
    /// skew param 2
    c: f32,
    /// scale param 2
    d: f32,
    /// pos x
    x: f32,
    /// pos y
    y: f32,
}

impl Tm {
    pub fn new(a: &f32, b: &f32, c: &f32, d: &f32, x: &f32, y: &f32) -> Self {
        Self { a: *a, b: *b, c: *c, d: *d, x: *x, y: *y }
    }
}

impl Default for Tm {
    fn default() -> Self {
        Self { a: 1_f32, b: 0_f32, c: 0_f32, d: 1_f32, x: 0_f32, y: 0_f32 }
    }
}

/// Tm
#[derive(Debug, Clone, PartialEq)]
struct Td {
    /// pos x
    x: i64,
    /// pos y
    y: i64,
}

impl Td {
    pub fn new(x: &i64, y: &i64) -> Self {
        Self { x: *x, y: *y }
    }
}

impl Default for Td {
    fn default() -> Self {
        Self { x: 0_i64, y: 0_i64 }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
struct PdfFont {
    pub font_name: String,
    pub font_subtype: String,
    pub base_font: String,
}

impl PdfFont {
    pub fn new(font_name: &str, font_subtype: &str, base_font: &str, ) -> Self {
        Self { font_name: font_name.to_string(), font_subtype: font_subtype.to_string(), base_font: base_font.to_string() }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
struct PdfText {
    /// Index of the operataion the text was found
    pub op: u32,
    /// BT operataion the text was found
    pub bt: u32,
    /// Text matrix
    pub tm: Tm,
    /// Text translation
    pub td: Td,
    /// Font
    pub font: PdfFont,
    pub font_size: i64,
    pub page_num: u32,
    pub text: String,
}

impl PdfText {
    pub fn text_mut(&mut self) -> &mut String {
        &mut self.text
    }
}

/// Drop-in replacement for `extract_text` that retains additional metadata including
/// page_number, pos_x, pos_y, font_name, font_size, ...
fn extract_text(doc: &Document, page_numbers: &[u32]) -> Result<Vec<PdfText>> {
    let text_fragments = extract_text_chunks(doc, page_numbers)?;

    // Merge text that are positioned in the same Tm since Font cannot be reliable used across PDFs
    let mut text = Vec::new();
    let mut current_text: Option<PdfText> = None;
    for maybe_text_fragment in text_fragments.into_iter() {
        let key = format!("{:?}", maybe_text_fragment.tm);
        if let Some(mut current) = current_text.take() {
            let current_key = format!("{:?}", current.tm);
            if key == current_key {
                current.text = [current.text, maybe_text_fragment.text].join("");
                current_text.replace(current);
            } else {
                text.push(current);
            }
        } else {
            current_text = Some(maybe_text_fragment);
        }
    }
    if let Some(current) = current_text.take() {
        text.push(current);
    }

    // Additional filters of junk text
    let text = text.into_iter()
        .filter(|pdf_text| {            
            // Heuristics from <https://doi.org/10.1371/journal.pcbi.1005962> suitable for most articles
            let n_chars = pdf_text.text.chars().count();
            let n_num = pdf_text.text.chars().filter(|c| c.is_numeric()).count();
            let num_frac = n_num as f32 / n_chars as f32;
            dbg!(num_frac);
            let num_check = num_frac < 0.1_f32;
            let n_sym = pdf_text.text.chars().filter(|c| !c.is_alphanumeric()).count();
            let sym_frac = n_sym as f32 / n_chars as f32;
            dbg!(sym_frac);
            let sym_check =  sym_frac < 0.1_f32;
            let n_lower = pdf_text.text.chars().filter(|c| c.is_lowercase()).count();
            let lower_frac = n_lower as f32 / n_chars as f32;
            dbg!(lower_frac);
            let lower_check = lower_frac > 0.5_f32;
            num_check & sym_check & lower_check
        })
        .collect::<Vec<_>>();

    Ok(text)
}
fn extract_text_chunks(doc: &Document, page_numbers: &[u32]) -> Result<Vec<PdfText>> {
    let pages: BTreeMap<u32, (u32, u16)> = doc.get_pages();
    page_numbers
        .iter()
        .flat_map(|page_number| {
            let result = extract_text_chunks_from_page(doc, &pages, *page_number);
            match result {
                std::result::Result::Ok(text_chunks) => text_chunks.into_iter()
                    .filter_map(|x| if let std::result::Result::Ok(x) = x {
                        Some(x)
                    } else {
                        None
                    })
                    .map(|mut pdf_text| {
                        pdf_text.page_num = *page_number;
                        pdf_text
                    })
                    .map(Ok)
                    .collect::<Vec<_>>(),
                Err(err) => vec![Err(err)],
            }
        })
        .collect()
}

fn extract_text_chunks_from_page(doc: &Document, pages: &BTreeMap<u32, (u32, u16)>, page_number: u32) -> Result<Vec<Result<PdfText>>> {
    let mut collected_chunks_and_errs = Vec::<Result<PdfText>>::new();
    let page_id = *pages.get(&page_number).ok_or(anyhow!("Page number {page_number} not found."))?;

    // extract out the fonts on the page
    let fonts_map = extract_fonts_from_page(doc, page_id)?
        .into_iter()
        .map(|(a, b, c)| (a, (b, c)))
        .collect::<HashMap<_, _>>();

    // extract out the font encodings
    let fonts = doc.get_page_fonts(page_id)?;    
    let encodings: BTreeMap<Vec<u8>, Encoding> = fonts
        .into_iter()
        .filter_map(|(name, font)| match font.get_font_encoding(doc) {
            std::result::Result::Ok(it) => Some((name, it)),
            Err(err) => {
                let err = anyhow!("{err:?}");
                collected_chunks_and_errs.push(Err(err));
                None
            },
        })
        .collect();
    let content_data = doc.get_page_content(page_id)?;
    let content = Content::decode(&content_data)?;

    // each text with different encoding is extracted as separate chunk
    let mut current_encoding = None;
    let mut current_text = PdfText::default();
    let mut index: u32 = 0;
    for operation in &content.operations {
        match operation.operator.as_ref() {
            "BT" => {
                current_text.bt = index;
            }
            "TD" => {
                dbg!(&operation);
            }
            "Tm" => {
                dbg!(&operation);
                let m = operation
                    .operands
                    .iter()
                    .map(|f| f.as_f32().unwrap_or_default())
                    .collect::<Vec<_>>();
                let tm = Tm::new(m.get(0).unwrap_or(&1_f32),
                    m.get(1).unwrap_or(&0_f32),
                    m.get(2).unwrap_or(&0_f32),
                    m.get(3).unwrap_or(&1_f32),
                    m.get(4).unwrap_or(&0_f32),
                    m.get(5).unwrap_or(&0_f32),
                );
                current_text.tm = tm;
            }
            // "T*" => {
            //     dbg!(&operation);
            // }
            // "'" => {
            //     dbg!(&operation);
            // }
            "Td" => {
                dbg!(&operation);
                let d = operation
                    .operands
                    .iter()
                    .map(|f| f.as_i64().unwrap_or_default())
                    .collect::<Vec<_>>();
                let td = Td::new(d.get(0).unwrap_or(&0_i64),
                    d.get(1).unwrap_or(&0_i64),
                );
                current_text.td = td;
            }
            "Tf" => {
                let current_font = operation
                    .operands
                    .first()
                    .ok_or_else(|| anyhow!("missing font operand".to_string()))?
                    .as_name();
                let font_size = operation
                    .operands
                    .last()
                    .ok_or_else(|| anyhow!("missing font size operand".to_string()))?
                    .as_i64()?;
                let (current_enc, font_name) = match current_font {
                    std::result::Result::Ok(font) => (encodings.get(font), String::from_utf8(font.to_vec()).unwrap()),
                    Err(err) => {
                        let err = anyhow!("{err:?}");
                        collected_chunks_and_errs.push(Err(err));
                        (None, String::new())
                    },
                };
                current_encoding = current_enc;
                let pdf_font = PdfFont::new(&font_name, fonts_map.get(&font_name).unwrap().0.as_str(), fonts_map.get(&font_name).unwrap().1.as_str());
                current_text.font = pdf_font;
                current_text.font_size = font_size;

                if !current_text.text.is_empty() {
                    current_text.op = index;
                    collected_chunks_and_errs.push(Ok(current_text.clone()));
                    current_text.text_mut().clear();
                }
            }
            "Tj" | "TJ" => match current_encoding {
                Some(encoding) => {
                    let res = collect_text(current_text.text_mut(), encoding, &operation.operands);
                    if let Err(err) = res {
                        let err = anyhow!("{err:?}");
                        dbg!(&err);
                        collected_chunks_and_errs.push(Err(err));
                    } else {
                        dbg!(&current_text);
                    }
                }
                None => {},
            },
            "ET" => {
                if !current_text.text.ends_with('\n') {
                    current_text.text_mut().push('\n')
                }
                if !current_text.text.is_empty() {
                    current_text.op = index;
                    collected_chunks_and_errs.push(Ok(current_text));
                }
                current_text = PdfText::default();
            }
            _ => {}
        }
        index += 1;
    }
    if !current_text.text.is_empty() {
        current_text.op = index;
        collected_chunks_and_errs.push(Ok(current_text));
    }

    Ok(collected_chunks_and_errs)
}

fn collect_text(text: &mut String, encoding: &Encoding, operands: &[Object]) -> Result<()> {
    for operand in operands.iter() {
        match operand {
            Object::String(bytes, _) => {
                text.push_str(&Document::decode_text(encoding, bytes)?);
            }
            Object::Array(arr) => {
                collect_text(text, encoding, arr)?;
                text.push(' ');
            }
            Object::Integer(i) => {
                if *i < -100 {
                    text.push(' ');
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract all fonts from the document pages
fn extract_fonts_from_page(doc: &Document, page_id: (u32, u16)) -> Result<Vec<(String, String, String)>> {
    let font_refs = doc.get_page_fonts(page_id)?;
    let mut fonts_found = Vec::<(String, String, String)>::new();
    for (font_name, &font_dict) in font_refs.iter() {
        let base_font = font_dict
            .get(b"BaseFont")
            .and_then(|bf| bf.as_name())
            .unwrap_or(b"<Unknown>");
        let subtype = font_dict
            .get(b"Subtype")
            .and_then(|st| st.as_name())
            .unwrap_or(b"<Unknown>");
        fonts_found.push((String::from_utf8_lossy(font_name).to_string(), 
            String::from_utf8_lossy(base_font).to_string(),
            String::from_utf8_lossy(subtype).to_string()),
        );
    }
    Ok(fonts_found)
}

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
    // DM: Change to phymes_schemas::embed::pdfs.rs
    // Extract document metadata

    // Extract the page number and text along with any errors from the documents
    let pages = docs
        .into_par_iter()
        .map(|(id, doc)| {
            doc.get_pages()
                .into_par_iter()
                .map(
                    |(page_num, page_id): (u32, (u32, u16))| -> Result<Vec<(String, PdfText)>, Error> {
                        // Extract text from the page
                        let text = extract_text(doc, &[page_num]).map_err(|e| {
                            Error::other(format!(
                                "Failed to extract text from page {page_num} id={page_id:?}: {e:}"
                            ))
                        })?
                        .into_iter()
                        .map(|pdf_text| (id.to_string(), pdf_text))
                        .collect::<Vec<_>>();
                        std::result::Result::Ok(text)
                    },
                )
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    // Create an ArrowTable to hold the extracted text
    // DM: migrate to phymes_schemas::embed::pdfs.rs 
    let mut document_id_vec = Vec::new();
    let mut chunk_id_vec = Vec::new(); // the page number
    let mut page_num_vec = Vec::new();
    let mut tm_a_vec = Vec::new();
    let mut tm_b_vec = Vec::new();
    let mut tm_c_vec = Vec::new();
    let mut tm_d_vec = Vec::new();
    let mut tm_x_vec = Vec::new();
    let mut tm_y_vec = Vec::new();
    let mut td_x_vec = Vec::new();
    let mut td_y_vec = Vec::new();
    let mut font_name_vec = Vec::new();
    let mut font_subtype_vec = Vec::new();
    let mut base_font_vec = Vec::new();
    let mut font_size_vec = Vec::new();
    let mut text_vec = Vec::new();
    for page in pages {
        match page {
            std::result::Result::Ok(chunks) => {
                for (id, mut pdf_text) in chunks {
                    page_num_vec.push(pdf_text.page_num);
                    tm_a_vec.push(pdf_text.tm.a);
                    tm_b_vec.push(pdf_text.tm.b);
                    tm_c_vec.push(pdf_text.tm.c);
                    tm_d_vec.push(pdf_text.tm.d);
                    tm_x_vec.push(pdf_text.tm.x);
                    tm_y_vec.push(pdf_text.tm.y);
                    td_x_vec.push(pdf_text.td.x);
                    td_y_vec.push(pdf_text.td.y);
                    font_name_vec.push(pdf_text.font.font_name.to_owned());
                    font_subtype_vec.push(pdf_text.font.font_subtype.to_owned());
                    base_font_vec.push(pdf_text.font.base_font.to_owned());
                    font_size_vec.push(pdf_text.font_size);
                    text_vec.push(pdf_text.text.drain(..).as_str().to_string());
                    chunk_id_vec.push(format!("{id}_{pdf_text:?}"));
                    document_id_vec.push(id);
                }
            }
            Err(e) => {
                event!(Level::ERROR, "{e:?}");
            }
        }
    }
    let document_id_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(document_id_vec));
    let chunk_id_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(chunk_id_vec));
    let page_num_arr: ArrayRef = Arc::new(arrow::array::UInt32Array::from(page_num_vec));
    let tm_a_arr: ArrayRef = Arc::new(arrow::array::Float32Array::from(tm_a_vec));
    let tm_b_arr: ArrayRef = Arc::new(arrow::array::Float32Array::from(tm_b_vec));
    let tm_c_arr: ArrayRef = Arc::new(arrow::array::Float32Array::from(tm_c_vec));
    let tm_d_arr: ArrayRef = Arc::new(arrow::array::Float32Array::from(tm_d_vec));
    let tm_x_arr: ArrayRef = Arc::new(arrow::array::Float32Array::from(tm_x_vec));
    let tm_y_arr: ArrayRef = Arc::new(arrow::array::Float32Array::from(tm_y_vec));
    let td_x_arr: ArrayRef = Arc::new(arrow::array::Int64Array::from(td_x_vec));
    let td_y_arr: ArrayRef = Arc::new(arrow::array::Int64Array::from(td_y_vec));
    let font_name_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(font_name_vec));
    let font_subtype_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(font_subtype_vec));
    let base_font_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(base_font_vec));
    let font_size_arr: ArrayRef = Arc::new(arrow::array::Int64Array::from(font_size_vec));
    let text_arr: ArrayRef = Arc::new(arrow::array::StringArray::from(text_vec));
    let batch = RecordBatch::try_from_iter(vec![
        ("chunk_id", chunk_id_arr),
        ("document_id", document_id_arr),
        // ("page_num", page_num_arr),
        // ("tm_a", tm_a_arr),
        // ("tm_b", tm_b_arr),
        // ("tm_c", tm_c_arr),
        // ("tm_d", tm_d_arr),
        // ("tm_x", tm_x_arr),
        // ("tm_y", tm_y_arr),
        // ("td_x", td_x_arr),
        // ("td_y", td_y_arr),
        // ("font_name", font_name_arr),
        // ("font_subtype", font_subtype_arr),
        // ("base_font", base_font_arr),
        // ("font_size", font_size_arr),
        ("text", text_arr),
    ])?;
    // dbg!(&batch);
    Ok(batch)
}

static IGNORE_TYPE_NAMES: &[&[u8]] = &[
    // b"Length",
    // b"BBox",
    b"FormType",
    // b"Matrix",
    // b"Type",
    // b"XObject",
    // b"Subtype",
    // b"Filter",
    // b"ColorSpace",
    // b"Width",
    // b"Height",
    // b"BitsPerComponent",
    // b"Length1",
    // b"Length2",
    // b"Length3",
    // b"PTEX.FileName",
    // b"PTEX.PageNumber",
    // b"PTEX.InfoDict",
    // b"FontDescriptor",
    // b"ExtGState",
    // b"MediaBox",
    // b"Annot",
];

static IGNORE_KEYS: &[&[u8]] = &[
    b"Producer",
    // b"ModDate",
    // b"Creator",
    // b"ProcSet",
    // b"XObject",
    // b"MediaBox",
    // b"Annots",
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
                    std::result::Result::Ok(dict) => dict
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
                    if let std::result::Result::Ok(dict_mut) = object.as_dict_mut() {
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
        std::result::Result::Ok(document) => Ok(document),
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
                .filter_map(|s| {
                    let bytes = s
                        .unwrap()
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap())
                        .collect::<Vec<u8>>();
                    match load_pdf_document(&bytes) {
                        std::result::Result::Ok(doc) => Some(filter_pdf(doc)),
                        Err(_err) => None
                    }
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
    use phymes_subject::{
        BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };

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
            ["doc_1_PdfText { op: 4, bt: 0, tm: Tm { a: 1.0, b: 0.0, c: 0.0, d: 1.0, x: 0.0, y: 0.0 }, td: Td { x: 100, y: 600 }, font: PdfFont { font_name: \"F1\", font_subtype: \"Courier\", base_font: \"Type1\" }, font_size: 48, page_num: 1, text: \"\" }", "doc_1_PdfText { op: 4, bt: 0, tm: Tm { a: 1.0, b: 0.0, c: 0.0, d: 1.0, x: 0.0, y: 0.0 }, td: Td { x: 100, y: 600 }, font: PdfFont { font_name: \"F1\", font_subtype: \"Courier\", base_font: \"Type1\" }, font_size: 48, page_num: 2, text: \"\" }", "doc_2_PdfText { op: 4, bt: 0, tm: Tm { a: 1.0, b: 0.0, c: 0.0, d: 1.0, x: 0.0, y: 0.0 }, td: Td { x: 100, y: 600 }, font: PdfFont { font_name: \"F1\", font_subtype: \"Courier\", base_font: \"Type1\" }, font_size: 48, page_num: 1, text: \"\" }", "doc_2_PdfText { op: 4, bt: 0, tm: Tm { a: 1.0, b: 0.0, c: 0.0, d: 1.0, x: 0.0, y: 0.0 }, td: Td { x: 100, y: 600 }, font: PdfFont { font_name: \"F1\", font_subtype: \"Courier\", base_font: \"Type1\" }, font_size: 48, page_num: 2, text: \"\" }"]        );
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

    #[test]
    fn test_extract_pdf_extract_fonts() {
        // Create a dummy PDF in memory
        let doc_1 = filter_pdf(make_pdf_document(&["1\n2\n3", "4\n5\n6"]));

        // Extract Fonts        
        let fonts = doc_1.get_pages()
            .into_iter()
            .map(|(_page_num, page_id): (u32, (u32, u16))| -> Result<Vec<(String, String, String)>> {
                extract_fonts_from_page(&doc_1, page_id)
            })
            .collect::<Vec<_>>();
        assert_eq!(fonts.len(), 2);
        assert_eq!(fonts.first().as_ref().unwrap().as_ref().unwrap().first().unwrap().0, "F1");
        assert_eq!(fonts.first().as_ref().unwrap().as_ref().unwrap().first().unwrap().1, "Courier");
        assert_eq!(fonts.first().as_ref().unwrap().as_ref().unwrap().first().unwrap().2, "Type1");
        assert_eq!(fonts.get(1).as_ref().unwrap().as_ref().unwrap().first().unwrap().0, "F1");
        assert_eq!(fonts.get(1).as_ref().unwrap().as_ref().unwrap().first().unwrap().1, "Courier");
        assert_eq!(fonts.get(1).as_ref().unwrap().as_ref().unwrap().first().unwrap().2, "Type1");
    }
}
