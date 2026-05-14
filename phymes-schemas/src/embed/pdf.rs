use arrow::datatypes::{DataType, Field};

use crate::DataEncoding;

/// Root PDF document container
pub struct PdfDocument {
    /// Document ID, PK
    doc_id: u32, 
    version: String,
    creation_date: i64,
    modification_date: i64,
    author: String,
    title: String,
}

/// Each PDF document has one or more pages
pub struct PdfPage {
    /// Page ID, PK
    page_id: u32,
    /// Document ID, FK
    doc_id: u32,
    page_number: u32,
    width: f32,
    height: f32, 
    units: String,
}

pub enum PdfObjectType {
    Dictionary,
    Stream,
    Array,
    String,
    Number,
    Boolean,
    Null
}

/// PDFs are build from "Indirect Objects" including Dictionaries, Streams, etc.)
pub struct PdfObject {
    /// Object ID, PK
    obj_id: u32,
    /// Document ID, FK
    doc_id: u32,
    object_number: u32,
    generation_number: u32,
    object_type: PdfObjectType,
    raw_content: String
}

/// Content Streams are the actual text and drawing instructions for the page
pub struct PdfStream {
    /// Stream ID, PK, AutoIncrement
    stream_id: u32,
    /// Object ID, FK
    obj_id: u32,
    length: u32,
    compression: DataEncoding,
    stream_data: Vec<u8>,
}

/// Resource dictionary
pub struct PdfResource {}

/// The decompressed and parsed stream for quering against
pub struct PdfContent {
    /// Content Step, PK along with Operator step
    content_step: u32,
    /// Operator Step, PK along with Content step
    operator_step: u32,
    /// Stream ID, FK
    stream_id: u32,
    operator_type: String,
    operand: serde_json::Value,
}


/// PDF Text Matrix (PdfTm) operator
#[derive(Debug, Clone, PartialEq)]
struct PdfTm {
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

impl PdfTm {
    pub fn new(a: &f32, b: &f32, c: &f32, d: &f32, x: &f32, y: &f32) -> Self {
        Self { a: *a, b: *b, c: *c, d: *d, x: *x, y: *y }
    }
}

impl Default for PdfTm {
    fn default() -> Self {
        Self { a: 1_f32, b: 0_f32, c: 0_f32, d: 1_f32, x: 0_f32, y: 0_f32 }
    }
}

fn create_pdf_tm_fields() -> Vec<Field> {
    let field_names = ["tm_a", "tm_b", "tm_c", "tm_d", "tm_x", "tm_y"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Float32, false))
        .collect::<Vec<_>>();
    fields_vec
}

/// PdfTm
#[derive(Debug, Clone, PartialEq)]
struct PdfTd {
    /// pos x
    x: i64,
    /// pos y
    y: i64,
}

impl PdfTd {
    pub fn new(x: &i64, y: &i64) -> Self {
        Self { x: *x, y: *y }
    }
}

impl Default for PdfTd {
    fn default() -> Self {
        Self { x: 0_i64, y: 0_i64 }
    }
}

fn create_pdf_td_fields() -> Vec<Field> {
    let field_names = ["td_x", "td_y"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>();
    fields_vec
}

/// PDF Font information
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

fn create_pdf_font_fields() -> Vec<Field> {
    let field_names = ["font_name", "font_subtype", "base_font"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec
}

#[derive(Default, Debug, Clone, PartialEq)]
struct PdfText {
    /// Index of the operataion the text was found
    pub op: u32,
    /// BT operataion the text was found
    pub bt: u32,
    /// Text matrix
    pub tm: PdfTm,
    /// Text translation
    pub td: PdfTd,
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

fn create_pdf_text_fields() -> Vec<Field> {
    let field_names = ["font_name", "font_subtype", "base_font"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec
}

fn create_pdf_manuscript_fields() -> Vec<Field> {
    let field_names = ["document", "section", "text"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["document", "page", "paragraph", "sentence"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    fields_vec
}