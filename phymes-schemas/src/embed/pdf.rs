use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use phymes_subject::MappableTrait;

use crate::{AvailableSchemaTrait, DataEncoding, create_schema_from_fields, embed::documents::create_documents_fields_vec};

/// PDF Text Matrix (PdfTm) operator
#[derive(Debug, Clone, PartialEq)]
pub struct PdfTm {
    /// scale param 1
    pub a: f32,
    /// skew param 1
    pub b: f32,
    /// skew param 2
    pub c: f32,
    /// scale param 2
    pub d: f32,
    /// pos x
    pub x: f32,
    /// pos y
    pub y: f32,
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

fn create_pdf_tm_fields_vec() -> Vec<Field> {
    let field_names = ["tm_a", "tm_b", "tm_c", "tm_d", "tm_x", "tm_y"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Float32, false))
        .collect::<Vec<_>>();
    fields_vec
}

/// PdfTm
#[derive(Debug, Clone, PartialEq)]
pub struct PdfTd {
    /// pos x
    pub x: i64,
    /// pos y
    pub y: i64,
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

fn create_pdf_td_fields_vec() -> Vec<Field> {
    let field_names = ["td_x", "td_y"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>();
    fields_vec
}

/// PDF Font information
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfFont {
    pub font_name: String,
    pub font_subtype: String,
    pub base_font: String,
}

impl PdfFont {
    pub fn new(font_name: &str, font_subtype: &str, base_font: &str, ) -> Self {
        Self { font_name: font_name.to_string(), font_subtype: font_subtype.to_string(), base_font: base_font.to_string() }
    }
}

fn create_pdf_font_fields_vec() -> Vec<Field> {
    let field_names = ["font_name", "font_subtype", "base_font"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfText {
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
    pub text: String,
}

impl PdfText {
    pub fn text_mut(&mut self) -> &mut String {
        &mut self.text
    }
    pub fn build_pdf_text_subject(self, document_id: &str, chunk_id: &str, page_number: &u32) -> PdfTextSubject {
        PdfTextSubject { 
            document_id: document_id.to_string(), 
            chunk_id: chunk_id.to_string(), 
            page_number: *page_number, 
            op: self.op, 
            bt: self.bt, 
            tm: self.tm, 
            td: self.td, 
            font: self.font, 
            font_size: self.font_size, 
            text: self.text 
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfTextSubject {
    pub document_id: String,
    pub chunk_id: String,
    pub page_number: u32,
    pub op: u32,
    pub bt: u32,
    pub tm: PdfTm,
    pub td: PdfTd,
    pub font: PdfFont,
    pub font_size: i64,
    pub text: String,
}

impl PdfTextSubject {
    fn to_fields() -> Fields {
        Fields::from_iter(create_pdf_text_fields_vec())
    }
}

impl MappableTrait for PdfTextSubject {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PdfTextSubject {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

fn create_pdf_text_fields_vec() -> Vec<Field> {
    let mut fields_vec = create_documents_fields_vec();
    let field_names = ["op", "bt", "page_number"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    let field_names = ["font_size"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>());
    fields_vec.extend(create_pdf_tm_fields_vec());
    fields_vec.extend(create_pdf_td_fields_vec());
    fields_vec.extend(create_pdf_font_fields_vec());
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

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfGraphics {
    // TODO
}

impl PdfGraphics {
    pub fn build_pdf_graphics_subject(self, document_id: &str, chunk_id: &str, page_number: &u32) -> PdfGraphicsSubject {
        PdfGraphicsSubject { document_id: document_id.to_string(), chunk_id: chunk_id.to_string(), page_number: *page_number }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfGraphicsSubject {
    pub document_id: String,
    pub chunk_id: String,
    pub page_number: u32,
    // TODO
}

impl PdfGraphicsSubject {
    fn to_fields() -> Fields {
        Fields::from_iter(create_pdf_graphics_fields_vec())
    }
}

impl MappableTrait for PdfGraphicsSubject {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PdfGraphicsSubject {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

fn create_pdf_graphics_fields_vec() -> Vec<Field> {
    let field_names = ["document_id", "chunk_id"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["page_number"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    // TODO
    fields_vec
}

/// Each PDF document has one or more pages
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfPage {
    pub page_number: u32,
    pub width: f32,
    pub height: f32, 
    pub units: String,
    pub text: Vec<PdfText>,
    pub graphics: Vec<PdfGraphics>,
}

impl PdfPage {
    /// Chunk ID to uniquely identify the text or graphic
    fn make_chunk_id(document_id: &str, page_number: &u32, unique_tag: &str) -> String {
        format!("{document_id}{page_number}{unique_tag}")
    }
    pub fn build_pdf_page_subject(self, document_id: &str) -> (PdfPageSubject, Vec<PdfTextSubject>, Vec<PdfGraphicsSubject>) {
        let page_number = self.page_number;
        let pdf_page_subject = PdfPageSubject { 
            document_id: document_id.to_string(),
            page_number: page_number,
            height: self.height,
            width: self.width,
            units: self.units.clone(),
        };
        let (text, graphics) = (self.text, self.graphics);
        let pdf_text_subjects = text.into_iter()
            .map(|s| {
                let chunk_id = PdfPage::make_chunk_id(document_id, &page_number, "");
                s.build_pdf_text_subject(document_id, &chunk_id, &page_number)
            })
            .collect::<Vec<_>>();
        let pdf_graphics_subjects = graphics.into_iter()
            .map(|s| {
                let chunk_id = PdfPage::make_chunk_id(document_id, &page_number, "");
                s.build_pdf_graphics_subject(document_id, &chunk_id, &page_number)
            })
            .collect::<Vec<_>>();
        (pdf_page_subject, pdf_text_subjects, pdf_graphics_subjects)
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfPageSubject {
    /// Document ID, FK
    pub document_id: String,
    /// Document ID and Page Number, PK
    pub page_number: u32,
    pub width: f32,
    pub height: f32, 
    pub units: String,
}

impl PdfPageSubject {
    fn to_fields() -> Fields {
        Fields::from_iter(create_pdf_pages_fields_vec())
    }
}

impl MappableTrait for PdfPageSubject {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PdfPageSubject {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

fn create_pdf_pages_fields_vec() -> Vec<Field> {
    let field_names = ["document_id", "units"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["page_number"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    let field_names = ["width", "height"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Float32, false))
        .collect::<Vec<_>>());
    fields_vec
}

/// Root PDF document container
pub struct PdfDocument {
    /// Document ID, PK
    pub document_id: u32, 
    pub version: String,
    pub creation_date: i64,
    pub modification_date: i64,
    pub author: String,
    pub title: String,
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
    document_id: u32,
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