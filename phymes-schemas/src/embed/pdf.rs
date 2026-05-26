use anyhow::Result;
use arrow::{array::RecordBatch, datatypes::{DataType, Field, Fields, SchemaRef}};
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait};
use serde::{Deserialize, Serialize};

use crate::{AvailableSchemaTrait, DataFormat, JsonSchemaTrait, create_route_bytes_record_batch, create_schema_from_fields, embed::documents::create_documents_fields_vec};

/// PDF Text Matrix (PdfTm) operator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub fn as_hash(&self) -> String {
        let mut cpy = self.clone();
        cpy.text.clear();
        format!("{cpy:?}")
    }
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
            tm_a: self.tm.a,
            tm_b: self.tm.b,
            tm_c: self.tm.c,
            tm_d: self.tm.d,
            tm_x: self.tm.x,
            tm_y: self.tm.y,
            td_x: self.td.x,
            td_y: self.td.y,
            font_name: self.font.font_name,
            font_subtype: self.font.font_subtype,
            base_font: self.font.base_font,
            font_size: self.font_size, 
            text: self.text 
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdfTextSubject {
    pub document_id: String,
    pub chunk_id: String,
    pub page_number: u32,
    pub op: u32,
    pub bt: u32,
    pub tm_a: f32,
    pub tm_b: f32,
    pub tm_c: f32,
    pub tm_d: f32,
    pub tm_x: f32,
    pub tm_y: f32,
    pub td_x: i64,
    pub td_y: i64,
    pub font_name: String,
    pub font_subtype: String,
    pub base_font: String,
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

// fn create_pdf_manuscript_fields() -> Vec<Field> {
//     let field_names = ["document", "section", "text"];
//     let mut fields_vec = field_names
//         .iter()
//         .map(|f| Field::new(*f, DataType::Utf8, false))
//         .collect::<Vec<_>>();
//     let field_names = ["document", "page", "paragraph", "sentence"];
//     fields_vec.extend(field_names
//         .iter()
//         .map(|f| Field::new(*f, DataType::UInt32, false))
//         .collect::<Vec<_>>());
//     fields_vec
// }

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfGraphics {
    // TODO
}

impl PdfGraphics {
    pub fn as_hash(&self) -> String {
        let mut cpy = self.clone();
        format!("{cpy:?}")
    }
    pub fn build_pdf_graphics_subject(self, document_id: &str, chunk_id: &str, page_number: &u32) -> PdfGraphicsSubject {
        PdfGraphicsSubject { document_id: document_id.to_string(), chunk_id: chunk_id.to_string(), page_number: *page_number }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub fn new(page_number: &u32, width: &f32, height: &f32, units: &str, text: &[PdfText], graphics: &[PdfGraphics]) -> Self {
        Self { page_number: *page_number, width: *width, height: *height, units: units.to_string(), text: text.to_vec(), graphics: graphics.to_vec() }
    }
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
                let chunk_id = PdfPage::make_chunk_id(document_id, &page_number, &s.as_hash());
                s.build_pdf_text_subject(document_id, &chunk_id, &page_number)
            })
            .collect::<Vec<_>>();
        let pdf_graphics_subjects = graphics.into_iter()
            .map(|s| {
                let chunk_id = PdfPage::make_chunk_id(document_id, &page_number, &s.as_hash());
                s.build_pdf_graphics_subject(document_id, &chunk_id, &page_number)
            })
            .collect::<Vec<_>>();
        (pdf_page_subject, pdf_text_subjects, pdf_graphics_subjects)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfDocument {
    /// Document ID, PK
    pub document_id: String,
    pub version: String,
    pub creation_date: i64,
    pub modification_date: i64,
    pub author: String,
    pub title: String,
    pub pages: Vec<PdfPage>,
}

impl PdfDocument {
    pub fn new(document_id: &str, version: &str, creation_date: &i64, modification_date: &i64, author: &str, title: &str, pages: &[PdfPage]) -> PdfDocument {
        Self { document_id: document_id.to_string(), version: version.to_string(), creation_date: *creation_date, modification_date: *modification_date, author: author.to_string(), title: title.to_string(), pages: pages.to_vec() }
    }
    pub fn build_pdf_document_subject(self) -> (PdfDocumentSubject, Vec<PdfPageSubject>, Vec<PdfTextSubject>, Vec<PdfGraphicsSubject>) {
        let document_id = self.document_id.clone();
        let pdf_document_subject = PdfDocumentSubject {
            document_id: self.document_id.clone(),
            version: self.version.clone(),
            creation_date: self.creation_date,
            modification_date: self.modification_date,
            author: self.author.clone(),
            title: self.title.clone(),
        };
        let ((pdf_page_subject, pdf_text_subject), pdf_graphics_subject): ((Vec<PdfPageSubject>, Vec<Vec<PdfTextSubject>>), Vec<Vec<PdfGraphicsSubject>>) = self.pages
            .into_iter()
            .map(|p| {
                let (page, text, graphics) = p.build_pdf_page_subject(&document_id);
                ((page, text), graphics)
            })
            .unzip();
        let pdf_text_subject = pdf_text_subject.into_iter().flatten().collect();
        let pdf_graphics_subject = pdf_graphics_subject.into_iter().flatten().collect();
        (pdf_document_subject, pdf_page_subject, pdf_text_subject, pdf_graphics_subject)
    }
}

/// Root PDF document container
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdfDocumentSubject {
    /// Document ID, PK
    pub document_id: String,
    pub version: String,
    pub creation_date: i64,
    pub modification_date: i64,
    pub author: String,
    pub title: String,
}

impl PdfDocumentSubject {
    fn to_fields() -> Fields {
        Fields::from_iter(create_pdf_document_fields_vec())
    }
}

impl MappableTrait for PdfDocumentSubject {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PdfDocumentSubject {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

fn create_pdf_document_fields_vec() -> Vec<Field> {
    let field_names = ["document_id", "version", "author", "title"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["creation_date", "modification_date"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>());
    fields_vec
}


/// Root PDF document container
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PdfDocumentsResponse {
    pub results: Vec<PdfDocument>,
}

impl PdfDocumentsResponse {
    pub fn new(documents: &[PdfDocument]) -> Self {
        Self { results: documents.to_vec() }
    }
}

impl JsonSchemaTrait for PdfDocumentsResponse {
    /// Parse the OpenAlexResponseWorks object into tables following the `create_ipc_fields` schema
    ///   where each row is routed to a different table
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        let mut docs_subjects = Vec::new();
        let mut pages_subjects = Vec::new();
        let mut texts_subjects = Vec::new();
        let mut graphics_subjects = Vec::new();
        for result in self.results {
            // Parse into individual subjects
            let (
                docs_subject,
                pages_subject,
                texts_subject,
                graphics_subject,
            ) = result.build_pdf_document_subject();

            // Handle each individual subjects
            docs_subjects.push(docs_subject);
            pages_subjects.extend(pages_subject);
            texts_subjects.extend(texts_subject);
            graphics_subjects.extend(graphics_subject);
        }

        // Wrap into IPC [RecordBatch]
        let mut names = Vec::new();
        let mut publishers = Vec::new();
        let mut subjects = Vec::new();
        let mut formats = Vec::new();
        let mut bytes = Vec::new();

        // Handle each individual subject
        if !docs_subjects.is_empty() {
            names.push(docs_subjects.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(docs_subjects.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(
                Subject::get_builder()
                    .with_name(docs_subjects.first().unwrap().get_name())
                    .with_schema(docs_subjects.first().unwrap().to_schema())
                    .with_struct::<PdfDocumentSubject>(&docs_subjects)?
                    .build()?
                    .to_ipc_stream()?,
            );
        }
        if !pages_subjects.is_empty() {
            names.push(pages_subjects.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(pages_subjects.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(
                Subject::get_builder()
                    .with_name(pages_subjects.first().unwrap().get_name())
                    .with_schema(pages_subjects.first().unwrap().to_schema())
                    .with_struct::<PdfPageSubject>(&pages_subjects)?
                    .build()?
                    .to_ipc_stream()?,
            );
        }
        if !texts_subjects.is_empty() {
            names.push(texts_subjects.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(texts_subjects.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(
                Subject::get_builder()
                    .with_name(texts_subjects.first().unwrap().get_name())
                    .with_schema(texts_subjects.first().unwrap().to_schema())
                    .with_struct::<PdfTextSubject>(&texts_subjects)?
                    .build()?
                    .to_ipc_stream()?,
            );
        }
        if !graphics_subjects.is_empty() {
            names.push(graphics_subjects.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(graphics_subjects.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(
                Subject::get_builder()
                    .with_name(graphics_subjects.first().unwrap().get_name())
                    .with_schema(graphics_subjects.first().unwrap().to_schema())
                    .with_struct::<PdfGraphicsSubject>(&graphics_subjects)?
                    .build()?
                    .to_ipc_stream()?,
            );
        }
        
        create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)
    }
}

// pub enum PdfObjectType {
//     Dictionary,
//     Stream,
//     Array,
//     String,
//     Number,
//     Boolean,
//     Null
// }

// /// PDFs are build from "Indirect Objects" including Dictionaries, Streams, etc.)
// pub struct PdfObject {
//     /// Object ID, PK
//     obj_id: u32,
//     /// Document ID, FK
//     document_id: u32,
//     object_number: u32,
//     generation_number: u32,
//     object_type: PdfObjectType,
//     raw_content: String
// }

// /// Content Streams are the actual text and drawing instructions for the page
// pub struct PdfStream {
//     /// Stream ID, PK, AutoIncrement
//     stream_id: u32,
//     /// Object ID, FK
//     obj_id: u32,
//     length: u32,
//     compression: DataEncoding,
//     stream_data: Vec<u8>,
// }

// /// Resource dictionary
// pub struct PdfResource {}

// /// The decompressed and parsed stream for quering against
// pub struct PdfContent {
//     /// Content Step, PK along with Operator step
//     content_step: u32,
//     /// Operator Step, PK along with Content step
//     operator_step: u32,
//     /// Stream ID, FK
//     stream_id: u32,
//     operator_type: String,
//     operand: serde_json::Value,
// }