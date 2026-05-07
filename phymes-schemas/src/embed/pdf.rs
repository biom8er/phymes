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

/// Font dictionary
pub struct PdfFont {}

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