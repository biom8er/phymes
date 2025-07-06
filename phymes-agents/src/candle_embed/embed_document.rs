use serde::{Deserialize, Serialize};

/// Document schema
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    document_id: String,
    document_text: String,
    document_metadata: serde_json::Value,
}

impl Document {
    pub fn new(id: String, text: String, metadata: serde_json::Value) -> Self {
        Document {
            document_id: id,
            document_text: text,
            document_metadata: metadata,
        }
    }
}

/// Document chunk schema
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentChunk {
    document_id: String,
    chunk_id: String,
    chunk_text: String,
    chunk_embedding: Vec<f32>,
}
