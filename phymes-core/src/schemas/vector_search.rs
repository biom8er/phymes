use serde::{Deserialize, Serialize};

#[derive(Default, Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Documents {
    document_id: String,
    chunk_id: u32,
    text: String,
    embeddings: Vec<Vec<f32>>,
}

#[derive(Default, Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Queries {
    query_id: String,
    text: String,
    embeddings: Vec<Vec<f32>>,
}
