use std::sync::Arc;

use arrow::{array::{ArrayRef, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use phymes_diagnostics::create_timestamp_str;
use serde::{Deserialize, Serialize};

use crate::table::TableBuilder;

pub(crate) fn create_queries_fields() -> Fields {
    let field_names = ["query_id", "text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_queries_batch(
    query_ids: Vec<String>,
    text: Vec<String>,
) -> Result<RecordBatch> {
    let query_ids: ArrayRef = Arc::new(StringArray::from(query_ids));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let batch = RecordBatch::try_from_iter(vec![
        ("query_id", query_ids),
        ("text", text),
    ])?;
    Ok(batch)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueriesSubject {
    pub query_id: String,
    pub text: String,
}

pub trait QueriesBuilderTraitExt: Sized {
    fn with_text(self, text: &str) -> Result<Self>;
}

impl QueriesBuilderTraitExt for TableBuilder {
    fn with_text(mut self, text: &str) -> Result<Self> {
        // Handle the query text
        // DM: note that the prompt for the query is specific to Qwen!
        let content = if cfg!(feature = "hf_hub") {
            format!(
                "{}{}",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                text
            )
        } else {
            text.to_string()
        };

        // Add the record batch to the table
        let batch = create_queries_batch(
            vec![create_timestamp_str()],
            vec![content],
        )?;
        match self.record_batches {
            Some(ref mut batches) => {
                batches.push(batch);
                Ok(self)
            }
            None => {
                self.schema = Some(batch.schema());
                self.record_batches = Some(vec![batch]);
                Ok(self)
            }
        }        
    }
}