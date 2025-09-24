use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{schemas::available_subjects::create_timestamp_micros, table::table::TableBuilder};

pub fn create_builder_fields() -> Fields {
    let session_context_name = Field::new("session_context_name", DataType::Utf8, false);
    let flowchart_diagram = Field::new("flowchart_diagram", DataType::Utf8, false);
    let er_diagram = Field::new("er_diagram", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    Fields::from(vec![
        session_context_name,
        flowchart_diagram,
        er_diagram,
        timestamp,
    ])
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BuilderSubject {
    pub session_context_name: String, 
    pub flowchart_diagram: String, 
    pub er_diagram: String,
    pub timestamp: i64,
}

pub fn create_builder_batch(
    session_context_name: Vec<String>,
    flowchart_diagram: Vec<String>,
    er_diagram: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let session_context_name: ArrayRef = Arc::new(StringArray::from(session_context_name));
    let flowchart_diagram: ArrayRef = Arc::new(StringArray::from(flowchart_diagram));
    let er_diagram: ArrayRef = Arc::new(StringArray::from(er_diagram));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_context_name", session_context_name),
        ("flowchart_diagram", flowchart_diagram),
        ("er_diagram", er_diagram),
        ("timestamp", timestamp),
    ])?;
    Ok(batch)
}

pub trait BuilderBuilderTraitExt: Sized {
    fn with_builder(self, session_context_name: Option<&str>, flowchart_diagram: Option<&str>, er_diagram: Option<&str>) -> Result<Self>;
}

impl BuilderBuilderTraitExt for TableBuilder {
    fn with_builder(mut self, session_context_name: Option<&str>, flowchart_diagram: Option<&str>, er_diagram: Option<&str>) -> Result<Self> {
        // Handle the er_diagram
        let session_context_name = session_context_name.unwrap_or_default().to_string();
        let flowchart_diagram = flowchart_diagram.unwrap_or_default().to_string();
        let er_diagram = er_diagram.unwrap_or_default().to_string();

        // Add the record batch to the table
        let batch = create_builder_batch(
            vec![session_context_name],
            vec![flowchart_diagram],
            vec![er_diagram],
            vec![create_timestamp_micros()],
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