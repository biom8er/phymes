use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub fn create_mermaid_fields() -> Fields {
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
pub struct MermaidSubject {
    pub session_context_name: String, 
    pub flowchart_diagram: String, 
    pub er_diagram: String,
    pub timestamp: i64,
}

pub fn create_mermaid_batch(
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

pub fn create_mermaid_content_template_fields() -> Fields {
    let field_names = ["content"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_gantt_template_fields() -> Fields {
    let field_names = ["section", "task", "start", "end"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_flowchart_nodes_template_fields() -> Fields {
    let field_names = ["node_name", "node_shape", "node_label"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_flowchart_links_template_fields() -> Fields {
    let field_names = ["subject_name", "object_name", "link_type", "link_text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_sequence_diagram_participants_template_fields() -> Fields {
    let field_names = ["participant_name", "participant_type"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_sequence_diagram_messages_template_fields() -> Fields {
    let field_names = ["subject_name", "object_name", "message_type", "activation_type", "message_content", "note_content", "note_location"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}