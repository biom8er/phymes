use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Fields},
};
use serde::{Deserialize, Serialize};

pub(crate) fn create_session_mermaid_fields() -> Fields {
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
pub struct SessionMermaidSubject {
    pub session_context_name: String,
    pub flowchart_diagram: String,
    pub er_diagram: String,
    pub timestamp: i64,
}

pub fn create_session_mermaid_batch(
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

pub(crate) fn create_mermaid_visualization_fields() -> Fields {
    let field_names = ["visualization_name", "visualization_string"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["timestamp"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_content_template_fields() -> Fields {
    let field_names = ["content"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_content_template_batch(content: Vec<String>) -> Result<RecordBatch> {
    let content: ArrayRef = Arc::new(StringArray::from(content));
    let batch = RecordBatch::try_from_iter(vec![("content", content)])?;
    Ok(batch)
}

pub(crate) fn create_mermaid_gantt_template_fields() -> Fields {
    let field_names = ["section", "task", "start", "end"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_flowchart_nodes_template_fields() -> Fields {
    let field_names = ["node_name", "node_shape", "node_label"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_flowchart_links_template_fields() -> Fields {
    let field_names = ["subject_name", "object_name", "link_type", "link_text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_sequence_diagram_participants_template_fields() -> Fields {
    let field_names = ["participant_name", "participant_type"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_mermaid_sequence_diagram_participants_template_batch(
    participant_name: Vec<String>,
    participant_type: Vec<String>,
) -> Result<RecordBatch> {
    let participant_name: ArrayRef = Arc::new(StringArray::from(participant_name));
    let participant_type: ArrayRef = Arc::new(StringArray::from(participant_type));
    let batch = RecordBatch::try_from_iter(vec![
        ("participant_name", participant_name),
        ("participant_type", participant_type),
    ])?;
    Ok(batch)
}

pub(crate) fn create_mermaid_sequence_diagram_messages_template_fields() -> Fields {
    let field_names = [
        "subject_name",
        "object_name",
        "message_type",
        "activation_type",
        "message_content",
        "note_content",
        "note_location",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

#[allow(dead_code)]
pub fn create_mermaid_sequence_diagram_messages_template_batch(
    subject_name: Vec<String>,
    object_name: Vec<String>,
    message_type: Vec<String>,
    activation_type: Vec<String>,
    message_content: Vec<String>,
    note_content: Vec<String>,
    note_location: Vec<String>,
) -> Result<RecordBatch> {
    let subject_name: ArrayRef = Arc::new(StringArray::from(subject_name));
    let object_name: ArrayRef = Arc::new(StringArray::from(object_name));
    let message_type: ArrayRef = Arc::new(StringArray::from(message_type));
    let activation_type: ArrayRef = Arc::new(StringArray::from(activation_type));
    let message_content: ArrayRef = Arc::new(StringArray::from(message_content));
    let note_content: ArrayRef = Arc::new(StringArray::from(note_content));
    let note_location: ArrayRef = Arc::new(StringArray::from(note_location));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_name),
        ("object_name", object_name),
        ("message_type", message_type),
        ("activation_type", activation_type),
        ("message_content", message_content),
        ("note_content", note_content),
        ("note_location", note_location),
    ])?;
    Ok(batch)
}

pub(crate) fn create_mermaid_kanban_template_fields() -> Fields {
    let field_names = [
        "column_name",
        "column_label",
        "task_name",
        "task_description",
        "task_assigned",
        "task_ticket",
        "task_priority",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_xychart_template_fields() -> Fields {
    let fields_vec = vec![
        Field::new("x", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ];
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_er_diagram_entities_template_fields() -> Fields {
    let field_names = [
        "entity_name",
        "entity_alias",
        "attribute_name",
        "attribute_type",
        "attribute_key",
        "attribute_comment",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub(crate) fn create_mermaid_er_diagram_relations_template_fields() -> Fields {
    let field_names = [
        "subject_name",
        "object_name",
        "relation_type",
        "relation_content",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}
