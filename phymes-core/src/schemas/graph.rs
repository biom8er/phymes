use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt32Array},
    datatypes::{DataType, Field, Fields},
};

pub(crate) fn create_parse_xml_fields() -> Fields {
    let field_names = [
        "document_id",
        "element_tag",
        "element_attr",
        "text",
        "child_tag",
        "child_attr",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["element_index", "child_index"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

#[allow(clippy::too_many_arguments)]
pub fn create_parse_xml_batch(
    document_id: Vec<String>,
    element_tag: Vec<String>,
    element_attr: Vec<String>,
    text: Vec<String>,
    child_tag: Vec<String>,
    child_attr: Vec<String>,
    element_index: Vec<u32>,
    child_index: Vec<u32>,
) -> Result<RecordBatch> {
    let document_id: ArrayRef = Arc::new(StringArray::from(document_id));
    let element_tag: ArrayRef = Arc::new(StringArray::from(element_tag));
    let element_attr: ArrayRef = Arc::new(StringArray::from(element_attr));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let child_tag: ArrayRef = Arc::new(StringArray::from(child_tag));
    let child_attr: ArrayRef = Arc::new(StringArray::from(child_attr));
    let element_index: ArrayRef = Arc::new(UInt32Array::from(element_index));
    let child_index: ArrayRef = Arc::new(UInt32Array::from(child_index));
    let batch = RecordBatch::try_from_iter(vec![
        ("document_id", document_id),
        ("element_tag", element_tag),
        ("element_attr", element_attr),
        ("text", text),
        ("child_tag", child_tag),
        ("child_attr", child_attr),
        ("element_index", element_index),
        ("child_index", child_index),
    ])?;
    Ok(batch)
}

fn create_n_triples_vec_fields() -> Vec<Field> {
    let field_names = ["subject", "predicate", "object"];
    field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>()
}

pub(crate) fn create_n_triples_fields() -> Fields {
    Fields::from(create_n_triples_vec_fields())
}

pub fn create_n_triples_batch(
    subject: Vec<String>,
    predicate: Vec<String>,
    object: Vec<String>,
) -> Result<RecordBatch> {
    let subject: ArrayRef = Arc::new(StringArray::from(subject));
    let predicate: ArrayRef = Arc::new(StringArray::from(predicate));
    let object: ArrayRef = Arc::new(StringArray::from(object));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject", subject),
        ("predicate", predicate),
        ("object", object),
    ])?;
    Ok(batch)
}

fn create_n_quads_vec_fields() -> Vec<Field> {
    let field_names = ["graph"];
    create_n_triples_vec_fields()
        .into_iter()
        .chain(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Utf8, false))
                .collect::<Vec<_>>(),
        )
        .collect::<Vec<_>>()
}

pub(crate) fn create_n_quads_fields() -> Fields {
    Fields::from(create_n_quads_vec_fields())
}

pub fn create_n_quads_batch(
    subject: Vec<String>,
    predicate: Vec<String>,
    object: Vec<String>,
    graph: Vec<String>,
) -> Result<RecordBatch> {
    let subject: ArrayRef = Arc::new(StringArray::from(subject));
    let predicate: ArrayRef = Arc::new(StringArray::from(predicate));
    let object: ArrayRef = Arc::new(StringArray::from(object));
    let graph: ArrayRef = Arc::new(StringArray::from(graph));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject", subject),
        ("predicate", predicate),
        ("object", object),
        ("graph", graph),
    ])?;
    Ok(batch)
}

fn create_parse_n_quads_vec_fields() -> Vec<Field> {
    let field_names = ["dataset"];
    create_n_quads_vec_fields()
        .into_iter()
        .chain(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Utf8, false))
                .collect::<Vec<_>>(),
        )
        .collect::<Vec<_>>()
}

#[allow(dead_code)]
pub(crate) fn create_parse_n_quads_fields() -> Fields {
    Fields::from(create_parse_n_quads_vec_fields())
}

pub fn create_parse_n_quads_batch(
    subject: Vec<String>,
    predicate: Vec<String>,
    object: Vec<String>,
    graph: Vec<String>,
    dataset: Vec<String>,
) -> Result<RecordBatch> {
    let subject: ArrayRef = Arc::new(StringArray::from(subject));
    let predicate: ArrayRef = Arc::new(StringArray::from(predicate));
    let object: ArrayRef = Arc::new(StringArray::from(object));
    let graph: ArrayRef = Arc::new(StringArray::from(graph));
    let dataset: ArrayRef = Arc::new(StringArray::from(dataset));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject", subject),
        ("predicate", predicate),
        ("object", object),
        ("graph", graph),
        ("dataset", dataset),
    ])?;
    Ok(batch)
}

#[allow(dead_code)]
pub(crate) fn create_dataset_graph_fields() -> Fields {
    let field_names = ["graph", "dataset"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

#[allow(dead_code)]
pub fn create_dataset_graph_batch(graph: Vec<String>, dataset: Vec<String>) -> Result<RecordBatch> {
    let graph: ArrayRef = Arc::new(StringArray::from(graph));
    let dataset: ArrayRef = Arc::new(StringArray::from(dataset));
    let batch = RecordBatch::try_from_iter(vec![("graph", graph), ("dataset", dataset)])?;
    Ok(batch)
}

fn create_parse_owl_vec_fields() -> Vec<Field> {
    let field_names = ["entity"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec.extend(create_parse_n_quads_vec_fields());
    fields_vec
}

pub(crate) fn create_parse_owl_fields() -> Fields {
    Fields::from(create_parse_owl_vec_fields())
}

pub fn create_parse_owl_batch(
    entity: Vec<String>,
    subject: Vec<String>,
    predicate: Vec<String>,
    object: Vec<String>,
    graph: Vec<String>,
    dataset: Vec<String>,
) -> Result<RecordBatch> {
    let entity: ArrayRef = Arc::new(StringArray::from(entity));
    let subject: ArrayRef = Arc::new(StringArray::from(subject));
    let predicate: ArrayRef = Arc::new(StringArray::from(predicate));
    let object: ArrayRef = Arc::new(StringArray::from(object));
    let graph: ArrayRef = Arc::new(StringArray::from(graph));
    let dataset: ArrayRef = Arc::new(StringArray::from(dataset));
    let batch = RecordBatch::try_from_iter(vec![
        ("entity", entity),
        ("subject", subject),
        ("predicate", predicate),
        ("object", object),
        ("graph", graph),
        ("dataset", dataset),
    ])?;
    Ok(batch)
}
