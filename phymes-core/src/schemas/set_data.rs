use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt32Array},
    datatypes::{DataType, Field, Fields},
};

pub(crate) fn create_parse_xml_fields() -> Fields {
    let field_names = [
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

pub fn create_parse_xml_batch(
    element_tag: Vec<String>,
    element_attr: Vec<String>,
    text: Vec<String>,
    child_tag: Vec<String>,
    child_attr: Vec<String>,
    element_index: Vec<u32>,
    child_index: Vec<u32>,
) -> Result<RecordBatch> {
    let element_tag: ArrayRef = Arc::new(StringArray::from(element_tag));
    let element_attr: ArrayRef = Arc::new(StringArray::from(element_attr));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let child_tag: ArrayRef = Arc::new(StringArray::from(child_tag));
    let child_attr: ArrayRef = Arc::new(StringArray::from(child_attr));
    let element_index: ArrayRef = Arc::new(UInt32Array::from(element_index));
    let child_index: ArrayRef = Arc::new(UInt32Array::from(child_index));
    let batch = RecordBatch::try_from_iter(vec![
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

pub(crate) fn create_parse_owl_fields() -> Fields {
    let field_names = ["subject", "predicate", "object"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_parse_owl_batch(
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
