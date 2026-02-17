use anyhow::Result;
use arrow::{
    array::{ArrayRef, ListBuilder, StringArray, UInt8Builder},
    datatypes::{DataType, Field, Fields},
    record_batch::RecordBatch,
};
use std::sync::Arc;

/// Fields where each row is a [RecordBatch]
pub fn create_route_bytes_fields() -> Fields {
    let field_names = ["name", "publisher", "subject", "format"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let list_data_type = DataType::List(
        Arc::new(Field::new_list_field(DataType::UInt8, false))
    );
    let field_names = ["bytes"];
    fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>());
    Fields::from(fields_vec)
}

/// Each row of a `values` [RecordBatch] is split into a seperate message
pub fn create_route_bytes_record_batch(
    names: Vec<String>,
    publishers: Vec<String>,
    subjects: Vec<String>,
    formats: Vec<String>,
    bytes: Vec<Vec<u8>>,
) -> Result<RecordBatch> {
    let names: ArrayRef = Arc::new(StringArray::from(names));
    let publishers: ArrayRef = Arc::new(StringArray::from(publishers));
    let subjects: ArrayRef = Arc::new(StringArray::from(subjects));
    let formats: ArrayRef = Arc::new(StringArray::from(formats));
    let value_builder = UInt8Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::UInt8, false));
    for values in bytes.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let bytes: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("name", names),
        ("publisher", publishers),
        ("subject", subjects),
        ("format", formats),
        ("bytes", bytes),
    ])?;
    Ok(batch)
}

/// Fields for a single [RecordBatch] per row
pub fn create_bytes_fields() -> Fields {
    let list_data_type = DataType::List(
        Arc::new(Field::new_list_field(DataType::UInt8, false))
    );
    let field_names = ["bytes"];
    let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

/// A single [RecordBatch] per row
pub fn create_bytes_record_batch(
    bytes: Vec<Vec<u8>>,
) -> Result<RecordBatch> {
    let value_builder = UInt8Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::UInt8, false));
    for values in bytes.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let bytes: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("bytes", bytes),
    ])?;
    Ok(batch)
}

/// Each row of a `tool`s [RecordBatch] is JSON Schema object describing a tool/function
///   that can be called during text generation inference
pub fn create_tools_fields() -> Fields {
    let field_names = ["tool_id", "tool"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

/// Each row of a `tool`s [RecordBatch] is JSON Schema object describing a tool/function
///   that can be called during text generation inference
pub fn create_tools_record_batch(tool_ids: Vec<String>, tools: Vec<String>) -> Result<RecordBatch> {
    let tool_ids: ArrayRef = Arc::new(StringArray::from(tool_ids));
    let tools: ArrayRef = Arc::new(StringArray::from(tools));
    let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_ids), ("tool", tools)])?;
    Ok(batch)
}