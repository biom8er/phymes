use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Fields},
};
use serde::{Deserialize, Serialize};

pub fn create_workspace_fields_vec() -> Vec<Field> {
    let field_names = ["path", "content"];
    field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>()
}

/// Minimal Blob schema needed for code generation application
///
/// # Notes
///
/// - `content` is used for both the source code and the source code diff/patch
pub fn create_workspace_fields() -> Fields {
    Fields::from(create_workspace_fields_vec())
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct WorkspaceSubject {
    pub path: String,
    pub content: String,
}

pub fn create_workspace_batch(path: Vec<String>, content: Vec<String>) -> Result<RecordBatch> {
    let path: ArrayRef = Arc::new(StringArray::from(path));
    let content: ArrayRef = Arc::new(StringArray::from(content));
    let batch = RecordBatch::try_from_iter(vec![("path", path), ("content", content)])?;
    Ok(batch)
}

fn create_repository_add_on_fields_vec() -> Vec<Field> {
    let field_names = ["repository", "branch", "hash", "metadata"];
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
    fields_vec
}

/// Expanded workspace to model code repositories
///
/// # Notes
///
/// - `content` is used for both the source code and the source code diff/patch
pub fn create_repository_fields() -> Fields {
    let mut fields_vec = create_workspace_fields_vec();
    fields_vec.extend(create_repository_add_on_fields_vec());
    Fields::from(fields_vec)
}

pub fn create_repository_batch(
    path: Vec<String>,
    content: Vec<String>,
    repository: Vec<String>,
    branch: Vec<String>,
    hash: Vec<String>,
    metadata: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let path: ArrayRef = Arc::new(StringArray::from(path));
    let content: ArrayRef = Arc::new(StringArray::from(content));
    let repository: ArrayRef = Arc::new(StringArray::from(repository));
    let branch: ArrayRef = Arc::new(StringArray::from(branch));
    let hash: ArrayRef = Arc::new(StringArray::from(hash));
    let metadata: ArrayRef = Arc::new(StringArray::from(metadata));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("path", path),
        ("content", content),
        ("repository", repository),
        ("branch", branch),
        ("hash", hash),
        ("metadata", metadata),
        ("timestamp", timestamp),
    ])?;
    Ok(batch)
}

fn create_patch_add_on_fields_vec() -> Vec<Field> {
    let field_names = ["operator"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec
}

/// Minimal Diff/Patch schema needed for code generation application
///
/// # Notes
///
/// - `content` is used for both the source code and the source code diff/patch
/// - Allowable `operators` are "Create", "Update", and "Delete"
pub fn create_workspace_patch_fields() -> Fields {
    let mut fields_vec = create_workspace_fields_vec();
    fields_vec.extend(create_patch_add_on_fields_vec());
    Fields::from(fields_vec)
}

pub fn create_workspace_patch_batch(
    path: Vec<String>,
    content: Vec<String>,
    operator: Vec<String>,
) -> Result<RecordBatch> {
    let path: ArrayRef = Arc::new(StringArray::from(path));
    let content: ArrayRef = Arc::new(StringArray::from(content));
    let operator: ArrayRef = Arc::new(StringArray::from(operator));
    let batch = RecordBatch::try_from_iter(vec![
        ("path", path),
        ("content", content),
        ("operator", operator),
    ])?;
    Ok(batch)
}

/// Expanded Diff/Patch schema to model code repositories
pub fn create_repository_patch_fields() -> Fields {
    let mut fields_vec = create_workspace_fields_vec();
    fields_vec.extend(create_patch_add_on_fields_vec());
    fields_vec.extend(create_repository_add_on_fields_vec());
    Fields::from(fields_vec)
}

pub fn create_repository_patch_batch(
    path: Vec<String>,
    content: Vec<String>,
    operator: Vec<String>,
    repository: Vec<String>,
    branch: Vec<String>,
    hash: Vec<String>,
    metadata: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let path: ArrayRef = Arc::new(StringArray::from(path));
    let content: ArrayRef = Arc::new(StringArray::from(content));
    let operator: ArrayRef = Arc::new(StringArray::from(operator));
    let repository: ArrayRef = Arc::new(StringArray::from(repository));
    let branch: ArrayRef = Arc::new(StringArray::from(branch));
    let hash: ArrayRef = Arc::new(StringArray::from(hash));
    let metadata: ArrayRef = Arc::new(StringArray::from(metadata));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("path", path),
        ("content", content),
        ("operator", operator),
        ("repository", repository),
        ("branch", branch),
        ("hash", hash),
        ("metadata", metadata),
        ("timestamp", timestamp),
    ])?;
    Ok(batch)
}
