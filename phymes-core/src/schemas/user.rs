use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub fn create_user_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let first_name = Field::new("first_name", DataType::Utf8, false);
    let last_name = Field::new("last_name", DataType::Utf8, false);
    let password_hash = Field::new("password_hash", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    // let list_data_type = DataType::List(
    //     Arc::new(Field::new_list_field(DataType::Utf8, false))
    // );
    // let session_contexts = Field::new("session_contexts", list_data_type, false);
    Fields::from(vec![
        email,
        first_name,
        last_name,
        password_hash,
        timestamp,
        // session_contexts,
    ])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserSubject {
    pub email: String,
    pub first_name: String,
    pub last_name: String,
    pub password_hash: String,
    pub timestamp: i64,
    // pub session_contexts: Vec<String>,
}

pub fn create_user_batch(
    email: Vec<String>,
    first_name: Vec<String>,
    last_name: Vec<String>,
    password_hash: Vec<String>,
    timestamp: Vec<i64>,
    // session_contexts: Vec<Vec<String>>,
) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let first_name: ArrayRef = Arc::new(StringArray::from(first_name));
    let last_name: ArrayRef = Arc::new(StringArray::from(last_name));
    let password_hash: ArrayRef = Arc::new(StringArray::from(password_hash));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    // let value_builder = StringBuilder::new();
    // let mut list_builder = ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    // for values in session_contexts.into_iter() {
    //     for value in values.into_iter() {
    //         list_builder.values().append_value(&value);
    //     }
    //     list_builder.append(true);
    // }
    // let session_contexts: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("email", email),
        ("first_name", first_name),
        ("last_name", last_name),
        ("password_hash", password_hash),
        ("timestamp", timestamp),
        // ("session_contexts", session_contexts),
    ])?;
    Ok(batch)
}

pub fn create_user_session_contexts_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let session_context_name = Field::new("session_context_name", DataType::Utf8, false);
    Fields::from(vec![
        email,
        session_context_name,
    ])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserSessionContextsSubject {
    pub email: String,
    pub session_context_name: String,
}

pub fn create_user_session_contexts_batch(
    email: Vec<String>,
    session_context_name: Vec<String>,
) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let session_context_name: ArrayRef = Arc::new(StringArray::from(session_context_name));
    let batch = RecordBatch::try_from_iter(vec![
        ("email", email),
        ("session_context_name", session_context_name),
    ])?;
    Ok(batch)
}

pub fn create_user_inbox_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    Fields::from(vec![
        email,
    ])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserInboxSubject {
    pub email: String,
}

pub fn create_user_inbox_batch(
    email: Vec<String>,
) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let batch = RecordBatch::try_from_iter(vec![
        ("email", email),
    ])?;
    Ok(batch)
}

pub fn create_join_user_inbox_session_contexts_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let session_context_name = Field::new("session_context_name", DataType::Utf8, false);
    Fields::from(vec![
        email,
        session_context_name,
    ])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct JoinUserInboxSessionContextsMermaidDiagrams {
    pub email: String,
    pub session_context_name: String,
    pub flowchart_diagram: String,
    pub er_diagram: String,
    pub timestamp: i64,
}

pub fn create_join_user_inbox_session_contexts_mermaid_diagrams_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let session_context_name = Field::new("session_context_name", DataType::Utf8, false);
    let flowchart_diagram = Field::new("flowchart_diagram", DataType::Utf8, false);
    let er_diagram = Field::new("er_diagram", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    Fields::from(vec![
        email,
        session_context_name,
        flowchart_diagram,
        er_diagram,
        timestamp,
    ])
}