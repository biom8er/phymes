use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array, ListBuilder, RecordBatch, StringArray, StringBuilder}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{schemas::available_subjects::create_timestamp_micros, table::table::TableBuilder};

pub fn create_user_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let first_name = Field::new("first_name", DataType::Utf8, false);
    let last_name = Field::new("last_name", DataType::Utf8, false);
    let password_hash = Field::new("password_hash", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    let list_data_type = DataType::List(
        Arc::new(Field::new_list_field(DataType::Utf8, false))
    );
    let session_contexts = Field::new("session_contexts", list_data_type, false);
    Fields::from(vec![
        email,
        first_name,
        last_name,
        password_hash,
        timestamp,
        session_contexts,
    ])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserSubject {
    pub email: String,
    pub first_name: String,
    pub last_name: String,
    pub password_hash: String,
    pub timestamp: i64,
    pub session_contexts: Vec<String>,
}

pub fn create_user_batch(
    email: Vec<String>,
    first_name: Vec<String>,
    last_name: Vec<String>,
    password_hash: Vec<String>,
    timestamp: Vec<i64>,
    session_contexts: Vec<Vec<String>>,
) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let first_name: ArrayRef = Arc::new(StringArray::from(first_name));
    let last_name: ArrayRef = Arc::new(StringArray::from(last_name));
    let password_hash: ArrayRef = Arc::new(StringArray::from(password_hash));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let value_builder = StringBuilder::new();
    let mut list_builder = ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in session_contexts.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let session_contexts: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("email", email),
        ("first_name", first_name),
        ("last_name", last_name),
        ("password_hash", password_hash),
        ("timestamp", timestamp),
        ("session_contexts", session_contexts),
    ])?;
    Ok(batch)
}

pub trait UserBuilderTraitExt: Sized {
    fn with_user(self, email: Option<&str>, first_name: Option<&str>, last_name: Option<&str>, password_hash: Option<&str>, session_contexts: Option<Vec<String>>) -> Result<Self>;
}

impl UserBuilderTraitExt for TableBuilder {
    fn with_user(mut self, email: Option<&str>, first_name: Option<&str>, last_name: Option<&str>, password_hash: Option<&str>, session_contexts: Option<Vec<String>>) -> Result<Self> {
        // Handle the metadata
        let email = email.unwrap_or_default().to_string();
        let first_name = first_name.unwrap_or_default().to_string();
        let last_name = last_name.unwrap_or_default().to_string();
        let password_hash = password_hash.unwrap_or_default().to_string();
        let session_contexts = session_contexts.unwrap_or_default();

        // Add the record batch to the table
        let batch = create_user_batch(
            vec![email],
            vec![first_name],
            vec![last_name],
            vec![password_hash],
            vec![create_timestamp_micros()],
            vec![session_contexts],
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