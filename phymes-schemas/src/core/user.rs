use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Fields},
};
use serde::{Deserialize, Serialize};

pub(crate) fn create_user_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let first_name = Field::new("first_name", DataType::Utf8, false);
    let last_name = Field::new("last_name", DataType::Utf8, false);
    let password_hash = Field::new("password_hash", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    // let list_data_type = DataType::List(
    //     Arc::new(Field::new_list_field(DataType::Utf8, false))
    // );
    // let networks = Field::new("networks", list_data_type, false);
    Fields::from(vec![
        email,
        first_name,
        last_name,
        password_hash,
        timestamp,
        // networks,
    ])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserSubject {
    pub email: String,
    pub first_name: String,
    pub last_name: String,
    pub password_hash: String,
    pub timestamp: i64,
    // pub networks: Vec<String>,
}

pub fn create_user_batch(
    email: Vec<String>,
    first_name: Vec<String>,
    last_name: Vec<String>,
    password_hash: Vec<String>,
    timestamp: Vec<i64>,
    // networks: Vec<Vec<String>>,
) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let first_name: ArrayRef = Arc::new(StringArray::from(first_name));
    let last_name: ArrayRef = Arc::new(StringArray::from(last_name));
    let password_hash: ArrayRef = Arc::new(StringArray::from(password_hash));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    // let value_builder = StringBuilder::new();
    // let mut list_builder = ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    // for values in networks.into_iter() {
    //     for value in values.into_iter() {
    //         list_builder.values().append_value(&value);
    //     }
    //     list_builder.append(true);
    // }
    // let networks: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("email", email),
        ("first_name", first_name),
        ("last_name", last_name),
        ("password_hash", password_hash),
        ("timestamp", timestamp),
        // ("networks", networks),
    ])?;
    Ok(batch)
}

pub(crate) fn create_user_networks_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let network_name = Field::new("network_name", DataType::Utf8, false);
    Fields::from(vec![email, network_name])
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserNetworksSubject {
    pub email: String,
    pub network_name: String,
}

pub fn create_user_networks_batch(
    email: Vec<String>,
    network_name: Vec<String>,
) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let network_name: ArrayRef = Arc::new(StringArray::from(network_name));
    let batch = RecordBatch::try_from_iter(vec![
        ("email", email),
        ("network_name", network_name),
    ])?;
    Ok(batch)
}

pub(crate) fn create_user_inbox_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    Fields::from(vec![email])
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserInboxSubject {
    pub email: String,
}

pub fn create_user_inbox_batch(email: Vec<String>) -> Result<RecordBatch> {
    let email: ArrayRef = Arc::new(StringArray::from(email));
    let batch = RecordBatch::try_from_iter(vec![("email", email)])?;
    Ok(batch)
}

pub(crate) fn create_join_user_inbox_networks_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let network_name = Field::new("network_name", DataType::Utf8, false);
    Fields::from(vec![email, network_name])
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct JoinUserInboxNetworksMermaidDiagrams {
    pub email: String,
    pub network_name: String,
    pub flowchart_diagram: String,
    pub er_diagram: String,
    pub timestamp: i64,
}

pub(crate) fn create_join_user_inbox_networks_mermaid_diagrams_fields() -> Fields {
    let email = Field::new("email", DataType::Utf8, false);
    let network_name = Field::new("network_name", DataType::Utf8, false);
    let flowchart_diagram = Field::new("flowchart_diagram", DataType::Utf8, false);
    let er_diagram = Field::new("er_diagram", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    Fields::from(vec![
        email,
        network_name,
        flowchart_diagram,
        er_diagram,
        timestamp,
    ])
}
