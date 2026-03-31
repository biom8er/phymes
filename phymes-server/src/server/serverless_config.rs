// General imports
use clap::Parser;
use phymes_core::ObjectStorageBackend;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about, long_about = None)]
pub struct ServerlessConfig {
    /// The application route to call e.g., app/v1/chat
    #[arg(long, default_value = "app/v1/sign_in")]
    pub route: String,

    /// Basic authentication credentials e.g., email:password
    #[arg(long)]
    pub basic_auth: Option<String>,

    /// Bearer authentication credentials e.g., JWT-abc
    #[arg(long)]
    pub bearer_auth: Option<String>,

    /// The data to send in JSON format e.g., '{"content": "Write a python function to count prime numbers", "session_name": "EMAILChat", "subject_name": "messages"}'
    #[arg(long)]
    pub data: Option<String>,

    /// The backend for the object store
    #[arg(long)]
    pub object_store_backend: Option<ObjectStorageBackend>,

    /// The bucket for the object store
    #[arg(long)]
    pub object_store_bucket: Option<String>,

    /// Additional object store configuration options not in the environmental variables
    #[arg(long)]
    pub object_store_config: Option<Map<String, Value>>,
}
