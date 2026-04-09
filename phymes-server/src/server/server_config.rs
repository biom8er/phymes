// General imports
use clap::{Parser, ValueEnum};
use phymes_subject::ObjectStorageBackend;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about, long_about = None)]
pub struct ServerConfig {
    /// Address to serve the application on
    #[arg(long, default_value = "127.0.0.1:4000")]
    pub address: String,

    /// Assets directory
    #[arg(long, default_value = ".")]
    pub assets_dir: String,

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

impl From<&HashMap<String, String>> for ServerConfig {
    fn from(values: &HashMap<String, String>) -> ServerConfig {
        let object_store_backend = values
            .get("object_store_backend")
            .map(|v| ObjectStorageBackend::from_str(v, false).unwrap());
        let object_store_config = values
            .get("object_store_config")
            .map(|v| serde_json::from_str::<Map<String, Value>>(v).unwrap());
        ServerConfig {
            address: values.get("address").unwrap().to_string(),
            assets_dir: values.get("assets_dir").unwrap().to_string(),
            object_store_backend,
            object_store_bucket: values.get("object_store_bucket").cloned(),
            object_store_config,
        }
    }
}
