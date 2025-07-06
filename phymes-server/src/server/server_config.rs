// General imports
use clap::Parser;
use serde::{Deserialize, Serialize};
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
}

impl From<&HashMap<String, String>> for ServerConfig {
    fn from(values: &HashMap<String, String>) -> ServerConfig {
        ServerConfig {
            address: values.get("address").unwrap().to_string(),
            assets_dir: values.get("assets_dir").unwrap().to_string(),
        }
    }
}
