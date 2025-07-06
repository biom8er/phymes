// General imports
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about, long_about = None)]
pub struct ServerlessConfig {
    /// The application route to call e.g., app/v1/chat
    #[arg(long)]
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

    /// Assets directory
    #[arg(long, default_value = ".")]
    pub assets_dir: String,
}
