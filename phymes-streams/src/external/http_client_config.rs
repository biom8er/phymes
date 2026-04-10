use std::{env, fmt::Display};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_subject::{MappableTrait, Subject, SubjectTrait};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

/// Schema to use when packaging the HTTP client request response
///
/// More complex parsing should be handled by one of the extractor `DataProcessor`s
///   e.g., tabular for JSON Line, xml for XML, and PDF for PDFs
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default, PartialEq)]
pub enum HTTPClientRequestSchemas {
    #[default]
    #[value(name = "Messages")]
    Messages,
    #[value(name = "Attachments")]
    Attachments,
    #[value(skip)]
    Custom(String),
}
impl Display for HTTPClientRequestSchemas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Messages => write!(f, "Messages"),
            Self::Attachments => write!(f, "Attachments"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// The HTTP client request types
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum HTTPClientRequestType {
    #[default]
    #[value(name = "Get")]
    Get,
    #[value(name = "Post")]
    Post,
    #[value(name = "Put")]
    Put,
    #[value(name = "Patch")]
    Patch,
    #[value(name = "Delete")]
    Delete,
    #[value(name = "Head")]
    Head,
}
impl Display for HTTPClientRequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Get => write!(f, "Get"),
            Self::Post => write!(f, "Post"),
            Self::Put => write!(f, "Put"),
            Self::Patch => write!(f, "Patch"),
            Self::Delete => write!(f, "Delete"),
            Self::Head => write!(f, "Head"),
        }
    }
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct HTTPClientConfig {
    /// The timeout in seconds
    #[arg(long, default_value_t = 15)]
    pub timeout: usize,

    /// The request type to the URL
    #[arg(long, default_value_t = HTTPClientRequestType::Get)]
    pub request_type: HTTPClientRequestType,

    /// The request header user agent type
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent_type: Option<String>,

    /// The request header content type or header value
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// The name of the environmental variable for the API key
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bearer_auth: Option<String>,

    /// The base URL of the request
    ///
    /// # Notes
    /// - Can range from just scheme to the port number and all the way to the query string separator or fragment
    #[arg(long)]
    pub base_url: String,

    /// The JSON application data to send in the request if POST
    /// or the query URL to join with the base URL if GET
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<String>,

    /// The name of the streaming subject with the JSON application data to send in the request if POST
    /// or the query URL to join with the base URL if GET
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_name: Option<String>,

    /// The request schema to try and parse responses into
    #[arg(long, default_value_t = HTTPClientRequestSchemas::Messages)]
    pub request_schema: HTTPClientRequestSchemas,
}

impl HTTPClientConfig {
    /// Retrieve the api key
    pub fn api_key(&self) -> Result<String> {
        if let Some(env_var) = &self.bearer_auth {
            match env::var(env_var) {
                Ok(key) => Ok(key),
                Err(e) => Err(anyhow!("{e:?}")),
            }
        } else {
            Err(anyhow!("No API key environmental variable specificied."))
        }
    }
    /// Make the full url for GET requests
    pub fn url(&self, query_url: Option<&str>) -> String {
        if let Some(query_url) = query_url {
            format!("{}{query_url}", &self.base_url)
        } else {
            self.base_url.to_string()
        }
    }
}

impl Default for HTTPClientConfig {
    fn default() -> Self {
        Self {
            timeout: 15,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            content_type: Some("application/json".to_string()),
            bearer_auth: None,
            base_url: "".to_string(),
            subject_name: None,
            json: None,
            request_schema: HTTPClientRequestSchemas::Messages,
        }
    }
}

impl DataConfigTrait for HTTPClientConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_subject(subject: &Subject) -> Result<Self>
    where
        Self: Sized,
    {
        if let Some(bytes) = Self::from_subject_as_bytes(subject) {
            // Try to build the config
            match serde_json::from_slice::<HTTPClientConfig>(&bytes) {
                Ok(config) => {
                    config.check_required_members(subject.get_name())?;
                    Ok(config)
                },
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(), 
                    subject.get_name()
                )),
            }
        } else {
            // Check for the required fields
            let required_fields = &["timeout", "request_type", "request_schema"];
            let column_names = subject
                .get_schema()
                .fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect::<HashSet<_>>();
            Self::check_required_fields(subject.get_name(), &column_names, required_fields)?;            

            // Try to build the config
            match subject.to_struct::<HTTPClientConfig>() {
                Ok(mut config_vec) => match config_vec.pop() {
                    Some(config) => Ok(config),
                    None => Err(anyhow!(
                        "No config data found for `{}` with subject {}",
                        Self::get_static_name(),
                        subject.get_name()
                    )),
                },
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(), 
                    subject.get_name()
                )),
            }
        }
    }
    
    fn check_required_members(&self, _subject_name: &str) -> Result<()> {
        Ok(())
    }
}

impl MappableTrait for HTTPClientConfig {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}
