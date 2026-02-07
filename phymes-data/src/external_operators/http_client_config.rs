use std::{env, fmt::Display};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_core::{MappableTrait, Table, TableTrait};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

use crate::DataConfigTrait;

/// The HTTP client request types
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default, PartialEq)]
pub enum HTTPClientRequestSchemas {
    /// No parsing of the response
    #[default]
    #[value(name = "None")]
    None,
    /// `works` OpenAlex API endpoint
    #[value(name = "OpenAlexWorks")]
    OpenAlexWorks,
    /// `find/works` OpenAlex API endpoint
    #[value(name = "OpenAlexFind")]
    OpenAlexFind,
    /// `group_by` OpenAlex API endpoint
    #[value(name = "OpenAlexGroupBy")]
    OpenAlexGroupBy,
    /// EUtils ESearch utility
    ///
    /// MUST use `retmode=json`
    #[value(name = "ESearch")]
    ESearch,
    /// EUtils Efetch utility
    ///
    /// MUST use `retmode=xml`
    #[value(name = "EFetch")]
    EFetch,
    /// Semantic Scholar Recomendations API
    #[value(name = "SemanticScholarRecomendations")]
    SemanticScholarRecomendations,
    /// A general API endpoint for PDFs
    #[value(name = "PDF")]
    PDF,
    #[value(skip)]
    Custom(String),
}
impl Display for HTTPClientRequestSchemas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::OpenAlexWorks => write!(f, "OpenAlexWorks"),
            Self::OpenAlexFind => write!(f, "OpenAlexFind"),
            Self::OpenAlexGroupBy => write!(f, "OpenAlexGroupBy"),
            Self::ESearch => write!(f, "ESearch"),
            Self::EFetch => write!(f, "EFetch"),
            Self::SemanticScholarRecomendations => write!(f, "SemanticScholarRecomendations"),
            Self::PDF => write!(f, "PDF"),
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
    #[arg(long, default_value_t = HTTPClientRequestSchemas::None)]
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
            request_schema: HTTPClientRequestSchemas::None,
        }
    }
}

impl DataConfigTrait for HTTPClientConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_table(table: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // Check for the required fields
        let column_names = table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<HashSet<_>>();
        if !(column_names.contains("timeout")
            && column_names.contains("request_type")
            && column_names.contains("content_type")
            && column_names.contains("request_schema"))
        {
            return Err(anyhow!(
                "Table {} is missing required Field for `timeout`, `request_type`, `content_type`, and `request_schema` in HTTPClientConfig.",
                table.get_name()
            ));
        }

        // Try to build the config
        match table.to_struct::<HTTPClientConfig>() {
            Ok(config_vec) => match config_vec.first() {
                Some(config) => Ok(config.to_owned()),
                None => Err(anyhow!(
                    "No config data found for HTTPClientConfig with subject {}",
                    table.get_name()
                )),
            },
            Err(err) => Err(anyhow!(
                "HTTPClientConfig could not be built for subject {}. {err}",
                table.get_name()
            )),
        }
    }
}
