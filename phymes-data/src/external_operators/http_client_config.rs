use std::{env, fmt::Display};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_core::{MappableTrait, Table, TableTrait};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

use crate::DataConfigTrait;

/// The HTTP client request types
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum HTTPClientRequestSchemas {
    /// No parsing of the response
    #[default]
    #[value(name = "None")]
    None,
    #[value(name = "OpenAlex")]
    OpenAlex,
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
    #[value(skip)]
    Custom(String),
}
impl Display for HTTPClientRequestSchemas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::OpenAlex => write!(f, "OpenAlex"),
            Self::ESearch => write!(f, "ESearch"),
            Self::EFetch => write!(f, "EFetch"),
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

    /// The request header content type or header value
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// The name of the environmental variable for the API key
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bearer_auth: Option<String>,

    /// The base URL of the request
    #[arg(long)]
    pub base_url: String,

    /// The JSON application data to send in the request if POST
    /// or the query URL to join with the base URL if GET
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<String>,

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
            format!("{}?{query_url}", &self.base_url)
        } else if let Some(query_url) = &self.json {
            format!("{}?{query_url}", &self.base_url)
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
            content_type: Some("application/json".to_string()),
            bearer_auth: None,
            base_url: "".to_string(),
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
        if !(column_names.contains("timeout") && column_names.contains("request_type") && column_names.contains("content_type") && column_names.contains("request_schema")) {
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

/// Collection of structs for parsing OpenAlex data
pub(crate) mod open_alex_schemas {
    use super::*;
    
    /// Struct for authorship info
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Authorship {
        author: Author,
        institutions: Vec<Institution>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Author {
        id: Option<String>,
        display_name: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Institution {
        id: Option<String>,
        display_name: Option<String>,
    }

    /// Struct for concept info (including MeSH)
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Concept {
        id: Option<String>,
        display_name: Option<String>,
        level: Option<u8>,
        score: Option<f64>,
    }

    /// Struct for host venue info
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct HostVenue {
        id: Option<String>,
        display_name: Option<String>,
        publisher: Option<String>,
        #[serde(rename = "type")]
        venue_type: Option<String>,
        url: Option<String>,
    }

    /// Struct for each work
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Work {
        id: String,
        display_name: String,
        publication_year: Option<u16>,
        publication_date: Option<String>,
        doi: Option<String>,
        language: Option<String>,
        type_: Option<String>,
        cited_by_count: Option<u32>,
        authorships: Vec<Authorship>,
        concepts: Vec<Concept>,
        mesh: Option<Vec<Concept>>,
        host_venue: Option<HostVenue>,
        open_access: Option<OpenAccess>,
        abstract_inverted_index: Option<serde_json::Value>, // Raw JSON for abstracts
    }

    /// Struct for open access info
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct OpenAccess {
        is_oa: bool,
        oa_status: Option<String>,
        oa_url: Option<String>,
    }

    /// Struct for API response
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct OpenAlexResponse {
        pub(crate) results: Vec<Work>,
        meta: Meta,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Meta {
        count: u32,
        per_page: u32,
        page: u32,
    }
}

/// Collection of structs for parsing EUtils data
pub(crate) mod e_utils_schemas {
    use super::*;

    /// Struct for parsing ESearch JSON response
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct ESearchResponse {
        pub(crate) esearchresult: ESearchResult,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct ESearchResult {
        pub(crate) idlist: Vec<String>,
    }

    /// Struct for parsing EFetch XML response (simplified)
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct PubmedArticleSet {
        #[serde(rename = "PubmedArticle", default)]
        pub(crate) articles: Vec<PubmedArticle>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct PubmedArticle {
        #[serde(rename = "MedlineCitation")]
        citation: MedlineCitation,
        #[serde(rename = "PubmedData")]
        pubmed_data: Option<PubmedData>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct MedlineCitation {
        #[serde(rename = "Article")]
        article: Article,
        #[serde(rename = "MeshHeadingList", default)]
        mesh_headings: Option<MeshHeadingList>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Article {
        #[serde(rename = "ArticleTitle")]
        title: String,
        #[serde(rename = "Abstract", default)]
        abstract_text: Option<AbstractText>,
        #[serde(rename = "Journal")]
        journal: Journal,
        #[serde(rename = "AuthorList", default)]
        authors: Option<AuthorList>,
        #[serde(rename = "Pagination", default)]
        pagination: Option<Pagination>,
        #[serde(rename = "ELocationID", default)]
        elocation_ids: Vec<ELocationID>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct Journal {
        #[serde(rename = "Title")]
        title: String,
        #[serde(rename = "ISSN", default)]
        issn: Option<String>,
        #[serde(rename = "JournalIssue")]
        issue: JournalIssue,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct JournalIssue {
        #[serde(rename = "PubDate")]
        pub_date: PubDate,
        #[serde(rename = "Volume", default)]
        volume: Option<String>,
        #[serde(rename = "Issue", default)]
        issue: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct PubDate {
        #[serde(rename = "Year", default)]
        year: Option<String>,
        #[serde(rename = "Month", default)]
        month: Option<String>,
        #[serde(rename = "Day", default)]
        day: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct AbstractText {
        #[serde(rename = "AbstractText", default)]
        text: Vec<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct AuthorList {
        #[serde(rename = "Author", default)]
        authors: Vec<Author>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Author {
        #[serde(rename = "LastName", default)]
        last_name: Option<String>,
        #[serde(rename = "ForeName", default)]
        fore_name: Option<String>,
        #[serde(rename = "AffiliationInfo", default)]
        affiliations: Vec<AffiliationInfo>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct AffiliationInfo {
        #[serde(rename = "Affiliation", default)]
        affiliation: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Pagination {
        #[serde(rename = "MedlinePgn", default)]
        pages: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct ELocationID {
        #[serde(rename = "$value")]
        value: String,
        #[serde(rename = "EIdType", default)]
        id_type: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct MeshHeadingList {
        #[serde(rename = "MeshHeading", default)]
        headings: Vec<MeshHeading>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct MeshHeading {
        #[serde(rename = "DescriptorName", default)]
        descriptor: Option<String>,
    }

    /// Struct for PMC ID extraction
    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct PubmedData {
        #[serde(rename = "ArticleIdList")]
        id_list: ArticleIdList,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct ArticleIdList {
        #[serde(rename = "ArticleId", default)]
        ids: Vec<ArticleId>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(crate) struct ArticleId {
        #[serde(rename = "$value")]
        value: String,
        #[serde(rename = "IdType", default)]
        id_type: Option<String>,
    }
}