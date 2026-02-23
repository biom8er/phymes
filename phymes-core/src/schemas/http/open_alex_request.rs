use std::fmt::Display;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// OpenAlex API Base URL
pub const OPENALEX_API: &str = "https://api.openalex.org/";

/// Struct for API requests
///
/// # Notes
/// - see paging documentation <https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging>
/// - see filters documentation <https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists>
/// - see search documentation <https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities>
/// - see semantic search with vector embeddings <https://docs.openalex.org/how-to-use-the-api/find-similar-works>
/// - see sort documentation <https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/sort-entity-lists>
/// - see select documentation <https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/select-fields>
/// - see sample entities <https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/sample-entity-lists>
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct OpenAlexRequest {
    pub page: Option<u32>,
    pub per_page: Option<u32>,
    pub cursor: Option<String>,
    pub filter: Option<Map<String, Value>>,
    pub search: Option<String>,
    pub query: Option<String>,
    pub api_key: Option<String>,
    pub entity: OpenAlexRequestEntity,
    pub sort: Option<Map<String, Value>>,
    pub select: Option<Vec<String>>,
    pub sample: Option<u32>,
    pub seed: Option<u32>,
}

impl OpenAlexRequest {
    /// OpenAlex GET Request Query
    ///
    /// # Example
    ///
    /// ```bash
    /// https://api.openalex.org/find/works?query=machine%20learning%20for%20drug%20discovery&api_key=YOUR_KEY
    /// ```
    pub fn to_get_query(&self) -> Result<String> {
        let mut query_list = Vec::new();
        if let Some(page) = self.page {
            let query = format!("page={page}");
            query_list.push(query);
        }
        if let Some(per_page) = self.per_page {
            let query = format!("per-page={per_page}");
            query_list.push(query);
        }
        if let Some(cursor) = self.cursor.as_ref() {
            let query = format!("cursor={cursor}");
            query_list.push(query);
        }
        if let Some(filter) = self.filter.as_ref() {
            let query = filter
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(",");
            let query = format!("filter={query}");
            query_list.push(query);
        }
        if let Some(search) = self.search.as_ref() {
            let query = format!("search={search}");
            query_list.push(query);
        }
        if let Some(api_key) = self.api_key.as_ref() {
            let query = format!("api_key={api_key}");
            query_list.push(query);
        }
        if let Some(sort) = self.sort.as_ref() {
            let query = sort
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(",");
            let query = format!("sort={query}");
            query_list.push(query);
        }
        if let Some(select) = self.select.as_ref() {
            let query = select.join(",");
            let query = format!("select={query}");
            query_list.push(query);
        }
        if let Some(sample) = self.sample {
            let query = format!("sample={sample}");
            query_list.push(query);
        }
        if let Some(seed) = self.seed {
            let query = format!("seed={seed}");
            query_list.push(query);
        }
        if query_list.is_empty() {
            Err(anyhow!(
                "Missing query parameters for OpenAlex GET request."
            ))
        } else {
            let query_str = query_list.join("&");
            Ok(query_str)
        }
    }

    /// OpenAlex Base URL
    pub fn to_base_url(&self) -> String {
        format!("{OPENALEX_API}/{}", self.entity)
    }
}

/// OpenAlex Entities
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OpenAlexRequestEntity {
    #[default]
    Works,
    Authors,
    Sources,
    Institutions,
    Topics,
    Keywords,
    Publishers,
    Funders,
    Awards,
    Geo,
    Concepts,
    #[serde(rename = "find/works")]
    FindWorks,
    Autocomplete,
    Text,
}

impl Display for OpenAlexRequestEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Works => write!(f, "works"),
            Self::Authors => write!(f, "authors"),
            Self::Sources => write!(f, "sources"),
            Self::Institutions => write!(f, "institutions"),
            Self::Topics => write!(f, "topics"),
            Self::Keywords => write!(f, "keywords"),
            Self::Publishers => write!(f, "publishers"),
            Self::Funders => write!(f, "funders"),
            Self::Awards => write!(f, "awards"),
            Self::Geo => write!(f, "geo"),
            Self::Concepts => write!(f, "concepts"),
            Self::FindWorks => write!(f, "find/works"),
            Self::Autocomplete => write!(f, "autocomplete"),
            Self::Text => write!(f, "text"),
        }
    }
}
