/// Collection of structs for parsing EUtils data
use serde::{Deserialize, Serialize};

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
