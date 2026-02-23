/// Collection of structs for SemanticScholar data
use serde::{Deserialize, Serialize};

/// Request body for Recommendations API.
/// You can provide positive and negative seed papers, a limit, and desired fields.
#[derive(Debug, Serialize, Deserialize)]
pub struct RecommendationsRequest {
    #[serde(rename = "positivePaperIds")]
    pub positive_papers: Option<Vec<String>>,
    #[serde(rename = "negativePaperIds")]
    pub negative_papers: Option<Vec<String>>,
    // /// Maximum number of recommendations to return.
    // pub(crate) limit: Option<u32>,
    // /// List of paper fields to return in the response.
    // pub(crate) fields: Option<Vec<String>>,
}

/// A seed paper can be specified by paperId or DOI.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum SeedPaper {
    PaperId {
        #[serde(rename = "paperId")]
        paper_id: String,
    },
    Doi {
        doi: String,
    },
}

/// Top-level response for recommendations.
///
/// The API returns a list of papers with requested metadata.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RecommendationsResponse {
    #[serde(rename = "recommendedPapers")]
    pub(crate) papers: Vec<Paper>,
}

/// Author metadata.
///
/// authorId may be absent for some records.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Author {
    #[serde(rename = "authorId")]
    author_id: Option<String>,
    name: String,
    aliases: Option<Vec<String>>,
    affiliations: Option<Vec<String>>,
    homepage: Option<String>,
    #[serde(rename = "paperCount")]
    paper_count: Option<u32>,
    #[serde(rename = "citationCount")]
    citation_count: Option<u32>,
    #[serde(rename = "hIndex")]
    h_index: Option<u32>,
    url: Option<String>,
}

/// TL;DR summary (if available).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Tldr {
    text: Option<String>,
}

/// External identifiers for a paper (DOI, ArXiv, DBLP, MAG, etc.)
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ExternalIds {
    #[serde(rename = "DOI")]
    pub doi: Option<String>,
    #[serde(rename = "DBLP")]
    pub dblp: Option<String>,
    #[serde(rename = "MAG")]
    pub mag: Option<String>,
    #[serde(rename = "ArXiv")]
    pub arxiv: Option<String>,
    #[serde(rename = "ACL")]
    pub acl: Option<String>,
    #[serde(rename = "CorpusId")]
    pub corpus_id: Option<String>,
    #[serde(rename = "PubMed")]
    pub pubmed: Option<String>,
    #[serde(rename = "Medline")]
    pub medline: Option<String>,
    #[serde(rename = "PubMedCentral")]
    pub pubmedcentral: Option<String>,
}

/// Publication venue metadata (conference, journal, etc.)
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PublicationVenue {
    pub id: Option<String>,
    pub name: Option<String>,
    pub r#type: Option<String>,
    pub alternate_names: Option<Vec<String>>,
    pub url: Option<String>,
}

/// Field of study classification
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FieldOfStudy {
    pub category: Option<String>, // e.g. "Computer Science"
    pub source: Option<String>,   // e.g. "s2-fos-model"
}

/// Journal metadata
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Journal {
    pub name: Option<String>,
    pub volume: Option<String>,
    pub pages: Option<String>,
    pub publisher: Option<String>,
    pub issn: Option<String>,
}

/// Open access PDF metadata
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAccessPdf {
    pub url: Option<String>,
    pub status: Option<String>,  // e.g. "HYBRID"
    pub license: Option<String>, // e.g. "CCBY"
    pub version: Option<String>, // e.g. "publishedVersion"
}

/// Citation styles
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CitationStyle {
    pub bibtex: Option<String>,
}

/// Paper metadata model with extended fields
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Paper {
    #[serde(rename = "paperId")]
    pub paper_id: String,
    pub title: String,
    pub r#abstract: Option<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    #[serde(rename = "publicationTypes")]
    pub publication_types: Option<Vec<String>>,
    #[serde(rename = "publicationDate")]
    pub publication_date: Option<String>,
    pub url: Option<String>,
    #[serde(rename = "isOpenAccess")]
    pub is_open_access: Option<bool>,
    #[serde(rename = "openAccessPdf")]
    pub open_access_pdf: Option<OpenAccessPdf>,
    #[serde(rename = "citationCount")]
    pub citation_count: Option<u32>,
    #[serde(rename = "influentialCitationCount")]
    pub influential_citation_count: Option<u32>,
    #[serde(rename = "referenceCount")]
    pub reference_count: Option<u32>,
    #[serde(rename = "fieldsOfStudy")]
    pub fields_of_study: Option<Vec<String>>,
    #[serde(rename = "s2FieldsOfStudy")]
    pub s2_fields_of_study: Option<Vec<FieldOfStudy>>,
    pub authors: Option<Vec<Author>>,
    pub tldr: Option<Tldr>,
    #[serde(rename = "externalIds")]
    pub external_ids: Option<ExternalIds>,
    #[serde(rename = "publicationVenue")]
    pub publication_venue: Option<PublicationVenue>,
    pub journal: Option<Journal>,
    #[serde(rename = "CitationStyles")]
    pub citation_styles: Option<CitationStyle>,
}

/// Author search response
/// "<https://api.semanticscholar.org/graph/v1/author/search?query=geoffrey+hinton&fields=name,url,affiliations,paperCount,citationCount,hIndex&limit=3>";
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AuthorSearchResponse {
    total: Option<u32>,
    data: Vec<Author>,
}
