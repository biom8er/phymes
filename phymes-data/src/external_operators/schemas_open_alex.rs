/// Collection of structs for parsing OpenAlex data
use serde::{Deserialize, Serialize};
// use phymes_diagnostics::HashMap;

//
// ===== Shared enums =====
//

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CountryCode {
    US,
    GB,
    DE,
    FR,
    NL,
    CN,
    JP,
    IN,
    ES,
    IT,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Currency {
    USD,
    EUR,
    GBP,
    JPY,
    CNY,
    AUD,
    CAD,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConceptLevel {
    Domain,
    Field,
    Subfield,
    Topic,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstitutionRelationship {
    Parent,
    Child,
    Related,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstitutionType {
    Education,
    Healthcare,
    Company,
    Archive,
    Nonprofit,
    Government,
    Facility,
    Other,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoleType {
    Institution,
    Funder,
    Publisher,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Journal,
    Repository,
    Conference,
    EbookPlatform,
    BookSeries,
    Metadata,
    Other,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeywordType {
    Phrase,
    Term,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkType {
    Article,
    Book,
    BookChapter,
    Dataset,
    Review,
    ReferenceEntry,
    Dissertation,
    Report,
    Other,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OaStatus {
    Gold,
    Hybrid,
    Green,
    Bronze,
    Closed,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageCode {
    En,
    De,
    Fr,
    Es,
    Zh,
    Ja,
    Ru,
    Pt,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthorPosition {
    First,
    Middle,
    Last,
    Solo,
    #[serde(other)]
    Unknown,
}

//
// ===== Shared structs =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct CountsByYear {
    pub year: u32,
    pub works_count: Option<u32>,
    pub cited_by_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SummaryStats {
    #[serde(rename = "2yr_mean_citedness")]
    pub two_year_mean_citedness: Option<f64>,
    pub h_index: Option<u32>,
    pub i10_index: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Concept {
    pub id: Option<String>,
    pub wikidata: Option<String>,
    pub display_name: Option<String>,
    pub level: Option<ConceptLevel>,
    pub score: Option<f64>,
}

//
// ===== Work and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Work {
    pub id: String,
    pub display_name: Option<String>,
    pub title: Option<String>,
    pub doi: Option<String>,
    #[serde(rename = "type")]
    pub type_: Option<WorkType>,
    pub publication_date: Option<String>,
    pub publication_year: Option<i32>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,

    pub abstract_inverted_index: Option<serde_json::Value>, // Raw JSON for abstracts
    // pub abstract_inverted_index: Option<HashMap<String, Vec<u32>>>,

    pub authorships: Vec<Authorship>,
    pub awards: Option<Vec<Award>>,
    pub funders: Option<Vec<WorkFunder>>,

    pub apc_list: Option<ApcInfo>,
    pub apc_paid: Option<ApcInfo>,

    pub best_oa_location: Option<Location>,
    pub primary_location: Option<Location>,
    pub locations: Option<Vec<Location>>,
    pub locations_count: Option<u32>,

    pub open_access: Option<OpenAccess>,

    pub biblio: Option<Biblio>,
    pub citation_normalized_percentile: Option<CitationPercentile>,
    pub cited_by_count: Option<u32>,
    pub cited_by_percentile_year: Option<CitedByPercentileYear>,
    pub counts_by_year: Option<Vec<CountsByYear>>,

    pub concepts: Option<Vec<Concept>>,
    pub topics: Option<Vec<WorkTopic>>,
    pub primary_topic: Option<WorkTopic>,
    pub keywords: Option<Vec<WorkKeyword>>,
    pub mesh: Option<Vec<MeshTag>>,
    pub sustainable_development_goals: Option<Vec<SdgTag>>,

    pub corresponding_author_ids: Option<Vec<String>>,
    pub corresponding_institution_ids: Option<Vec<String>>,
    pub countries_distinct_count: Option<u32>,
    pub institutions_distinct_count: Option<u32>,

    pub indexed_in: Option<Vec<String>>,
    pub ids: Option<WorkIds>,

    pub is_paratext: Option<bool>,
    pub is_retracted: Option<bool>,
    pub is_xpac: Option<bool>,

    pub referenced_works: Option<Vec<String>>,
    pub referenced_works_count: Option<u32>,
    pub related_works: Option<Vec<String>>,

    pub language: Option<LanguageCode>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Authorship {
    pub author_position: Option<AuthorPosition>,
    pub author: Option<WorkAuthorRef>,
    pub institutions: Option<Vec<WorkInstitutionRef>>,
    pub is_corresponding: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkAuthorRef {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub orcid: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkInstitutionRef {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub ror: Option<String>,
    pub country_code: Option<CountryCode>,
    #[serde(rename = "type")]
    pub type_: Option<InstitutionType>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Award {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub funder_award_id: Option<String>,
    pub funder_id: Option<String>,
    pub funder_display_name: Option<String>,
    pub doi: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkFunder {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub ror: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApcInfo {
    pub value: Option<u32>,
    pub currency: Option<Currency>,
    pub value_usd: Option<u32>,
    pub provenance: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Location {
    pub is_oa: Option<bool>,
    pub landing_page_url: Option<String>,
    pub pdf_url: Option<String>,
    pub source: Option<SourceRef>,
    pub license: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceRef {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub issn_l: Option<String>,
    pub issn: Option<Vec<String>>,
    pub host_organization: Option<String>,
    #[serde(rename = "type")]
    pub type_: Option<SourceType>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAccess {
    pub is_oa: Option<bool>,
    pub oa_status: Option<OaStatus>,
    pub oa_url: Option<String>,
    pub any_repository_has_fulltext: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Biblio {
    pub volume: Option<String>,
    pub issue: Option<String>,
    pub first_page: Option<String>,
    pub last_page: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CitationPercentile {
    pub value: Option<f64>,
    pub is_in_top_1_percent: Option<bool>,
    pub is_in_top_10_percent: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CitedByPercentileYear {
    pub min: Option<u32>,
    pub max: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkTopic {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub score: Option<f64>,
    pub subfield: Option<Subfield>,
    pub field: Option<Field>,
    pub domain: Option<Domain>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Subfield {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Field {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Domain {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkKeyword {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MeshTag {
    pub descriptor_ui: Option<String>,
    pub descriptor_name: Option<String>,
    pub qualifier_ui: Option<String>,
    pub qualifier_name: Option<String>,
    pub is_major_topic: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SdgTag {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkIds {
    pub openalex: Option<String>,
    pub doi: Option<String>,
    pub mag: Option<u64>,
    pub pmid: Option<String>,
    pub pmcid: Option<String>,
}

//
// ===== Author and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Author {
    pub id: String,
    pub display_name: Option<String>,
    pub display_name_alternatives: Option<Vec<String>>,
    pub orcid: Option<String>,
    pub works_count: Option<u32>,
    pub cited_by_count: Option<u32>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,

    pub affiliations: Option<Vec<Affiliation>>,
    pub last_known_institutions: Option<Vec<AuthorInstitutionRef>>,
    pub ids: Option<AuthorIds>,
    pub summary_stats: Option<SummaryStats>,
    pub counts_by_year: Option<Vec<CountsByYear>>,
    pub x_concepts: Option<Vec<Concept>>,

    pub works_api_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Affiliation {
    pub institution: AuthorInstitutionRef,
    pub years: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorInstitutionRef {
    pub id: Option<String>,
    pub ror: Option<String>,
    pub display_name: Option<String>,
    pub country_code: Option<CountryCode>,
    #[serde(rename = "type")]
    pub type_: Option<InstitutionType>,
    pub lineage: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorIds {
    pub openalex: Option<String>,
    pub orcid: Option<String>,
    pub scopus: Option<String>,
    pub twitter: Option<String>,
    pub wikipedia: Option<String>,
}

//
// ===== Source and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub display_name: Option<String>,
    pub abbreviated_title: Option<String>,
    pub alternate_titles: Option<Vec<String>>,

    pub apc_prices: Option<Vec<ApcPrice>>,
    pub apc_usd: Option<u32>,

    pub cited_by_count: Option<u32>,
    pub country_code: Option<CountryCode>,

    pub counts_by_year: Option<Vec<CountsByYear>>,

    pub created_date: Option<String>,
    pub updated_date: Option<String>,

    pub homepage_url: Option<String>,

    pub host_organization: Option<String>,
    pub host_organization_name: Option<String>,
    pub host_organization_lineage: Option<Vec<String>>,

    pub ids: Option<SourceIds>,

    pub is_core: Option<bool>,
    pub is_in_doaj: Option<bool>,
    pub is_oa: Option<bool>,

    pub issn: Option<Vec<String>>,
    pub issn_l: Option<String>,

    pub societies: Option<Vec<Society>>,

    pub summary_stats: Option<SummaryStats>,

    #[serde(rename = "type")]
    pub type_: Option<SourceType>,

    pub works_api_url: Option<String>,
    pub works_count: Option<u32>,

    pub x_concepts: Option<Vec<Concept>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApcPrice {
    pub price: Option<u32>,
    pub currency: Option<Currency>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Society {
    pub url: Option<String>,
    pub organization: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceIds {
    pub fatcat: Option<String>,
    pub issn: Option<Vec<String>>,
    pub issn_l: Option<String>,
    pub mag: Option<u64>,
    pub openalex: Option<String>,
    pub wikidata: Option<String>,
}

//
// ===== Institution and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Institution {
    pub id: String,
    pub ror: Option<String>,
    pub display_name: Option<String>,
    pub display_name_acronyms: Option<Vec<String>>,
    pub display_name_alternatives: Option<Vec<String>>,

    pub country_code: Option<CountryCode>,
    pub type_: Option<InstitutionType>,

    pub cited_by_count: Option<u64>,
    pub works_count: Option<u64>,

    pub created_date: Option<String>,
    pub updated_date: Option<String>,

    pub homepage_url: Option<String>,
    pub image_url: Option<String>,
    pub image_thumbnail_url: Option<String>,

    pub geo: Option<Geo>,

    pub ids: Option<InstitutionIds>,

    pub associated_institutions: Option<Vec<AssociatedInstitution>>,
    pub counts_by_year: Option<Vec<CountsByYear>>,
    pub lineage: Option<Vec<String>>,
    pub repositories: Option<Vec<Repository>>,
    pub roles: Option<Vec<Role>>,
    pub summary_stats: Option<SummaryStats>,
    pub x_concepts: Option<Vec<Concept>>,

    pub international: Option<InternationalNames>,

    pub is_super_system: Option<bool>,

    pub works_api_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AssociatedInstitution {
    pub id: Option<String>,
    pub ror: Option<String>,
    pub display_name: Option<String>,
    pub country_code: Option<CountryCode>,
    pub type_: Option<InstitutionType>,

    pub relationship: Option<InstitutionRelationship>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Geo {
    pub city: Option<String>,
    pub geonames_city_id: Option<String>,
    pub region: Option<String>,
    pub country_code: Option<CountryCode>,
    pub country: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionIds {
    pub openalex: Option<String>,
    pub ror: Option<String>,
    pub grid: Option<String>,
    pub mag: Option<u64>,
    pub wikipedia: Option<String>,
    pub wikidata: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Repository {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub host_organization: Option<String>,
    pub host_organization_name: Option<String>,
    pub host_organization_lineage: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Role {
    pub role: RoleType,
    pub id: String,
    pub works_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InternationalNames {
    pub display_name: Option<serde_json::Value>,
    // pub display_name: HashMap<String, String>,
}

//
// ===== Topic and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Topic {
    pub id: String,
    pub display_name: String,
    pub description: Option<String>,

    pub domain: TopicDomain,
    pub field: TopicField,
    pub subfield: TopicSubfield,

    pub ids: Option<TopicIds>,
    pub keywords: Option<Vec<TopicKeyword>>,

    pub updated_date: Option<String>,
    pub works_count: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicDomain {
    pub id: u64,
    pub display_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicField {
    pub id: u64,
    pub display_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicSubfield {
    pub id: u64,
    pub display_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicIds {
    pub openalex: Option<String>,
    pub wikipedia: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicKeyword {
    pub display_name: String,
    pub r#type: Option<KeywordType>,
}

//
// ===== Publisher and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Publisher {
    pub id: String,
    pub display_name: String,

    pub alternate_titles: Option<Vec<String>>,
    pub country_codes: Option<Vec<CountryCode>>,

    pub cited_by_count: Option<u64>,
    pub works_count: Option<u64>,

    pub created_date: Option<String>,
    pub updated_date: Option<String>,

    pub hierarchy_level: Option<u32>,
    pub parent_publisher: Option<String>,
    pub lineage: Option<Vec<String>>,

    pub ids: Option<PublisherIds>,
    pub roles: Option<Vec<Role>>,
    pub counts_by_year: Option<Vec<CountsByYear>>,
    pub summary_stats: Option<SummaryStats>,

    pub image_url: Option<String>,
    pub image_thumbnail_url: Option<String>,

    pub sources_api_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherIds {
    pub openalex: Option<String>,
    pub ror: Option<String>,
    pub wikidata: Option<String>,
}

//
// ===== Funder and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Funder {
    pub id: String,
    pub display_name: String,

    pub alternate_titles: Option<Vec<String>>,
    pub description: Option<String>,

    pub country_code: Option<CountryCode>,

    pub cited_by_count: Option<u64>,
    pub works_count: Option<u64>,
    pub grants_count: Option<u64>,

    pub created_date: Option<String>,
    pub updated_date: Option<String>,

    pub homepage_url: Option<String>,
    pub image_url: Option<String>,
    pub image_thumbnail_url: Option<String>,

    pub ids: Option<FunderIds>,
    pub roles: Option<Vec<Role>>,
    pub counts_by_year: Option<Vec<CountsByYear>>,
    pub summary_stats: Option<SummaryStats>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunderIds {
    pub openalex: Option<String>,
    pub ror: Option<String>,
    pub wikidata: Option<String>,
    pub crossref: Option<String>,
    pub doi: Option<String>,
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