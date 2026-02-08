use std::{fmt::Display, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::{ArrayRef, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// OpenAlex API Base URL
const OPENALEX_API: &str = "https://api.openalex.org/";

/// OpenAlex Entities
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OpenAlexEntity {
    Work(Work),
    Author(Author),
    Source(Source),
    Institution(Institution),
    Topic(Topic),
    Keyword(Keyword),
    Publisher(Publisher),
    Funder(Funder),
    Award(Award),
    Geo(Geo),
    Concept(Concept),
    #[default]
    #[serde(other)]
    Unknown,
}

//
// ===== Shared enums =====
//

#[derive(Debug, Serialize, Deserialize, Default)]
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
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum Currency {
    USD,
    EUR,
    GBP,
    JPY,
    CNY,
    AUD,
    CAD,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ConceptLevel {
    Domain,
    Field,
    Subfield,
    Topic,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InstitutionRelationship {
    Parent,
    Child,
    Related,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
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
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RoleType {
    Institution,
    Funder,
    Publisher,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Journal,
    Repository,
    Conference,
    EbookPlatform,
    BookSeries,
    Metadata,
    Other,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KeywordType {
    Phrase,
    Term,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
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
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OaStatus {
    Gold,
    Hybrid,
    Green,
    Bronze,
    Closed,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
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
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AuthorPosition {
    First,
    Middle,
    Last,
    Solo,
    #[default]
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
    // pub level: Option<ConceptLevel>,
    pub level: Option<u32>,
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
    pub publication_year: Option<u32>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,
    pub abstract_inverted_index: Option<serde_json::Value>, // Raw JSON for abstracts
    // pub abstract_inverted_index: Option<HashMap<String, Vec<u32>>>,
    pub authorships: Vec<Authorship>,
    pub awards: Option<Vec<Award>>,
    pub funders: Option<Vec<Funder>>,
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
    pub concepts: Option<Vec<WorkConcept>>,
    pub topics: Option<Vec<WorkTopic>>,
    pub primary_topic: Option<WorkTopic>,
    pub keywords: Option<Vec<Keyword>>,
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

impl Work {
    pub fn to_tables(self) -> () {
        // WorkTable
        let abstract_ = if let Some(abstract_inverted_index) = self.abstract_inverted_index {
            let abstract_inverted_index = serde_json::from_value::<Map<String, Value>>(abstract_inverted_index).unwrap()
                .into_iter()
                .map(|(k,v)| (k, serde_json::from_value::<Vec<usize>>(v).unwrap()))
                .collect::<HashMap<_, _>>();
            abstract_from_inverted_index(&abstract_inverted_index)
        } else {
            String::new()
        };
        

        // WorkAuthorshipTable
    
        // WorkAwardTable

        // WorkFunderTable

        // WorkApcInfoTable

        // WorkLocationTable

        // WorkOpenAccessTable

        // WorkBiblioTable

        // WorkCitationPercentileTable

        // WorkCitedByPercentileYearTable

        // WorkCountsByYearTable

        // WorkConceptTable

        // WorkTopicTable

        // WorkKeywordTable

        // WorkMeshTagTable

        // WorkSdgTagTable

        // WorkCorrespondingAuthorTable

        // WorkCorrespondingInstitutionTable

        // WorkIdsTable

        // WorkReferenceWorksTable

        // WorkRelatedWorksTable
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkTable {
    pub id: String,
    pub display_name: String,
    pub title: String,
    pub doi: String,
    pub type_: WorkType,
    pub publication_date: String,
    pub publication_year: u32,
    pub created_date: String,
    pub updated_date: String,
    pub abstract_: String, // todo: conversion from inverse abstract
    // pub authorships: Vec<Authorship>, // todo: WorkAuthorshipTable
    // pub awards: Option<Vec<Award>>, // todo: WorkAwardTable
    // pub funders: Option<Vec<Funder>>, // todo: WorkFunderTable
    // pub apc_list: Option<ApcInfo>, // todo: WorkApcInfoTable
    pub apc_list_id: String,
    pub apc_paid: Option<ApcInfo>, // todo: WorkApcInfoTable
    pub apc_paid_id: String,
    // pub best_oa_location: Option<Location>, // todo: WorkLocationTable
    pub best_oa_location_id: String,
    // pub primary_location: Option<Location>, // todo: WorkLocationTable
    pub primary_location_id: String,
    // pub locations: Option<Vec<Location>>, // todo: WorkLocationTable
    pub locations_count: u32,
    // pub open_access: OpenAccess, // todo: WorkOpenAccessTable
    // pub biblio: Option<Biblio>, // todo: WorkBiblioTable
    // pub citation_normalized_percentile: Option<CitationPercentile>, // todo: WorkCitationPercentileTable
    pub cited_by_count: u32,
    // pub cited_by_percentile_year: Option<CitedByPercentileYear>, // todo: WorkCitedByPercentileYearTable
    // pub counts_by_year: Option<Vec<CountsByYear>>, // todo: WorkCountsByYearTable
    // pub concepts: Option<Vec<WorkConcept>>, // todo: WorkConceptTable
    // pub topics: Option<Vec<WorkTopic>>, // todo: WorkTopicTable
    // pub primary_topic: Option<WorkTopic>, // todo: WorkTopicTable
    pub primary_topic_id: String,
    // pub keywords: Option<Vec<Keyword>>, // todo: WorkKeywordTable
    // pub mesh: Option<Vec<MeshTag>>, // todo: WorkMeshTagTable
    // pub sustainable_development_goals: Option<Vec<SdgTag>>, // todo: WorkSdgTagTable
    // pub corresponding_author_ids: Vec<String>, // todo: WorkCorrespondingAuthorTable
    // pub corresponding_institution_ids: Vec<String>, // todo: WorkCorrespondingInstitutionTable
    pub countries_distinct_count: Option<u32>,
    pub institutions_distinct_count: Option<u32>,
    pub indexed_in: Vec<String>,
    // pub ids: Option<WorkIds>, // todo: WorkIdsTable
    pub is_paratext: bool,
    pub is_retracted: bool,
    pub is_xpac: bool,
    pub referenced_works: Option<Vec<String>>, // todo: WorkReferenceWorksTable
    pub referenced_works_count: Option<u32>,
    pub related_works: Option<Vec<String>>, // todo: WorkRelatedWorksTable
    pub language: Option<LanguageCode>,
}

// DM: Vec -> new table, object -> id that references a table
pub fn create_open_alex_work_fields() -> Fields {
    let field_names = ["work_id", 
        "display_name", 
        "title", 
        "doi", 
        "type_", 
        "publication_date", 
        "created_date", 
        "updated_date", 
        "abstract", 
        "work_authorship_id", // new table `WorkAuthorship`
        "work_award_id",  // new table `WorkAward`
        "work_funder_id",  // new table `WorkFunder`
        "work_apc_list_id",  // new table `WorkApcInfo`
        "work_apc_paid_id",  // new table (same as work_apc_list_id)
        "work_best_oa_location_id", // new table (same as work_apc_list_id)
        "work_primary_location", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["publication_year", 
        "doi", "type_", 
        "publication_date", 
        "locations_count", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_", 
        "type_"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    Fields::from(fields_vec)
}

pub fn create_open_alex_work_record_batch(
    names: Vec<String>,
    publishers: Vec<String>,
    subjects: Vec<String>,
    values: Vec<String>,
) -> Result<RecordBatch> {
    let names: ArrayRef = Arc::new(StringArray::from(names));
    let publishers: ArrayRef = Arc::new(StringArray::from(publishers));
    let subjects: ArrayRef = Arc::new(StringArray::from(subjects));
    let values: ArrayRef = Arc::new(StringArray::from(values));
    let batch = RecordBatch::try_from_iter(vec![
        ("name", names),
        ("publisher", publishers),
        ("subject", subjects),
        ("values", values),
    ])?;
    Ok(batch)
}

/// The Authorship object represents a single author and her institutional affiliations in the context of a given work
#[derive(Debug, Serialize, Deserialize)]
pub struct Authorship {
    pub author_position: Option<AuthorPosition>,
    pub author: Option<Author>,
    pub institutions: Option<Vec<Institution>>,
    pub is_corresponding: Option<bool>,
    pub countries: Option<Vec<CountryCode>>,
    pub raw_affiliation_strings: Option<Vec<String>>,
    pub raw_author_name: Option<Vec<String>>,
}

impl Authorship {
    pub fn to_work_authorship_table(self, work_id: &str) -> WorkAuthorshipTable {
        let author_id = if let Some(author) = self.author {
            author.id.unwrap_or_default()
        } else {
            String::new()
        };
        let institution_ids = if let Some(institutions) = self.institutions {
            institutions.into_iter().map(|i| i.id).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        WorkAuthorshipTable { 
            work_id: work_id.to_string(), 
            author_position: self.author_position.unwrap_or_default(), 
            author_id, 
            institution_ids,
            is_corresponding: self.is_corresponding.unwrap_or_default(), 
            countries: self.countries.unwrap_or_default(), 
            raw_affiliation_strings: self.raw_affiliation_strings.unwrap_or_default(), 
            raw_author_name: self.raw_author_name.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkAuthorshipTable {
    pub work_id: String,
    pub author_position: AuthorPosition,
    pub author_id: String,
    pub institution_ids: Vec<String>,
    pub is_corresponding: bool,
    pub countries: Vec<CountryCode>,
    pub raw_affiliation_strings: Vec<String>,
    pub raw_author_name: Vec<String>,
}

/// WorkAuthorship
pub fn create_work_authorship_fields() -> Fields {
    let field_names = ["work_id", 
        "author_position", 
        "author_id"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
    let field_names = [
        "institutions",
        "countries",
        "raw_affiliation_strings",
        "raw_author_name",
    ];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>(),
    );
    let field_names = [
        "is_corresponding",
    ];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Boolean, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
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
    pub source: Option<Source>,
    pub license: Option<String>,
    pub version: Option<String>,
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
pub struct TopicSubfield {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicField {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicDomain {
    pub id: Option<String>,
    pub display_name: Option<String>,
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
    pub mag: Option<String>,
    pub pmid: Option<String>,
    pub pmcid: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkTopic {
    // work_id in SQL
    // topic_id in SQL
    pub id: Option<String>,
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkConcept {
    // work_id in SQL
    // topic_id in SQL
    pub id: Option<String>,
    pub score: Option<f64>,
}


//
// ===== Author and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Author {
    pub id: Option<String>, // DM: Optional for dehydrated responses
    pub orcid: Option<String>,
    pub display_name: Option<String>,
    pub display_name_alternatives: Option<Vec<String>>,
    pub works_count: Option<u32>,
    pub cited_by_count: Option<u32>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,
    pub affiliations: Option<Vec<Affiliation>>,
    pub last_known_institutions: Option<Vec<Institution>>,
    pub ids: Option<AuthorIds>,
    pub summary_stats: Option<SummaryStats>,
    pub counts_by_year: Option<Vec<CountsByYear>>,
    pub x_concepts: Option<Vec<Concept>>,
    pub works_api_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Affiliation {
    pub institution: Institution,
    pub years: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorIds {
    pub openalex: Option<String>,
    pub orcid: Option<String>,
    pub scopus: Option<String>,
    pub twitter: Option<String>,
    pub wikipedia: Option<String>,
}

/// Flattened SQL tables
/// see <https://github.com/ourresearch/openalex-documentation-scripts/blob/main/openalex-pg-schema.sql>
/// see <https://docs.openalex.org/download-all-data/upload-to-your-database/load-to-a-relational-database>
// AuthorsCountsByYear = CountsByYear + author_id
// AuthorIds = AuthorIDs + author_id

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
    pub mag: Option<String>,
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
    pub mag: Option<String>,
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
    pub keywords: Option<Vec<Keyword>>,
    pub updated_date: Option<String>,
    pub works_count: Option<u64>,
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicIds {
    pub openalex: Option<String>,
    pub wikipedia: Option<String>,
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
// ===== Keywords and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Keyword {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,
    pub cited_by_count: Option<u32>,
    pub works_count: Option<u32>,
    pub score: Option<f64>,
    #[serde(rename = "type")]
    pub type_: Option<KeywordType>,
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

//
// ===== Award and related =====
//

#[derive(Debug, Serialize, Deserialize)]
pub struct Award {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub funder_award_id: Option<String>,
    pub funder: Option<Funder>,
    pub funded_outputs: Option<Vec<String>>,
    pub funded_outputs_count: Option<u32>,
    pub amount: Option<f32>,
    pub currency: Option<Currency>,
    pub funding_type: Option<String>,
    pub funder_scheme: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub start_year: Option<u32>,
    pub end_year: Option<u32>,
    pub landing_page_url: Option<String>,
    pub doi: Option<String>,
    pub provenance: Option<String>,
    pub lead_investigator: Option<Investigator>,
    pub co_lead_investigator: Option<Investigator>,
    pub investigators: Option<Vec<Investigator>>,
    pub works_api_url: Option<String>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Investigator {
    pub given_name: Option<String>,
    pub family_name: Option<String>,
    pub orcid: Option<String>,
    pub role_start: Option<String>,
    pub affiliation: Option<Affiliation>,
}

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
            let query = filter.iter().map(|(k,v)| format!("{k}:{v}")).collect::<Vec<_>>().join(",");
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
            let query = sort.iter().map(|(k,v)| format!("{k}:{v}")).collect::<Vec<_>>().join(",");
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
            Err(anyhow!("Missing query parameters for OpenAlex GET request."))
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

/// Struct for API response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAlexResponseWorks {
    pub(crate) results: Vec<Work>,
    pub(crate) meta: Meta,
}

impl OpenAlexResponseWorks {
    /// Parse the OpenAlexResponseWorks object into tables following the [create_values_fields] schema
    ///   where each row is a different table
    pub(crate) fn to_record_batches(&self) -> Result<RecordBatch> {
        for work in results {

        }
        Ok()
    }
}

// todo!(): OpenAlexResponseAuthors, ...

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAlexResponseFind {
    pub(crate) results: Vec<FindResponse>,
    pub(crate) meta: Meta,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FindResponse {
    pub(crate) score: Option<f32>,
    pub(crate) entity: OpenAlexEntity,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAlexResponseGroupBy {
    pub(crate) group_by: Vec<GroupByResponse>,
    pub(crate) meta: Meta,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GroupByResponse {
    pub(crate) key: String,
    pub(crate) key_display_name: String,
    pub(crate) count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Meta {
    pub(crate) count: u32,
    pub(crate) groups_count: Option<u32>,
    pub(crate) db_response_time_ms: u32,
    pub(crate) page: Option<u32>,
    pub(crate) per_page: u32,
    pub(crate) next_cursor: Option<String>,
    pub(crate) query: Option<String>,
    pub(crate) filters_applied: Option<Map<String, Value>>,
    pub(crate) timing: Option<Timing>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RateLimitResponse {
    pub(crate) api_key: String,
    pub(crate) rate_limit: u32,
    pub(crate) page: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RateLimit {
    pub(crate) credits_limit: u32,
    pub(crate) credits_remaining: u32,
    pub(crate) resets_at: String,
    pub(crate) resets_in_seconds: u32,
    pub(crate) credit_costs: CreditCosts,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CreditCosts {
    pub(crate) singleton: u32,
    pub(crate) list: u32,
    pub(crate) content: u32,
    pub(crate) vector: u32,
    pub(crate) text: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Timing {
    pub(crate) embed_ms: u32,
    pub(crate) search_ms: u32,
    pub(crate) hydrate_ms: u32,
    pub(crate) total_ms: u32,
}

// Documentation for the OpenAlex API in MarkDown from <https://docs.openalex.org/>
const OPENALEX_API_DOCUMENTATION: &str = r#"# Get lists of entities

It's easy to get a list of entity objects from from the API:`/<entity_name>`. Here's an example:

* Get a list of *all* the topics in OpenAlex:\
  [`https://api.openalex.org/topics`](https://api.openalex.org/topics)

This query returns a `meta` object with details about the query, a `results` list of [`Topic`](https://docs.openalex.org/api-entities/topics/topic-object) objects, and an empty [`group_by`](https://docs.openalex.org/how-to-use-the-api/get-groups-of-entities) list:

```json
meta: {
    count: 4516,
    db_response_time_ms: 81,
    page: 1,
    per_page: 25
    },
results: [
    // long list of Topic entities
 ],
group_by: [] // empty
```

Listing entities is a lot more useful when you add parameters to [page](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging), [filter](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists), [search](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities), and [sort](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/sort-entity-lists) them. Keep reading to learn how to do that.

# Paging

{% hint style="info" %}
You can see executable examples of paging in [this user-contributed Jupyter notebook!](https://github.com/ourresearch/openalex-api-tutorials/blob/main/notebooks/getting-started/paging.ipynb)
{% endhint %}

### Basic paging

Use the `page` query parameter to control which page of results you want (eg `page=1`, `page=2`, etc). By default there are 25 results per page; you can use the `per-page` parameter to change that to any number between 1 and 200.

* Get the 2nd page of a list:\
  [`https://api.openalex.org/works?page=2`](https://api.openalex.org/works?page=2)
* Get 200 results on the second page:\
  [`https://api.openalex.org/works?page=2&per-page=200`](https://api.openalex.org/works?page=2\&per-page=200)

Basic paging only works to get the first 10,000 results of any list. If you want to see more than 10,000 results, you'll need to use [cursor paging](#cursor-paging).

### Cursor paging

Cursor paging is a bit more complicated than [basic paging](#basic-paging), but it allows you to access as many records as you like.

To use cursor paging, you request a cursor by adding the `cursor=*` parameter-value pair to your query.

* Get a cursor in order to start cursor pagination:\
  [`https://api.openalex.org/works?filter=publication_year:2020&per-page=100&cursor=*`](https://api.openalex.org/works?filter=publication_year:2020\&per-page=100\&cursor=*)

The response to your query will include a `next_cursor` value in the response's `meta` object. Here's what it looks like:

```json
{
  "meta": {
    "count": 8695857,
    "db_response_time_ms": 28,
    "page": null,
    "per_page": 100,
    "next_cursor": "IlsxNjA5MzcyODAwMDAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI0ODg0OTk3NjQnXSI="
  },
  "results" : [
    // the first page of results
  ]
}
```

To retrieve the next page of results, copy the `meta.next_cursor` value into the cursor field of your next request.

* Get the next page of results using a cursor value:\
  [`https://api.openalex.org/works?filter=publication_year:2020&per-page=100&cursor=IlsxNjA5MzcyODAwMDAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI0ODg0OTk3NjQnXSI=`](https://api.openalex.org/works?filter=publication_year:2020\&per-page=100\&cursor=IlsxNjA5MzcyODAwMDAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI0ODg0OTk3NjQnXSI=)

This second page of results will have a new value for `meta.next_cursor`. You'll use this new value the same way you did the first, and it'll give you the second page of results. To get *all* the results, keep repeating this process until `meta.next_cursor` is null and the `results` set is empty.

Besides using cursor paging to get entities, you can also use it in [`group_by` queries](https://docs.openalex.org/how-to-use-the-api/get-groups-of-entities).

{% hint style="danger" %}
**Don't use cursor paging to download the whole dataset.**

* It's bad for you because it will take many days to page through a long list like /works or /authors.
* It's bad for us (and other users!) because it puts a massive load on our servers.

Instead, download everything at once, using the [OpenAlex snapshot](https://docs.openalex.org/download-all-data/openalex-snapshot). It's free, easy, fast, and you get all the results in same format you'd get from the API.
{% endhint %}

# Filter entity lists

Filters narrow the list down to just entities that meet a particular condition--specifically, a particular value for a particular attribute.

A list of filters are set using the `filter` parameter, formatted like this: `filter=attribute:value,attribute2:value2`. Examples:

* Get the works whose [type](https://docs.openalex.org/api-entities/works/work-object#type) is `book`:\
  [`https://api.openalex.org/works?filter=type:book`](https://api.openalex.org/works?filter=type:book)
* Get the authors whose name is Einstein:\
  [`https://api.openalex.org/authors?filter=display_name.search:einstein`](https://api.openalex.org/authors?filter=display_name.search:einstein)

Filters are case-insensitive.

## Logical expressions

### Inequality

For numerical filters, use the less-than (`<`) and greater-than (`>`) symbols to filter by inequalities. Example:

* Get sources that host more than 1000 works:\
  [`https://api.openalex.org/sources?filter=works_count:>1000`](https://api.openalex.org/sources?filter=works_count:%3E1000)

Some attributes have special filters that act as syntactic sugar around commonly-expressed inequalities: for example, the `from_publication_date` filter on `works`. See the endpoint-specific documentation below for more information. Example:

* Get all works published between 2022-01-01 and 2022-01-26 (inclusive):\
  [`https://api.openalex.org/works?filter=from_publication_date:2022-01-01,to_publication_date:2022-01-26`](https://api.openalex.org/works?filter=from_publication_date:2022-01-01,to_publication_date:2022-01-26)

### Negation (NOT)

You can negate any filter, numerical or otherwise, by prepending the exclamation mark symbol (`!`) to the filter value. Example:

* Get all institutions *except* for ones located in the US:\
  [`https://api.openalex.org/institutions?filter=country_code:!us`](https://api.openalex.org/institutions?filter=country_code:!us)

### Intersection (AND)

By default, the returned result set includes only records that satisfy *all* the supplied filters. In other words, filters are combined as an AND query. Example:

* Get all works that have been cited more than once *and* are free to read:\
  [`https://api.openalex.org/works?filter=cited_by_count:>1,is_oa:true`](https://api.openalex.org/works?filter=cited_by_count:%3E1,is_oa:true)

To create an AND query within a single attribute, you can either repeat a filter, or use the plus symbol (`+`):

* Get all the works that have an author from France *and* an author from the UK:
  * Using repeating filters: [`https://api.openalex.org/works?filter=institutions.country_code:fr,institutions.country_code:gb`](https://api.openalex.org/works?filter=institutions.country_code:fr,institutions.country_code:gb)
  * Using the plus symbol (`+`): [`https://api.openalex.org/works?filter=institutions.country_code:fr+gb`](https://api.openalex.org/works?filter=institutions.country_code:fr+gb)

Note that the plus symbol (`+`) syntax will not work for search filters, boolean filters, or numeric filters.

### Addition (OR)

Use the pipe symbol (`|`) to input lists of values such that *any* of the values can be satisfied--in other words, when you separate filter values with a pipe, they'll be combined as an `OR` query. Example:

* Get all the works that have an author from France or an author from the UK:\
  [`https://api.openalex.org/works?filter=institutions.country_code:fr|gb`](https://api.openalex.org/works?filter=institutions.country_code:fr|gb)

This is particularly useful when you want to retrieve a many records by ID all at once. Instead of making a whole bunch of singleton calls in a loop, you can make one call, like this:

* Get the works with DOI `10.1371/journal.pone.0266781` *or* with DOI `10.1371/journal.pone.0267149` (note the pipe separator between the two DOIs):\
  [`https://api.openalex.org/works?filter=doi:https://doi.org/10.1371/journal.pone.0266781|https://doi.org/10.1371/journal.pone.0267149`](https://api.openalex.org/works?filter=doi:https://doi.org/10.1371/journal.pone.0266781|https://doi.org/10.1371/journal.pone.0267149)

You can combine up to 100 values for a given filter in this way. You will also need to use the parameter `per-page=100` to get all of the results per query. See our [blog post](https://blog.ourresearch.org/fetch-multiple-dois-in-one-openalex-api-request/) for a tutorial.

{% hint style="danger" %}
You can use OR for values *within* a given filter, but not *between* different filters. So this, for example, **doesn't work and will return an error**:

* Get either French works *or* ones published in the journal with ISSN 0957-1558:\
  [`https://api.openalex.org/works?filter=institutions.country_code:fr|primary_location.source.issn:0957-1558`](https://api.openalex.org/works?filter=institutions.country_code:fr|primary_location.source.issn:0957-1558)
  {% endhint %}

## Available Filters

The filters for each entity can be found here:

* [Works](https://docs.openalex.org/api-entities/works/filter-works)
* [Authors](https://docs.openalex.org/api-entities/authors/filter-authors)
* [Sources](https://docs.openalex.org/api-entities/sources/filter-sources)
* [Institutions](https://docs.openalex.org/api-entities/institutions/filter-institutions)
* [Concepts](https://docs.openalex.org/api-entities/concepts/filter-concepts)
* [Publishers](https://docs.openalex.org/api-entities/publishers/filter-publishers)
* [Funders](https://docs.openalex.org/api-entities/funders/filter-funders)

{% hint style="info" %}
**Looking for text search?** Filters match exact attribute values. If you want to search for words in titles, abstracts, or other text fields, see [Search entities](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities). Or, for AI-powered semantic search that finds conceptually related works even when they use different terminology, check out [Find similar works](https://docs.openalex.org/how-to-use-the-api/find-similar-works).
{% endhint %}

# Search entities

## The `search` parameter

The `search` query parameter finds results that match a given text search. Example:

* Get works with search term "dna" in the title, abstract, or fulltext:\
  [`https://api.openalex.org/works?search=dna`](https://api.openalex.org/works?search=dna)

When you [search `works`](https://docs.openalex.org/api-entities/works/search-works), the API looks for matches in titles, abstracts, and [fulltext](https://docs.openalex.org/api-entities/works/work-object#has_fulltext). When you [search `concepts`](https://docs.openalex.org/api-entities/concepts/search-concepts), we look in each concept's `display_name` and `description` fields. When you [search `sources`](https://docs.openalex.org/api-entities/sources/search-sources), we look at the `display_name`*,* `alternate_titles`, and `abbreviated_title` fields. When you [search `authors`](https://docs.openalex.org/api-entities/authors/search-authors), we look at the `display_name` and `display_name_alternatives` fields. When you [search `institutions`](https://docs.openalex.org/api-entities/institutions/search-institutions), we look at the `display_name`, `display_name_alternatives`, and `display_name_acronyms` fields.

For most text search we remove [stop words](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-stop-tokenfilter.html) and use [stemming](https://en.wikipedia.org/wiki/Stemming) (specifically, the [Kstem token filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-kstem-tokenfilter.html)) to improve results. So words like "the" and "an" are transparently removed, and a search for "possums" will also return records using the word "possum." With the exception of raw affiliation strings, we do not search within words but rather try to match whole words. So a search with "lun" will not match the word "lunar".

### Search without stemming

To disable stemming and the removal of stop words for searches on titles and abstracts, you can add `.no_stem` to the search filter. So, for example, if you want to search for "surgery" and not get "surgeries" too:

* [`https://api.openalex.org/works?filter=display_name.search.no_stem:surgery`](https://api.openalex.org/works?filter=display_name.search.no_stem:surgery)
* [`https://api.openalex.org/works?filter=title.search.no_stem:surgery`](https://api.openalex.org/works?filter=title.search.no_stem:surgery)
* [`https://api.openalex.org/works?filter=abstract.search.no_stem:surgery`](https://api.openalex.org/works?filter=abstract.search.no_stem:surgery)
* [`https://api.openalex.org/works?filter=title_and_abstract.search.no_stem:surgery`](https://api.openalex.org/works?filter=title_and_abstract.search.no_stem:surgery)

### Boolean searches

Including any of the words `AND`, `OR`, or `NOT` in any of your searches will enable boolean search. Those words must be UPPERCASE. You can use this in all searches, including using the `search` parameter, and using [search filters](#the-search-filter).

This allows you to craft complex queries using those boolean operators along with parentheses and quotation marks. Surrounding a phrase with quotation marks will search for an exact match of that phrase, after stemming and stop-word removal (be sure to use **double quotation marks** — `"`). Using parentheses will specify order of operations for the boolean operators. Words that are not separated by one of the boolean operators will be interpreted as `AND`.

Behind the scenes, the boolean search is using Elasticsearch's [query string query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html) on the searchable fields (such as title, abstract, and fulltext for works; see each individual entity page for specifics about that entity). Wildcard and fuzzy searches using `*`, `?` or `~` are not allowed; these characters will be removed from any searches. These searches, even when using quotation marks, will go through the same cleaning as desscribed above, including stemming and removal of stop words.

* Search for works that mention "elmo" and "sesame street," but not the words "cookie" or "monster": [`https://api.openalex.org/works?search=(elmo AND "sesame street") NOT (cookie OR monster)`](https://api.openalex.org/works?search=%28elmo%20AND%20%22sesame%20street%22%29%20NOT%20%28cookie%20OR%20monster%29)

## Relevance score

When you use search, each returned entity in the results lists gets an extra property called `relevance_score`, and the list is by default sorted in descending order of `relevance_score`. The `relevance_score` is based on text similarity to your search term. It also includes a weighting term for citation counts: more highly-cited entities score higher, all else being equal.

If you search for a multiple-word phrase, the algorithm will treat each word separately, and rank results higher when the words appear close together. If you want to return only results where the exact phrase is used, just enclose your phrase within quotes. Example:

* Get works with the exact phrase "fierce creatures" in the title or abstract (returns just a few results):\
  [`https://api.openalex.org/works?search="fierce%20creatures"`](https://api.openalex.org/works?search=%22fierce%20creatures%22)
* Get works with the words "fierce" and "creatures" in the title or abstract, with works that have the two words close together ranked higher by `relevance_score` (returns way more results):\
  [`https://api.openalex.org/works?search=fierce%20creatures`](https://api.openalex.org/works?search=fierce%20creatures)

## The search filter

You can also use search as a [filter](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists), allowing you to fine-tune the fields you're searching over. To do this, you append `.search` to the end of the property you are filtering for:

* Get authors who have "Einstein" as part of their name:\
  [`https://api.openalex.org/authors?filter=display_name.search:einstein`](https://api.openalex.org/authors?filter=display_name.search:einstein)
* Get works with "cubist" in the title:\
  [`https://api.openalex.org/works?filter=title.search:cubist`](https://api.openalex.org/works?filter=title.search:cubist)

Additionally, the filter `default.search` is available on all entities; this works the same as the [`search` parameter](#the-search-parameter).

{% hint style="info" %}
You might be tempted to use the search filter to power an autocomplete or typeahead. Instead, we recommend you use the [autocomplete endpoint](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/autocomplete-entities), which is much faster.\
\
👎 [`https://api.openalex.org/institutions?filter=display_name.search:florida`](https://api.openalex.org/institutions?filter=display_name.search:florida)

👍 [`https://api.openalex.org/autocomplete/institutions?q=Florida`](https://api.openalex.org/autocomplete/institutions?q=Florida)
{% endhint %}

## Keyword search vs. semantic search

The keyword search described on this page finds works containing specific words or phrases. Use it when you know the exact terminology you're looking for, or when you need to combine search with other filters and sorting options.

If you want to find works that are *conceptually similar*—even when they use different terminology—use [Find similar works](https://docs.openalex.org/how-to-use-the-api/find-similar-works) instead. Semantic search uses AI embeddings to match by meaning, so a query about "machine learning in healthcare" will find relevant papers even if they use terms like "AI-driven diagnosis" or "computational medicine."

| Use keyword search when...               | Use semantic search when...           |
| ---------------------------------------- | ------------------------------------- |
| You know the exact terms to search for   | You want conceptually related works   |
| You need to combine with filters/sorting | You're exploring a new research area  |
| You want to search specific fields       | Your query is a sentence or paragraph |

# Sort entity lists

Use the `?sort` parameter to specify the property you want your list sorted by. You can sort by these properties, where they exist:

* `display_name`
* `cited_by_count`
* `works_count`
* `publication_date`
* `relevance_score` (only exists if there's a [search filter](#search) active)

By default, sort direction is ascending. You can reverse this by appending `:desc` to the sort key like `works_count:desc`. You can sort by multiple properties by providing multiple sort keys, separated by commas. Examples:

* All works, sorted by `cited_by_count` (highest counts first)\
  [`https://api.openalex.org/works?sort=cited_by_count:desc`](https://api.openalex.org/works?sort=cited_by_count:desc)
* All sources, in alphabetical order by title:\
  [`https://api.openalex.org/sources?sort=display_name`](https://api.openalex.org/sources?sort=display_name)

You can sort by relevance\_score when searching:

* Sort by year, then by relevance\_score when searching for "bioplastics":\
  [`https://api.openalex.org/works?filter=display_name.search:bioplastics&sort=publication_year:desc,relevance_score:desc`](https://api.openalex.org/works?filter=display_name.search:bioplastics\&sort=publication_year:desc,relevance_score:desc)

An error is thrown if attempting to sort by `relevance_score` without a search query.

# Select fields

You can use `select` to limit the fields that are returned in results.

* Display works with only the `id`, `doi`, and `display_name` returned in the results\
  [`https://api.openalex.org/works?select=id,doi,display\_name`](https://api.openalex.org/works?select=id,doi,display_name)

```json
"results": [
  {
    "id": "https://openalex.org/W1775749144",
    "doi": "https://doi.org/10.1016/s0021-9258(19)52451-6",
    "display_name": "PROTEIN MEASUREMENT WITH THE FOLIN PHENOL REAGENT"
  },
  {
    "id": "https://openalex.org/W2100837269",
    "doi": "https://doi.org/10.1038/227680a0",
    "display_name": "Cleavage of Structural Proteins during the Assembly of the Head of Bacteriophage T4"
  },
  // more results removed for brevity
]
```

## Limitations

The fields you choose must exist within the entity (of course). You can only select root-level fields.

So if we have a record like so:

```
"id": "https://openalex.org/W2138270253",
"open_access": {
  "is_oa": true,
  "oa_status": "bronze",
  "oa_url": "http://www.pnas.org/content/74/12/5463.full.pdf"
}
```

You can choose to display `id` and `open_access`, but you will get an error if you try to choose `open_access.is_oa`.

You can use select fields when getting lists of entities or a [single entity](https://docs.openalex.org/how-to-use-the-api/get-single-entities/select-fields). It does not work with [group-by](https://docs.openalex.org/how-to-use-the-api/get-groups-of-entities) or [autocomplete](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/autocomplete-entities).

# Sample entity lists

You can use `sample` to get a random list of up to 10,000 results.

* Get 100 random works\
  <https://api.openalex.org/works?sample=100&per-page=100>
* Get 50 random works that are open access and published in 2021\
  <https://api.openalex.org/works?filter=open_access.is_oa:true,publication_year:2021&sample=50&per-page=50>

You can add a `seed` value in order to retrieve the same set of random records, in the same order, multiple times.

* Get 20 random sources with a seed value\
  <https://api.openalex.org/sources?sample=20&seed=123>

{% hint style="info" %}
Depending on your query, random results with a seed value *may* change over time due to new records coming into OpenAlex.
{% endhint %}

## Limitations

* The sample size is limited to 10,000 results.
* You must provide a `seed` value when paging beyond the first page of results. Without a seed value, you might get duplicate records in your results.
* You must use [basic paging](https://docs.openalex.org/how-to-use-the-api/paging#basic-paging) when sampling. Cursor pagination is not supported.
"#;

use std::borrow::Borrow;

pub fn abstract_from_inverted_index<K>(
    inverted: &HashMap<K, Vec<usize>>,
) -> String
where
    K: Borrow<str>,
{
    // Find the maximum index to size the vector
    let max_index = inverted
        .values()
        .flat_map(|positions| positions.iter())
        .copied()
        .max()
        .unwrap_or(0);

    // Preallocate vector of words
    let mut words = vec![String::new(); max_index + 1];

    // Fill in words at their positions
    for (key, positions) in inverted {
        let word = key.borrow();
        for &pos in positions {
            if pos < words.len() {
                words[pos] = word.to_owned();
            }
        }
    }

    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_abstract_from_inverted_index_string_keys() {
        let mut idx: HashMap<String, Vec<usize>> = HashMap::new();
        idx.insert("hello".into(), vec![0]);
        idx.insert("world".into(), vec![1]);

        assert_eq!(
            abstract_from_inverted_index(&idx),
            "hello world"
        );
    }

    #[test]
    fn test_abstract_from_inverted_index_str_keys() {
        let mut idx: HashMap<&str, Vec<usize>> = HashMap::new();
        idx.insert("foo", vec![0]);
        idx.insert("bar", vec![1]);

        assert_eq!(
            abstract_from_inverted_index(&idx),
            "foo bar"
        );
    }

    #[test]
    fn test_abstract_from_inverted_index_cow_keys() {
        let mut idx: HashMap<Cow<'static, str>, Vec<usize>> = HashMap::new();
        idx.insert(Cow::Borrowed("alpha"), vec![1]);
        idx.insert(Cow::Owned("beta".into()), vec![0]);

        assert_eq!(
            abstract_from_inverted_index(&idx),
            "beta alpha"
        );
    }
}
