use std::{fmt::Display, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::RecordBatch, datatypes::{DataType, Field, Fields, SchemaRef}};
use crate::{AvailableSchemaTrait, BuildableTrait, BuilderTrait, DataFormat, JsonSchemaTrait, MappableTrait, Table, TableBuilderTrait, TableTrait, create_route_bytes_record_batch, create_schema_from_fields};
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

impl Display for CountryCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::US => write!(f, "US"),
            Self::GB => write!(f, "GB"),
            Self::DE => write!(f, "DE"),
            Self::FR => write!(f, "FR"),
            Self::NL => write!(f, "NL"),
            Self::CN => write!(f, "CN"),
            Self::JP => write!(f, "JP"),
            Self::IN => write!(f, "IN"),
            Self::ES => write!(f, "ES"),
            Self::IT => write!(f, "IT"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InstitutionRelationship {
    Parent,
    Child,
    Related,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RoleType {
    Institution,
    Funder,
    Publisher,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum KeywordType {
    Phrase,
    Term,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
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

impl Display for AuthorPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::First => write!(f, "First"),
            Self::Middle => write!(f, "Middle"),
            Self::Last => write!(f, "Last"),
            Self::Solo => write!(f, "Solo"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

//
// ===== Shared structs =====
//

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CountsByYear {
    pub year: u32,
    pub works_count: Option<u32>,
    pub cited_by_count: u32,
}

impl CountsByYear {
    pub fn to_work_counts_by_year(self, work_id: &str) -> WorkCountsByYearTable {
        WorkCountsByYearTable { work_id: work_id.to_string(), year: self.year, cited_by_count: self.cited_by_count }
    }
    pub fn to_author_counts_by_year(self, author_id: &str) -> AuthorCountsByYearTable {
        AuthorCountsByYearTable { author_id: author_id.to_string(), year: self.year, cited_by_count: self.cited_by_count }
    }
    pub fn to_institution_counts_by_year(self, institution_id: &str) -> InstitutionCountsByYearTable {
        InstitutionCountsByYearTable { institution_id: institution_id.to_string(), year: self.year, cited_by_count: self.cited_by_count }
    }
    pub fn to_funder_counts_by_year(self, funder_id: &str) -> FunderCountsByYearTable {
        FunderCountsByYearTable { funder_id: funder_id.to_string(), year: self.year, cited_by_count: self.cited_by_count }
    }
    pub fn to_publisher_counts_by_year(self, publisher_id: &str) -> PublisherCountsByYearTable {
        PublisherCountsByYearTable { publisher_id: publisher_id.to_string(), year: self.year, cited_by_count: self.cited_by_count }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SummaryStats {
    #[serde(rename = "2yr_mean_citedness")]
    pub two_year_mean_citedness: Option<f64>,
    pub h_index: Option<u32>,
    pub i10_index: Option<u32>,
}

impl SummaryStats {
    pub fn to_author_summary_stats_table(self, author_id: &str) -> AuthorSummaryStatsTable {
        AuthorSummaryStatsTable { author_id: author_id.to_string(), 
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(), 
            h_index: self.h_index.unwrap_or_default(), 
            i10_index: self.i10_index.unwrap_or_default() 
        }
    }
    pub fn to_institution_summary_stats_table(self, institution_id: &str) -> InstitutionSummaryStatsTable {
        InstitutionSummaryStatsTable { institution_id: institution_id.to_string(), 
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(), 
            h_index: self.h_index.unwrap_or_default(), 
            i10_index: self.i10_index.unwrap_or_default() 
        }
    }
    pub fn to_funder_summary_stats_table(self, funder_id: &str) -> FunderSummaryStatsTable {
        FunderSummaryStatsTable { funder_id: funder_id.to_string(), 
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(), 
            h_index: self.h_index.unwrap_or_default(), 
            i10_index: self.i10_index.unwrap_or_default() 
        }
    }
    pub fn to_publisher_summary_stats_table(self, publisher_id: &str) -> PublisherSummaryStatsTable {
        PublisherSummaryStatsTable { publisher_id: publisher_id.to_string(), 
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(), 
            h_index: self.h_index.unwrap_or_default(), 
            i10_index: self.i10_index.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
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
    pub abstract_inverted_index: Option<serde_json::Value>,
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

impl Work {
    pub fn to_tables(self) -> (WorkTable, 
        Vec<WorkAuthorshipTable>, 
        Vec<WorkAwardTable>, 
        Vec<WorkFunderTable>, 
        Vec<WorkApcInfoTable>, 
        Vec<WorkLocationTable>,
        Option<WorkOpenAccessTable>,
        Option<WorkBiblioTable>,
        Option<WorkCitationPercentileTable>,
        Option<WorkCitedByPercentileYearTable>,
        Vec<WorkCountsByYearTable>,
        Vec<WorkConceptTable>,
        Vec<WorkTopicTable>,
        Vec<WorkKeywordTable>,
        Vec<WorkMeshTagTable>,
        Vec<WorkSdgTagTable>,
        Vec<WorkCorrespondingAuthorTable>,
        Vec<WorkCorrespondingInstitutionTable>,
        Vec<WorkIndexedInTable>,
        Option<WorkIdsTable>,
        Vec<WorkReferencedWorksTable>,
        Vec<WorkRelatedWorksTable>,
    ) {
        // WorkAuthorshipTable
        let work_authorship_table = self.authorships.into_iter()
            .map(|t| t.to_work_authorship_table(&self.id))
            .collect::<Vec<_>>();
    
        // WorkAwardTable
        let work_award_table = self.awards.unwrap_or_default().into_iter()
            .map(|t| t.to_work_award_table(&self.id))
            .collect::<Vec<_>>();

        // WorkFunderTable
        let work_funder_table = self.funders.unwrap_or_default().into_iter()
            .map(|t| t.to_work_funder_table(&self.id))
            .collect::<Vec<_>>();

        // WorkApcInfoTable
        let mut work_apc_info_table = Vec::new();
        if let Some(apc_info) = self.apc_list {
            let t = apc_info.to_work_apc_info_table(&self.id, true, false);
            work_apc_info_table.push(t);
        }
        if let Some(apc_info) = self.apc_paid {
            let t = apc_info.to_work_apc_info_table(&self.id, false, true);
            work_apc_info_table.push(t);
        }

        // WorkLocationTable
        let work_location_table = self.locations.unwrap_or_default().into_iter()
            .map(|t| if self.best_oa_location.is_some() && self.best_oa_location.as_ref().unwrap() == &t {
                t.to_work_location_table(&self.id, true, false)
            } else if self.primary_location.is_some() && self.primary_location.as_ref().unwrap() == &t {
                t.to_work_location_table(&self.id, false, true)
            } else {
                t.to_work_location_table(&self.id, false, false)
            })
            .collect::<Vec<_>>();

        // WorkOpenAccessTable
        let work_open_access_table = self.open_access.map(|t| t.to_work_open_access_table(&self.id));

        // WorkBiblioTable
        let work_biblio_table = self.biblio.map(|t| t.to_work_biblio_table(&self.id));

        // WorkCitationPercentileTable
        let work_citation_normalized_percentile_table = self.citation_normalized_percentile.map(|t| t.to_work_citation_percentile_table(&self.id));

        // WorkCitedByPercentileYearTable
        let work_cited_percentile_year_table = self.cited_by_percentile_year.map(|t| t.to_work_cited_by_percentile_year(&self.id));

        // WorkCountsByYearTable
        let work_counts_by_year_table = self.counts_by_year.unwrap_or_default().into_iter()
            .map(|t| t.to_work_counts_by_year(&self.id))
            .collect::<Vec<_>>();

        // WorkConceptTable
        let work_concepts_table = self.concepts.unwrap_or_default().into_iter()
            .map(|t| t.to_work_concept_table(&self.id))
            .collect::<Vec<_>>();

        // WorkTopicTable
        let work_topics_table = self.topics.unwrap_or_default().into_iter()
            .map(|t| if self.primary_topic.is_some() && self.primary_topic.as_ref().unwrap() == &t {
                t.to_work_topic_table(&self.id, true)
            } else {
                t.to_work_topic_table(&self.id, false)
            })
            .collect::<Vec<_>>();

        // WorkKeywordTable
        let work_keywords_table = self.keywords.unwrap_or_default().into_iter()
            .map(|t| t.to_work_keyword_table(&self.id))
            .collect::<Vec<_>>();

        // WorkMeshTagTable
        let work_mesh_tag_table = self.mesh.unwrap_or_default().into_iter()
            .map(|t| t.to_work_mesh_tag_table(&self.id))
            .collect::<Vec<_>>();

        // WorkSdgTagTable
        let work_sdg_tag_table = self.sustainable_development_goals.unwrap_or_default().into_iter()
            .map(|t| t.to_work_sdg_tag_table(&self.id))
            .collect::<Vec<_>>();

        // WorkCorrespondingAuthorTable
        let work_corresponding_author_table = self.corresponding_author_ids.unwrap_or_default().into_iter()
            .map(|t| WorkCorrespondingAuthorTable { work_id: self.id.to_owned(), corresponding_author_id: t})
            .collect::<Vec<_>>();

        // WorkCorrespondingInstitutionTable
        let work_corresponding_insitution_table = self.corresponding_institution_ids.unwrap_or_default().into_iter()
            .map(|t| WorkCorrespondingInstitutionTable { work_id: self.id.to_owned(), corresponding_institution_id: t})
            .collect::<Vec<_>>();

        // WorkIndexedInTable
        let work_indexed_in_table = self.indexed_in.unwrap_or_default().into_iter()
            .map(|t| WorkIndexedInTable { work_id: self.id.to_owned(), indexed_in: t})
            .collect::<Vec<_>>();

        // WorkIdsTable
        let work_ids_table = self.ids.map(|t| t.to_work_ids_table(&self.id));

        // WorkReferencedWorksTable
        let work_referenced_works_table = self.referenced_works.unwrap_or_default().into_iter()
            .map(|t| WorkReferencedWorksTable { work_id: self.id.to_owned(), referenced_work_id: t})
            .collect::<Vec<_>>();

        // WorkRelatedWorksTable
        let work_related_works_table = self.related_works.unwrap_or_default().into_iter()
            .map(|t| WorkRelatedWorksTable { work_id: self.id.to_owned(), related_work_id: t})
            .collect::<Vec<_>>();

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
        let work_table = WorkTable {
            work_id: self.id,
            display_name: self.display_name.unwrap_or_default(),
            title: self.title.unwrap_or_default(),
            doi: self.doi.unwrap_or_default(),
            type_: self.type_.unwrap_or_default(),
            publication_date: self.publication_date.unwrap_or_default(),
            publication_year: self.publication_year.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            abstract_,
            locations_count: self.locations_count.unwrap_or_default(),
            cited_by_count: self.cited_by_count.unwrap_or_default(),
            countries_distinct_count: self.countries_distinct_count.unwrap_or_default(),
            institutions_distinct_count: self.institutions_distinct_count.unwrap_or_default(),
            is_paratext: self.is_paratext.unwrap_or_default(),
            is_retracted: self.is_retracted.unwrap_or_default(),
            is_xpac: self.is_xpac.unwrap_or_default(),
            referenced_works_count: self.referenced_works_count.unwrap_or_default(),
            language: self.language.unwrap_or_default(),
        };

        (work_table, 
            work_authorship_table, 
            work_award_table, 
            work_funder_table, 
            work_apc_info_table, 
            work_location_table, 
            work_open_access_table, 
            work_biblio_table, 
            work_citation_normalized_percentile_table,
            work_cited_percentile_year_table,
            work_counts_by_year_table,
            work_concepts_table,
            work_topics_table,
            work_keywords_table,
            work_mesh_tag_table,
            work_sdg_tag_table,
            work_corresponding_author_table,
            work_corresponding_insitution_table,
            work_indexed_in_table,
            work_ids_table,
            work_referenced_works_table,
            work_related_works_table
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkTable {
    pub work_id: String,
    pub display_name: String,
    pub title: String,
    pub doi: String,
    pub type_: WorkType,
    pub publication_date: String,
    pub publication_year: u32,
    pub created_date: String,
    pub updated_date: String,
    pub abstract_: String,
    pub locations_count: u32,
    pub cited_by_count: u32,
    pub countries_distinct_count: u32,
    pub institutions_distinct_count: u32,
    pub is_paratext: bool,
    pub is_retracted: bool,
    pub is_xpac: bool,
    pub referenced_works_count: u32,
    pub language: LanguageCode,
}

impl WorkTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "display_name", 
            "title", 
            "doi", 
            "type_", 
            "publication_date", 
            "created_date", 
            "updated_date", 
            "abstract_", 
            "language"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["publication_year", 
            "locations_count", 
            "countries_distinct_count", 
            "institutions_distinct_count", 
            "referenced_works_count"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        let field_names = ["is_paratext", 
            "is_retracted", 
            "is_xpac"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Boolean, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
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
    pub raw_author_name: Option<String>,
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
    pub raw_author_name: String,
}

impl WorkAuthorshipTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "author_position", 
            "author_id",
            "raw_author_name"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
        let field_names = [
            "institution_ids",
            "countries",
            "raw_affiliation_strings",
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
}

impl MappableTrait for WorkAuthorshipTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkAuthorshipTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApcInfo {
    pub value: Option<u32>,
    pub currency: Option<Currency>,
    pub value_usd: Option<u32>,
    pub provenance: Option<String>,
}

impl ApcInfo {
    pub fn to_work_apc_info_table(self, work_id: &str, is_list: bool, is_paid: bool) -> WorkApcInfoTable {
        WorkApcInfoTable { 
            work_id: work_id.to_string(), 
            is_list, 
            is_paid, 
            value: self.value.unwrap_or_default(), 
            currency: self.currency.unwrap_or_default(), 
            value_usd: self.value_usd.unwrap_or_default(), 
            provenance: self.provenance.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkApcInfoTable {
    pub work_id: String,
    pub is_list: bool,
    pub is_paid: bool,
    pub value: u32,
    pub currency: Currency,
    pub value_usd: u32,
    pub provenance: String,
}

impl WorkApcInfoTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "Currency", 
            "provenance"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "value",
            "value_usd",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "is_list",
            "is_paid",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkApcInfoTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkApcInfoTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkAwardTable {
    pub work_id: String,
    pub award_id: String,
}

impl WorkAwardTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "award_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkAwardTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkAwardTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkFunderTable {
    pub work_id: String,
    pub funder_id: String,
}

impl WorkFunderTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "funder_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkFunderTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkFunderTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct Location {
    pub is_oa: Option<bool>,
    pub landing_page_url: Option<String>,
    pub pdf_url: Option<String>,
    pub source: Option<Source>,
    pub license: Option<String>,
    pub version: Option<String>,
}

impl Location {
    pub fn to_work_location_table(self, work_id: &str, is_best_oa: bool, is_primary: bool) -> WorkLocationTable {
        let source_id = if let Some(source) = self.source {
            source.id
        } else {
            String::new()
        };
        WorkLocationTable { 
            work_id: work_id.to_string(), 
            is_best_oa,
            is_primary, 
            is_oa: self.is_oa.unwrap_or_default(), 
            landing_page_url: self.landing_page_url.unwrap_or_default(), 
            pdf_url: self.pdf_url.unwrap_or_default(), 
            source_id, 
            license: self.license.unwrap_or_default(), 
            version: self.version.unwrap_or_default(), 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkLocationTable {
    pub work_id: String,
    pub is_best_oa: bool,
    pub is_primary: bool,
    pub is_oa: bool,
    pub landing_page_url: String,
    pub pdf_url: String,
    pub source_id: String,
    pub license: String,
    pub version: String,
}

impl WorkLocationTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "landing_page_url", 
            "pdf_url", 
            "source_id", 
            "license", 
            "version"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "is_best_oa",
            "is_primary",
            "is_oa",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkLocationTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkLocationTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAccess {
    pub is_oa: Option<bool>,
    pub oa_status: Option<OaStatus>,
    pub oa_url: Option<String>,
    pub any_repository_has_fulltext: Option<bool>,
}

impl OpenAccess {
    pub fn to_work_open_access_table(self, work_id: &str) -> WorkOpenAccessTable {
        WorkOpenAccessTable { 
            work_id: work_id.to_string(), 
            is_oa: self.is_oa.unwrap_or_default(), 
            oa_status: self.oa_status.unwrap_or_default(), 
            oa_url: self.oa_url.unwrap_or_default(), 
            any_repository_has_fulltext: self.any_repository_has_fulltext.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkOpenAccessTable {
    pub work_id: String,
    pub is_oa: bool,
    pub oa_status: OaStatus,
    pub oa_url: String,
    pub any_repository_has_fulltext: bool,
}

impl WorkOpenAccessTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "oa_status", 
            "oa_url"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "is_oa",
            "any_repository_has_fulltext",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkOpenAccessTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkOpenAccessTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Biblio {
    pub volume: Option<String>,
    pub issue: Option<String>,
    pub first_page: Option<String>,
    pub last_page: Option<String>,
}

impl Biblio {
    pub fn to_work_biblio_table(self, work_id: &str) -> WorkBiblioTable {
        WorkBiblioTable { work_id: work_id.to_string(), 
            volume: self.volume.unwrap_or_default(), 
            issue: self.issue.unwrap_or_default(), 
            first_page: self.first_page.unwrap_or_default(), 
            last_page: self.last_page.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkBiblioTable {
    pub work_id: String,
    pub volume: String,
    pub issue: String,
    pub first_page: String,
    pub last_page: String,
}

impl WorkBiblioTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "volume", 
            "issue", 
            "first_page", 
            "last_page"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkBiblioTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkBiblioTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CitationPercentile {
    pub value: Option<f64>,
    pub is_in_top_1_percent: Option<bool>,
    pub is_in_top_10_percent: Option<bool>,
}

impl CitationPercentile {
    pub fn to_work_citation_percentile_table(self, work_id: &str) -> WorkCitationPercentileTable {
        WorkCitationPercentileTable { work_id: work_id.to_string(), 
            value: self.value.unwrap_or_default(), 
            is_in_top_1_percent: self.is_in_top_1_percent.unwrap_or_default(), 
            is_in_top_10_percent: self.is_in_top_10_percent.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkCitationPercentileTable {
    pub work_id: String,
    pub value: f64,
    pub is_in_top_1_percent: bool,
    pub is_in_top_10_percent: bool,
}

impl WorkCitationPercentileTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "value",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float64, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "is_in_top_1_percent",
            "is_in_top_10_percent",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkCitationPercentileTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkCitationPercentileTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CitedByPercentileYear {
    pub min: Option<u32>,
    pub max: Option<u32>,
}

impl CitedByPercentileYear {
    pub fn to_work_cited_by_percentile_year(self, work_id: &str) -> WorkCitedByPercentileYearTable {
        WorkCitedByPercentileYearTable {
            work_id: work_id.to_string(),
            min: self.min.unwrap_or_default(),
            max: self.max.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkCitedByPercentileYearTable {
    pub work_id: String,
    pub min: u32,
    pub max: u32,
}

impl WorkCitedByPercentileYearTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "min",
            "max",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkCitedByPercentileYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkCitedByPercentileYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct WorkCountsByYearTable {
    pub work_id: String,
    pub year: u32,
    pub cited_by_count: u32,
}

impl WorkCountsByYearTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "year",
            "cited_by_count",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkCountsByYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkCountsByYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct WorkKeyword {
    pub id: Option<String>,
    pub score: Option<f32>,
}

impl WorkKeyword {
    pub fn to_work_keyword_table(self, work_id: &str) -> WorkKeywordTable {
        WorkKeywordTable { work_id: work_id.to_string(), keyword_id: self.id.unwrap_or_default(), score: self.score.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkKeywordTable {
    pub work_id: String,
    pub keyword_id: String,
    pub score: f32,
}

impl WorkKeywordTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "keyword_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkKeywordTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkKeywordTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MeshTag {
    pub descriptor_ui: Option<String>,
    pub descriptor_name: Option<String>,
    pub qualifier_ui: Option<String>,
    pub qualifier_name: Option<String>,
    pub is_major_topic: Option<bool>,
}

impl MeshTag {
    pub fn to_work_mesh_tag_table(self, work_id: &str) -> WorkMeshTagTable {
        WorkMeshTagTable { 
            work_id:work_id.to_string(), 
            descriptor_ui: self.descriptor_ui.unwrap_or_default(), 
            descriptor_name: self.descriptor_name.unwrap_or_default(), 
            qualifier_ui: self.qualifier_ui.unwrap_or_default(), 
            qualifier_name: self.qualifier_name.unwrap_or_default(), 
            is_major_topic: self.is_major_topic.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkMeshTagTable {
    pub work_id: String,
    pub descriptor_ui: String,
    pub descriptor_name: String,
    pub qualifier_ui: String,
    pub qualifier_name: String,
    pub is_major_topic: bool,
}

impl WorkMeshTagTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "descriptor_ui", "descriptor_name", "qualifier_ui", "qualifier_name"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["is_major_topic"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkMeshTagTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkMeshTagTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SdgTag {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub score: Option<f32>,
}

impl SdgTag {
    pub fn to_work_sdg_tag_table(self, work_id: &str) -> WorkSdgTagTable {
        WorkSdgTagTable { 
            work_id: work_id.to_string(), 
            sdg_tag_id: self.id.unwrap_or_default(), 
            display_name: self.display_name.unwrap_or_default(), 
            score: self.score.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkSdgTagTable {
    pub work_id: String,
    pub sdg_tag_id: String,
    pub display_name: String,
    pub score: f32,
}

impl WorkSdgTagTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "sdg_tag_id", "display_name"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkSdgTagTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkSdgTagTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkIds {
    pub openalex: Option<String>,
    pub doi: Option<String>,
    pub mag: Option<String>,
    pub pmid: Option<String>,
    pub pmcid: Option<String>,
}

impl WorkIds {
    pub fn to_work_ids_table(self, work_id: &str) -> WorkIdsTable {
        WorkIdsTable { 
            work_id: work_id.to_string(), 
            openalex: self.openalex.unwrap_or_default(), 
            doi: self.doi.unwrap_or_default(), 
            mag: self.mag.unwrap_or_default(), 
            pmid: self.pmid.unwrap_or_default(), 
            pmcid: self.pmcid.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkIdsTable {
    pub work_id: String,
    pub openalex: String,
    pub doi: String,
    pub mag: String,
    pub pmid: String,
    pub pmcid: String,
}

impl WorkIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id",
        "openalex",
        "doi",
        "mag",
        "pmid",
        "pmcid"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct WorkTopic {
    pub id: Option<String>,
    pub score: Option<f32>,
}

impl WorkTopic {
    pub fn to_work_topic_table(self, work_id: &str, is_primary: bool) -> WorkTopicTable {
        WorkTopicTable { 
            work_id: work_id.to_string(), 
            topic_id: self.id.unwrap_or_default(), 
            is_primary, 
            score: self.score.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkTopicTable {
    pub work_id: String,
    pub topic_id: String,
    pub is_primary: bool,
    pub score: f32,
}

impl WorkTopicTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "topic_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        let field_names = ["is_primary"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkTopicTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkTopicTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkConcept {
    pub id: Option<String>,
    pub score: Option<f32>,
}

impl WorkConcept {
    pub fn to_work_concept_table(self, work_id: &str) -> WorkConceptTable {
        WorkConceptTable { 
            work_id: work_id.to_string(), 
            concept_id: self.id.unwrap_or_default(), 
            score: self.score.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkConceptTable {
    pub work_id: String,
    pub concept_id: String,
    pub score: f32,
}

impl WorkConceptTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "concept_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkConceptTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkConceptTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkCorrespondingAuthorTable {
    pub work_id: String,
    pub corresponding_author_id: String,
}

impl WorkCorrespondingAuthorTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "corresponding_author_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkCorrespondingAuthorTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkCorrespondingAuthorTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkCorrespondingInstitutionTable {
    pub work_id: String,
    pub corresponding_institution_id: String,
}

impl WorkCorrespondingInstitutionTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "corresponding_institution_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkCorrespondingInstitutionTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkCorrespondingInstitutionTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkIndexedInTable {
    pub work_id: String,
    pub indexed_in: String,
}

impl WorkIndexedInTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "indexed_in"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkIndexedInTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkIndexedInTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkReferencedWorksTable {
    pub work_id: String,
    pub referenced_work_id: String,
}

impl WorkReferencedWorksTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "referenced_work_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkReferencedWorksTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkReferencedWorksTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkRelatedWorksTable {
    pub work_id: String,
    pub related_work_id: String,
}

impl WorkRelatedWorksTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "related_work_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for WorkRelatedWorksTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for WorkRelatedWorksTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
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
    pub x_concepts: Option<Vec<AuthorConcept>>,
    pub works_api_url: Option<String>,
}

impl Author {
    pub fn to_tables(self) -> (AuthorTable,
        Vec<AuthorDisplayNameAlternativesTable>, 
        Vec<AuthorAffiliationTable>, 
        Vec<AuthorLastKnownInstitutionsTable>, 
        Option<AuthorIdsTable>,
        Option<AuthorSummaryStatsTable>,
        Vec<AuthorCountsByYearTable>,
        Vec<AuthorConceptTable>,
    ) {
        let author_display_name_alternatives = self.display_name_alternatives.unwrap_or_default().into_iter()
            .map(|t| AuthorDisplayNameAlternativesTable { author_id: self.id.clone().unwrap_or_default(), display_name: t})
            .collect::<Vec<_>>();
        let author_affiliation = self.affiliations.unwrap_or_default().into_iter()
            .map(|t| t.to_author_affiliation_table(&self.id.clone().unwrap_or_default()))
            .collect::<Vec<_>>();
        let author_last_known_institutions = self.last_known_institutions.unwrap_or_default().into_iter()
            .map(|t| t.to_author_last_known_institutions_table(&self.id.clone().unwrap_or_default()))
            .collect::<Vec<_>>();
        let author_ids = self.ids.map(|t| t.to_author_ids_table(&self.id.clone().unwrap_or_default()));
        let author_summary_stats = self.summary_stats.map(|t| t.to_author_summary_stats_table(&self.id.clone().unwrap_or_default()));
        let author_counts_by_year = self.counts_by_year.unwrap_or_default().into_iter()
            .map(|t| t.to_author_counts_by_year(&self.id.clone().unwrap_or_default()))
            .collect::<Vec<_>>();
        let author_concepts = self.x_concepts.unwrap_or_default().into_iter()
            .map(|t| t.to_author_concept_table(&self.id.clone().unwrap_or_default()))
            .collect::<Vec<_>>();
        let author = AuthorTable {
            author_id: self.id.unwrap_or_default(),
            orcid: self.orcid.unwrap_or_default(),
            display_name: self.display_name.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default(),
            cited_by_count: self.cited_by_count.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            works_api_url: self.works_api_url.unwrap_or_default()
        };
        (author, author_display_name_alternatives, author_affiliation, author_last_known_institutions, author_ids, author_summary_stats, author_counts_by_year, author_concepts)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorTable {
    pub author_id: String,
    pub orcid: String,
    pub display_name: String,
    pub works_count: u32,
    pub cited_by_count: u32,
    pub created_date: String,
    pub updated_date: String,
    pub works_api_url: String,
}

impl AuthorTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "orcid", 
            "display_name", 
            "created_date", 
            "updated_date", 
            "works_api_url"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["works_count", 
            "cited_by_count"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorDisplayNameAlternativesTable {
    pub author_id: String,
    pub display_name: String,
}

impl AuthorDisplayNameAlternativesTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorDisplayNameAlternativesTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorDisplayNameAlternativesTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Affiliation {
    pub institution: Institution,
    pub years: Vec<u32>,
}

impl Affiliation {
    pub fn to_author_affiliation_table(self, author_id: &str) -> AuthorAffiliationTable {
        AuthorAffiliationTable {
            author_id: author_id.to_string(),
            institution_id: self.institution.id,
            years: self.years,
        }
    }
    pub fn to_award_affiliation_table(self, award_id: &str, orcid: &str) -> AwardAffiliationTable {
        AwardAffiliationTable { award_id: award_id.to_string(), orcid: orcid.to_string(), institution_id: self.institution.id, years: self.years }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorAffiliationTable {
    pub author_id: String,
    pub institution_id: String,
    pub years: Vec<u32>,
}

impl AuthorAffiliationTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "institution_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)));
        let field_names = ["years"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, list_data_type.clone(), false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorAffiliationTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorAffiliationTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorLastKnownInstitutionsTable {
    pub author_id: String,
    pub institution_id: String,
}

impl AuthorLastKnownInstitutionsTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "institution_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorLastKnownInstitutionsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorLastKnownInstitutionsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorIds {
    pub openalex: Option<String>,
    pub orcid: Option<String>,
    pub scopus: Option<String>,
    pub twitter: Option<String>,
    pub wikipedia: Option<String>,
}

impl AuthorIds {
    pub fn to_author_ids_table(self, author_id: &str) -> AuthorIdsTable {
        AuthorIdsTable { 
            author_id: author_id.to_string(), 
            openalex: self.openalex.unwrap_or_default(), 
            orcid: self.orcid.unwrap_or_default(), 
            scopus: self.scopus.unwrap_or_default(), 
            twitter: self.twitter.unwrap_or_default(), 
            wikipedia: self.wikipedia.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorIdsTable {
    pub author_id: String,
    pub openalex: String,
    pub orcid: String,
    pub scopus: String,
    pub twitter: String,
    pub wikipedia: String,
}

impl AuthorIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "openalex", "orcid", "scopus", "twitter", "wikipedia"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AuthorSummaryStatsTable {
    pub author_id: String,
    pub two_year_mean_citedness: f64,
    pub h_index: u32,
    pub i10_index: u32,
}

impl AuthorSummaryStatsTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["two_year_mean_citedness"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float64, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "h_index",
            "i10_index",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorSummaryStatsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorSummaryStatsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AuthorCountsByYearTable {
    pub author_id: String,
    pub year: u32,
    pub cited_by_count: u32,
}

impl AuthorCountsByYearTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "year",
            "cited_by_count",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorCountsByYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorCountsByYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorConcept {
    pub id: Option<String>,
    pub score: Option<f32>,
}

impl AuthorConcept {
    pub fn to_author_concept_table(self, author_id: &str) -> AuthorConceptTable {
        AuthorConceptTable { 
            author_id: author_id.to_string(), 
            concept_id: self.id.unwrap_or_default(), 
            score: self.score.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorConceptTable {
    pub author_id: String,
    pub concept_id: String,
    pub score: f32,
}

impl AuthorConceptTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "concept_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AuthorConceptTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AuthorConceptTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

//
// ===== Source and related =====
//

#[derive(Debug, Serialize, Deserialize, PartialEq)]
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

impl Source {
    pub fn to_table(self) -> (SourceTable,
        Vec<SourceAlternativeTitlesTable>,
        Vec<SourceApcInfoTable>,
        Vec<SourceCountsByYearTable>,
        Vec<SourceLineageTable>,
        Option<SourceIdsTable>,
        Vec<SourceIssnTable>,
        Vec<SourceSocietyTable>,
        Option<SourceSummaryStatsTable>,
        Vec<SourceConceptTable>,
    ) {
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceTable {
    pub source_id: String,
    pub display_name: String,
    pub abbreviated_title: String,
    pub cited_by_count: u32,
    pub country_code: CountryCode,
    pub created_date: String,
    pub updated_date: String,
    pub homepage_url: String,
    pub host_organization: String,
    pub host_organization_name: String,
    pub is_core: bool,
    pub is_in_doaj: bool,
    pub is_oa: bool,
    pub issn_l: String,
    pub type_: SourceType,
    pub works_api_url: String,
    pub works_count: u32,
}

impl SourceTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", 
            "display_name", 
            "abbreviated_title", 
            "country_code", 
            "created_date", 
            "updated_date", 
            "homepage_url", 
            "host_organization", 
            "host_organization_name", 
            "type_"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["cited_by_count", 
            "works_count"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        let field_names = ["is_core", "is_in_doaj", "is_oa"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Boolean, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ApcPrice {
    pub price: Option<u32>,
    pub currency: Option<Currency>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceApcInfoTable {
    pub source_id: String,
    pub value: u32,
    pub currency: Currency,
    pub value_usd: u32,
    pub provenance: String,
}

impl SourceApcInfoTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", 
            "Currency", 
            "provenance"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "value",
            "value_usd",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceApcInfoTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceApcInfoTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Society {
    pub url: Option<String>,
    pub organization: Option<String>,
}

impl Society {
    pub fn to_source_society_table(self, source_id: &str) -> SourceSocietyTable {
        SourceSocietyTable { source_id: source_id.to_string(), url: self.url.unwrap_or_default(), organization: self.organization.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceSocietyTable {
    pub source_id: String,
    pub url: String,
    pub organization: String,
}

impl SourceSocietyTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", "url", "organization"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceSocietyTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceSocietyTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceIds {
    pub fatcat: Option<String>,
    pub issn: Option<Vec<String>>,
    pub issn_l: Option<String>,
    pub mag: Option<String>,
    pub openalex: Option<String>,
    pub wikidata: Option<String>,
}

impl SourceIds {
    pub fn to_source_ids_table(self, source_id: &str) -> SourceIdsTable {
        SourceIdsTable { source_id: source_id.to_string(), 
            fatcat: self.fatcat.unwrap_or_default(), 
            issn: self.issn.unwrap_or_default(), 
            issn_l: self.issn_l.unwrap_or_default(), 
            mag: self.mag.unwrap_or_default(), 
            openalex: self.openalex.unwrap_or_default(), 
            wikidata: self.wikidata.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceIdsTable {
    pub source_id: String,
    pub fatcat: String,
    pub issn: Vec<String>,
    pub issn_l: String,
    pub mag: String,
    pub openalex: String,
    pub wikidata: String,
}

impl SourceIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", "fatcat", "issn_l", "mag", "openalex", "wikidata"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
        let field_names = ["issn"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, list_data_type.clone(), false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceAlternativeTitlesTable {
    pub source_id: String,
    pub display_name: String,
}

impl SourceAlternativeTitlesTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceAlternativeTitlesTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceAlternativeTitlesTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceSummaryStatsTable {
    pub source_id: String,
    pub two_year_mean_citedness: f64,
    pub h_index: u32,
    pub i10_index: u32,
}

impl SourceSummaryStatsTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["two_year_mean_citedness"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float64, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "h_index",
            "i10_index",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceSummaryStatsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceSummaryStatsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceCountsByYearTable {
    pub source_id: String,
    pub year: u32,
    pub cited_by_count: u32,
}

impl SourceCountsByYearTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "year",
            "cited_by_count",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceCountsByYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceCountsByYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceConcept {
    pub id: Option<String>,
    pub score: Option<f32>,
}

impl SourceConcept {
    pub fn to_source_concept_table(self, source_id: &str) -> SourceConceptTable {
        SourceConceptTable { 
            source_id: source_id.to_string(), 
            concept_id: self.id.unwrap_or_default(), 
            score: self.score.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceConceptTable {
    pub source_id: String,
    pub concept_id: String,
    pub score: f32,
}

impl SourceConceptTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", "concept_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceConceptTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceConceptTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceLineageTable {
    pub source_id: String,
    pub lineage_id: String,
}

impl SourceLineageTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", "lineage_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceLineageTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceLineageTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceIssnTable {
    pub source_id: String,
    pub issn: String,
}

impl SourceIssnTable {
    fn to_fields() -> Fields {
        let field_names = ["source_id", "issn"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceIssnTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceIssnTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

//
// ===== Institution and related =====
//

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Institution {
    pub id: String,
    pub ror: Option<String>,
    pub display_name: Option<String>,
    pub display_name_acronyms: Option<Vec<String>>,
    pub display_name_alternatives: Option<Vec<String>>,
    pub country_code: Option<CountryCode>,
    pub type_: Option<InstitutionType>,
    pub cited_by_count: Option<u32>,
    pub works_count: Option<u32>,
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
    pub x_concepts: Option<Vec<InstitutionConcept>>,
    pub international: Option<InternationalNames>,
    pub is_super_system: Option<bool>,
    pub works_api_url: Option<String>,
}

impl Institution {
    pub fn to_author_last_known_institutions_table(self, author_id: &str) -> AuthorLastKnownInstitutionsTable {
        AuthorLastKnownInstitutionsTable {
            author_id: author_id.to_string(),
            institution_id: self.id
        }
    }
    pub fn to_table(self) -> (InstitutionTable,
        Vec<InstitutionDisplayNameAcronymsTable>,
        Vec<InstitutionDisplayNameAlternativesTable>,
        Option<InstitutionGeoTable>,
        Option<InstitutionIdsTable>,
        Vec<InstitutionAssociatedInstitutionTable>,
        Vec<InstitutionRepositoryTable>,
        Vec<InstitutionRoleTable>,
        Option<InstitutionInternationalNamesTable>,
        Option<InstitutionSummaryStatsTable>,
        Vec<InstitutionCountsByYearTable>,
        Vec<InstitutionConceptTable>,
        Vec<InstitutionLineageTable>,
    ) {
        let institution_display_name_acronyms = self.display_name_acronyms.unwrap_or_default().into_iter()
            .map(|t| InstitutionDisplayNameAcronymsTable { institution_id: self.id.clone(), display_name: t})
            .collect::<Vec<_>>();
        let institution_display_name_alternatives = self.display_name_alternatives.unwrap_or_default().into_iter()
            .map(|t| InstitutionDisplayNameAlternativesTable { institution_id: self.id.clone(), display_name: t})
            .collect::<Vec<_>>();
        let institution_geo = self.geo.map(|t| t.to_institution_geo_table(&self.id.clone()));
        let institution_ids = self.ids.map(|t| t.to_institution_ids_table(&self.id.clone()));
        let institution_associated_institution = self.associated_institutions.unwrap_or_default().into_iter()
            .map(|t| t.to_institution_associated_institution_table(&self.id))
            .collect::<Vec<_>>();
        let institution_repository = self.repositories.unwrap_or_default().into_iter()
            .map(|t| t.to_institution_repository_table(&self.id))
            .collect::<Vec<_>>();
        let institution_role = self.roles.unwrap_or_default().into_iter()
            .map(|t| t.to_institution_role_table(&self.id))
            .collect::<Vec<_>>();
        let institution_international_names = self.international.map(|t| t.to_insitution_international_names_table(&self.id.clone()));
        let institution_summary_stats = self.summary_stats.map(|t| t.to_institution_summary_stats_table(&self.id));
        let institution_counts_by_year = self.counts_by_year.unwrap_or_default().into_iter()
            .map(|t| t.to_institution_counts_by_year(&self.id))
            .collect::<Vec<_>>();
        let institution_concepts = self.x_concepts.unwrap_or_default().into_iter()
            .map(|t| t.to_institution_concept_table(&self.id))
            .collect::<Vec<_>>();
        let institution_lineage = self.lineage.unwrap_or_default().into_iter()
            .map(|t| InstitutionLineageTable { institution_id: self.id.clone(), lineage_id: t})
            .collect::<Vec<_>>();
        let institution = InstitutionTable {
            institution_id: self.id,
            ror: self.ror.unwrap_or_default(),
            display_name: self.display_name.unwrap_or_default(),
            country_code: self.country_code.unwrap_or_default(),
            type_: self.type_.unwrap_or_default(),
            cited_by_count: self.cited_by_count.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            homepage_url: self.homepage_url.unwrap_or_default(),
            image_url: self.image_url.unwrap_or_default(),
            image_thumbnail_url: self.image_thumbnail_url.unwrap_or_default(),
            is_super_system: self.is_super_system.unwrap_or_default(),
            works_api_url: self.works_api_url.unwrap_or_default()
        };
        (institution, institution_display_name_acronyms, institution_display_name_alternatives, institution_geo, institution_ids, institution_associated_institution, institution_repository, institution_role,
        institution_international_names, institution_summary_stats, institution_counts_by_year, institution_concepts, institution_lineage)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionTable {
    pub institution_id: String,
    pub ror: String,
    pub display_name: String,
    pub country_code: CountryCode,
    pub type_: InstitutionType,
    pub cited_by_count: u32,
    pub works_count: u32,
    pub created_date: String,
    pub updated_date: String,
    pub homepage_url: String,
    pub image_url: String,
    pub image_thumbnail_url: String,
    pub is_super_system: bool,
    pub works_api_url: String,
}

impl InstitutionTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", 
            "ror", 
            "display_name", 
            "country_code", 
            "type_", 
            "created_date", 
            "updated_date", 
            "homepage_url", 
            "image_url", 
            "image_thumbnail_url", 
            "works_api_url", ];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["cited_by_count", 
            "works_count"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        let field_names = ["is_super_system"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Boolean, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionDisplayNameAcronymsTable {
    pub institution_id: String,
    pub display_name: String,
}

impl InstitutionDisplayNameAcronymsTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionDisplayNameAcronymsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionDisplayNameAcronymsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionDisplayNameAlternativesTable {
    pub institution_id: String,
    pub display_name: String,
}

impl InstitutionDisplayNameAlternativesTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionDisplayNameAlternativesTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionDisplayNameAlternativesTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionLineageTable {
    pub institution_id: String,
    pub lineage_id: String,
}

impl InstitutionLineageTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "lineage_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionLineageTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionLineageTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AssociatedInstitution {
    pub id: Option<String>,
    pub ror: Option<String>,
    pub display_name: Option<String>,
    pub country_code: Option<CountryCode>,
    pub type_: Option<InstitutionType>,
    pub relationship: Option<InstitutionRelationship>,
}

impl AssociatedInstitution {
    pub fn to_institution_associated_institution_table(self, institution_id: &str) -> InstitutionAssociatedInstitutionTable {
        InstitutionAssociatedInstitutionTable { 
            institution_id: institution_id.to_string(), 
            ror: self.ror.unwrap_or_default(), 
            display_name: self.display_name.unwrap_or_default(), 
            country_code: self.country_code.unwrap_or_default(), 
            type_: self.type_.unwrap_or_default(), 
            relationship: self.relationship.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionAssociatedInstitutionTable {
    pub institution_id: String,
    pub ror: String,
    pub display_name: String,
    pub country_code: CountryCode,
    pub type_: InstitutionType,
    pub relationship: InstitutionRelationship,
}

impl InstitutionAssociatedInstitutionTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "ror", "display_name", "country_code", "type_", "relationship"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionAssociatedInstitutionTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionAssociatedInstitutionTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Geo {
    pub city: Option<String>,
    pub geonames_city_id: Option<String>,
    pub region: Option<String>,
    pub country_code: Option<CountryCode>,
    pub country: Option<String>,
    pub latitude: Option<f32>,
    pub longitude: Option<f32>,
}

impl Geo {
    pub fn to_institution_geo_table(self, institution_id: &str) -> InstitutionGeoTable {
        InstitutionGeoTable { 
            institution_id: institution_id.to_string(), 
            city: self.city.unwrap_or_default(), 
            geonames_city_id: self.geonames_city_id.unwrap_or_default(), 
            region: self.region.unwrap_or_default(), 
            country_code: self.country_code.unwrap_or_default(), 
            country: self.country.unwrap_or_default(), 
            latitude: self.latitude.unwrap_or_default(), 
            longitude: self.longitude.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionGeoTable {
    pub institution_id: String,
    pub city: String,
    pub geonames_city_id: String,
    pub region: String,
    pub country_code: CountryCode,
    pub country: String,
    pub latitude: f32,
    pub longitude: f32,
}

impl InstitutionGeoTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "city", "geonames_city_id", "region", "country_code", "country"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
        let field_names = ["latitude", "longitude"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, list_data_type.clone(), false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionGeoTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionGeoTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InstitutionIds {
    pub openalex: Option<String>,
    pub ror: Option<String>,
    pub grid: Option<String>,
    pub mag: Option<String>,
    pub wikipedia: Option<String>,
    pub wikidata: Option<String>,
}

impl InstitutionIds {
    pub fn to_institution_ids_table(self, institution_id: &str) -> InstitutionIdsTable {
        InstitutionIdsTable { 
            institution_id: institution_id.to_string(), 
            openalex: self.openalex.unwrap_or_default(), 
            ror: self.ror.unwrap_or_default(), 
            grid: self.grid.unwrap_or_default(), 
            mag: self.mag.unwrap_or_default(), 
            wikipedia: self.wikipedia.unwrap_or_default(), 
            wikidata: self.wikidata.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionIdsTable {
    pub institution_id: String,
    pub openalex: String,
    pub ror: String,
    pub grid: String,
    pub mag: String,
    pub wikipedia: String,
    pub wikidata: String,
}

impl InstitutionIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "openalex", "ror", "grid", "mag", "wikipedia", "wikidata"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Repository {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub host_organization: Option<String>,
    pub host_organization_name: Option<String>,
    pub host_organization_lineage: Option<Vec<String>>,
}

impl Repository {
    pub fn to_institution_repository_table(self, institution_id: &str) -> InstitutionRepositoryTable {
        InstitutionRepositoryTable { 
            institution_id: institution_id.to_string(), 
            display_name: self.display_name.unwrap_or_default(), 
            host_organization: self.host_organization.unwrap_or_default(), 
            host_organization_name: self.host_organization_name.unwrap_or_default(), 
            host_organization_lineage: self.host_organization_lineage.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionRepositoryTable {
    pub institution_id: String,
    pub display_name: String,
    pub host_organization: String,
    pub host_organization_name: String,
    pub host_organization_lineage: Vec<String>,
}

impl InstitutionRepositoryTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "display_name", "host_organization", "host_organization_name"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
        let field_names = ["host_organization_lineage"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, list_data_type.clone(), false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionRepositoryTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionRepositoryTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Role {
    pub role: RoleType,
    pub id: String,
    pub works_count: u32,
}

impl Role {
    pub fn to_institution_role_table(self, institution_id: &str) -> InstitutionRoleTable {
        InstitutionRoleTable { institution_id: institution_id.to_string(), role: self.role, id: self.id, works_count: self.works_count }
    }
    pub fn to_publisher_role_table(self, publisher_id:&str) -> PublisherRoleTable {
        PublisherRoleTable { publisher_id: publisher_id.to_string(), role: self.role, id: self.id, works_count: self.works_count }
    }
    pub fn to_funder_role_table(self, funder_id:&str) -> FunderRoleTable {
        FunderRoleTable { funder_id: funder_id.to_string(), role: self.role, id: self.id, works_count: self.works_count }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionRoleTable {
    pub institution_id: String,
    pub role: RoleType,
    pub id: String,
    pub works_count: u32,
}

impl InstitutionRoleTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "role", "id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["works_count"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionRoleTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionRoleTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InternationalNames {
    pub display_name: Option<serde_json::Value>,
}

impl InternationalNames {
    pub fn to_insitution_international_names_table(self, institution_id: &str) -> InstitutionInternationalNamesTable {
        InstitutionInternationalNamesTable { institution_id: institution_id.to_string(), display_name: self.display_name.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionInternationalNamesTable {
    pub institution_id: String,
    pub display_name: serde_json::Value,
}

impl InstitutionInternationalNamesTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionInternationalNamesTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionInternationalNamesTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InstitutionSummaryStatsTable {
    pub institution_id: String,
    pub two_year_mean_citedness: f64,
    pub h_index: u32,
    pub i10_index: u32,
}

impl InstitutionSummaryStatsTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["two_year_mean_citedness"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float64, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "h_index",
            "i10_index",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionSummaryStatsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionSummaryStatsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InstitutionCountsByYearTable {
    pub institution_id: String,
    pub year: u32,
    pub cited_by_count: u32,
}

impl InstitutionCountsByYearTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "year",
            "cited_by_count",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionCountsByYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionCountsByYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InstitutionConcept {
    pub id: Option<String>,
    pub score: Option<f32>,
}

impl InstitutionConcept {
    pub fn to_institution_concept_table(self, institution_id: &str) -> InstitutionConceptTable {
        InstitutionConceptTable { 
            institution_id: institution_id.to_string(), 
            concept_id: self.id.unwrap_or_default(), 
            score: self.score.unwrap_or_default() 
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstitutionConceptTable {
    pub institution_id: String,
    pub concept_id: String,
    pub score: f32,
}

impl InstitutionConceptTable {
    fn to_fields() -> Fields {
        let field_names = ["institution_id", "concept_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["score"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for InstitutionConceptTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for InstitutionConceptTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
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
    pub keywords: Option<Vec<String>>,
    pub updated_date: Option<String>,
    pub works_count: Option<u32>,
}

impl Topic {
    pub fn to_work_topic_table(self, work_id: &str, is_primary: bool, score: f32) -> WorkTopicTable {
        WorkTopicTable { 
            work_id: work_id.to_string(), 
            topic_id: self.id, 
            is_primary,
            score
        }
    }
    pub fn to_table(self) -> (TopicTable,
        TopicDomainTable,
        TopicFieldTable,
        TopicSubfieldTable,
        Option<TopicIdsTable>,
        Vec<TopicKeywordTable>
    ) {
        let topic_domain = self.domain.to_topic_domain_table(&self.id);
        let topic_field = self.field.to_topic_field_table(&self.id);
        let topic_subfield = self.subfield.to_topic_subfield_table(&self.id);
        let topic_ids = self.ids.map(|t| t.to_topic_ids_table(&self.id));
        let topic_keyword = self.keywords.unwrap_or_default().into_iter().map(|k| TopicKeywordTable {topic_id: self.id.clone(), keyword: k}).collect::<Vec<_>>();
        let topic = TopicTable {
            topic_id: self.id,
            display_name: self.display_name,
            description: self.description.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default()
        };
        (topic, topic_domain, topic_field, topic_subfield, topic_ids, topic_keyword)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicTable {
    pub topic_id: String,
    pub display_name: String,
    pub description: String,
    pub updated_date: String,
    pub works_count: u32,
}

impl TopicTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", 
            "display_name", 
            "description", 
            "updated_date"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["works_count"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicIds {
    pub openalex: Option<String>,
    pub wikipedia: Option<String>,
}

impl TopicIds {
    pub fn to_topic_ids_table(self, topic_id: &str) -> TopicIdsTable {
        TopicIdsTable { topic_id: topic_id.to_string(), openalex: self.openalex.unwrap_or_default(), wikipedia: self.wikipedia.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicIdsTable {
    pub topic_id: String,
    pub openalex: String,
    pub wikipedia: String,
}

impl TopicIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id",
        "openalex",
        "wikipedia"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicSubfield {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

impl TopicSubfield {
    pub fn to_topic_subfield_table(self, topic_id: &str) -> TopicSubfieldTable {
        TopicSubfieldTable { topic_id: topic_id.to_string(), topic_subfield_id: self.id.unwrap_or_default(), display_name: self.display_name.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicSubfieldTable {
    pub topic_id: String,
    pub topic_subfield_id: String,
    pub display_name: String,
}

impl TopicSubfieldTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id",
        "topic_subfield_id",
        "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicSubfieldTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicSubfieldTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicField {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

impl TopicField {
    pub fn to_topic_field_table(self, topic_id: &str) -> TopicFieldTable {
        TopicFieldTable { topic_id: topic_id.to_string(), topic_field_id: self.id.unwrap_or_default(), display_name: self.display_name.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicFieldTable {
    pub topic_id: String,
    pub topic_field_id: String,
    pub display_name: String,
}

impl TopicFieldTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id",
        "topic_field_id",
        "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicFieldTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicFieldTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicDomain {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

impl TopicDomain {
    pub fn to_topic_domain_table(self, topic_id: &str) -> TopicDomainTable {
        TopicDomainTable { topic_id: topic_id.to_string(), topic_domain_id: self.id.unwrap_or_default(), display_name: self.display_name.unwrap_or_default() }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicDomainTable {
    pub topic_id: String,
    pub topic_domain_id: String,
    pub display_name: String,
}

impl TopicDomainTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id",
        "topic_domain_id",
        "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicDomainTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicDomainTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicKeywordTable {
    pub topic_id: String,
    pub keyword: String,
}

impl TopicKeywordTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "keyword"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicKeywordTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicKeywordTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
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
    pub cited_by_count: Option<u32>,
    pub works_count: Option<u32>,
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

impl Publisher {
    pub fn to_table(self) -> (PublisherTable,
        Vec<PublisherAlternativeTitlesTable>,
        Vec<PublisherCountryCodeTable>,
        Vec<PublisherLineageTable>,
        Option<PublisherIdsTable>,
        Vec<PublisherRoleTable>,
        Vec<PublisherCountsByYearTable>,
        Option<PublisherSummaryStatsTable>,
    ) {
        let publisher_alternative_titles = self.alternate_titles.unwrap_or_default().into_iter().map(|t| PublisherAlternativeTitlesTable {publisher_id: self.id.to_owned(), title: t}).collect::<Vec<_>>();
        let publisher_country_code = self.country_codes.unwrap_or_default().into_iter().map(|t| PublisherCountryCodeTable {publisher_id: self.id.to_owned(), country_code: t}).collect::<Vec<_>>();
        let publisher_lineage = self.lineage.unwrap_or_default().into_iter()
            .map(|t| PublisherLineageTable { publisher_id: self.id.clone(), lineage_id: t})
            .collect::<Vec<_>>();
        let publisher_ids = self.ids.map(|i| i.to_publisher_ids_table(&self.id));
        let publisher_role = self.roles.unwrap_or_default().into_iter().map(|t| t.to_publisher_role_table(&self.id)).collect::<Vec<_>>();
        let publisher_counts_by_year = self.counts_by_year.unwrap_or_default().into_iter().map(|t| t.to_publisher_counts_by_year(&self.id)).collect::<Vec<_>>();
        let publisher_summary_stats = self.summary_stats.map(|t| t.to_publisher_summary_stats_table(&self.id));
        let publisher = PublisherTable {
            publisher_id: self.id,
            display_name: self.display_name,
            cited_by_count: self.cited_by_count.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            hierarchy_level: self.hierarchy_level.unwrap_or_default(),
            parent_publisher: self.parent_publisher.unwrap_or_default(),
            image_url: self.image_url.unwrap_or_default(),
            image_thumbnail_url: self.image_thumbnail_url.unwrap_or_default(),
            sources_api_url: self.sources_api_url.unwrap_or_default()
        };
        (publisher, publisher_alternative_titles, publisher_country_code, publisher_lineage, publisher_ids, publisher_role, publisher_counts_by_year, publisher_summary_stats)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherTable {
    pub publisher_id: String,
    pub display_name: String,
    pub cited_by_count: u32,
    pub works_count: u32,
    pub created_date: String,
    pub updated_date: String,
    pub hierarchy_level: u32,
    pub parent_publisher: String,
    pub image_url: String,
    pub image_thumbnail_url: String,
    pub sources_api_url: String,
}

impl PublisherTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id", 
            "display_name", 
            "created_date", 
            "updated_date", 
            "parent_publisher", 
            "image_url", 
            "image_thumbnail_url", 
            "sources_api_url"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["cited_by_count", 
            "works_count", 
            "hierarchy_level"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherIds {
    pub openalex: Option<String>,
    pub ror: Option<String>,
    pub wikidata: Option<String>,
}

impl PublisherIds {
    pub fn to_publisher_ids_table(self, publisher_id: &str) -> PublisherIdsTable {
        PublisherIdsTable { publisher_id: publisher_id.to_string(), 
            openalex: self.openalex.unwrap_or_default(), 
            ror: self.ror.unwrap_or_default(), 
            wikidata: self.wikidata.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherIdsTable {
    pub publisher_id: String,
    pub openalex: String,
    pub ror: String,
    pub wikidata: String,
}

impl PublisherIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id", "openalex", "ror", "wikidata"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherLineageTable {
    pub publisher_id: String,
    pub lineage_id: String,
}

impl PublisherLineageTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id", "lineage_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherLineageTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherLineageTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherCountryCodeTable {
    pub publisher_id: String,
    pub country_code: CountryCode,
}

impl PublisherCountryCodeTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id", "country_code"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherCountryCodeTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherCountryCodeTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherRoleTable {
    pub publisher_id: String,
    pub role: RoleType,
    pub id: String,
    pub works_count: u32,
}

impl PublisherRoleTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id", "role", "id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["works_count"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherRoleTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherRoleTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PublisherAlternativeTitlesTable {
    pub publisher_id: String,
    pub title: String,
}

impl PublisherAlternativeTitlesTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id", "title"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherAlternativeTitlesTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherAlternativeTitlesTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PublisherSummaryStatsTable {
    pub publisher_id: String,
    pub two_year_mean_citedness: f64,
    pub h_index: u32,
    pub i10_index: u32,
}

impl PublisherSummaryStatsTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["two_year_mean_citedness"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float64, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "h_index",
            "i10_index",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherSummaryStatsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherSummaryStatsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PublisherCountsByYearTable {
    pub publisher_id: String,
    pub year: u32,
    pub cited_by_count: u32,
}

impl PublisherCountsByYearTable {
    fn to_fields() -> Fields {
        let field_names = ["publisher_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "year",
            "cited_by_count",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for PublisherCountsByYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for PublisherCountsByYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

//
// ===== Keywords and related =====
//

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Keyword {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,
    pub cited_by_count: Option<u32>,
    pub works_count: Option<u32>,
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
    pub cited_by_count: Option<u32>,
    pub works_count: Option<u32>,
    pub grants_count: Option<u32>,
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

impl Funder {
    pub fn to_work_funder_table(self, work_id: &str) -> WorkFunderTable {
        WorkFunderTable { work_id: work_id.to_string(), funder_id: self.id }
    }
    pub fn to_award_funder_table(self, award_id: &str) -> AwardFunderTable {
        AwardFunderTable { award_id: award_id.to_string(), funder_id: self.id }
    }
    pub fn to_table(self) -> (FunderTable,
        Vec<FunderAlternativeTitlesTable>,
        Option<FunderIdsTable>,
        Vec<FunderRoleTable>,
        Vec<FunderCountsByYearTable>,
        Option<FunderSummaryStatsTable>,
    ) {
        let funder_alternative_titles = self.alternate_titles.unwrap_or_default().into_iter().map(|t| FunderAlternativeTitlesTable {funder_id: self.id.to_owned(), title: t}).collect::<Vec<_>>();
        let funder_ids = self.ids.map(|i| i.to_funder_ids_table(&self.id));
        let funder_role = self.roles.unwrap_or_default().into_iter().map(|t| t.to_funder_role_table(&self.id)).collect::<Vec<_>>();
        let funder_counts_by_year = self.counts_by_year.unwrap_or_default().into_iter().map(|t| t.to_funder_counts_by_year(&self.id)).collect::<Vec<_>>();
        let funder_summary_stats = self.summary_stats.map(|t| t.to_funder_summary_stats_table(&self.id));
        let funder = FunderTable {
            funder_id: self.id,
            display_name: self.display_name,
            description: self.description.unwrap_or_default(),
            country_code: self.country_code.unwrap_or_default(),
            cited_by_count: self.cited_by_count.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default(),
            grants_count: self.grants_count.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            homepage_url: self.homepage_url.unwrap_or_default(),
            image_url: self.image_url.unwrap_or_default(),
            image_thumbnail_url: self.image_thumbnail_url.unwrap_or_default(),
        };
        (funder, funder_alternative_titles, funder_ids, funder_role, funder_counts_by_year, funder_summary_stats)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunderTable {
    pub funder_id: String,
    pub display_name: String,
    pub description: String,
    pub country_code: CountryCode,
    pub cited_by_count: u32,
    pub works_count: u32,
    pub grants_count: u32,
    pub created_date: String,
    pub updated_date: String,
    pub homepage_url: String,
    pub image_url: String,
    pub image_thumbnail_url: String,
}

impl FunderTable {
    fn to_fields() -> Fields {
        let field_names = ["funder_id", 
            "display_name", 
            "description", 
            "country_code", 
            "created_date", 
            "updated_date", 
            "homepage_url", 
            "image_url", 
            "image_thumbnail_url"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["cited_by_count", 
            "works_count", 
            "grants_count"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for FunderTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for FunderTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunderIds {
    pub openalex: Option<String>,
    pub ror: Option<String>,
    pub wikidata: Option<String>,
    pub crossref: Option<String>,
    pub doi: Option<String>,
}

impl FunderIds {
    pub fn to_funder_ids_table(self, funder_id: &str) -> FunderIdsTable {
        FunderIdsTable { funder_id: funder_id.to_string(), 
            openalex: self.openalex.unwrap_or_default(), 
            ror: self.ror.unwrap_or_default(), 
            wikidata: self.wikidata.unwrap_or_default(), 
            crossref: self.crossref.unwrap_or_default(), 
            doi: self.doi.unwrap_or_default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunderIdsTable {
    pub funder_id: String,
    pub openalex: String,
    pub ror: String,
    pub wikidata: String,
    pub crossref: String,
    pub doi: String,
}

impl FunderIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["funder_id", "openalex", "ror", "wikidata", "crossref", "doi"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for FunderIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for FunderIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunderAlternativeTitlesTable {
    pub funder_id: String,
    pub title: String,
}

impl FunderAlternativeTitlesTable {
    fn to_fields() -> Fields {
        let field_names = ["funder_id", "title"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for FunderAlternativeTitlesTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for FunderAlternativeTitlesTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunderRoleTable {
    pub funder_id: String,
    pub role: RoleType,
    pub id: String,
    pub works_count: u32,
}

impl FunderRoleTable {
    fn to_fields() -> Fields {
        let field_names = ["funder_id", "role", "id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["works_count"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for FunderRoleTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for FunderRoleTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct FunderSummaryStatsTable {
    pub funder_id: String,
    pub two_year_mean_citedness: f64,
    pub h_index: u32,
    pub i10_index: u32,
}

impl FunderSummaryStatsTable {
    fn to_fields() -> Fields {
        let field_names = ["funder_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["two_year_mean_citedness"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float64, false))
                .collect::<Vec<_>>(),
        );
        let field_names = [
            "h_index",
            "i10_index",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for FunderSummaryStatsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for FunderSummaryStatsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct FunderCountsByYearTable {
    pub funder_id: String,
    pub year: u32,
    pub cited_by_count: u32,
}

impl FunderCountsByYearTable {
    fn to_fields() -> Fields {
        let field_names = ["funder_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = [
            "year",
            "cited_by_count",
        ];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for FunderCountsByYearTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for FunderCountsByYearTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
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

impl Award {
    pub fn to_work_award_table(self, work_id: &str) -> WorkAwardTable {
        WorkAwardTable {
            work_id: work_id.to_string(),
            award_id: self.id.unwrap_or_default()
        }
    }
    pub fn to_table(self) -> (AwardTable,
        Option<AwardFunderTable>,
        Vec<AwardFundedOutputsTable>,
        Vec<AwardInvestigatorTable>,
        Vec<AwardAffiliationTable>
    ) {
        let award_funder = self.funder.map(|f| f.to_award_funder_table(&self.id.clone().unwrap_or_default()));
        let award_funded_outputs = self.funded_outputs.unwrap_or_default().into_iter().map(|f| AwardFundedOutputsTable { award_id: self.id.clone().unwrap_or_default(), work_id: f}).collect::<Vec<_>>();
        let (award_investigator, award_affiliation): (Vec<_>, Vec<_>) = self.investigators.unwrap_or_default().into_iter().map(|i| if self.lead_investigator.is_some() && self.lead_investigator.as_ref().unwrap() == &i {
            i.to_award_investigator_table(&self.id.clone().unwrap_or_default(), true, false)
        } else if self.co_lead_investigator.is_some() && self.co_lead_investigator.as_ref().unwrap() == &i {
            i.to_award_investigator_table(&self.id.clone().unwrap_or_default(), false, true)
        } else {
            i.to_award_investigator_table(&self.id.clone().unwrap_or_default(), false, false)
        }).unzip();
        let award_affiliation = award_affiliation.into_iter().filter_map(|a| if let Some(a_i) = a {
            Some(a_i)
        } else {
            None
        }).collect::<Vec<_>>();
        let award = AwardTable {
            award_id: self.id.unwrap_or_default(),
            display_name: self.display_name.unwrap_or_default(),
            description: self.description.unwrap_or_default(),
            funder_award_id: self.funder_award_id.unwrap_or_default(),
            funded_outputs_count: self.funded_outputs_count.unwrap_or_default(),
            amount: self.amount.unwrap_or_default(),
            currency: self.currency.unwrap_or_default(),
            funding_type: self.funding_type.unwrap_or_default(),
            funder_scheme: self.funder_scheme.unwrap_or_default(),
            start_date: self.start_date.unwrap_or_default(),
            end_date: self.end_date.unwrap_or_default(),
            start_year: self.start_year.unwrap_or_default(),
            end_year: self.end_year.unwrap_or_default(),
            landing_page_url: self.landing_page_url.unwrap_or_default(),
            doi: self.doi.unwrap_or_default(),
            provenance: self.provenance.unwrap_or_default(),
            works_api_url: self.works_api_url.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default()
        };
        (award, award_funder, award_funded_outputs, award_investigator, award_affiliation)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AwardTable {
    pub award_id: String,
    pub display_name: String,
    pub description: String,
    pub funder_award_id: String,
    pub funded_outputs_count: u32,
    pub amount: f32,
    pub currency: Currency,
    pub funding_type: String,
    pub funder_scheme: String,
    pub start_date: String,
    pub end_date: String,
    pub start_year: u32,
    pub end_year: u32,
    pub landing_page_url: String,
    pub doi: String,
    pub provenance: String,
    pub works_api_url: String,
    pub created_date: String,
    pub updated_date: String,
}

impl AwardTable {
    fn to_fields() -> Fields {
        let field_names = ["award_id", 
            "display_name", 
            "description", 
            "funder_award_id", 
            "currency", 
            "funding_type", 
            "funder_scheme", 
            "start_date", 
            "end_date", 
            "landing_page_url", 
            "doi", 
            "provenance", 
            "works_api_url", 
            "created_date", 
            "updated_date"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["funded_outputs_count", 
            "start_year", 
            "end_year"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>());
        let field_names = ["amount"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Float32, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AwardTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AwardTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Investigator {
    pub given_name: Option<String>,
    pub family_name: Option<String>,
    pub orcid: Option<String>,
    pub role_start: Option<String>,
    pub affiliation: Option<Affiliation>,
}

impl Investigator {
    pub fn to_award_investigator_table(self, award_id: &str, is_lead_investigator: bool, is_co_lead_investigator: bool) -> (AwardInvestigatorTable, Option<AwardAffiliationTable>) {
        let award_affiliation = self.affiliation.map(|a| a.to_award_affiliation_table(award_id, &self.orcid.clone().unwrap_or_default()));
        let award_investigator = AwardInvestigatorTable { 
            award_id: award_id.to_string(), 
            is_lead_investigator,
            is_co_lead_investigator,
            given_name: self.given_name.unwrap_or_default(), 
            family_name: self.family_name.unwrap_or_default(), 
            orcid: self.orcid.unwrap_or_default(), 
            role_start: self.role_start.unwrap_or_default(), 
        };
        (award_investigator, award_affiliation)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AwardInvestigatorTable {
    pub award_id: String,
    pub is_lead_investigator: bool,
    pub is_co_lead_investigator: bool,
    pub given_name: String,
    pub family_name: String,
    pub orcid: String,
    pub role_start: String,
}

impl AwardInvestigatorTable {
    fn to_fields() -> Fields {
        let field_names = ["award_id", 
            "given_name", 
            "family_name", 
            "orcid", 
            "role_start"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["is_lead_investigator", 
            "is_co_lead_investigator"];
        fields_vec.extend(field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Boolean, false))
            .collect::<Vec<_>>());
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AwardInvestigatorTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AwardInvestigatorTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AwardAffiliationTable {
    pub award_id: String,
    pub orcid: String,
    pub institution_id: String,
    pub years: Vec<u32>,
}

impl AwardAffiliationTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "orcid", "institution_id"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)));
        let field_names = ["years"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, list_data_type.clone(), false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AwardAffiliationTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AwardAffiliationTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AwardFunderTable {
    pub award_id: String,
    pub funder_id: String,
}

impl AwardFunderTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "funder_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AwardFunderTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AwardFunderTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AwardFundedOutputsTable {
    pub award_id: String,
    pub work_id: String,
}

impl AwardFundedOutputsTable {
    fn to_fields() -> Fields {
        let field_names = ["author_id", "work_id"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for AwardFundedOutputsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for AwardFundedOutputsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
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
pub struct OpenAlexResponseWorks {
    pub results: Vec<Work>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseWorks {
    /// Parse the OpenAlexResponseWorks object into tables following the [create_ipc_fields] schema
    ///   where each row is routed to a different table
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        let mut work_tables = Vec::new();
        let mut work_authorship_tables = Vec::new();
        let mut work_award_tables = Vec::new();
        let mut work_funder_tables = Vec::new();
        let mut work_apc_info_tables = Vec::new();
        let mut work_location_tables = Vec::new();
        let mut work_open_access_tables = Vec::new();
        let mut work_biblio_tables = Vec::new();
        let mut work_citation_normalized_percentile_tables = Vec::new();
        let mut work_cited_percentile_year_tables = Vec::new();
        let mut work_counts_by_year_tables = Vec::new();
        let mut work_concepts_tables = Vec::new();
        let mut work_topics_tables = Vec::new();
        let mut work_keywords_tables = Vec::new();
        let mut work_mesh_tag_tables = Vec::new();
        let mut work_sdg_tag_tables = Vec::new();
        let mut work_corresponding_author_tables = Vec::new();
        let mut work_corresponding_insitution_tables = Vec::new();
        let mut work_indexed_in_tables = Vec::new();
        let mut work_ids_tables = Vec::new();
        let mut work_referenced_works_tables = Vec::new();
        let mut work_related_works_tables = Vec::new();
        for work in self.results {
            // Parse into individual tables
            let (work_table, 
                work_authorship_table, 
                work_award_table, 
                work_funder_table, 
                work_apc_info_table, 
                work_location_table, 
                work_open_access_table, 
                work_biblio_table, 
                work_citation_normalized_percentile_table,
                work_cited_percentile_year_table,
                work_counts_by_year_table,
                work_concepts_table,
                work_topics_table,
                work_keywords_table,
                work_mesh_tag_table,
                work_sdg_tag_table,
                work_corresponding_author_table,
                work_corresponding_insitution_table,
                work_indexed_in_table,
                work_ids_table,
                work_referenced_works_table,
                work_related_works_table) = work.to_tables();

            // Handle each individual table
            work_tables.push(work_table);
            work_authorship_tables.extend(work_authorship_table);
            work_award_tables.extend(work_award_table);
            work_funder_tables.extend(work_funder_table);
            work_apc_info_tables.extend(work_apc_info_table);
            work_location_tables.extend(work_location_table);
            if let Some(work_open_access_table) = work_open_access_table {
                work_open_access_tables.push(work_open_access_table);
            }
            if let Some(work_biblio_table) = work_biblio_table {
                work_biblio_tables.push(work_biblio_table);
            }
            if let Some(work_citation_normalized_percentile_table) = work_citation_normalized_percentile_table {
                work_citation_normalized_percentile_tables.push(work_citation_normalized_percentile_table);
            }
            if let Some(work_cited_percentile_year_table) = work_cited_percentile_year_table {
                work_cited_percentile_year_tables.push(work_cited_percentile_year_table);
            }
            work_counts_by_year_tables.extend(work_counts_by_year_table);
            work_concepts_tables.extend(work_concepts_table);
            work_topics_tables.extend(work_topics_table);
            work_keywords_tables.extend(work_keywords_table);
            work_mesh_tag_tables.extend(work_mesh_tag_table);
            work_sdg_tag_tables.extend(work_sdg_tag_table);
            work_corresponding_author_tables.extend(work_corresponding_author_table);
            work_corresponding_insitution_tables.extend(work_corresponding_insitution_table);
            work_indexed_in_tables.extend(work_indexed_in_table);
            if let Some(work_ids_table) = work_ids_table {
                work_ids_tables.push(work_ids_table);
            }
            work_referenced_works_tables.extend(work_referenced_works_table);
            work_related_works_tables.extend(work_related_works_table);
        }

        // Wrap into IPC [RecordBatch]
        let mut names = Vec::new();
        let mut publishers = Vec::new();
        let mut subjects = Vec::new();
        let mut formats = Vec::new();
        let mut bytes = Vec::new();

        // Handle each individual table
        if !work_tables.is_empty() {
            names.push(work_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_tables.first().unwrap().get_name())
                .with_schema(work_tables.first().unwrap().to_schema())
                .with_struct::<WorkTable>(&work_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_authorship_tables.is_empty() {
            names.push(work_authorship_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_authorship_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_authorship_tables.first().unwrap().get_name())
                .with_schema(work_authorship_tables.first().unwrap().to_schema())
                .with_struct::<WorkAuthorshipTable>(&work_authorship_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_award_tables.is_empty() {
            names.push(work_award_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_award_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_award_tables.first().unwrap().get_name())
                .with_schema(work_award_tables.first().unwrap().to_schema())
                .with_struct::<WorkAwardTable>(&work_award_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_funder_tables.is_empty() {
            names.push(work_funder_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_funder_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_funder_tables.first().unwrap().get_name())
                .with_schema(work_funder_tables.first().unwrap().to_schema())
                .with_struct::<WorkFunderTable>(&work_funder_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_apc_info_tables.is_empty() {
            names.push(work_apc_info_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_apc_info_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_apc_info_tables.first().unwrap().get_name())
                .with_schema(work_apc_info_tables.first().unwrap().to_schema())
                .with_struct::<WorkApcInfoTable>(&work_apc_info_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_location_tables.is_empty() {
            names.push(work_location_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_location_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_location_tables.first().unwrap().get_name())
                .with_schema(work_location_tables.first().unwrap().to_schema())
                .with_struct::<WorkLocationTable>(&work_location_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_open_access_tables.is_empty() {
            names.push(work_open_access_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_open_access_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_open_access_tables.first().unwrap().get_name())
                .with_schema(work_open_access_tables.first().unwrap().to_schema())
                .with_struct::<WorkOpenAccessTable>(&work_open_access_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_biblio_tables.is_empty() {
            names.push(work_biblio_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_biblio_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_biblio_tables.first().unwrap().get_name())
                .with_schema(work_biblio_tables.first().unwrap().to_schema())
                .with_struct::<WorkBiblioTable>(&work_biblio_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_citation_normalized_percentile_tables.is_empty() {
            names.push(work_citation_normalized_percentile_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_citation_normalized_percentile_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_citation_normalized_percentile_tables.first().unwrap().get_name())
                .with_schema(work_citation_normalized_percentile_tables.first().unwrap().to_schema())
                .with_struct::<WorkCitationPercentileTable>(&work_citation_normalized_percentile_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_cited_percentile_year_tables.is_empty() {
            names.push(work_cited_percentile_year_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_cited_percentile_year_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_cited_percentile_year_tables.first().unwrap().get_name())
                .with_schema(work_cited_percentile_year_tables.first().unwrap().to_schema())
                .with_struct::<WorkCitedByPercentileYearTable>(&work_cited_percentile_year_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_counts_by_year_tables.is_empty() {
            names.push(work_counts_by_year_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_counts_by_year_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_counts_by_year_tables.first().unwrap().get_name())
                .with_schema(work_counts_by_year_tables.first().unwrap().to_schema())
                .with_struct::<WorkCountsByYearTable>(&work_counts_by_year_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_concepts_tables.is_empty() {
            names.push(work_concepts_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_concepts_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_concepts_tables.first().unwrap().get_name())
                .with_schema(work_concepts_tables.first().unwrap().to_schema())
                .with_struct::<WorkConceptTable>(&work_concepts_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_topics_tables.is_empty() {
            names.push(work_topics_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_topics_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_topics_tables.first().unwrap().get_name())
                .with_schema(work_topics_tables.first().unwrap().to_schema())
                .with_struct::<WorkTopicTable>(&work_topics_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_keywords_tables.is_empty() {
            names.push(work_keywords_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_keywords_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_keywords_tables.first().unwrap().get_name())
                .with_schema(work_keywords_tables.first().unwrap().to_schema())
                .with_struct::<WorkKeywordTable>(&work_keywords_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_mesh_tag_tables.is_empty() {
            names.push(work_mesh_tag_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_mesh_tag_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_mesh_tag_tables.first().unwrap().get_name())
                .with_schema(work_mesh_tag_tables.first().unwrap().to_schema())
                .with_struct::<WorkMeshTagTable>(&work_mesh_tag_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_sdg_tag_tables.is_empty() {
            names.push(work_sdg_tag_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_sdg_tag_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_sdg_tag_tables.first().unwrap().get_name())
                .with_schema(work_sdg_tag_tables.first().unwrap().to_schema())
                .with_struct::<WorkSdgTagTable>(&work_sdg_tag_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_corresponding_author_tables.is_empty() {
            names.push(work_corresponding_author_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_corresponding_author_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_corresponding_author_tables.first().unwrap().get_name())
                .with_schema(work_corresponding_author_tables.first().unwrap().to_schema())
                .with_struct::<WorkCorrespondingAuthorTable>(&work_corresponding_author_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_corresponding_insitution_tables.is_empty() {
            names.push(work_corresponding_insitution_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_corresponding_insitution_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_corresponding_insitution_tables.first().unwrap().get_name())
                .with_schema(work_corresponding_insitution_tables.first().unwrap().to_schema())
                .with_struct::<WorkCorrespondingInstitutionTable>(&work_corresponding_insitution_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_indexed_in_tables.is_empty() {
            names.push(work_indexed_in_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_indexed_in_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_indexed_in_tables.first().unwrap().get_name())
                .with_schema(work_indexed_in_tables.first().unwrap().to_schema())
                .with_struct::<WorkIndexedInTable>(&work_indexed_in_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_ids_tables.is_empty() {
            names.push(work_ids_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_ids_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_ids_tables.first().unwrap().get_name())
                .with_schema(work_ids_tables.first().unwrap().to_schema())
                .with_struct::<WorkIdsTable>(&work_ids_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_referenced_works_tables.is_empty() {
            names.push(work_referenced_works_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_referenced_works_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_referenced_works_tables.first().unwrap().get_name())
                .with_schema(work_referenced_works_tables.first().unwrap().to_schema())
                .with_struct::<WorkReferencedWorksTable>(&work_referenced_works_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !work_related_works_tables.is_empty() {
            names.push(work_related_works_tables.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(work_related_works_tables.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(work_related_works_tables.first().unwrap().get_name())
                .with_schema(work_related_works_tables.first().unwrap().to_schema())
                .with_struct::<WorkRelatedWorksTable>(&work_related_works_tables)?
                .build()?
                .to_ipc_stream()?
            );
        }

        let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        Ok(batch)
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
pub struct Meta {
    pub count: u32,
    pub groups_count: Option<u32>,
    pub db_response_time_ms: u32,
    pub page: Option<u32>,
    pub per_page: u32,
    pub next_cursor: Option<String>,
    pub query: Option<String>,
    pub filters_applied: Option<Map<String, Value>>,
    pub timing: Option<Timing>,
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
pub struct Timing {
    pub embed_ms: u32,
    pub search_ms: u32,
    pub hydrate_ms: u32,
    pub total_ms: u32,
}

// Documentation for the OpenAlex API in MarkDown from <https://docs.openalex.org/>
#[allow(dead_code)]
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
