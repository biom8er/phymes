use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use crate::{AvailableSchemaTrait, BuilderTrait, MappableTrait, create_schema_from_fields, schemas::http::{AwardAffiliationTable, open_alex_common::{CountsByYear, SummaryStats}, open_alex_institution::Institution}};
use serde::{Deserialize, Serialize};

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