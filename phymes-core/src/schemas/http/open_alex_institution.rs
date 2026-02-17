use std::{fmt::Display, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::RecordBatch, datatypes::{DataType, Field, Fields, SchemaRef}};
use crate::{AvailableSchemaTrait, BuildableTrait, BuilderTrait, DataFormat, JsonSchemaTrait, MappableTrait, Table, TableBuilderTrait, TableTrait, create_route_bytes_record_batch, create_schema_from_fields, open_alex::OaStatus};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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