use std::{fmt::Display, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::RecordBatch, datatypes::{DataType, Field, Fields, SchemaRef}};
use crate::{AvailableSchemaTrait, BuildableTrait, BuilderTrait, DataFormat, JsonSchemaTrait, MappableTrait, Table, TableBuilderTrait, TableTrait, create_route_bytes_record_batch, create_schema_from_fields, schemas::http::{AwardFunderTable, WorkFunderTable, open_alex_common::{CountryCode, CountsByYear, RoleType, SummaryStats}, open_alex_institution::Role}};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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