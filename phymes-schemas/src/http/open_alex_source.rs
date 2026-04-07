use std::sync::Arc;

use phymes_core::MappableTrait;
use crate::{
    AvailableSchemaTrait, create_schema_from_fields,
    http::open_alex_common::{
        CountryCode, CountsByYear, Currency, SourceType, SummaryStats,
    },
};
use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use serde::{Deserialize, Serialize};

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
    pub host_organization_lineage: Option<Vec<Option<String>>>,
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
    pub x_concepts: Option<Vec<SourceConcept>>,
}

impl Source {
    #[allow(clippy::type_complexity)]
    pub fn build_tables(
        self,
    ) -> (
        SourceTable,
        Vec<SourceAlternativeTitlesTable>,
        Vec<SourceApcPriceTable>,
        Vec<SourceCountsByYearTable>,
        Vec<SourceLineageTable>,
        Option<SourceIdsTable>,
        Vec<SourceIssnTable>,
        Vec<SourceSocietyTable>,
        Option<SourceSummaryStatsTable>,
        Vec<SourceConceptTable>,
    ) {
        let source_alternative_titles = self
            .alternate_titles
            .unwrap_or_default()
            .into_iter()
            .map(|t| SourceAlternativeTitlesTable {
                source_id: self.id.clone(),
                display_name: t,
            })
            .collect::<Vec<_>>();
        let source_apc_price = self
            .apc_prices
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.build_source_apc_price_table(&self.id))
            .collect::<Vec<_>>();
        let source_counts_by_year = self
            .counts_by_year
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.build_source_counts_by_year(&self.id))
            .collect::<Vec<_>>();
        let source_lineage = self
            .host_organization_lineage
            .unwrap_or_default()
            .into_iter()
            .filter_map(|t| {
                if let Some(t) = t {
                    Some(SourceLineageTable {
                        source_id: self.id.clone(),
                        lineage_id: t,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let source_ids = self.ids.map(|t| t.build_source_ids_table(&self.id));
        let source_issn = self
            .issn
            .unwrap_or_default()
            .into_iter()
            .map(|t| SourceIssnTable {
                source_id: self.id.clone(),
                issn: t,
            })
            .collect::<Vec<_>>();
        let source_society = self
            .societies
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.build_source_society_table(&self.id))
            .collect::<Vec<_>>();
        let source_summary_stats = self
            .summary_stats
            .map(|t| t.build_source_summary_stats_table(&self.id));
        let source_concept = self
            .x_concepts
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.build_source_concept_table(&self.id))
            .collect::<Vec<_>>();
        let source = SourceTable {
            source_id: self.id,
            display_name: self.display_name.unwrap_or_default(),
            abbreviated_title: self.abbreviated_title.unwrap_or_default(),
            cited_by_count: self.cited_by_count.unwrap_or_default(),
            country_code: self.country_code.unwrap_or_default(),
            created_date: self.created_date.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            homepage_url: self.homepage_url.unwrap_or_default(),
            host_organization: self.host_organization.unwrap_or_default(),
            host_organization_name: self.host_organization_name.unwrap_or_default(),
            is_core: self.is_core.unwrap_or_default(),
            is_in_doaj: self.is_in_doaj.unwrap_or_default(),
            is_oa: self.is_oa.unwrap_or_default(),
            issn_l: self.issn_l.unwrap_or_default(),
            type_: self.type_.unwrap_or_default(),
            works_api_url: self.works_api_url.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default(),
        };
        (
            source,
            source_alternative_titles,
            source_apc_price,
            source_counts_by_year,
            source_lineage,
            source_ids,
            source_issn,
            source_society,
            source_summary_stats,
            source_concept,
        )
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
        let field_names = [
            "source_id",
            "display_name",
            "abbreviated_title",
            "country_code",
            "created_date",
            "updated_date",
            "homepage_url",
            "host_organization",
            "host_organization_name",
            "type_",
        ];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["cited_by_count", "works_count"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        let field_names = ["is_core", "is_in_doaj", "is_oa"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
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

impl ApcPrice {
    pub fn build_source_apc_price_table(self, source_id: &str) -> SourceApcPriceTable {
        SourceApcPriceTable {
            source_id: source_id.to_string(),
            price: self.price.unwrap_or_default(),
            currency: self.currency.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceApcPriceTable {
    pub source_id: String,
    pub price: u32,
    pub currency: Currency,
}

impl SourceApcPriceTable {
    fn to_fields() -> Fields {
        let field_names = ["work_id", "Currency"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["price"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for SourceApcPriceTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for SourceApcPriceTable {
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
    pub fn build_source_society_table(self, source_id: &str) -> SourceSocietyTable {
        SourceSocietyTable {
            source_id: source_id.to_string(),
            url: self.url.unwrap_or_default(),
            organization: self.organization.unwrap_or_default(),
        }
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
    pub fn build_source_ids_table(self, source_id: &str) -> SourceIdsTable {
        SourceIdsTable {
            source_id: source_id.to_string(),
            fatcat: self.fatcat.unwrap_or_default(),
            issn: self.issn.unwrap_or_default(),
            issn_l: self.issn_l.unwrap_or_default(),
            mag: self.mag.unwrap_or_default(),
            openalex: self.openalex.unwrap_or_default(),
            wikidata: self.wikidata.unwrap_or_default(),
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
        let field_names = [
            "source_id",
            "fatcat",
            "issn_l",
            "mag",
            "openalex",
            "wikidata",
        ];
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
        let field_names = ["h_index", "i10_index"];
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
        let field_names = ["year", "cited_by_count"];
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
    pub fn build_source_concept_table(self, source_id: &str) -> SourceConceptTable {
        SourceConceptTable {
            source_id: source_id.to_string(),
            concept_id: self.id.unwrap_or_default(),
            score: self.score.unwrap_or_default(),
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
