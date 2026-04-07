use phymes_core::MappableTrait;
use crate::{
    AvailableSchemaTrait, create_schema_from_fields,
    schemas::http::{
        open_alex_common::{CountryCode, CountsByYear, RoleType, SummaryStats},
        open_alex_institution::Role,
    },
};
use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use serde::{Deserialize, Serialize};

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
    #[allow(clippy::type_complexity)]
    pub fn build_tables(
        self,
    ) -> (
        PublisherTable,
        Vec<PublisherAlternativeTitlesTable>,
        Vec<PublisherCountryCodeTable>,
        Vec<PublisherLineageTable>,
        Option<PublisherIdsTable>,
        Vec<PublisherRoleTable>,
        Vec<PublisherCountsByYearTable>,
        Option<PublisherSummaryStatsTable>,
    ) {
        let publisher_alternative_titles = self
            .alternate_titles
            .unwrap_or_default()
            .into_iter()
            .map(|t| PublisherAlternativeTitlesTable {
                publisher_id: self.id.to_owned(),
                title: t,
            })
            .collect::<Vec<_>>();
        let publisher_country_code = self
            .country_codes
            .unwrap_or_default()
            .into_iter()
            .map(|t| PublisherCountryCodeTable {
                publisher_id: self.id.to_owned(),
                country_code: t,
            })
            .collect::<Vec<_>>();
        let publisher_lineage = self
            .lineage
            .unwrap_or_default()
            .into_iter()
            .map(|t| PublisherLineageTable {
                publisher_id: self.id.clone(),
                lineage_id: t,
            })
            .collect::<Vec<_>>();
        let publisher_ids = self.ids.map(|i| i.build_publisher_ids_table(&self.id));
        let publisher_role = self
            .roles
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.build_publisher_role_table(&self.id))
            .collect::<Vec<_>>();
        let publisher_counts_by_year = self
            .counts_by_year
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.build_publisher_counts_by_year(&self.id))
            .collect::<Vec<_>>();
        let publisher_summary_stats = self
            .summary_stats
            .map(|t| t.build_publisher_summary_stats_table(&self.id));
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
            sources_api_url: self.sources_api_url.unwrap_or_default(),
        };
        (
            publisher,
            publisher_alternative_titles,
            publisher_country_code,
            publisher_lineage,
            publisher_ids,
            publisher_role,
            publisher_counts_by_year,
            publisher_summary_stats,
        )
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
        let field_names = [
            "publisher_id",
            "display_name",
            "created_date",
            "updated_date",
            "parent_publisher",
            "image_url",
            "image_thumbnail_url",
            "sources_api_url",
        ];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["cited_by_count", "works_count", "hierarchy_level"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
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
    pub fn build_publisher_ids_table(self, publisher_id: &str) -> PublisherIdsTable {
        PublisherIdsTable {
            publisher_id: publisher_id.to_string(),
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
