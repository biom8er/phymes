use std::sync::Arc;

use crate::{
    AvailableSchemaTrait, MappableTrait, create_schema_from_fields,
    schemas::http::{
        WorkAwardTable, open_alex_author::Affiliation, open_alex_common::Currency,
        open_alex_funder::Funder,
    },
};
use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use serde::{Deserialize, Serialize};

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
    pub fn build_work_award_table(self, work_id: &str) -> WorkAwardTable {
        WorkAwardTable {
            work_id: work_id.to_string(),
            award_id: self.id.unwrap_or_default(),
        }
    }
    pub fn build_tables(
        self,
    ) -> (
        AwardTable,
        Option<AwardFunderTable>,
        Vec<AwardFundedOutputsTable>,
        Vec<AwardInvestigatorTable>,
        Vec<AwardAffiliationTable>,
    ) {
        let award_funder = self
            .funder
            .map(|f| f.build_award_funder_table(&self.id.clone().unwrap_or_default()));
        let award_funded_outputs = self
            .funded_outputs
            .unwrap_or_default()
            .into_iter()
            .map(|f| AwardFundedOutputsTable {
                award_id: self.id.clone().unwrap_or_default(),
                work_id: f,
            })
            .collect::<Vec<_>>();
        let (award_investigator, award_affiliation): (Vec<_>, Vec<_>) = self
            .investigators
            .unwrap_or_default()
            .into_iter()
            .map(|i| {
                if self.lead_investigator.is_some()
                    && self.lead_investigator.as_ref().unwrap() == &i
                {
                    i.build_award_investigator_table(
                        &self.id.clone().unwrap_or_default(),
                        true,
                        false,
                    )
                } else if self.co_lead_investigator.is_some()
                    && self.co_lead_investigator.as_ref().unwrap() == &i
                {
                    i.build_award_investigator_table(
                        &self.id.clone().unwrap_or_default(),
                        false,
                        true,
                    )
                } else {
                    i.build_award_investigator_table(
                        &self.id.clone().unwrap_or_default(),
                        false,
                        false,
                    )
                }
            })
            .unzip();
        let award_affiliation = award_affiliation.into_iter().flatten().collect::<Vec<_>>();
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
            updated_date: self.updated_date.unwrap_or_default(),
        };
        (
            award,
            award_funder,
            award_funded_outputs,
            award_investigator,
            award_affiliation,
        )
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
        let field_names = [
            "award_id",
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
            "updated_date",
        ];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["funded_outputs_count", "start_year", "end_year"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        let field_names = ["amount"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Float32, false))
                .collect::<Vec<_>>(),
        );
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
    pub fn build_award_investigator_table(
        self,
        award_id: &str,
        is_lead_investigator: bool,
        is_co_lead_investigator: bool,
    ) -> (AwardInvestigatorTable, Option<AwardAffiliationTable>) {
        let award_affiliation = self.affiliation.map(|a| {
            a.build_award_affiliation_table(award_id, &self.orcid.clone().unwrap_or_default())
        });
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
        let field_names = [
            "award_id",
            "given_name",
            "family_name",
            "orcid",
            "role_start",
        ];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["is_lead_investigator", "is_co_lead_investigator"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::Boolean, false))
                .collect::<Vec<_>>(),
        );
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
        let list_data_type =
            DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)));
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
        let field_names = ["award_id", "funder_id"];
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
        let field_names = ["award_id", "work_id"];
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
