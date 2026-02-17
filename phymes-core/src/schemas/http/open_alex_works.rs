use std::sync::Arc;

use arrow::{datatypes::{DataType, Field, Fields, SchemaRef}};
use crate::{AvailableSchemaTrait, MappableTrait, create_schema_from_fields, 
    schemas::http::{open_alex_author::Author, open_alex_award::Award, open_alex_common::{AuthorPosition, CountryCode, CountsByYear, Currency, LanguageCode, OaStatus, WorkType, abstract_from_inverted_index}, open_alex_funder::Funder, open_alex_institution::Institution, open_alex_source::Source}};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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