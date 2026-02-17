use anyhow::{anyhow, Result};
use arrow::array::RecordBatch;
use crate::{AvailableSchemaTrait, BuildableTrait, BuilderTrait, DataFormat, JsonSchemaTrait, MappableTrait, Table, TableBuilderTrait, TableTrait, create_route_bytes_record_batch, create_schema_from_fields, 
    schemas::http::{open_alex_common::OpenAlexEntity, open_alex_works::{Work, WorkApcInfoTable, WorkAuthorshipTable, WorkAwardTable, WorkBiblioTable, WorkCitationPercentileTable, WorkCitedByPercentileYearTable, WorkConceptTable, WorkCorrespondingAuthorTable, WorkCorrespondingInstitutionTable, WorkCountsByYearTable, WorkFunderTable, WorkIdsTable, WorkIndexedInTable, WorkKeywordTable, WorkLocationTable, WorkMeshTagTable, WorkOpenAccessTable, WorkReferencedWorksTable, WorkRelatedWorksTable, WorkSdgTagTable, WorkTable, WorkTopicTable}}, 
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
pub struct OpenAlexResponseFind {
    pub results: Vec<FindResponse>,
    pub meta: Meta,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FindResponse {
    pub score: Option<f32>,
    pub entity: OpenAlexEntity,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseGroupBy {
    pub group_by: Vec<GroupByResponse>,
    pub meta: Meta,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GroupByResponse {
    pub key: String,
    pub key_display_name: String,
    pub count: u32,
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
pub struct RateLimitResponse {
    pub api_key: String,
    pub rate_limit: u32,
    pub page: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RateLimit {
    pub credits_limit: u32,
    pub credits_remaining: u32,
    pub resets_at: String,
    pub resets_in_seconds: u32,
    pub credit_costs: CreditCosts,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreditCosts {
    pub singleton: u32,
    pub list: u32,
    pub content: u32,
    pub vector: u32,
    pub text: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Timing {
    pub embed_ms: u32,
    pub search_ms: u32,
    pub hydrate_ms: u32,
    pub total_ms: u32,
}