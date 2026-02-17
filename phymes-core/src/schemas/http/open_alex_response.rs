use anyhow::Result;
use arrow::array::RecordBatch;
use crate::{AvailableSchemaTrait, BuildableTrait, BuilderTrait, DataFormat, JsonSchemaTrait, MappableTrait, Table, TableBuilderTrait, TableTrait, create_route_bytes_record_batch, open_alex::{AuthorAffiliationTable, AuthorConceptTable, AuthorCountsByYearTable, AuthorDisplayNameAlternativesTable, AuthorIdsTable, AuthorLastKnownInstitutionsTable, AuthorSummaryStatsTable, AuthorTable, InstitutionAssociatedInstitutionTable, InstitutionConceptTable, InstitutionCountsByYearTable, InstitutionDisplayNameAcronymsTable, InstitutionDisplayNameAlternativesTable, InstitutionGeoTable, InstitutionIdsTable, InstitutionInternationalNamesTable, InstitutionLineageTable, InstitutionRepositoryTable, InstitutionRoleTable, InstitutionSummaryStatsTable, InstitutionTable, TopicDomainTable, TopicFieldTable, TopicIdsTable, TopicKeywordTable, TopicSubfieldTable, TopicTable}, schemas::http::{open_alex_author::{Author, AuthorIds}, open_alex_award::Award, open_alex_common::OpenAlexEntity, open_alex_funder::Funder, open_alex_institution::Institution, open_alex_publisher::Publisher, open_alex_source::Source, open_alex_topic::Topic, open_alex_works::{Work, WorkApcInfoTable, WorkAuthorshipTable, WorkAwardTable, WorkBiblioTable, WorkCitationPercentileTable, WorkCitedByPercentileYearTable, WorkConceptTable, WorkCorrespondingAuthorTable, WorkCorrespondingInstitutionTable, WorkCountsByYearTable, WorkFunderTable, WorkIdsTable, WorkIndexedInTable, WorkKeywordTable, WorkLocationTable, WorkMeshTagTable, WorkOpenAccessTable, WorkReferencedWorksTable, WorkRelatedWorksTable, WorkSdgTagTable, WorkTable, WorkTopicTable}} 
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseAuthors {
    pub results: Vec<Author>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseAuthors {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        let mut author_vec = Vec::new();
        let mut author_display_name_alternatives_vec = Vec::new(); 
        let mut author_affiliation_vec = Vec::new();
        let mut author_last_known_institutions_vec = Vec::new();
        let mut author_ids_vec = Vec::new();
        let mut author_summary_stats_vec = Vec::new();
        let mut author_counts_by_year_vec = Vec::new();
        let mut author_concepts_vec = Vec::new();  
        for result in self.results {
            let (author, 
                author_display_name_alternatives, 
                author_affiliation, 
                author_last_known_institutions, 
                author_ids, 
                author_summary_stats, 
                author_counts_by_year, 
                author_concepts) = result.to_tables();
            author_vec.push(author);
            author_display_name_alternatives_vec.extend(author_display_name_alternatives);
            author_affiliation_vec.extend(author_affiliation);
            author_last_known_institutions_vec.extend(author_last_known_institutions);
            if let Some(author_ids) = author_ids {
                author_ids_vec.push(author_ids);
            }
            if let Some(author_summary_stats) = author_summary_stats {
                author_summary_stats_vec.push(author_summary_stats);
            }
            author_counts_by_year_vec.extend(author_counts_by_year);
            author_concepts_vec.extend(author_concepts);            
        }

        // Wrap into IPC [RecordBatch]
        let mut names = Vec::new();
        let mut publishers = Vec::new();
        let mut subjects = Vec::new();
        let mut formats = Vec::new();
        let mut bytes = Vec::new();

        // Handle each individual table
        if !author_vec.is_empty() {
            names.push(author_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_vec.first().unwrap().get_name())
                .with_schema(author_vec.first().unwrap().to_schema())
                .with_struct::<AuthorTable>(&author_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_display_name_alternatives_vec.is_empty() {
            names.push(author_display_name_alternatives_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_display_name_alternatives_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_display_name_alternatives_vec.first().unwrap().get_name())
                .with_schema(author_display_name_alternatives_vec.first().unwrap().to_schema())
                .with_struct::<AuthorDisplayNameAlternativesTable>(&author_display_name_alternatives_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_affiliation_vec.is_empty() {
            names.push(author_affiliation_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_affiliation_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_affiliation_vec.first().unwrap().get_name())
                .with_schema(author_affiliation_vec.first().unwrap().to_schema())
                .with_struct::<AuthorAffiliationTable>(&author_affiliation_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_last_known_institutions_vec.is_empty() {
            names.push(author_last_known_institutions_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_last_known_institutions_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_last_known_institutions_vec.first().unwrap().get_name())
                .with_schema(author_last_known_institutions_vec.first().unwrap().to_schema())
                .with_struct::<AuthorLastKnownInstitutionsTable>(&author_last_known_institutions_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_ids_vec.is_empty() {
            names.push(author_ids_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_ids_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_ids_vec.first().unwrap().get_name())
                .with_schema(author_ids_vec.first().unwrap().to_schema())
                .with_struct::<AuthorIdsTable>(&author_ids_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_summary_stats_vec.is_empty() {
            names.push(author_summary_stats_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_summary_stats_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_summary_stats_vec.first().unwrap().get_name())
                .with_schema(author_summary_stats_vec.first().unwrap().to_schema())
                .with_struct::<AuthorSummaryStatsTable>(&author_summary_stats_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_counts_by_year_vec.is_empty() {
            names.push(author_counts_by_year_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_counts_by_year_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_counts_by_year_vec.first().unwrap().get_name())
                .with_schema(author_counts_by_year_vec.first().unwrap().to_schema())
                .with_struct::<AuthorCountsByYearTable>(&author_counts_by_year_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !author_concepts_vec.is_empty() {
            names.push(author_concepts_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(author_concepts_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(author_concepts_vec.first().unwrap().get_name())
                .with_schema(author_concepts_vec.first().unwrap().to_schema())
                .with_struct::<AuthorConceptTable>(&author_concepts_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }

        let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        Ok(batch)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseInstitution {
    pub results: Vec<Institution>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseInstitution {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        let mut institution_vec = Vec::new();
        let mut institution_display_name_acronyms_vec = Vec::new();
        let mut institution_display_name_alternatives_vec = Vec::new();
        let mut institution_geo_vec = Vec::new();
        let mut institution_ids_vec = Vec::new();
        let mut institution_associated_institution_vec = Vec::new();
        let mut institution_repository_vec = Vec::new();
        let mut institution_role_vec = Vec::new();
        let mut institution_international_names_vec = Vec::new();
        let mut institution_summary_stats_vec = Vec::new();
        let mut institution_counts_by_year_vec = Vec::new();
        let mut institution_concepts_vec = Vec::new();
        let mut institution_lineage_vec = Vec::new();
        for result in self.results {
            let (institution, 
                institution_display_name_acronyms, 
                institution_display_name_alternatives, 
                institution_geo, 
                institution_ids, 
                institution_associated_institution, 
                institution_repository, 
                institution_role,
                institution_international_names, 
                institution_summary_stats, 
                institution_counts_by_year, 
                institution_concepts, 
                institution_lineage) = result.to_tables();
            institution_vec.push(institution);
            institution_display_name_acronyms_vec.extend(institution_display_name_acronyms);
            institution_display_name_alternatives_vec.extend(institution_display_name_alternatives);
            if let Some(institution_geo) = institution_geo {
                institution_geo_vec.push(institution_geo);
            }
            if let Some(institution_ids) = institution_ids {
                institution_ids_vec.push(institution_ids);
            }
            institution_associated_institution_vec.extend(institution_associated_institution);
            institution_repository_vec.extend(institution_repository);
            institution_role_vec.extend(institution_role);
            if let Some(institution_international_names) = institution_international_names {
                institution_international_names_vec.push(institution_international_names);
            }
            if let Some(institution_summary_stats) = institution_summary_stats {
                institution_summary_stats_vec.push(institution_summary_stats);
            }
            institution_counts_by_year_vec.extend(institution_counts_by_year);
            institution_concepts_vec.extend(institution_concepts);
            institution_lineage_vec.extend(institution_lineage);
        }

        // Wrap into IPC [RecordBatch]
        let mut names = Vec::new();
        let mut publishers = Vec::new();
        let mut subjects = Vec::new();
        let mut formats = Vec::new();
        let mut bytes = Vec::new();
        if !institution_vec.is_empty() {
            names.push(institution_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_vec.first().unwrap().get_name())
                .with_schema(institution_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionTable>(&institution_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_display_name_acronyms_vec.is_empty() {
            names.push(institution_display_name_acronyms_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_display_name_acronyms_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_display_name_acronyms_vec.first().unwrap().get_name())
                .with_schema(institution_display_name_acronyms_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionDisplayNameAcronymsTable>(&institution_display_name_acronyms_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_display_name_alternatives_vec.is_empty() {
            names.push(institution_display_name_alternatives_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_display_name_alternatives_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_display_name_alternatives_vec.first().unwrap().get_name())
                .with_schema(institution_display_name_alternatives_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionDisplayNameAlternativesTable>(&institution_display_name_alternatives_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_geo_vec.is_empty() {
            names.push(institution_geo_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_geo_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_geo_vec.first().unwrap().get_name())
                .with_schema(institution_geo_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionGeoTable>(&institution_geo_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_ids_vec.is_empty() {
            names.push(institution_ids_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_ids_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_ids_vec.first().unwrap().get_name())
                .with_schema(institution_ids_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionIdsTable>(&institution_ids_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_associated_institution_vec.is_empty() {
            names.push(institution_associated_institution_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_associated_institution_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_associated_institution_vec.first().unwrap().get_name())
                .with_schema(institution_associated_institution_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionAssociatedInstitutionTable>(&institution_associated_institution_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_repository_vec.is_empty() {
            names.push(institution_repository_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_repository_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_repository_vec.first().unwrap().get_name())
                .with_schema(institution_repository_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionRepositoryTable>(&institution_repository_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_role_vec.is_empty() {
            names.push(institution_role_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_role_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_role_vec.first().unwrap().get_name())
                .with_schema(institution_role_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionRoleTable>(&institution_role_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_international_names_vec.is_empty() {
            names.push(institution_international_names_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_international_names_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_international_names_vec.first().unwrap().get_name())
                .with_schema(institution_international_names_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionInternationalNamesTable>(&institution_international_names_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_summary_stats_vec.is_empty() {
            names.push(institution_summary_stats_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_summary_stats_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_summary_stats_vec.first().unwrap().get_name())
                .with_schema(institution_summary_stats_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionSummaryStatsTable>(&institution_summary_stats_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_counts_by_year_vec.is_empty() {
            names.push(institution_counts_by_year_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_counts_by_year_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_counts_by_year_vec.first().unwrap().get_name())
                .with_schema(institution_counts_by_year_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionCountsByYearTable>(&institution_counts_by_year_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_concepts_vec.is_empty() {
            names.push(institution_concepts_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_concepts_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_concepts_vec.first().unwrap().get_name())
                .with_schema(institution_concepts_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionConceptTable>(&institution_concepts_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !institution_lineage_vec.is_empty() {
            names.push(institution_lineage_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(institution_lineage_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(institution_lineage_vec.first().unwrap().get_name())
                .with_schema(institution_lineage_vec.first().unwrap().to_schema())
                .with_struct::<InstitutionLineageTable>(&institution_lineage_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }

        let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        Ok(batch)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseTopic {
    pub results: Vec<Topic>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseTopic {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        let mut topic_vec = Vec::new();
        let mut topic_domain_vec = Vec::new();
        let mut topic_field_vec = Vec::new();
        let mut topic_subfield_vec = Vec::new();
        let mut topic_ids_vec = Vec::new();
        let mut topic_keyword_vec = Vec::new();
        for result in self.results {
            let (topic, topic_domain, topic_field, topic_subfield, topic_ids, topic_keyword) = result.to_tables();
            topic_vec.push(topic);
            topic_domain_vec.push(topic_domain);
            topic_field_vec.push(topic_field);
            topic_subfield_vec.push(topic_subfield);
            if let Some(topic_ids) = topic_ids {
                topic_ids_vec.push(topic_ids);
            }
            topic_keyword_vec.extend(topic_keyword);
        }

        // Wrap into IPC [RecordBatch]
        let mut names = Vec::new();
        let mut publishers = Vec::new();
        let mut subjects = Vec::new();
        let mut formats = Vec::new();
        let mut bytes = Vec::new();
        if !topic_vec.is_empty() {
            names.push(topic_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(topic_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(topic_vec.first().unwrap().get_name())
                .with_schema(topic_vec.first().unwrap().to_schema())
                .with_struct::<TopicTable>(&topic_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !topic_domain_vec.is_empty() {
            names.push(topic_domain_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(topic_domain_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(topic_domain_vec.first().unwrap().get_name())
                .with_schema(topic_domain_vec.first().unwrap().to_schema())
                .with_struct::<TopicDomainTable>(&topic_domain_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !topic_field_vec.is_empty() {
            names.push(topic_field_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(topic_field_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(topic_field_vec.first().unwrap().get_name())
                .with_schema(topic_field_vec.first().unwrap().to_schema())
                .with_struct::<TopicFieldTable>(&topic_field_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !topic_subfield_vec.is_empty() {
            names.push(topic_subfield_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(topic_subfield_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(topic_subfield_vec.first().unwrap().get_name())
                .with_schema(topic_subfield_vec.first().unwrap().to_schema())
                .with_struct::<TopicSubfieldTable>(&topic_subfield_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !topic_ids_vec.is_empty() {
            names.push(topic_ids_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(topic_ids_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(topic_ids_vec.first().unwrap().get_name())
                .with_schema(topic_ids_vec.first().unwrap().to_schema())
                .with_struct::<TopicIdsTable>(&topic_ids_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }
        if !topic_keyword_vec.is_empty() {
            names.push(topic_keyword_vec.first().unwrap().get_name().to_string());
            publishers.push(publisher.to_string());
            subjects.push(topic_keyword_vec.first().unwrap().get_name().to_string());
            formats.push(DataFormat::Ipc.to_string());
            bytes.push(Table::get_builder()
                .with_name(topic_keyword_vec.first().unwrap().get_name())
                .with_schema(topic_keyword_vec.first().unwrap().to_schema())
                .with_struct::<TopicKeywordTable>(&topic_keyword_vec)?
                .build()?
                .to_ipc_stream()?
            );
        }

        let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        Ok(batch)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseSource {
    pub results: Vec<Source>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseSource {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        // let mut topic_vec = Vec::new();
        // for result in self.results {
        //     let (topic, topic_domain, topic_field, topic_subfield, topic_ids, topic_keyword) = result.to_tables();
        //     topic_vec.push(topic);
        //     if let Some(topic_ids) = topic_ids {
        //         topic_ids_vec.push(topic_ids);
        //     }
        //     topic_keyword_vec.extend(topic_keyword);
        // }

        // // Wrap into IPC [RecordBatch]
        // let mut names = Vec::new();
        // let mut publishers = Vec::new();
        // let mut subjects = Vec::new();
        // let mut formats = Vec::new();
        // let mut bytes = Vec::new();
        // if !topic_vec.is_empty() {
        //     names.push(topic_vec.first().unwrap().get_name().to_string());
        //     publishers.push(publisher.to_string());
        //     subjects.push(topic_vec.first().unwrap().get_name().to_string());
        //     formats.push(DataFormat::Ipc.to_string());
        //     bytes.push(Table::get_builder()
        //         .with_name(topic_vec.first().unwrap().get_name())
        //         .with_schema(topic_vec.first().unwrap().to_schema())
        //         .with_struct::<TopicTable>(&topic_vec)?
        //         .build()?
        //         .to_ipc_stream()?
        //     );
        // }

        // let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        // Ok(batch)
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponsePublisher {
    pub results: Vec<Publisher>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponsePublisher {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        // let mut topic_vec = Vec::new();
        // for result in self.results {
        //     let (topic, topic_domain, topic_field, topic_subfield, topic_ids, topic_keyword) = result.to_tables();
        //     topic_vec.push(topic);
        //     if let Some(topic_ids) = topic_ids {
        //         topic_ids_vec.push(topic_ids);
        //     }
        //     topic_keyword_vec.extend(topic_keyword);
        // }

        // // Wrap into IPC [RecordBatch]
        // let mut names = Vec::new();
        // let mut publishers = Vec::new();
        // let mut subjects = Vec::new();
        // let mut formats = Vec::new();
        // let mut bytes = Vec::new();
        // if !topic_vec.is_empty() {
        //     names.push(topic_vec.first().unwrap().get_name().to_string());
        //     publishers.push(publisher.to_string());
        //     subjects.push(topic_vec.first().unwrap().get_name().to_string());
        //     formats.push(DataFormat::Ipc.to_string());
        //     bytes.push(Table::get_builder()
        //         .with_name(topic_vec.first().unwrap().get_name())
        //         .with_schema(topic_vec.first().unwrap().to_schema())
        //         .with_struct::<TopicTable>(&topic_vec)?
        //         .build()?
        //         .to_ipc_stream()?
        //     );
        // }

        // let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        // Ok(batch)
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseAward {
    pub results: Vec<Award>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseAward {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        // let mut topic_vec = Vec::new();
        // for result in self.results {
        //     let (topic, topic_domain, topic_field, topic_subfield, topic_ids, topic_keyword) = result.to_tables();
        //     topic_vec.push(topic);
        //     if let Some(topic_ids) = topic_ids {
        //         topic_ids_vec.push(topic_ids);
        //     }
        //     topic_keyword_vec.extend(topic_keyword);
        // }

        // // Wrap into IPC [RecordBatch]
        // let mut names = Vec::new();
        // let mut publishers = Vec::new();
        // let mut subjects = Vec::new();
        // let mut formats = Vec::new();
        // let mut bytes = Vec::new();
        // if !topic_vec.is_empty() {
        //     names.push(topic_vec.first().unwrap().get_name().to_string());
        //     publishers.push(publisher.to_string());
        //     subjects.push(topic_vec.first().unwrap().get_name().to_string());
        //     formats.push(DataFormat::Ipc.to_string());
        //     bytes.push(Table::get_builder()
        //         .with_name(topic_vec.first().unwrap().get_name())
        //         .with_schema(topic_vec.first().unwrap().to_schema())
        //         .with_struct::<TopicTable>(&topic_vec)?
        //         .build()?
        //         .to_ipc_stream()?
        //     );
        // }

        // let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        // Ok(batch)
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAlexResponseFunder {
    pub results: Vec<Funder>,
    pub meta: Meta,
}

impl JsonSchemaTrait for OpenAlexResponseFunder {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch> {
        // let mut topic_vec = Vec::new();
        // for result in self.results {
        //     let (topic, topic_domain, topic_field, topic_subfield, topic_ids, topic_keyword) = result.to_tables();
        //     topic_vec.push(topic);
        //     if let Some(topic_ids) = topic_ids {
        //         topic_ids_vec.push(topic_ids);
        //     }
        //     topic_keyword_vec.extend(topic_keyword);
        // }

        // // Wrap into IPC [RecordBatch]
        // let mut names = Vec::new();
        // let mut publishers = Vec::new();
        // let mut subjects = Vec::new();
        // let mut formats = Vec::new();
        // let mut bytes = Vec::new();
        // if !topic_vec.is_empty() {
        //     names.push(topic_vec.first().unwrap().get_name().to_string());
        //     publishers.push(publisher.to_string());
        //     subjects.push(topic_vec.first().unwrap().get_name().to_string());
        //     formats.push(DataFormat::Ipc.to_string());
        //     bytes.push(Table::get_builder()
        //         .with_name(topic_vec.first().unwrap().get_name())
        //         .with_schema(topic_vec.first().unwrap().to_schema())
        //         .with_struct::<TopicTable>(&topic_vec)?
        //         .build()?
        //         .to_ipc_stream()?
        //     );
        // }

        // let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        // Ok(batch)
        todo!()
    }
}

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