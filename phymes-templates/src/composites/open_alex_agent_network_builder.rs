use object_store::aws::AmazonS3ConfigKey;
use serde_json::{Map, Value};

/// OpenAlex agent
pub struct OpenAlexAgentNetworkBuilder<'a> {
    /// Session
    pub network_name: &'a str,
}

impl Default for OpenAlexAgentNetworkBuilder<'_> {
    fn default() -> Self {
        OpenAlexAgentNetworkBuilder {
            network_name: "open_alex_agent_network",
        }
    }
}

impl<'a> OpenAlexAgentNetworkBuilder<'a> {
    pub fn as_mermaid_flowchart(&self) -> String {
        let network_name = self.network_name;
        format!(
            r#"flowchart TD
    {network_name}_r-rt@{{shape: subproc, label: {network_name}_r}}
	%% ------------------------------------------------------------------------------
	%% OpenAlex download from AWS
    %% 1. Read from the AWS bucket
    %% 2. Extract to OpenAlex Schemas
	%% ------------------------------------------------------------------------------
	subgraph get_open_alex_aws_bucket_t
		list_open_alex_aws_bucket_s-subject-.->|AllRecordBatches|get_open_alex_aws_bucket_p-subscribe
		get_open_alex_aws_bucket_p-subscribe-->get_open_alex_aws_bucket_p-processor
		get_open_alex_aws_bucket_p-processor-->get_open_alex_aws_bucket_p-publish
		get_open_alex_aws_bucket_p-publish-->|Replace|get_open_alex_aws_bucket_s-subject
	end
	{network_name}_r-rt-->get_open_alex_aws_bucket_t
	list_open_alex_aws_bucket_s-subject@{{shape: doc, label: list_open_alex_aws_bucket_s}}
	get_open_alex_aws_bucket_p-subscribe@{{shape: diamond, label: All}}
	get_open_alex_aws_bucket_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	get_open_alex_aws_bucket_p-publish@{{shape: fork}}
	get_open_alex_aws_bucket_s-subject@{{shape: doc, label: get_open_alex_aws_bucket_s}}
	%% ------------------------------------------------------------------------------
	subgraph extract_open_alex_aws_bucket_t
		get_open_alex_aws_bucket_s-subject-.->|AllRecordBatches|extract_open_alex_aws_bucket_p-subscribe
		extract_open_alex_aws_bucket_p-subscribe-->extract_open_alex_aws_bucket_p-processor
		extract_open_alex_aws_bucket_p-processor-->extract_open_alex_aws_bucket_p-publish
		extract_open_alex_aws_bucket_p-publish-->|Replace|extract_open_alex_aws_bucket_s-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkAwardTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkAuthorshipTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkFunderTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkApcInfoTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkLocationTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkOpenAccessTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkBiblioTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkCitationPercentileTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkCitedByPercentileYearTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkCountsByYearTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkConceptTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkTopicTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkKeywordTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkMeshTagTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkSdgTagTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkCorrespondingAuthorTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkCorrespondingInstitutionTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkIndexedInTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkIdsTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkReferencedWorksTable-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|WorkRelatedWorksTable-subject
	end
	{network_name}_r-rt-->extract_open_alex_aws_bucket_t
	extract_open_alex_aws_bucket_p-subscribe@{{shape: diamond, label: All}}
	extract_open_alex_aws_bucket_p-processor@{{shape: rect, label: ExtractTabular}}
	extract_open_alex_aws_bucket_p-publish@{{shape: fork}}
	extract_open_alex_aws_bucket_s-subject@{{shape: doc, label: extract_open_alex_aws_bucket_s}}
	WorkTable-subject@{{shape: doc, label: WorkTable}}
	WorkAwardTable-subject@{{shape: doc, label: WorkAwardTable}}
	WorkAuthorshipTable-subject@{{shape: doc, label: WorkAuthorshipTable}}
	WorkFunderTable-subject@{{shape: doc, label: WorkFunderTable}}
	WorkApcInfoTable-subject@{{shape: doc, label: WorkApcInfoTable}}
	WorkLocationTable-subject@{{shape: doc, label: WorkLocationTable}}
	WorkOpenAccessTable-subject@{{shape: doc, label: WorkOpenAccessTable}}
	WorkBiblioTable-subject@{{shape: doc, label: WorkBiblioTable}}
	WorkCitationPercentileTable-subject@{{shape: doc, label: WorkCitationPercentileTable}}
	WorkCitedByPercentileYearTable-subject@{{shape: doc, label: WorkCitedByPercentileYearTable}}
	WorkCountsByYearTable-subject@{{shape: doc, label: WorkCountsByYearTable}}
	WorkConceptTable-subject@{{shape: doc, label: WorkConceptTable}}
	WorkTopicTable-subject@{{shape: doc, label: WorkTopicTable}}
	WorkKeywordTable-subject@{{shape: doc, label: WorkKeywordTable}}
	WorkMeshTagTable-subject@{{shape: doc, label: WorkMeshTagTable}}
	WorkSdgTagTable-subject@{{shape: doc, label: WorkSdgTagTable}}
	WorkCorrespondingAuthorTable-subject@{{shape: doc, label: WorkCorrespondingAuthorTable}}
	WorkCorrespondingInstitutionTable-subject@{{shape: doc, label: WorkCorrespondingInstitutionTable}}
	WorkIndexedInTable-subject@{{shape: doc, label: WorkIndexedInTable}}
	WorkIdsTable-subject@{{shape: doc, label: WorkIdsTable}}
	WorkReferencedWorksTable-subject@{{shape: doc, label: WorkReferencedWorksTable}}
	WorkRelatedWorksTable-subject@{{shape: doc, label: WorkRelatedWorksTable}}
	%% ------------------------------------------------------------------------------
	%% OpenAlex search for OpenAccess articles by topic
    %% 1. Extract `WorkTopicTable`
    %% 2. Filter works by Topic
	%% ------------------------------------------------------------------------------
	subgraph filter_work_topic_table_t
		WorkTopicTable-subject-.->|AllRecordBatches|cmp_work_topic_table_p-subscribe
		cmp_work_topic_table_p-subscribe-->cmp_work_topic_table_p-processor
		cmp_work_topic_table_p-processor-->cmp_work_topic_table_p-publish
		cmp_work_topic_table_p-publish-->|Replace|cmp_work_topic_table_s-subject
		cmp_work_topic_table_s-subject-->|AllRecordBatches|filter_work_topic_table_p-subscribe
		filter_work_topic_table_p-subscribe-->filter_work_topic_table_p-processor
		filter_work_topic_table_p-processor-->filter_work_topic_table_p-publish
		filter_work_topic_table_p-publish-->|Replace|filter_work_topic_table_s-subject
		filter_work_topic_table_s-subject-->|AllRecordBatches|select_work_topic_table_p-subscribe
		select_work_topic_table_p-subscribe-->select_work_topic_table_p-processor
		select_work_topic_table_p-processor-->select_work_topic_table_p-publish
		select_work_topic_table_p-publish-->|Replace|select_work_topic_table_s-subject
		open_alex_topics_s-subject-->|AllRecordBatches|join_work_topic_table_p-subscribe
		select_work_topic_table_s-subject-->|AllRecordBatches|join_work_topic_table_p-subscribe
		join_work_topic_table_p-subscribe-->join_work_topic_table_p-processor
		join_work_topic_table_p-processor-->join_work_topic_table_p-publish
		join_work_topic_table_p-publish-->|Replace|join_work_topic_table_s-subject
	end
	{network_name}_r-rt-->filter_work_topic_table_t
	cmp_work_topic_table_p-subscribe@{{shape: diamond, label: All}}
	cmp_work_topic_table_p-processor@{{shape: rect, label: Select}}
	cmp_work_topic_table_p-publish@{{shape: fork}}
	cmp_work_topic_table_s-subject@{{shape: doc, label: cmp_work_topic_table_s}}
	filter_work_topic_table_p-subscribe@{{shape: diamond, label: All}}
	filter_work_topic_table_p-processor@{{shape: rect, label: Filter}}
	filter_work_topic_table_p-publish@{{shape: fork}}
	filter_work_topic_table_s-subject@{{shape: doc, label: filter_work_topic_table_s}}
	select_work_topic_table_p-subscribe@{{shape: diamond, label: All}}
	select_work_topic_table_p-processor@{{shape: rect, label: Select}}
	select_work_topic_table_p-publish@{{shape: fork}}
	select_work_topic_table_s-subject@{{shape: doc, label: select_work_topic_table_s}}
	open_alex_topics_s-subject@{{shape: doc, label: open_alex_topics_s}}
	join_work_topic_table_p-subscribe@{{shape: diamond, label: All}}
	join_work_topic_table_p-processor@{{shape: rect, label: Join}}
	join_work_topic_table_p-publish@{{shape: fork}}
	join_work_topic_table_s-subject@{{shape: doc, label: join_work_topic_table_s}}
	%% ------------------------------------------------------------------------------
	%% OpenAlex search for OpenAccess PDF URLs
    %% 1. Extract `WorkLocationTable`
    %% 2. Join filtered `WorkTopicTable` and `WorkLocationTable` on work_id
    %% 3. List OpenAccess PDF URLs
	%% ------------------------------------------------------------------------------
	subgraph select_open_access_pdf_url_t
		WorkLocationTable-subject-.->|AllRecordBatches|cmp_work_location_table_p-subscribe
		cmp_work_location_table_p-subscribe-->cmp_work_location_table_p-processor
		cmp_work_location_table_p-processor-->cmp_work_location_table_p-publish
		cmp_work_location_table_p-publish-->|Replace|cmp_work_location_table_s-subject
		cmp_work_location_table_s-subject-->|AllRecordBatches|filter_work_location_table_p-subscribe
		filter_work_location_table_p-subscribe-->filter_work_location_table_p-processor
		filter_work_location_table_p-processor-->filter_work_location_table_p-publish
		filter_work_location_table_p-publish-->|Replace|filter_work_location_table_s-subject
		filter_work_location_table_s-subject-->|AllRecordBatches|select_work_location_table_p-subscribe
		select_work_location_table_p-subscribe-->select_work_location_table_p-processor
		select_work_location_table_p-processor-->select_work_location_table_p-publish
		select_work_location_table_p-publish-->|Replace|select_work_location_table_s-subject
		join_work_topic_table_s-subject-.->|AllRecordBatches|join_work_location_table_p-subscribe
		select_work_location_table_s-subject-->|AllRecordBatches|join_work_location_table_p-subscribe
		join_work_location_table_p-subscribe-->join_work_location_table_p-processor
		join_work_location_table_p-processor-->join_work_location_table_p-publish
		join_work_location_table_p-publish-->|Replace|join_work_location_table_s-subject
		join_work_location_table_s-subject-->|AllRecordBatches|select_open_acces_pdf_url_p-subscribe
		select_open_acces_pdf_url_p-subscribe-->select_open_acces_pdf_url_p-processor
		select_open_acces_pdf_url_p-processor-->select_open_acces_pdf_url_p-publish
		select_open_acces_pdf_url_p-publish-->|Replace|select_open_acces_pdf_url_s-subject
	end
	{network_name}_r-rt-->select_open_access_pdf_url_t
	cmp_work_location_table_p-subscribe@{{shape: diamond, label: All}}
	cmp_work_location_table_p-processor@{{shape: rect, label: Select}}
	cmp_work_location_table_p-publish@{{shape: fork}}
	cmp_work_location_table_s-subject@{{shape: doc, label: cmp_work_location_table_s}}
	filter_work_location_table_p-subscribe@{{shape: diamond, label: All}}
	filter_work_location_table_p-processor@{{shape: rect, label: Filter}}
	filter_work_location_table_p-publish@{{shape: fork}}
	filter_work_location_table_s-subject@{{shape: doc, label: filter_work_location_table_s}}
	select_work_location_table_p-subscribe@{{shape: diamond, label: All}}
	select_work_location_table_p-processor@{{shape: rect, label: Select}}
	select_work_location_table_p-publish@{{shape: fork}}
	select_work_location_table_s-subject@{{shape: doc, label: select_work_location_table_s}}
	join_work_location_table_p-subscribe@{{shape: diamond, label: All}}
	join_work_location_table_p-processor@{{shape: rect, label: Join}}
	join_work_location_table_p-publish@{{shape: fork}}
	join_work_location_table_s-subject@{{shape: doc, label: join_work_location_table_s}}
	select_open_acces_pdf_url_p-subscribe@{{shape: diamond, label: All}}
	select_open_acces_pdf_url_p-processor@{{shape: rect, label: Select}}
	select_open_acces_pdf_url_p-publish@{{shape: fork}}
	select_open_acces_pdf_url_s-subject@{{shape: doc, label: select_open_acces_pdf_url_s}}
	%% ------------------------------------------------------------------------------
	%% HTTP OpenAccess PDF download
    %% 1. Download PDF
    %% 2. ExtractPDFNetwork
    %% 3. EmbedTextNetwork (where query = Ontology term)
    %% 4. RetrieveTextNetwork
	%% ------------------------------------------------------------------------------
	subgraph download_open_access_pdf_t
		select_open_acces_pdf_url_s-subject-.->|AllRecordBatches|download_open_access_pdf_p-subscribe
		download_open_access_pdf_p-subscribe-->download_open_access_pdf_p-processor
		download_open_access_pdf_p-processor-->download_open_access_pdf_p-publish
		download_open_access_pdf_p-publish-->|Extend|UserPdf-subject
	end
	{network_name}_r-rt-->download_open_access_pdf_t
	download_open_access_pdf_p-processor@{{shape: rect, label: HTTPClientRequestProcessor}}
	download_open_access_pdf_p-publish@{{shape: fork}}
	download_open_access_pdf_p-subscribe@{{shape: diamond, label: All}}
	UserPdf-subject@{{shape: doc, label: UserPdf}}
	%% ------------------------------------------------------------------------------
	%% Extensions
    %% 1. ExtractPDFNetwork
    %% 2. EmbedTextNetwork (where query = Ontology term)
    %% 3. RetrieveTextNetwork
	%% ------------------------------------------------------------------------------"#
        )
    }
    pub fn as_mermaid_erdiagram(&self) -> String {
        let bucket = "openalex";
        let mut config = Map::<String, Value>::new();
        let _ = config.insert(
            AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
            Value::String("true".to_string()),
        );
        let _ = config.insert(
            AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
            Value::String("https://s3.amazonaws.com".to_string()),
        );
        let backend_config = serde_json::to_string(&config).unwrap().replace('"', "'");
        // List-UInt8 bytes
        format!(
            r#"erDiagram
    list_open_alex_aws_bucket_s["list_open_alex_aws_bucket_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    get_open_alex_aws_bucket_p["get_open_alex_aws_bucket_p"] {{
        UInt32 timeout "15"
        Utf8 ops_type "Get"
        Utf8 backend "Aws"
        Utf8 bucket "{bucket}"
        Utf8 backend_config "{backend_config}"
        Utf8 subject_name "list_open_alex_aws_bucket_s"
    }}
    get_open_alex_aws_bucket_s["get_open_alex_aws_bucket_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 metadata
        Int64 last_modified
        List-UInt8 bytes
    }}
    extract_open_alex_aws_bucket_p["extract_open_alex_aws_bucket_p"] {{
	    Boolean cpu "false"
	    Utf8 format "JsonSchema"
        Utf8 schema "OpenAlexResponseWorks"
        Utf8 encoding "Gz"
	    Utf8 lhs_name "get_open_alex_aws_bucket_s"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractTabular"
	    Utf8 lhs_stream "Accumulate"
    }}
    extract_open_alex_aws_bucket_s["extract_open_alex_aws_bucket_s"] {{
        List-UInt8 bytes
    }}
    WorkTable["WorkTable"] {{
        List-UInt8 bytes
    }}
    WorkAwardTable["WorkAwardTable"] {{
        List-UInt8 bytes
    }}
    WorkAuthorshipTable["WorkAuthorshipTable"] {{
        List-UInt8 bytes
    }}
    WorkFunderTable["WorkFunderTable"] {{
        List-UInt8 bytes
    }}
    WorkApcInfoTable["WorkApcInfoTable"] {{
        List-UInt8 bytes
    }}
    WorkLocationTable["WorkLocationTable"] {{
        List-UInt8 bytes
    }}
    WorkOpenAccessTable["WorkOpenAccessTable"] {{
        List-UInt8 bytes
    }}
    WorkBiblioTable["WorkBiblioTable"] {{
        List-UInt8 bytes
    }}
    WorkCitationPercentileTable["WorkCitationPercentileTable"] {{
        List-UInt8 bytes
    }}
    WorkCitedByPercentileYearTable["WorkCitedByPercentileYearTable"] {{
        List-UInt8 bytes
    }}
    WorkCountsByYearTable["WorkCountsByYearTable"] {{
        List-UInt8 bytes
    }}
    WorkConceptTable["WorkConceptTable"] {{
        List-UInt8 bytes
    }}
    WorkTopicTable["WorkTopicTable"] {{
        List-UInt8 bytes
    }}
    WorkKeywordTable["WorkKeywordTable"] {{
        List-UInt8 bytes
    }}
    WorkMeshTagTable["WorkMeshTagTable"] {{
        List-UInt8 bytes
    }}
    WorkSdgTagTable["WorkSdgTagTable"] {{
        List-UInt8 bytes
    }}
    WorkCorrespondingAuthorTable["WorkCorrespondingAuthorTable"] {{
        List-UInt8 bytes
    }}
    WorkCorrespondingInstitutionTable["WorkCorrespondingInstitutionTable"] {{
        List-UInt8 bytes
    }}
    WorkIndexedInTable["WorkIndexedInTable"] {{
        List-UInt8 bytes
    }}
    WorkIdsTable["WorkIdsTable"] {{
        List-UInt8 bytes
    }}
    WorkReferencedWorksTable["WorkReferencedWorksTable"] {{
        List-UInt8 bytes
    }}
    WorkRelatedWorksTable["WorkRelatedWorksTable"] {{
        List-UInt8 bytes
    }}
    cmp_work_topic_table_p["cmp_work_topic_table_p"] {{
        List-Utf8 as_columns "['work_id','topic_id','is_primary','score','cmp_is_primary','cmp_score']"
		List-Utf8 cast_templates "['','','','','1','0.5']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','UInt8','Float32','UInt8','Float32']"
        List-Utf8 column_operators "['None','None','None','None','Value','Value']"
        Boolean cpu "false"
        Utf8 lhs_name "WorkTopicTable"
        List-Utf8 lhs_values "['work_id','topic_id','is_primary','score','cmp_is_primary','cmp_score']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
    }}
    filter_work_topic_table_p["filter_work_topic_table_p"] {{
        List-Utf8 cmp_columns "['cmp_is_primary', 'cmp_score']"
        List-Utf8 cmp_operators "['Equals', 'GreaterThan']"
        Utf8 cmp_predicate "All"
        Boolean cpu "true"
        Utf8 lhs_name "cmp_work_topic_table_s"
        List-Utf8 lhs_values "['is_primary', 'score']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
    }}
    select_work_topic_table_p["select_work_topic_table_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "filter_work_topic_table_s"
        List-Utf8 lhs_values "['work_id','topic_id','is_primary','score']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
    }}
    open_alex_topics_s["open_alex_topics_s"] {{
        Utf8 topic_id "https://openalex.org/T10123"
    }}
    join_work_topic_table_p["join_work_topic_table_p"] {{
        Boolean cpu "false"
        Utf8 lhs_fk "topic_id"
        Utf8 lhs_name "open_alex_topics_s"
        Utf8 lhs_pk "topic_id"
        Utf8 operator "Join"
        Utf8 rhs_fk "topic_id"
        Utf8 rhs_name "select_work_topic_table_s"
        Utf8 rhs_pk "topic_id"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }}
    join_work_topic_table_s["join_work_topic_table_s"] {{
        Utf8 topic_id
        Utf8 work_id
        UInt8 is_primary
        Float32 score
    }}
    cmp_work_location_table_p["cmp_work_location_table_p"] {{
        List-Utf8 as_columns "['work_id','landing_page_url','pdf_url','source_id','license','version','is_best_oa','is_primary','is_oa','cmp_is_best_oa','pdf_url_len','cmp_pdf_url_len']"
		List-Utf8 cast_templates "['','','','','','','','','','1','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8','UInt8','UInt8','UInt32','UInt32']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','None','Value','Len','Zeros']"
        Boolean cpu "false"
        Utf8 lhs_name "WorkLocationTable"
        List-Utf8 lhs_values "['work_id','landing_page_url','pdf_url','source_id','license','version','is_best_oa','is_primary','is_oa','cmp_is_best_oa','pdf_url','cmp_pdf_url_len']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
    }}
    filter_work_location_table_p["filter_work_location_table_p"] {{
        List-Utf8 cmp_columns "['cmp_is_best_oa','cmp_pdf_url_len']"
        List-Utf8 cmp_operators "['Equals','GreaterThan']"
        Utf8 cmp_predicate "All"
        Boolean cpu "true"
        Utf8 lhs_name "cmp_work_location_table_s"
        List-Utf8 lhs_values "['is_best_oa','pdf_url_len']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
    }}
    select_work_location_table_p["select_work_location_table_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "filter_work_location_table_s"
        List-Utf8 lhs_values "['work_id','landing_page_url','pdf_url','source_id','license','version','is_best_oa','is_primary','is_oa']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
    }}
    join_work_location_table_p["join_work_location_table_p"] {{
        Boolean cpu "false"
        Utf8 lhs_fk "work_id"
        Utf8 lhs_name "join_work_topic_table_s"
        Utf8 lhs_pk "work_id"
        Utf8 operator "Join"
        Utf8 rhs_fk "work_id"
        Utf8 rhs_name "select_work_location_table_s"
        Utf8 rhs_pk "work_id"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }}
    select_open_acces_pdf_url_p["select_open_acces_pdf_url_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "join_work_location_table_s"
        List-Utf8 as_columns "['','','','content','','']"
        List-Utf8 lhs_values "['work_id','topic_id','score','pdf_url','source_id','version']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    select_open_acces_pdf_url_s["select_open_acces_pdf_url_s"] {{
        Utf8 work_id
        Utf8 topic_id
        Float32 score
        Utf8 content
        Utf8 source_id
        Utf8 version
    }}
    download_open_access_pdf_p["download_open_access_pdf_p"] {{
        UInt32 timeout "15"
        Utf8 request_type "Get"
        Utf8 user_agent_type "rust-openalex-client/2.0"
        Utf8 subject_name "select_open_acces_pdf_url_s"
        Utf8 request_schema "Attachments"
        Utf8 base_url ""
    }}
    UserPdf["UserPdf"] {{
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }}"#
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait,
        create_object_store_meta_batch,
    };
    use phymes_streams::ChatBuilderTraitExt;
    use phymes_task::SubscriptionTrait;

    use super::*;
    use crate::{EmbedTextNetwork, ExtractPDFNetwork, RetrieveTextNetwork};

    #[ignore = "In progress... Some issues with embeddings and retrieval."]
    #[tokio::test(flavor = "current_thread")]
    async fn test_open_alex_agent_network() -> Result<()> {
        // Extract PDF session
        let extract_pdf_session = ExtractPDFNetwork::default();
        let extract_pdf_network_builder = NetworkBuilder::from_mermaid_flowchart(
            extract_pdf_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            extract_pdf_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(extract_pdf_session.network_name);

        // Embed text session
        let embed_text_network = EmbedTextNetwork::default();
        let embed_text_network_builder = NetworkBuilder::from_mermaid_flowchart(
            &embed_text_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &embed_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(embed_text_network.network_name);

        // Retrieve text session
        let retrieve_text_network = RetrieveTextNetwork::default();
        let retrieve_text_builder = NetworkBuilder::from_mermaid_flowchart(
            retrieve_text_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            retrieve_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(retrieve_text_network.network_name);

        // Initialize the session
        let open_alex_agent_network = OpenAlexAgentNetworkBuilder::default();
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &open_alex_agent_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &open_alex_agent_network.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(open_alex_agent_network.network_name)
        .extend(extract_pdf_network_builder)?
        .extend(embed_text_network_builder)?
        .extend(retrieve_text_builder)?
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test session data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let messages = "list_open_alex_aws_bucket_s";

        // // Make the Get config
        // let bucket_name = "openalex";
        // let mut store_config = Map::<String, Value>::new();
        // let _ = store_config.insert(
        //     AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
        //     Value::String("true".to_string()),
        // );
        // let _ = store_config.insert(
        //     AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
        //     Value::String("https://s3.amazonaws.com".to_string()),
        // );
        // let config = ObjectStoreConfig {
        //     timeout: 5,
        //     ops_type: ObjectStoreOptsType::Get,
        //     backend: ObjectStorageBackend::Aws,
        //     bucket: Some(bucket_name.to_string()),
        //     backend_config: Some(store_config.clone()),
        //     locations: None,
        //     chunk_size: None,
        //     subject_name: Some(messages.to_string()),
        //     ..Default::default()
        // };
        // let config_json = serde_json::to_vec(&config)?;
        // let config_batch = create_bytes_record_batch(vec![config_json])?;
        // let config_table = SubjectBuilder::new()
        //     .with_name("get_open_alex_aws_bucket_p")
        //     .with_record_batches(vec![config_batch])?
        //     .build()?;
        // let _ = message_map.insert(
        //     config_table.get_name().to_string(),
        //     IPCMessage::get_builder()
        //         .with_name(config_table.get_name())
        //         .with_publisher(open_alex_agent_network.network_name)
        //         .with_subject(config_table.get_name())
        //         .with_update(&Publication::Replace {
        //             subject_name: config_table.get_name().to_string(),
        //         })
        //         .with_message(config_table.to_ipc_stream()?)
        //         .build()?,
        // );

        // Make the list of paths to Get
        // let location = vec!["data/works/updated_date=2018-01-12/part_0000.gz".to_string()];
        let location = vec!["data/works/updated_date=2026-03-10/part_0005.gz".to_string()];
        // let location = vec!["data/works/manifest".to_string()];
        let bucket = vec!["openalex".to_string()];
        let e_tag = vec![String::new()];
        let version = vec![String::new()];
        let size = vec![0_u32];
        let last_modified = vec![0_i64];
        let message_batch =
            create_object_store_meta_batch(location, bucket, e_tag, version, size, last_modified)?;
        let message_subject = Subject::get_builder()
            .with_name(&messages)
            .with_record_batches(vec![message_batch])?
            .build()?;
        let _ = message_map.insert(
            messages.to_string(),
            IPCMessage::get_builder()
                .with_name(&messages)
                .with_publisher(open_alex_agent_network.network_name)
                .with_subject(&messages)
                .with_update(&Publication::Replace {
                    subject_name: messages.to_string(),
                })
                .with_message(message_subject.to_ipc_stream()?)
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the session
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionErrors,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionEvents.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionEvents,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }

        assert_eq!(response.len(), 0);

        // Test AWS object store GET
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "WorkTable".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("WorkTable")
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 34973);
        let column = subject.get_column_as_vec_str("work_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/W4367307220");

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "WorkTopicTable".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("WorkTopicTable")
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 91321);
        let column = subject.get_column_as_vec_str("work_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/W4367307220");
        let column = subject.get_column_as_vec_str("topic_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/T10123");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/T13802");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        assert_eq!(column.first().unwrap(), &0.9994);
        assert_eq!(column.last().unwrap(), &0.2251);
        let column = subject.get_column_as_vec_primitive::<u8>("is_primary")?;
        assert_eq!(column.first().unwrap(), &1);
        assert_eq!(column.last().unwrap(), &1);

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "WorkLocationTable".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("WorkLocationTable")
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 48958);
        let column = subject.get_column_as_vec_str("work_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/W4367307220");
        let column = subject.get_column_as_vec_primitive::<u8>("is_best_oa")?;
        assert_eq!(column.first().unwrap(), &0);
        assert_eq!(column.last().unwrap(), &0);
        let column = subject.get_column_as_vec_primitive::<u8>("is_primary")?;
        assert_eq!(column.first().unwrap(), &1);
        assert_eq!(column.last().unwrap(), &1);
        let column = subject.get_column_as_vec_primitive::<u8>("is_oa")?;
        assert_eq!(column.first().unwrap(), &0);
        assert_eq!(column.last().unwrap(), &0);
        let column = subject.get_column_as_vec_str("landing_page_url");
        assert_eq!(
            column.first().unwrap(),
            &"https://doi.org/10.1016/j.str.2014.09.012"
        );
        assert_eq!(
            column.last().unwrap(),
            &"http://dx.doi.org/10.2307/jj.2430693"
        );
        let column = subject.get_column_as_vec_str("pdf_url");
        assert_eq!(column.first().unwrap(), &"");
        assert_eq!(column.last().unwrap(), &"");
        let column = subject.get_column_as_vec_str("source_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/S7112016");
        assert_eq!(column.last().unwrap(), &"");
        let column = subject.get_column_as_vec_str("license");
        assert_eq!(column.first().unwrap(), &"");
        assert_eq!(column.last().unwrap(), &"cc-by");
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.first().unwrap(), &"publishedVersion");
        assert_eq!(column.last().unwrap(), &"publishedVersion");

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "extract_open_alex_aws_bucket_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        assert!(batches.is_empty());

        // Test join work topic with user defined topics
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "join_work_topic_table_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("join_work_topic_table_s")
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 13);
        let column = subject.get_column_as_vec_str("work_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/W2036680792");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/W2037563286");
        let column = subject.get_column_as_vec_str("topic_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/T10123");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/T10123");
        let column = subject.get_column_as_vec_primitive::<u8>("is_primary")?;
        assert_eq!(column.first().unwrap(), &1);
        assert_eq!(column.last().unwrap(), &1);
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        assert_eq!(column.first().unwrap(), &0.9998);
        assert_eq!(column.last().unwrap(), &0.9998);

        // Test select open access PDF url as content
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "select_open_acces_pdf_url_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("select_open_acces_pdf_url_s")
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 6);
        let column = subject.get_column_as_vec_str("work_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/W2036554147");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/W4408584426");
        let column = subject.get_column_as_vec_str("topic_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/T10123");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/T10123");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        assert_eq!(column.first().unwrap(), &0.9998);
        assert_eq!(column.last().unwrap(), &0.9688);
        let column = subject.get_column_as_vec_str("content");
        assert_eq!(
            column.first().unwrap(),
            &"http://www.jidonline.org/article/S0022202X15321485/pdf"
        );
        assert_eq!(
            column.last().unwrap(),
            &"https://doi.org/10.37184/jlnh.2959-1805.3.9"
        );
        let column = subject.get_column_as_vec_str("source_id");
        assert_eq!(column.first().unwrap(), &"https://openalex.org/S28607811");
        assert_eq!(column.last().unwrap(), &"https://openalex.org/S4387288081");
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.first().unwrap(), &"publishedVersion");
        assert_eq!(column.last().unwrap(), &"publishedVersion");

        // Test HTTP request of open access PDF
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 6);
        let column = subject.get_column_as_vec_str("filename");
        assert_eq!(
            column.first().unwrap(),
            &"http://www.jidonline.org/article/S0022202X15321485/pdf"
        );
        assert_eq!(
            column.last().unwrap(),
            &"https://doi.org/10.37184/jlnh.2959-1805.3.9"
        );
        let column = subject.get_column_as_vec_str("extension");
        assert_eq!(column.first().unwrap(), &"text/html; charset=UTF-8");
        assert_eq!(column.last().unwrap(), &"application/pdf");
        let column = subject.get_column_as_vec_str("metadata");
        assert_eq!(column.first().unwrap(), &"tool");
        assert_eq!(column.last().unwrap(), &"tool");
        let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
        for c in column {
            assert!(c > 0);
        }
        let column = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert!(column.len() > 100);

        // Make the query data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let chat = AvailableInterfaceSubjects::UserMessages
            .to_subject_builder(None)
            .append_new_user_query_str(
                "What are two XPC-causing mutations identified in Chinese patients?",
                "user",
            )?
            .build()?;
        let _ = message_map.insert(
            chat.get_name().to_string(),
            IPCMessage::get_builder()
                .with_message(chat.to_ipc_stream()?)
                .with_subject(chat.get_name())
                .with_update(&Publication::Extend {
                    subject_name: chat.get_name().to_string(),
                })
                .with_publisher(open_alex_agent_network.network_name)
                .make_name()?
                .build()?,
        );

        // 2. Run the session
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionErrors,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionEvents.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionEvents,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionTraces.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionTraces,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }

        assert_eq!(response.len(), 0);

        // Test PDF extraction, embedding, and retrieval
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::EmbeddingScores.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::EmbeddingScores.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        dbg!(&subject.count_rows());
        // assert_eq!(subject.count_rows(), 5);
        let column = subject.get_column_as_vec_str("chunk_id");
        dbg!(column.first().unwrap());
        // assert_eq!(column.first().unwrap(), &""https://doi.org/10.37184/jlnh.2959-1805.3.9_4_1");
        let column = subject.get_column_as_vec_str("query_id");
        dbg!(column.first().unwrap());
        // assert_eq!(column.first().unwrap(), &"1775410537711065");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        for t in column {
            assert!(t > 0.15); // Threshold used for filtering
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(
                AvailableInterfaceSubjects::ToolMessages
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(batches)?
            .build()?;
        dbg!(&subject.count_rows());
        // assert_eq!(subject.count_rows(), 1);
        let column = subject.get_column_as_vec_str("role");
        dbg!(column.first().unwrap());
        // assert_eq!(column.first().unwrap(), &"tool");
        let column = subject.get_column_as_vec_str("content");
        dbg!(column.first().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"[{\"text\":\"Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.The two \"}]"
        // );
        let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
        for t in column {
            assert!(t > 0);
        }

        Ok(())
    }
}
