use std::fmt::Display;

use anyhow::{Result, anyhow};
use arrow::datatypes::DataType;
use clap::ValueEnum;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, IPCMessageMap, MessageBuilderTrait, OwlFormat, Table, TableBuilderTrait, TablePublication, TableTrait, create_session_tasks_subscribe_publish_batch, items_to_list, make_random_id
};
use serde::{Deserialize, Serialize};

use crate::create_message_map;

/// A session for melting a `Study Dataset` from a single workflow step
/// 
/// # Notes
/// 
/// * Does not consider pre-filtering by ontology before vector search
pub struct OntologyRAGSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// The ontology predicate tags to extract
    pub ontology_predicate_tags: Vec<String>,
    /// The ontology properties predicate tags to extract
    pub properties_predicate_tags: Vec<String>,
    /// The ontology classes predicate tags to extract
    pub classes_predicate_tags: Vec<String>,
}

impl<'a> OntologyRAGSession<'a> {
    /// New [OntologyRAGSession]
    pub fn new(session_context_name: Option<&'a str>,
        ontology_predicate_tags: Option<&[&str]>,
        properties_predicate_tags: Option<&[&str]>,
        classes_predicate_tags: Option<&[&str]>,
    ) -> Result<Self> {
        let session_context_name = if let Some(name) = session_context_name {
            name
        } else {
            "ontology_rag_session"
        };
        let ontology_predicate_tags = if let Some(tags) = ontology_predicate_tags {
            tags.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()
        } else {
            OwlFormat::owl_format_ontology().predicate_tags
        };
        let properties_predicate_tags = if let Some(tags) = properties_predicate_tags {
            tags.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()
        } else {
            OwlFormat::owl_format_object_property().predicate_tags
        };
        let classes_predicate_tags = if let Some(tags) = classes_predicate_tags {
            tags.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()
        } else {
            OwlFormat::owl_format_class().predicate_tags
        };
        Ok(Self {
            session_context_name,
            ontology_predicate_tags,
            properties_predicate_tags,
            classes_predicate_tags,
        })
    }

    /// Make the variable columns
    fn variable_columns(&self) -> Result<String> {
        items_to_list(self.variable_names)
    }

    /// Make the datatype columns
    fn data_type_columns(&self) -> Result<String> {
        let items = self.data_types.into_iter().map(|d| d.to_string()).collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Make cast operator columns for variables
    fn cast_operator_columns(&self) -> Result<String> {
        let items = self.data_types.iter().map(|_| "Cast".to_string()).collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Make cast template columns for variables
    fn cast_templates_columns(&self) -> Result<String> {
        let items = self.data_types.iter().map(|_| "".to_string()).collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// make column operator columns for variables
    fn column_operators_columns(&self) -> Result<String> {
        let items = self.data_types.iter().map(|_| "None".to_string()).collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD	
	%% -------------------------------------
	%% ontologies attachments and extraction
	%% -------------------------------------
	subgraph ontologies_extraction
	    UserScript-subject-.->|FullTable|ontologies_extract-subscribe
	    ontologies_extract-subscribe-->ontologies_extract-processor
	    ontologies_extract-processor-->ontologies_extract-publish
	    ontologies_extract-publish-->|Replace|OntologiesExtract-subject
	    OntologiesExtract-subject-->|FullTable|ontologies_extract_predicates-subscribe
	    ontologies_extract_predicates-subscribe-->ontologies_extract_predicates-processor
	    ontologies_extract_predicates-processor-->ontologies_extract_predicates-publish
	    ontologies_extract_predicates-publish-->|Replace|OntologiesExtractPredicates-subject
	    OntologiesExtractPredicates-subject-->|FullTable|ontologies_extract_filter-subscribe
	    ontologies_extract_filter-subscribe-->ontologies_extract_filter-processor
	    ontologies_extract_filter-processor-->ontologies_extract_filter-publish
	    ontologies_extract_filter-publish-->|Replace|OntologiesExtractFilter-subject
	    OntologiesExtractFilter-subject-->|FullTable|ontologies_extract_triples-subscribe
	    ontologies_extract_triples-subscribe-->ontologies_extract_triples-processor
	    ontologies_extract_triples-processor-->ontologies_extract_triples-publish
	    ontologies_extract_triples-publish-->|Replace|OntologiesExtractTriples-subject
	    OntologiesExtractTriples-subject-->|FullTable|ontologies_extract_pivot-subscribe
	    ontologies_extract_pivot-subscribe-->ontologies_extract_pivot-processor
	    ontologies_extract_pivot-processor-->ontologies_extract_pivot-publish
	    ontologies_extract_pivot-publish-->|Replace|OntologiesExtractPivot-subject
	    OntologiesExtractPivot-subject-->|FullTable|ontologies_extract_select-subscribe
	    ontologies_extract_select-subscribe-->ontologies_extract_select-processor
	    ontologies_extract_select-processor-->ontologies_extract_select-publish
	    ontologies_extract_select-publish-->|Extend|ONTOLOGIES-subject
	end
	ontologies_extraction-rt@{shape: subproc, label: ontologies_extraction}
	ontologies_extraction-rt-->ontologies_extraction
	UserScript-subject@{shape: doc, label: UserScript}
	ontologies_extract-processor@{shape: rect, label: ExtractXML}
	ontologies_extract-publish@{shape: fork}
	ontologies_extract-subscribe@{shape: diamond, label: All}
	OntologiesExtract-subject@{shape: doc, label: OntologiesExtract}
	ontologies_extract_predicates-processor@{shape: rect, label: Select}
	ontologies_extract_predicates-publish@{shape: fork}
	ontologies_extract_predicates-subscribe@{shape: diamond, label: All}
	OntologiesExtractPredicates-subject@{shape: doc, label: OntologiesExtractPredicates}
	ontologies_extract_filter-processor@{shape: rect, label: Filter}
	ontologies_extract_filter-publish@{shape: fork}
	ontologies_extract_filter-subscribe@{shape: diamond, label: All}
	OntologiesExtractFilter-subject@{shape: doc, label: OntologiesExtractFilter}
	ontologies_extract_triples-processor@{shape: rect, label: Select}
	ontologies_extract_triples-publish@{shape: fork}
	ontologies_extract_triples-subscribe@{shape: diamond, label: All}
	OntologiesExtractTriples-subject@{shape: doc, label: OntologiesExtractTriples}
	ontologies_extract_pivot-processor@{shape: rect, label: Pivot}
	ontologies_extract_pivot-publish@{shape: fork}
	ontologies_extract_pivot-subscribe@{shape: diamond, label: All}
	OntologiesExtractPivot-subject@{shape: doc, label: OntologiesExtractPivot}
	ontologies_extract_select-processor@{shape: rect, label: Select}
	ontologies_extract_select-publish@{shape: fork}
	ontologies_extract_select-subscribe@{shape: diamond, label: All}	
	ONTOLOGIES-subject@{shape: doc, label: ONTOLOGIES}
	%% -------------------------------------
	
	%% -------------------------------------
	%% classes attachments and extraction
	%% -------------------------------------
	subgraph classes_extraction
	    UserScript-subject-.->|FullTable|classes_extract-subscribe
	    classes_extract-subscribe-->classes_extract-processor
	    classes_extract-processor-->classes_extract-publish
	    classes_extract-publish-->|Replace|ClassesExtract-subject
	    ClassesExtract-subject-->|FullTable|classes_extract_coalesce-subscribe
	    classes_extract_coalesce-subscribe-->classes_extract_coalesce-processor
	    classes_extract_coalesce-processor-->classes_extract_coalesce-publish
	    classes_extract_coalesce-publish-->|Replace|ClassesExtractCoalesce-subject
	    ClassesExtractCoalesce-subject-->|FullTable|classes_extract_predicates-subscribe
	    classes_extract_predicates-subscribe-->classes_extract_predicates-processor
	    classes_extract_predicates-processor-->classes_extract_predicates-publish
	    classes_extract_predicates-publish-->|Replace|ClassesExtractPredicates-subject
	    ClassesExtractPredicates-subject-->|FullTable|classes_extract_filter-subscribe
	    classes_extract_filter-subscribe-->classes_extract_filter-processor
	    classes_extract_filter-processor-->classes_extract_filter-publish
	    classes_extract_filter-publish-->|Replace|ClassesExtractFilter-subject
	    ClassesExtractFilter-subject-->|FullTable|classes_extract_triples-subscribe
	    classes_extract_triples-subscribe-->classes_extract_triples-processor
	    classes_extract_triples-processor-->classes_extract_triples-publish
	    classes_extract_triples-publish-->|Replace|ClassesExtractTriples-subject
	    ClassesExtractTriples-subject-->|FullTable|classes_extract_pivot-subscribe
	    classes_extract_pivot-subscribe-->classes_extract_pivot-processor
	    classes_extract_pivot-processor-->classes_extract_pivot-publish
	    classes_extract_pivot-publish-->|Replace|ClassesExtractPivot-subject
	    ClassesExtractPivot-subject-->|FullTable|classes_extract_select-subscribe
	    classes_extract_select-subscribe-->classes_extract_select-processor
	    classes_extract_select-processor-->classes_extract_select-publish
	    classes_extract_select-publish-->|Replace|ClassesExtractSelect-subject
	end
	classes_extraction-rt@{shape: subproc, label: classes_extraction}
	classes_extraction-rt-->classes_extraction
	classes_extract-processor@{shape: rect, label: ExtractXML}
	classes_extract-publish@{shape: fork}
	classes_extract-subscribe@{shape: diamond, label: All}
	ClassesExtract-subject@{shape: doc, label: ClassesExtract}
	classes_extract_coalesce-processor@{shape: rect, label: CoalesceProcessor}
	classes_extract_coalesce-publish@{shape: fork}
	classes_extract_coalesce-subscribe@{shape: diamond, label: All}
	ClassesExtractCoalesce-subject@{shape: doc, label: ClassesExtractCoalesce}
	classes_extract_predicates-processor@{shape: rect, label: Select}
	classes_extract_predicates-publish@{shape: fork}
	classes_extract_predicates-subscribe@{shape: diamond, label: All}
	ClassesExtractPredicates-subject@{shape: doc, label: ClassesExtractPredicates}
	classes_extract_filter-processor@{shape: rect, label: Filter}
	classes_extract_filter-publish@{shape: fork}
	classes_extract_filter-subscribe@{shape: diamond, label: All}
	ClassesExtractFilter-subject@{shape: doc, label: ClassesExtractFilter}
	classes_extract_triples-processor@{shape: rect, label: Select}
	classes_extract_triples-publish@{shape: fork}
	classes_extract_triples-subscribe@{shape: diamond, label: All}
	ClassesExtractTriples-subject@{shape: doc, label: ClassesExtractTriples}
	classes_extract_pivot-processor@{shape: rect, label: Pivot}
	classes_extract_pivot-publish@{shape: fork}
	classes_extract_pivot-subscribe@{shape: diamond, label: All}
	ClassesExtractPivot-subject@{shape: doc, label: ClassesExtractPivot}
	classes_extract_select-processor@{shape: rect, label: Select}
	classes_extract_select-publish@{shape: fork}
	classes_extract_select-subscribe@{shape: diamond, label: All}
	ClassesExtractSelect-subject@{shape: doc, label: ClassesExtractSelect}
	%% -------------------------------------
	
	%% -------------------------------------
	%% classes post-pivot cleanup
	%% -------------------------------------
	subgraph classes_pivot
	    ClassesExtractSelect-subject-.->|FullTable|classes_pivot_coalesce-subscribe
	    classes_pivot_coalesce-subscribe-->classes_pivot_coalesce-processor
	    classes_pivot_coalesce-processor-->classes_pivot_coalesce-publish
	    classes_pivot_coalesce-publish-->|Replace|ClassesPivotCoalesce-subject
	    ClassesPivotCoalesce-subject-->|FullTable|classes_pivot_group_by-subscribe
	    classes_pivot_group_by-subscribe-->classes_pivot_group_by-processor
	    classes_pivot_group_by-processor-->classes_pivot_group_by-publish
	    classes_pivot_group_by-publish-->|Replace|ClassesPivotGroupBy-subject
	    ClassesPivotGroupBy-subject-->|FullTable|classes_pivot_select-subscribe
	    classes_pivot_select-subscribe-->classes_pivot_select-processor
	    classes_pivot_select-processor-->classes_pivot_select-publish
	    classes_pivot_select-publish-->|Extend|CLASSES-subject
	end
	classes_pivot-rt@{shape: subproc, label: classes_pivot}
	classes_pivot-rt-->classes_pivot
	classes_pivot_coalesce-processor@{shape: rect, label: CoalesceProcessor}
	classes_pivot_coalesce-publish@{shape: fork}
	classes_pivot_coalesce-subscribe@{shape: diamond, label: All}
	ClassesPivotCoalesce-subject@{shape: doc, label: ClassesPivotCoalesce}
	classes_pivot_group_by-processor@{shape: rect, label: GroupBy}
	classes_pivot_group_by-publish@{shape: fork}
	classes_pivot_group_by-subscribe@{shape: diamond, label: All}
	ClassesPivotGroupBy-subject@{shape: doc, label: ClassesPivotGroupBy}
	classes_pivot_select-processor@{shape: rect, label: Select}
	classes_pivot_select-publish@{shape: fork}
	classes_pivot_select-subscribe@{shape: diamond, label: All}
	CLASSES-subject@{shape: doc, label: CLASSES}
	%% -------------------------------------
	
	%% -------------------------------------
	%% properties attachments and extraction
	%% -------------------------------------
	subgraph properties_extraction
	    UserScript-subject-.->|FullTable|properties_extract-subscribe
	    properties_extract-subscribe-->properties_extract-processor
	    properties_extract-processor-->properties_extract-publish
	    properties_extract-publish-->|Replace|PropertiesExtract-subject
	    PropertiesExtract-subject-->|FullTable|properties_extract_coalesce-subscribe
	    properties_extract_coalesce-subscribe-->properties_extract_coalesce-processor
	    properties_extract_coalesce-processor-->properties_extract_coalesce-publish
	    properties_extract_coalesce-publish-->|Replace|PropertiesExtractCoalesce-subject
	    PropertiesExtractCoalesce-subject-->|FullTable|properties_extract_predicates-subscribe
	    properties_extract_predicates-subscribe-->properties_extract_predicates-processor
	    properties_extract_predicates-processor-->properties_extract_predicates-publish
	    properties_extract_predicates-publish-->|Replace|PropertiesExtractPredicates-subject
	    PropertiesExtractPredicates-subject-->|FullTable|properties_extract_filter-subscribe
	    properties_extract_filter-subscribe-->properties_extract_filter-processor
	    properties_extract_filter-processor-->properties_extract_filter-publish
	    properties_extract_filter-publish-->|Replace|PropertiesExtractFilter-subject
	    PropertiesExtractFilter-subject-->|FullTable|properties_extract_triples-subscribe
	    properties_extract_triples-subscribe-->properties_extract_triples-processor
	    properties_extract_triples-processor-->properties_extract_triples-publish
	    properties_extract_triples-publish-->|Replace|PropertiesExtractTriples-subject
	    PropertiesExtractTriples-subject-->|FullTable|properties_extract_pivot-subscribe
	    properties_extract_pivot-subscribe-->properties_extract_pivot-processor
	    properties_extract_pivot-processor-->properties_extract_pivot-publish
	    properties_extract_pivot-publish-->|Replace|PropertiesExtractPivot-subject
	    PropertiesExtractPivot-subject-->|FullTable|properties_extract_select-subscribe
	    properties_extract_select-subscribe-->properties_extract_select-processor
	    properties_extract_select-processor-->properties_extract_select-publish
	    properties_extract_select-publish-->|Replace|PropertiesExtractSelect-subject
	end
	properties_extraction-rt@{shape: subproc, label: properties_extraction}
	properties_extraction-rt-->properties_extraction
	properties_extract-processor@{shape: rect, label: ExtractXML}
	properties_extract-publish@{shape: fork}
	properties_extract-subscribe@{shape: diamond, label: All}
	PropertiesExtract-subject@{shape: doc, label: PropertiesExtract}
	properties_extract_coalesce-processor@{shape: rect, label: CoalesceProcessor}
	properties_extract_coalesce-publish@{shape: fork}
	properties_extract_coalesce-subscribe@{shape: diamond, label: All}
	PropertiesExtractCoalesce-subject@{shape: doc, label: PropertiesExtractCoalesce}
	properties_extract_predicates-processor@{shape: rect, label: Select}
	properties_extract_predicates-publish@{shape: fork}
	properties_extract_predicates-subscribe@{shape: diamond, label: All}
	PropertiesExtractPredicates-subject@{shape: doc, label: PropertiesExtractPredicates}
	properties_extract_filter-processor@{shape: rect, label: Filter}
	properties_extract_filter-publish@{shape: fork}
	properties_extract_filter-subscribe@{shape: diamond, label: All}
	PropertiesExtractFilter-subject@{shape: doc, label: PropertiesExtractFilter}
	properties_extract_triples-processor@{shape: rect, label: Select}
	properties_extract_triples-publish@{shape: fork}
	properties_extract_triples-subscribe@{shape: diamond, label: All}
	PropertiesExtractTriples-subject@{shape: doc, label: PropertiesExtractTriples}
	properties_extract_pivot-processor@{shape: rect, label: Pivot}
	properties_extract_pivot-publish@{shape: fork}
	properties_extract_pivot-subscribe@{shape: diamond, label: All}
	PropertiesExtractPivot-subject@{shape: doc, label: PropertiesExtractPivot}
	properties_extract_select-processor@{shape: rect, label: Select}
	properties_extract_select-publish@{shape: fork}
	properties_extract_select-subscribe@{shape: diamond, label: All}
	PropertiesExtractSelect-subject@{shape: doc, label: PropertiesExtractSelect}
	%% -------------------------------------
	
	%% -------------------------------------
	%% properties post-pivot cleanup
	%% -------------------------------------
	subgraph properties_pivot
	    PropertiesExtractSelect-subject-.->|FullTable|properties_pivot_coalesce-subscribe
	    properties_pivot_coalesce-subscribe-->properties_pivot_coalesce-processor
	    properties_pivot_coalesce-processor-->properties_pivot_coalesce-publish
	    properties_pivot_coalesce-publish-->|Replace|PropertiesPivotCoalesce-subject
	    PropertiesPivotCoalesce-subject-->|FullTable|properties_pivot_group_by-subscribe
	    properties_pivot_group_by-subscribe-->properties_pivot_group_by-processor
	    properties_pivot_group_by-processor-->properties_pivot_group_by-publish
	    properties_pivot_group_by-publish-->|Replace|PropertiesPivotGroupBy-subject
	    PropertiesPivotGroupBy-subject-->|FullTable|properties_pivot_select-subscribe
	    properties_pivot_select-subscribe-->properties_pivot_select-processor
	    properties_pivot_select-processor-->properties_pivot_select-publish
	    properties_pivot_select-publish-->|Extend|PROPERTIES-subject
	end
	properties_pivot-rt@{shape: subproc, label: properties_pivot}
	properties_pivot-rt-->properties_pivot
	properties_pivot_coalesce-processor@{shape: rect, label: CoalesceProcessor}
	properties_pivot_coalesce-publish@{shape: fork}
	properties_pivot_coalesce-subscribe@{shape: diamond, label: All}
	PropertiesPivotCoalesce-subject@{shape: doc, label: PropertiesPivotCoalesce}
	properties_pivot_group_by-processor@{shape: rect, label: GroupBy}
	properties_pivot_group_by-publish@{shape: fork}
	properties_pivot_group_by-subscribe@{shape: diamond, label: All}
	PropertiesPivotGroupBy-subject@{shape: doc, label: PropertiesPivotGroupBy}
	properties_pivot_select-processor@{shape: rect, label: Select}
	properties_pivot_select-publish@{shape: fork}
	properties_pivot_select-subscribe@{shape: diamond, label: All}
	PROPERTIES-subject@{shape: doc, label: PROPERTIES}
	%% -------------------------------------
	
	%% -------------------------------------
	%% classes embeddings
	%% -------------------------------------
	subgraph classes_doc_chunking
	    CLASSES-subject-.->|FullTable|classes_doc_transform-subscribe
	    classes_doc_transform-subscribe-->classes_doc_transform-processor
	    classes_doc_transform-processor-->classes_doc_transform-publish
	    classes_doc_transform-publish-->|Replace|ClassesDocTransform-subject
	    ClassesDocTransform-subject-->|FullTable|classes_doc_select-subscribe
	    classes_doc_select-subscribe-->classes_doc_select-processor
	    classes_doc_select-processor-->classes_doc_select-publish
	    classes_doc_select-publish-->|Replace|ClassesDocSelect-subject
	    ClassesDocSelect-subject-->|FullTable|classes_doc_chunk-subscribe
	    classes_doc_chunk-subscribe-->classes_doc_chunk-processor
	    classes_doc_chunk-processor-->classes_doc_chunk-publish
	    classes_doc_chunk-publish-->|Extend|ClassesDocChunks-subject
	end
	classes_doc_chunking-rt@{shape: subproc, label: classes_doc_chunking}
	classes_doc_chunking-rt-->classes_doc_chunking
	classes_doc_transform-processor@{shape: rect, label: Select}
	classes_doc_transform-publish@{shape: fork}
	classes_doc_transform-subscribe@{shape: diamond, label: All}
	ClassesDocTransform-subject@{shape: doc, label: ClassesDocTransform}
	classes_doc_select-processor@{shape: rect, label: Select}
	classes_doc_select-publish@{shape: fork}
	classes_doc_select-subscribe@{shape: diamond, label: All}
	ClassesDocSelect-subject@{shape: doc, label: ClassesDocSelect}
	classes_doc_chunk-processor@{shape: rect, label: ChunkDocuments}
	classes_doc_chunk-publish@{shape: fork}
	classes_doc_chunk-subscribe@{shape: diamond, label: All}
	ClassesDocChunks-subject@{shape: doc, label: ClassesDocChunks}
	subgraph classes_embeddings
	    ClassesDocChunks-subject-.->|FullTable|classes_doc_coalesce-subscribe
	    classes_doc_coalesce-subscribe-->classes_doc_coalesce-processor
	    classes_doc_coalesce-processor-->classes_doc_coalesce-publish
	    classes_doc_coalesce-publish-->|Extend|ClassesDocCoalesce-subject
	    ClassesDocCoalesce-subject-->|FullTable|classes_doc_embed-subscribe
	    classes_doc_embed-subscribe-->classes_doc_embed-processor
	    classes_doc_embed-processor-->classes_doc_embed-publish
	    classes_doc_embed-publish-->|Extend|ClassesDocEmbeddings-subject
	end
	classes_embeddings-rt@{shape: subproc, label: classes_embeddings}
	classes_embeddings-rt-->classes_embeddings
	classes_doc_coalesce-processor@{shape: rect, label: CoalesceProcessor}
	classes_doc_coalesce-publish@{shape: fork}
	classes_doc_coalesce-subscribe@{shape: diamond, label: All}
	ClassesDocCoalesce-subject@{shape: doc, label: ClassesDocCoalesce}
	classes_doc_embed-processor@{shape: rect, label: CandleEmbedProcessor}
	classes_doc_embed-publish@{shape: fork}
	classes_doc_embed-subscribe@{shape: diamond, label: All}
	ClassesDocEmbeddings-subject@{shape: doc, label: ClassesDocEmbeddings}
	%% -------------------------------------
	
	%% -------------------------------------
	%% properties embeddings
	%% -------------------------------------
	subgraph properties_doc_chunking
	    PROPERTIES-subject-.->|FullTable|properties_doc_transform-subscribe
	    properties_doc_transform-subscribe-->properties_doc_transform-processor
	    properties_doc_transform-processor-->properties_doc_transform-publish
	    properties_doc_transform-publish-->|Replace|PropertiesDocTransform-subject
	    PropertiesDocTransform-subject-->|FullTable|properties_doc_select-subscribe
	    properties_doc_select-subscribe-->properties_doc_select-processor
	    properties_doc_select-processor-->properties_doc_select-publish
	    properties_doc_select-publish-->|Replace|PropertiesDocSelect-subject
	    PropertiesDocSelect-subject-->|FullTable|properties_doc_chunk-subscribe
	    properties_doc_chunk-subscribe-->properties_doc_chunk-processor
	    properties_doc_chunk-processor-->properties_doc_chunk-publish
	    properties_doc_chunk-publish-->|Extend|PropertiesDocChunks-subject
	end
	properties_doc_chunking-rt@{shape: subproc, label: properties_doc_chunking}
	properties_doc_chunking-rt-->properties_doc_chunking
	properties_doc_transform-processor@{shape: rect, label: Select}
	properties_doc_transform-publish@{shape: fork}
	properties_doc_transform-subscribe@{shape: diamond, label: All}
	PropertiesDocTransform-subject@{shape: doc, label: PropertiesDocTransform}
	properties_doc_select-processor@{shape: rect, label: Select}
	properties_doc_select-publish@{shape: fork}
	properties_doc_select-subscribe@{shape: diamond, label: All}
	PropertiesDocSelect-subject@{shape: doc, label: PropertiesDocSelect}
	properties_doc_chunk-processor@{shape: rect, label: ChunkDocuments}
	properties_doc_chunk-publish@{shape: fork}
	properties_doc_chunk-subscribe@{shape: diamond, label: All}
	PropertiesDocChunks-subject@{shape: doc, label: PropertiesDocChunks}
	subgraph properties_embeddings
	    PropertiesDocChunks-subject-.->|FullTable|properties_doc_coalesce-subscribe
	    properties_doc_coalesce-subscribe-->properties_doc_coalesce-processor
	    properties_doc_coalesce-processor-->properties_doc_coalesce-publish
	    properties_doc_coalesce-publish-->|Extend|PropertiesDocCoalesce-subject
	    PropertiesDocCoalesce-subject-->|FullTable|properties_doc_embed-subscribe
	    properties_doc_embed-subscribe-->properties_doc_embed-processor
	    properties_doc_embed-processor-->properties_doc_embed-publish
	    properties_doc_embed-publish-->|Extend|PropertiesDocEmbeddings-subject
	end
	properties_embeddings-rt@{shape: subproc, label: properties_embeddings}
	properties_embeddings-rt-->properties_embeddings
	properties_doc_coalesce-processor@{shape: rect, label: CoalesceProcessor}
	properties_doc_coalesce-publish@{shape: fork}
	properties_doc_coalesce-subscribe@{shape: diamond, label: All}
	PropertiesDocCoalesce-subject@{shape: doc, label: PropertiesDocCoalesce}
	properties_doc_embed-processor@{shape: rect, label: CandleEmbedProcessor}
	properties_doc_embed-publish@{shape: fork}
	properties_doc_embed-subscribe@{shape: diamond, label: All}
	PropertiesDocEmbeddings-subject@{shape: doc, label: PropertiesDocEmbeddings}
	%% -------------------------------------
	
	%% -------------------------------------
	%% Query embeddings
	%% -------------------------------------
	subgraph query_embeddings
	    UserQueries-subject-.->|LastRecordBatch|query_embed-subscribe
	    query_embed-subscribe-->query_embed-processor
	    query_embed-processor-->query_embed-publish
	    query_embed-publish-->|Replace|QueryEmbeddings-subject
	end
	query_embeddings-rt@{shape: subproc, label: query_embeddings}
	query_embeddings-rt-->query_embeddings
	UserQueries-subject@{shape: doc, label: UserQueries}
	query_embed-processor@{shape: rect, label: CandleEmbedProcessor}
	query_embed-publish@{shape: fork}
	query_embed-subscribe@{shape: diamond, label: All}
	QueryEmbeddings-subject@{shape: doc, label: QueryEmbeddings}
	%% -------------------------------------
	
	%% ------------------------
	%% Classes vector search
	%% ------------------------
	subgraph classes_vs
	    ClassesDocEmbeddings-subject-->|FullTable|classes_vector_distance-subscribe
	    QueryEmbeddings-subject-.->|FullTable|classes_vector_distance-subscribe
	    classes_vector_distance-subscribe-->classes_vector_distance-processor
	    classes_vector_distance-processor-->classes_vector_distance-publish
	    classes_vector_distance-publish-->|Replace|ClassesVectorDistance-subject
	    ClassesVectorDistance-subject-->|FullTable|classes_select_scores-subscribe
	    classes_select_scores-subscribe-->classes_select_scores-processor
	    classes_select_scores-processor-->classes_select_scores-publish
	    classes_select_scores-publish-->|Replace|ClassesSelectedScores-subject
	    ClassesSelectedScores-subject-->|FullTable|classes_filter_scores-subscribe
	    classes_filter_scores-subscribe-->classes_filter_scores-processor
	    classes_filter_scores-processor-->classes_filter_scores-publish
	    classes_filter_scores-publish-->|Replace|ClassesFilteredScores-subject
	    ClassesFilteredScores-subject-->|FullTable|classes_sort_scores-subscribe
	    classes_sort_scores-subscribe-->classes_sort_scores-processor
	    classes_sort_scores-processor-->classes_sort_scores-publish
	    classes_sort_scores-publish-->|Replace|ClassesSortedScores-subject
	    ClassesSortedScores-subject-->|FullTable|classes_limit_scores-subscribe
	    classes_limit_scores-subscribe-->classes_limit_scores-processor
	    classes_limit_scores-processor-->classes_limit_scores-publish
	    classes_limit_scores-publish-->|Replace|ClassesLimitedScores-subject
	    ClassesDocChunks-subject-->|FullTable|classes_chunks_join-subscribe
	    ClassesLimitedScores-subject-->|FullTable|classes_chunks_join-subscribe
	    classes_chunks_join-subscribe-->classes_chunks_join-processor
	    classes_chunks_join-processor-->classes_chunks_join-publish
	    classes_chunks_join-publish-->|Replace|ClassesChunksJoin-subject
	    ClassesChunksJoin-subject-->|FullTable|classes_chunks_select-subscribe
	    classes_chunks_select-subscribe-->classes_chunks_select-processor
	    classes_chunks_select-processor-->classes_chunks_select-publish
	    classes_chunks_select-publish-->|Extend|ClassesChunksSelect-subject		
	    ClassesChunksSelect-subject-->|FullTable|classes_chunks_summary-subscribe
	    classes_chunks_summary-subscribe-->classes_chunks_summary-processor
	    classes_chunks_summary-processor-->classes_chunks_summary-publish
	    classes_chunks_summary-publish-->|Extend|ToolMessages-subject
	end
	classes_vs-rt@{shape: subproc, label: classes_vs}
	classes_vs-rt-->classes_vs
	classes_vector_distance-processor@{shape: rect, label: VectorDistance}
	classes_vector_distance-subscribe@{shape: diamond, label: All}
	classes_vector_distance-publish@{shape: fork}
	ClassesVectorDistance-subject@{shape: doc, label: ClassesVectorDistance}
	classes_select_scores-processor@{shape: rect, label: Select}
	classes_select_scores-subscribe@{shape: diamond, label: All}
	classes_select_scores-publish@{shape: fork}
	ClassesSelectedScores-subject@{shape: doc, label: ClassesSelectedScores}
	classes_filter_scores-processor@{shape: rect, label: Filter}
	classes_filter_scores-subscribe@{shape: diamond, label: All}
	classes_filter_scores-publish@{shape: fork}
	ClassesFilteredScores-subject@{shape: doc, label: ClassesFilteredScores}
	classes_sort_scores-processor@{shape: rect, label: Sort}
	classes_sort_scores-subscribe@{shape: diamond, label: All}
	classes_sort_scores-publish@{shape: fork}
	ClassesSortedScores-subject@{shape: doc, label: ClassesSortedScores}
	classes_limit_scores-processor@{shape: rect, label: LimitProcessor}
	classes_limit_scores-subscribe@{shape: diamond, label: All}
	classes_limit_scores-publish@{shape: fork}
	ClassesLimitedScores-subject@{shape: doc, label: ClassesLimitedScores}
	classes_chunks_join-processor@{shape: rect, label: Join}
	classes_chunks_join-subscribe@{shape: diamond, label: All}
	classes_chunks_join-publish@{shape: fork}
	ClassesChunksJoin-subject@{shape: doc, label: ClassesChunksJoin}
	classes_chunks_select-processor@{shape: rect, label: Select}
	classes_chunks_select-subscribe@{shape: diamond, label: All}
	classes_chunks_select-publish@{shape: fork}
	ClassesChunksSelect-subject@{shape: doc, label: ClassesChunksSelect}	
	classes_chunks_summary-processor@{shape: rect, label: DataSummaryProcessor}
	classes_chunks_summary-subscribe@{shape: diamond, label: All}
	classes_chunks_summary-publish@{shape: fork}
	%% ------------------------
	
	%% ------------------------
	%% Properties vector search
	%% ------------------------
	subgraph properties_vs
	    PropertiesDocEmbeddings-subject-->|FullTable|properties_vector_distance-subscribe
	    QueryEmbeddings-subject-.->|FullTable|properties_vector_distance-subscribe
	    properties_vector_distance-subscribe-->properties_vector_distance-processor
	    properties_vector_distance-processor-->properties_vector_distance-publish
	    properties_vector_distance-publish-->|Replace|PropertiesVectorDistance-subject
	    PropertiesVectorDistance-subject-->|FullTable|properties_select_scores-subscribe
	    properties_select_scores-subscribe-->properties_select_scores-processor
	    properties_select_scores-processor-->properties_select_scores-publish
	    properties_select_scores-publish-->|Replace|PropertiesSelectedScores-subject
	    PropertiesSelectedScores-subject-->|FullTable|properties_filter_scores-subscribe
	    properties_filter_scores-subscribe-->properties_filter_scores-processor
	    properties_filter_scores-processor-->properties_filter_scores-publish
	    properties_filter_scores-publish-->|Replace|PropertiesFilteredScores-subject
	    PropertiesFilteredScores-subject-->|FullTable|properties_sort_scores-subscribe
	    properties_sort_scores-subscribe-->properties_sort_scores-processor
	    properties_sort_scores-processor-->properties_sort_scores-publish
	    properties_sort_scores-publish-->|Replace|PropertiesSortedScores-subject
	    PropertiesSortedScores-subject-->|FullTable|properties_limit_scores-subscribe
	    properties_limit_scores-subscribe-->properties_limit_scores-processor
	    properties_limit_scores-processor-->properties_limit_scores-publish
	    properties_limit_scores-publish-->|Replace|PropertiesLimitedScores-subject
	    PropertiesDocChunks-subject-->|FullTable|properties_chunks_join-subscribe
	    PropertiesLimitedScores-subject-->|FullTable|properties_chunks_join-subscribe
	    properties_chunks_join-subscribe-->properties_chunks_join-processor
	    properties_chunks_join-processor-->properties_chunks_join-publish
	    properties_chunks_join-publish-->|Replace|PropertiesChunksJoin-subject
	    PropertiesChunksJoin-subject-->|FullTable|properties_chunks_select-subscribe
	    properties_chunks_select-subscribe-->properties_chunks_select-processor
	    properties_chunks_select-processor-->properties_chunks_select-publish
	    properties_chunks_select-publish-->|Extend|PropertiesChunksSelect-subject		
	    PropertiesChunksSelect-subject-->|FullTable|properties_chunks_summary-subscribe
	    properties_chunks_summary-subscribe-->properties_chunks_summary-processor
	    properties_chunks_summary-processor-->properties_chunks_summary-publish
	    properties_chunks_summary-publish-->|Extend|ToolMessages-subject
	end
	properties_vs-rt@{shape: subproc, label: properties_vs}
	properties_vs-rt-->properties_vs
	properties_vector_distance-processor@{shape: rect, label: VectorDistance}
	properties_vector_distance-subscribe@{shape: diamond, label: All}
	properties_vector_distance-publish@{shape: fork}
	PropertiesVectorDistance-subject@{shape: doc, label: PropertiesVectorDistance}
	properties_select_scores-processor@{shape: rect, label: Select}
	properties_select_scores-subscribe@{shape: diamond, label: All}
	properties_select_scores-publish@{shape: fork}
	PropertiesSelectedScores-subject@{shape: doc, label: PropertiesSelectedScores}
	properties_filter_scores-processor@{shape: rect, label: Filter}
	properties_filter_scores-subscribe@{shape: diamond, label: All}
	properties_filter_scores-publish@{shape: fork}
	PropertiesFilteredScores-subject@{shape: doc, label: PropertiesFilteredScores}
	properties_sort_scores-processor@{shape: rect, label: Sort}
	properties_sort_scores-subscribe@{shape: diamond, label: All}
	properties_sort_scores-publish@{shape: fork}
	PropertiesSortedScores-subject@{shape: doc, label: PropertiesSortedScores}
	properties_limit_scores-processor@{shape: rect, label: LimitProcessor}
	properties_limit_scores-subscribe@{shape: diamond, label: All}
	properties_limit_scores-publish@{shape: fork}
	PropertiesLimitedScores-subject@{shape: doc, label: PropertiesLimitedScores}
	properties_chunks_join-processor@{shape: rect, label: Join}
	properties_chunks_join-subscribe@{shape: diamond, label: All}
	properties_chunks_join-publish@{shape: fork}
	PropertiesChunksJoin-subject@{shape: doc, label: PropertiesChunksJoin}
	properties_chunks_select-processor@{shape: rect, label: Select}
	properties_chunks_select-subscribe@{shape: diamond, label: All}
	properties_chunks_select-publish@{shape: fork}
	PropertiesChunksSelect-subject@{shape: doc, label: PropertiesChunksSelect}	
	properties_chunks_summary-processor@{shape: rect, label: DataSummaryProcessor}
	properties_chunks_summary-subscribe@{shape: diamond, label: All}
	properties_chunks_summary-publish@{shape: fork}
	ToolMessages-subject@{shape: doc, label: ToolMessages}
	%% ------------------------"#
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
	ONTOLOGIES["ONTOLOGIES"] {
	    Utf8 name "OBO Relations Ontology"
	    Utf8 description "OBO Relations Ontology"
	    Utf8 uri "RO"
	    Utf8 url "http://purl.obolibrary.org/obo/ro.owl"
	    Utf8 version "2025-06-24"
	}
	CLASSES["CLASSES"] {
	    Utf8 ontology "ChEBI"
	    Utf8 name "statin"
	    Utf8 uri "CHEBI_87631"
	    Utf8 url "http://purl.obolibrary.org/obo/CHEBI_87631"
	    Utf8 definition "statin"
	    Utf8 synonyms "statin"
	}
	PROPERTIES["PROPERTIES"] {
	    Utf8 ontology "RO"
	    Utf8 name "acts upstream of or within"
	    Utf8 uri "RO_0002264"
	    Utf8 url "http://purl.obolibrary.org/obo/RO_0002264"
	    Utf8 definition "RO"
	    Utf8 synonyms "RO"
	    Utf8 subproperty_of "RO"
	    Utf8 inverse_of "RO"
	    Utf8 domain "RO"
	    Utf8 range "RO"
	}
	ToolMessages["ToolMessages"] {
	    Utf8 role
	    Utf8 content
	    Int64 timestamp
	}
	ontologies_extract["ontologies_extract"] {
	    Boolean cpu "false"
	    Utf8 format "OwlOntology"
	    Utf8 lhs_name "UserScript"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractXML"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ontologies_extract_predicates["ontologies_extract_predicates"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None','None','None']"
	    List-Utf8 column_operators "['None','None','None','Value','Value','Value']"
	    List-Utf8 cast_templates "['','','','dc:title','dc:description','owl:versionInfo']"
	    List-Utf8 lhs_values "['subject','predicate','object','dc:title','dc:description','owl:versionInfo']"
	    Boolean cpu "false"
	    Utf8 lhs_name "OntologiesExtract"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ontologies_extract_filter["ontologies_extract_filter"] {
	    List-Utf8 lhs_values "['predicate','predicate','predicate']"
	    List-Utf8 cmp_columns "['dc:title','dc:description','owl:versionInfo']"
	    List-Utf8 cmp_operators "['Like','Like','Like']"
	    Utf8 cmp_predicate "Any"
	    Boolean cpu "false"
	    Utf8 lhs_name "OntologiesExtractPredicates"
	    Utf8 operator "Filter"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ontologies_extract_triples["ontologies_extract_triples"] {
	    List-Utf8 lhs_values "['subject','predicate','object']"
	    Boolean cpu "false"
	    Utf8 lhs_name "OntologiesExtractFilter"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ontologies_extract_pivot["ontologies_extract_pivot"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['ConcatSemicolonSeperator']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "OntologiesExtractTriples"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ontologies_extract_select["ontologies_extract_select"] {
	    List-Utf8 as_columns "['name','description','uri','url','version']"
	    Boolean cpu "false"
	    Utf8 lhs_name "OntologiesExtractPivot"
	    List-Utf8 lhs_values "['dc:title-object-ConcatSemicolonSeperator','dc:description-object-ConcatSemicolonSeperator','subject','subject','owl:versionInfo-object-ConcatSemicolonSeperator']"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_extract["classes_extract"] {
	    Boolean cpu "false"
	    Utf8 format "OwlClass"
	    Utf8 lhs_name "UserScript"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractXML"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_extract_coalesce["classes_extract_coalesce"] {
	    Int64 fetch "512"
	    Utf8 summary_format "None"
	}
	classes_extract_predicates["classes_extract_predicates"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None','None','None']"
	    List-Utf8 column_operators "['None','None','None','Value','Value','Value']"
	    List-Utf8 cast_templates "['','','','rdfs:label','obo:IAO_0000115','oboInOwl:hasExactSynonym']"
	    List-Utf8 lhs_values "['subject','predicate','object','rdfs:label','obo:IAO_0000115','oboInOwl:hasExactSynonym']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesExtractCoalesce"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_extract_filter["classes_extract_filter"] {
	    List-Utf8 lhs_values "['predicate','predicate','predicate']"
	    List-Utf8 cmp_columns "['rdfs:label','obo:IAO_0000115','oboInOwl:hasExactSynonym']"
	    List-Utf8 cmp_operators "['Like','Like','Like']"
	    Utf8 cmp_predicate "Any"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesExtractPredicates"
	    Utf8 operator "Filter"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_extract_triples["classes_extract_triples"] {
	    List-Utf8 lhs_values "['subject','predicate','object']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesExtractFilter"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ClassesExtractTriples["ClassesExtractTriples"] {
	    Utf8 subject
	    Utf8 predicate
	    Utf8 object
	}
	classes_extract_pivot["classes_extract_pivot"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['ConcatSemicolonSeperator']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "ClassesExtractTriples"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_extract_select["classes_extract_select"] {
	    List-Utf8 as_columns "['subject','rdfs-label','obo-IAO_0000115','oboInOwl-hasExactSynonym']"
	    List-Utf8 lhs_values "['subject','rdfs:label-object-ConcatSemicolonSeperator','obo:IAO_0000115-object-ConcatSemicolonSeperator','oboInOwl:hasExactSynonym-object-ConcatSemicolonSeperator']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesExtractPivot"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ClassesExtractSelect["ClassesExtractSelect"] {
	    Utf8 subject
	    Utf8 rdfs-label
	    Utf8 obo-IAO_0000115
	    Utf8 oboInOwl-hasExactSynonym
	}
	classes_pivot_coalesce["classes_pivot_coalesce"] {
	    Int64 fetch "512"
	    Utf8 summary_format "None"
	}
	classes_pivot_group_by["classes_pivot_group_by"] {
	    List-Utf8 agg_columns "['rdfs-label','obo-IAO_0000115','oboInOwl-hasExactSynonym']"
	    List-Utf8 agg_operators "['Concat','Concat','Concat']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesPivotCoalesce"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "GroupBy"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_pivot_select["classes_pivot_select"] {
	    List-Utf8 as_columns "['ontology','name','uri','url','definition','synonyms']"
	    List-Utf8 lhs_values "['subject','rdfs-label-Concat','subject','subject','obo-IAO_0000115-Concat','oboInOwl-hasExactSynonym-Concat']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesPivotGroupBy"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_extract["properties_extract"] {
	    Boolean cpu "false"
	    Utf8 format "OwlObjectProperty"
	    Utf8 lhs_name "UserScript"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractXML"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_extract_coalesce["properties_extract_coalesce"] {
	    Int64 fetch "512"
	    Utf8 summary_format "None"
	}
	properties_extract_predicates["properties_extract_predicates"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None','None','None','None','None','None','None']"
	    List-Utf8 column_operators "['None','None','None','Value','Value','Value','Value','Value','Value','Value']"
	    List-Utf8 cast_templates "['','','','rdfs:label','obo:IAO_0000115','oboInOwl:hasExactSynonym','rdfs:subPropertyOf','owl:inverseOf','rdfs:domain','rdfs:range']"
	    List-Utf8 lhs_values "['subject','predicate','object','rdfs:label','obo:IAO_0000115','oboInOwl:hasExactSynonym','rdfs:subPropertyOf','owl:inverseOf','rdfs:domain','rdfs:range']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesExtractCoalesce"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_extract_filter["properties_extract_filter"] {
	    List-Utf8 lhs_values "['predicate','predicate','predicate','predicate','predicate','predicate','predicate']"
	    List-Utf8 cmp_columns "['rdfs:label','obo:IAO_0000115','oboInOwl:hasExactSynonym','rdfs:subPropertyOf','owl:inverseOf','rdfs:domain','rdfs:range']"
	    List-Utf8 cmp_operators "['Like','Like','Like','Like','Like','Like','Like']"
	    Utf8 cmp_predicate "Any"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesExtractPredicates"
	    Utf8 operator "Filter"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_extract_triples["properties_extract_triples"] {
	    List-Utf8 lhs_values "['subject','predicate','object']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesExtractFilter"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	PropertiesExtractTriples["PropertiesExtractTriples"] {
	    Utf8 subject
	    Utf8 predicate
	    Utf8 object
	}
	properties_extract_pivot["properties_extract_pivot"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['ConcatSemicolonSeperator']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "PropertiesExtractTriples"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_extract_select["properties_extract_select"] {
	    List-Utf8 as_columns "['subject','rdfs-label','obo-IAO_0000115','oboInOwl-hasExactSynonym','rdfs-subPropertyOf','owl-inverseOf','rdfs-domain','rdfs-range']"
	    List-Utf8 lhs_values "['subject','rdfs:label-object-ConcatSemicolonSeperator','obo:IAO_0000115-object-ConcatSemicolonSeperator','oboInOwl:hasExactSynonym-object-ConcatSemicolonSeperator','rdfs:subPropertyOf-object-ConcatSemicolonSeperator','owl:inverseOf-object-ConcatSemicolonSeperator','rdfs:domain-object-ConcatSemicolonSeperator','rdfs:range-object-ConcatSemicolonSeperator']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesExtractPivot"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	PropertiesExtractSelect["PropertiesExtractSelect"] {
	    Utf8 subject
	    Utf8 rdfs-label
	    Utf8 obo-IAO_0000115
	    Utf8 oboInOwl-hasExactSynonym
	    Utf8 rdfs-subPropertyOf
	    Utf8 owl-inverseOf
	    Utf8 rdfs-domain
	    Utf8 rdfs-range
	}
	properties_pivot_coalesce["properties_pivot_coalesce"] {
	    Int64 fetch "512"
	    Utf8 summary_format "None"
	}
	properties_pivot_group_by["properties_pivot_group_by"] {
	    List-Utf8 agg_columns "['rdfs-label','obo-IAO_0000115','oboInOwl-hasExactSynonym','rdfs-subPropertyOf','owl-inverseOf','rdfs-domain','rdfs-range']"
	    List-Utf8 agg_operators "['Concat','Concat','Concat','Concat','Concat','Concat','Concat']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesPivotCoalesce"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "GroupBy"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_pivot_select["properties_pivot_select"] {
	    List-Utf8 as_columns "['ontology','name','uri','url','definition','synonyms','subproperty_of','inverse_of','domain','range']"
	    List-Utf8 lhs_values "['subject','rdfs-label-Concat','subject','subject','obo-IAO_0000115-Concat','oboInOwl-hasExactSynonym-Concat','rdfs-subPropertyOf-Concat','owl-inverseOf-Concat','rdfs-domain-Concat','rdfs-range-Concat']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesPivotGroupBy"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_doc_transform["classes_doc_transform"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None']"
	    List-Utf8 cast_templates "['','{{name}}; ','{{synonyms}}; ']"
	    List-Utf8 column_operators "['None','Concat','Concat']"
	    Boolean cpu "false"
	    Utf8 lhs_name "CLASSES"
	    List-Utf8 lhs_values "['uri','name','synonyms']"
	    List-Utf8 rhs_values "['','synonyms','definition']"
	    List-Utf8 as_columns "['uri','synonyms','definition']"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_doc_select["classes_doc_select"] {
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesDocTransform"
	    List-Utf8 lhs_values "['uri','definition']"
	    List-Utf8 as_columns "['document_id','text']"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	classes_doc_chunk["classes_doc_chunk"] {
	    Boolean cpu "false"
	    Utf8 lhs_fk "document_id"
	    Utf8 lhs_name "ClassesDocSelect"
	    Utf8 lhs_pk "document_id"
	    List-Utf8 lhs_values "['text']"
	    Utf8 operator "ChunkDocuments"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	ClassesDocChunks["ClassesDocChunks"] {
	    Utf8 chunk_id
	    Utf8 document_id
	    Utf8 text
	}
	classes_doc_coalesce["classes_doc_coalesce"] {
	    Int64 fetch "1"
	    Utf8 summary_format "None"
	}
	classes_doc_embed["classes_doc_embed"] {
	    Utf8 documents "ClassesDocCoalesce"
	    Utf8 candle_asset "QuantizedBertEmbed"
	    Utf8 weights_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf"
	    Utf8 tokenizer_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json"
	    Utf8 tokenizer_config_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json"
	    Utf8 weights_config_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json"
	    Boolean cpu "false"
	    Utf8 encoding_format "float"
	    Utf8 input_type "passage"
	    Utf8 modality "text"
	}
	ClassesDocEmbeddings["ClassesDocEmbeddings"] {
	    Utf8 chunk_id
	    Utf8 document_id
	    List-Float32 embedding
	}
	properties_doc_transform["properties_doc_transform"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None']"
	    List-Utf8 cast_templates "['','{{name}}; ','{{synonyms}}; ']"
	    List-Utf8 column_operators "['None','Concat','Concat']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PROPERTIES"
	    List-Utf8 lhs_values "['uri','name','synonyms']"
	    List-Utf8 rhs_values "['','synonyms','definition']"
	    List-Utf8 as_columns "['uri','synonyms','definition']"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_doc_select["properties_doc_select"] {
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesDocTransform"
	    List-Utf8 lhs_values "['uri','definition']"
	    List-Utf8 as_columns "['document_id','text']"
	    Utf8 operator "Select"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	properties_doc_chunk["properties_doc_chunk"] {
	    Boolean cpu "false"
	    Utf8 lhs_fk "document_id"
	    Utf8 lhs_name "PropertiesDocSelect"
	    Utf8 lhs_pk "document_id"
	    List-Utf8 lhs_values "['text']"
	    Utf8 operator "ChunkDocuments"
	    Utf8 stream "StreamLHSStreamRHS"
	}
	PropertiesDocChunks["PropertiesDocChunks"] {
	    Utf8 chunk_id
	    Utf8 document_id
	    Utf8 text
	}
	properties_doc_coalesce["properties_doc_coalesce"] {
	    Int64 fetch "1"
	    Utf8 summary_format "None"
	}
	properties_doc_embed["properties_doc_embed"] {
	    Utf8 documents "PropertiesDocCoalesce"
	    Utf8 candle_asset "QuantizedBertEmbed"
	    Utf8 weights_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf"
	    Utf8 tokenizer_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json"
	    Utf8 tokenizer_config_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json"
	    Utf8 weights_config_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json"
	    Boolean cpu "false"
	    Utf8 encoding_format "float"
	    Utf8 input_type "passage"
	    Utf8 modality "text"
	}
	PropertiesDocEmbeddings["PropertiesDocEmbeddings"] {
	    Utf8 chunk_id
	    Utf8 document_id
	    List-Float32 embedding
	}
	UserQueries["UserQueries"] {
	    Utf8 query_id
	    Utf8 text
	}
	QueryEmbeddings["QueryEmbeddings"] {
	    Utf8 query_id
	    List-Float32 embedding
	}
	query_embed["query_embed"] {
	    Utf8 documents "UserQueries"
	    Utf8 candle_asset "QuantizedBertEmbed"
	    Utf8 weights_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf"
	    Utf8 tokenizer_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json"
	    Utf8 tokenizer_config_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json"
	    Utf8 weights_config_file "/home/dmccloskey/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json"
	    Boolean cpu "false"
	    Utf8 encoding_format "float"
	    Utf8 input_type "query"
	    Utf8 modality "text"
	}
	classes_vector_distance["classes_vector_distance"] {
	    Boolean cpu "false"
	    Utf8 dist_operator "NormalizedDotProduct"
	    Utf8 lhs_fk "query_id"
	    Utf8 lhs_name "QueryEmbeddings"
	    Utf8 lhs_pk "query_id"
	    List-Utf8 lhs_values "['embedding']"
	    Utf8 operator "VectorDistance"
	    Utf8 rhs_fk "chunk_id"
	    Utf8 rhs_name "ClassesDocEmbeddings"
	    Utf8 rhs_pk "chunk_id"
	    List-Utf8 rhs_values "['embedding']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	classes_select_scores["classes_select_scores"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Float32','Float32']"
	    List-Utf8 cast_operators "['None','None','None','None']"
	    List-Utf8 cast_templates "['','','','0.5']"
	    List-Utf8 column_operators "['None','None','None','Value']"
	    List-Utf8 lhs_values "['query_id','chunk_id','score','threshold']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesVectorDistance"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	classes_filter_scores["classes_filter_scores"] {    
	    List-Utf8 cmp_columns "['threshold']"
	    List-Utf8 cmp_operators "['GreaterThan']"
	    Utf8 cmp_predicate "All"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesSelectedScores"
	    List-Utf8 lhs_values "['score']"
	    Utf8 operator "Filter"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	classes_sort_scores["classes_sort_scores"] {
	    Boolean cpu "false"
	    Utf8 lhs_fk "chunk_id"
	    Utf8 lhs_name "ClassesFilteredScores"
	    Utf8 lhs_pk "chunk_id"
	    List-Utf8 lhs_values "['score']"
	    Utf8 operator "Sort"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	classes_limit_scores["classes_limit_scores"] {
	    Int64 fetch "5"
	    Int64 skip "0"
	    Utf8 summary_format "None"
	}
	classes_chunks_join["classes_chunks_join"] {
	    Boolean cpu "false"
	    Utf8 lhs_fk "chunk_id"
	    Utf8 lhs_name "ClassesLimitedScores"
	    Utf8 lhs_pk "chunk_id"
	    List-Utf8 lhs_values "['score']"
	    Utf8 operator "Join"
	    Utf8 rhs_fk "chunk_id"
	    Utf8 rhs_name "ClassesDocChunks"
	    Utf8 rhs_pk "chunk_id"
	    List-Utf8 rhs_values "['text']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	classes_chunks_select["classes_chunks_select"] {
	    List-Utf8 lhs_values "['text']"
	    Boolean cpu "false"
	    Utf8 lhs_name "ClassesChunksJoin"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	classes_chunks_summary["classes_chunks_summary"] {
	    Utf8 summary_format "None"
	}
	properties_vector_distance["properties_vector_distance"] {
	    Boolean cpu "false"
	    Utf8 dist_operator "NormalizedDotProduct"
	    Utf8 lhs_fk "query_id"
	    Utf8 lhs_name "QueryEmbeddings"
	    Utf8 lhs_pk "query_id"
	    List-Utf8 lhs_values "['embedding']"
	    Utf8 operator "VectorDistance"
	    Utf8 rhs_fk "chunk_id"
	    Utf8 rhs_name "PropertiesDocEmbeddings"
	    Utf8 rhs_pk "chunk_id"
	    List-Utf8 rhs_values "['embedding']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	properties_select_scores["properties_select_scores"] {
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Float32','Float32']"
	    List-Utf8 cast_operators "['None','None','None','None']"
	    List-Utf8 cast_templates "['','','','0.5']"
	    List-Utf8 column_operators "['None','None','None','Value']"
	    List-Utf8 lhs_values "['query_id','chunk_id','score','threshold']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesVectorDistance"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	properties_filter_scores["properties_filter_scores"] {    
	    List-Utf8 cmp_columns "['threshold']"
	    List-Utf8 cmp_operators "['GreaterThan']"
	    Utf8 cmp_predicate "All"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesSelectedScores"
	    List-Utf8 lhs_values "['score']"
	    Utf8 operator "Filter"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	properties_sort_scores["properties_sort_scores"] {
	    Boolean cpu "false"
	    Utf8 lhs_fk "chunk_id"
	    Utf8 lhs_name "PropertiesFilteredScores"
	    Utf8 lhs_pk "chunk_id"
	    List-Utf8 lhs_values "['score']"
	    Utf8 operator "Sort"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	properties_limit_scores["properties_limit_scores"] {
	    Int64 fetch "5"
	    Int64 skip "0"
	    Utf8 summary_format "None"
	}
	properties_chunks_join["properties_chunks_join"] {
	    Boolean cpu "false"
	    Utf8 lhs_fk "chunk_id"
	    Utf8 lhs_name "PropertiesLimitedScores"
	    Utf8 lhs_pk "chunk_id"
	    List-Utf8 lhs_values "['score']"
	    Utf8 operator "Join"
	    Utf8 rhs_fk "chunk_id"
	    Utf8 rhs_name "PropertiesDocChunks"
	    Utf8 rhs_pk "chunk_id"
	    List-Utf8 rhs_values "['text']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	properties_chunks_select["properties_chunks_select"] {
	    List-Utf8 lhs_values "['text']"
	    Boolean cpu "false"
	    Utf8 lhs_name "PropertiesChunksJoin"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	properties_chunks_summary["properties_chunks_summary"] {
	    Utf8 summary_format "None"
	}"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        AvailableSubjects, AvailableSubjectsTrait, BlobBuilderTraitExt, BuildableTrait, BuilderTrait, CsvFormat, IPCMessage, MappableTrait, MessageBuilderTrait, TablePublication, TableTrait, create_session_supersteps_batch
    };
    use phymes_diagnostics::HashMap;

    use crate::{
        AvailableInterfaceSubjects, SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream, create_message_map
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_melt_study_data_session() -> Result<()> {
        // Make the anticipated pivot table values
        let variable_names = &["Age","Gender","Ethnicity","RFFT","VAT","BMI","Statin"];
        let data_types = &[DataType::Int64,DataType::Int64,DataType::Int64,DataType::Int64,DataType::Int64,DataType::Float64,DataType::Int64];

        // Initialize the session
        let ontology_rag_session = OntologyRAGSession::new(None,
            "Casenr",
            None,
            variable_names,
            data_types)?;
        // dbg!(&ontology_rag_session.as_mermaid_erdiagram()?);
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            ontology_rag_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            ontology_rag_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(ontology_rag_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the tabular data
        let csv_format = CsvFormat::default();
        let sample_names = ["4088","4089","4090","4091","4092","4093","4094","4095"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ages = vec![82,82,82,82,82,82,82,82];
        let genders = vec![0,0,0,0,1,1,1,1];
        let ethnicities = vec![0,0,0,0,0,0,0,0];
        let rffs = vec![52,40,53,33,47,35,67,25];
        let vats = vec![9,11,4,10,11,7,6,10];
        let bmis = vec![31.8734311,24.25867407,26.0932752,26.3958034,23.70110632,30.54380794,26.0261749,26.72929708];
        let statins = vec![1,1,0,1,1,1,0,0];
        let sample_names: ArrayRef = Arc::new(StringArray::from(sample_names));
        let ages: ArrayRef = Arc::new(Int64Array::from(ages));
        let genders: ArrayRef = Arc::new(Int64Array::from(genders));
        let ethnicities: ArrayRef = Arc::new(Int64Array::from(ethnicities));
        let rffs: ArrayRef = Arc::new(Int64Array::from(rffs));
        let vats: ArrayRef = Arc::new(Int64Array::from(vats));
        let bmis: ArrayRef = Arc::new(Float64Array::from(bmis));
        let statins: ArrayRef = Arc::new(Int64Array::from(statins));
        let batch = RecordBatch::try_from_iter(vec![("Casenr",sample_names),
            ("Age",ages),
            ("Gender",genders),
            ("Ethnicity",ethnicities),
            ("RFFT",rffs),
            ("VAT",vats),
            ("BMI",bmis),
            ("Statin",statins)])?;
        let table = Table::get_builder()
            .with_name("PivotTable")
            .with_record_batches(vec![batch])?
            .build()?;
        let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
        let blob = AvailableInterfaceSubjects::UserCsv
            .to_table_builder(None)
            .with_blob(None, Some("csv"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&TablePublication::Extend {
                table_name: blob.get_name().to_string(),
            })
            .with_publisher(ontology_rag_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![blob_message]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test session context
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get("StudySamplesMelt")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("sample_name");
            assert_eq!(column, ["4088","4089","4090","4091","4092","4093","4094","4095"]);
            let column = table_reading.get_column_as_vec_primitive::<u32>("study_id")?;
            assert!(!column.is_empty());

            let table_reading = session_reading
                .get_states()
                .get("SamplesVariablesMelt")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("sample_name");
            assert_eq!(column, ["4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095"]);
            let column = table_reading.get_column_as_vec_str("variable_name");
            assert_eq!(column, ["Age", "Age", "Age", "Age", "Age", "Age", "Age", "Age", "Gender", "Gender", "Gender", "Gender", "Gender", "Gender", "Gender", "Gender", "Ethnicity", "Ethnicity", "Ethnicity", "Ethnicity", "Ethnicity", "Ethnicity", "Ethnicity", "Ethnicity", "RFFT", "RFFT", "RFFT", "RFFT", "RFFT", "RFFT", "RFFT", "RFFT", "VAT", "VAT", "VAT", "VAT", "VAT", "VAT", "VAT", "VAT", "BMI", "BMI", "BMI", "BMI", "BMI", "BMI", "BMI", "BMI", "Statin", "Statin", "Statin", "Statin", "Statin", "Statin", "Statin", "Statin"]);
            let column = table_reading.get_column_as_vec_str("value");
            assert_eq!(column, ["82", "82", "82", "82", "82", "82", "82", "82", "0", "0", "0", "0", "1", "1", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "52", "40", "53", "33", "47", "35", "67", "25", "9", "11", "4", "10", "11", "7", "6", "10", "31.8734311", "24.25867407", "26.0932752", "26.3958034", "23.70110632", "30.54380794", "26.0261749", "26.72929708", "1", "1", "0", "1", "1", "1", "0", "0"]);
            let column = table_reading.get_column_as_vec_primitive::<u32>("study_id")?;
            assert!(!column.is_empty());

            let table_reading = session_reading
                .get_states()
                .get("StudyVariablesMelt")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("variable_name");
            assert_eq!(column, ["Age", "BMI", "Ethnicity", "Gender", "RFFT", "Statin", "VAT"]);
            let column = table_reading.get_column_as_vec_str("data_type");
            assert_eq!(column, ["Int64","Float64","Int64","Int64","Int64","Int64","Int64"]);
            let column = table_reading.get_column_as_vec_primitive::<u32>("study_id")?;
            assert!(!column.is_empty());
        }

        Ok(())
    }
}
