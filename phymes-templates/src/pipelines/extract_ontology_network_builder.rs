/// A session for melting a `Study Dataset` from a single workflow step
///
/// # Notes
///
/// * Does not consider pre-filtering by ontology before vector search
pub struct ExtractOntologyNetworkBuilder<'a> {
    /// Session
    pub network_name: &'a str,
}

impl<'a> Default for ExtractOntologyNetworkBuilder<'a> {
    fn default() -> Self {
        Self {
            network_name: "extract_ontology_network",
        }
    }
}

impl<'a> ExtractOntologyNetworkBuilder<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	%% ------------------------------------------------------------------------------
	%% OWL ontology extraction
	%% ------------------------------------------------------------------------------
	subgraph extract_owl_t
	    UserScript-subject-.->|AllRecordBatches|extract_owl_p-subscribe
	    extract_owl_p-subscribe-->extract_owl_p-processor
	    extract_owl_p-processor-->extract_owl_p-publish
	    extract_owl_p-publish-->|Extend|ParseOwl-subject
	end
	extract_owl_r-rt@{shape: subproc, label: extract_owl_r}
	extract_owl_r-rt-->extract_owl_t
	UserScript-subject@{shape: doc, label: UserScript}
	extract_owl_p-processor@{shape: rect, label: ExtractXML}
	extract_owl_p-publish@{shape: fork}
	extract_owl_p-subscribe@{shape: diamond, label: All}
	ParseOwl-subject@{shape: doc, label: ParseOwl}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:Ontology entities
	%% ------------------------------------------------------------------------------
	subgraph filter_ontology_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_ontology_entity_p-subscribe
	    comparator_ontology_entity_p-subscribe-->comparator_ontology_entity_p-processor
	    comparator_ontology_entity_p-processor-->comparator_ontology_entity_p-publish
	    comparator_ontology_entity_p-publish-->|Replace|comparator_ontology_entity_s-subject
	    comparator_ontology_entity_s-subject-->|AllRecordBatches|filter_ontology_entity_p-subscribe
	    filter_ontology_entity_p-subscribe-->filter_ontology_entity_p-processor
	    filter_ontology_entity_p-processor-->filter_ontology_entity_p-publish
	    filter_ontology_entity_p-publish-->|Replace|filter_ontology_entity_s-subject
	    filter_ontology_entity_s-subject-->|AllRecordBatches|select_ontology_entity_p-subscribe
	    select_ontology_entity_p-subscribe-->select_ontology_entity_p-processor
	    select_ontology_entity_p-processor-->select_ontology_entity_p-publish
	    select_ontology_entity_p-publish-->|Extend|select_ontology_entity_s-subject
	end
	extract_owl_r-rt-->filter_ontology_entity_t
	comparator_ontology_entity_p-processor@{shape: rect, label: Select}
	comparator_ontology_entity_p-publish@{shape: fork}
	comparator_ontology_entity_p-subscribe@{shape: diamond, label: All}
	comparator_ontology_entity_s-subject@{shape: doc, label: comparator_ontology_entity_s}
	filter_ontology_entity_p-processor@{shape: rect, label: Filter}
	filter_ontology_entity_p-publish@{shape: fork}
	filter_ontology_entity_p-subscribe@{shape: diamond, label: All}
	filter_ontology_entity_s-subject@{shape: doc, label: filter_ontology_entity_s}
	select_ontology_entity_p-processor@{shape: rect, label: Select}
	select_ontology_entity_p-publish@{shape: fork}
	select_ontology_entity_p-subscribe@{shape: diamond, label: All}
	select_ontology_entity_s-subject@{shape: doc, label: select_ontology_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:AnnotationProperty entities
	%% ------------------------------------------------------------------------------
	subgraph filter_annotation_property_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_annotation_property_entity_p-subscribe
	    comparator_annotation_property_entity_p-subscribe-->comparator_annotation_property_entity_p-processor
	    comparator_annotation_property_entity_p-processor-->comparator_annotation_property_entity_p-publish
	    comparator_annotation_property_entity_p-publish-->|Replace|comparator_annotation_property_entity_s-subject
	    comparator_annotation_property_entity_s-subject-->|AllRecordBatches|filter_annotation_property_entity_p-subscribe
	    filter_annotation_property_entity_p-subscribe-->filter_annotation_property_entity_p-processor
	    filter_annotation_property_entity_p-processor-->filter_annotation_property_entity_p-publish
	    filter_annotation_property_entity_p-publish-->|Replace|filter_annotation_property_entity_s-subject
	    filter_annotation_property_entity_s-subject-->|AllRecordBatches|select_annotation_property_entity_p-subscribe
	    select_annotation_property_entity_p-subscribe-->select_annotation_property_entity_p-processor
	    select_annotation_property_entity_p-processor-->select_annotation_property_entity_p-publish
	    select_annotation_property_entity_p-publish-->|Extend|select_annotation_property_entity_s-subject
	end
	extract_owl_r-rt-->filter_annotation_property_entity_t
	comparator_annotation_property_entity_p-processor@{shape: rect, label: Select}
	comparator_annotation_property_entity_p-publish@{shape: fork}
	comparator_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_annotation_property_entity_s-subject@{shape: doc, label: comparator_annotation_property_entity_s}
	filter_annotation_property_entity_p-processor@{shape: rect, label: Filter}
	filter_annotation_property_entity_p-publish@{shape: fork}
	filter_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_annotation_property_entity_s-subject@{shape: doc, label: filter_annotation_property_entity_s}
	select_annotation_property_entity_p-processor@{shape: rect, label: Select}
	select_annotation_property_entity_p-publish@{shape: fork}
	select_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	select_annotation_property_entity_s-subject@{shape: doc, label: select_annotation_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:DatatypeProperty entities
	%% ------------------------------------------------------------------------------
	subgraph filter_datatype_property_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_datatype_property_entity_p-subscribe
	    comparator_datatype_property_entity_p-subscribe-->comparator_datatype_property_entity_p-processor
	    comparator_datatype_property_entity_p-processor-->comparator_datatype_property_entity_p-publish
	    comparator_datatype_property_entity_p-publish-->|Replace|comparator_datatype_property_entity_s-subject
	    comparator_datatype_property_entity_s-subject-->|AllRecordBatches|filter_datatype_property_entity_p-subscribe
	    filter_datatype_property_entity_p-subscribe-->filter_datatype_property_entity_p-processor
	    filter_datatype_property_entity_p-processor-->filter_datatype_property_entity_p-publish
	    filter_datatype_property_entity_p-publish-->|Replace|filter_datatype_property_entity_s-subject
	    filter_datatype_property_entity_s-subject-->|AllRecordBatches|select_datatype_property_entity_p-subscribe
	    select_datatype_property_entity_p-subscribe-->select_datatype_property_entity_p-processor
	    select_datatype_property_entity_p-processor-->select_datatype_property_entity_p-publish
	    select_datatype_property_entity_p-publish-->|Extend|select_datatype_property_entity_s-subject
	end
	extract_owl_r-rt-->filter_datatype_property_entity_t
	comparator_datatype_property_entity_p-processor@{shape: rect, label: Select}
	comparator_datatype_property_entity_p-publish@{shape: fork}
	comparator_datatype_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_datatype_property_entity_s-subject@{shape: doc, label: comparator_datatype_property_entity_s}
	filter_datatype_property_entity_p-processor@{shape: rect, label: Filter}
	filter_datatype_property_entity_p-publish@{shape: fork}
	filter_datatype_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_datatype_property_entity_s-subject@{shape: doc, label: filter_datatype_property_entity_s}
	select_datatype_property_entity_p-processor@{shape: rect, label: Select}
	select_datatype_property_entity_p-publish@{shape: fork}
	select_datatype_property_entity_p-subscribe@{shape: diamond, label: All}
	select_datatype_property_entity_s-subject@{shape: doc, label: select_datatype_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:Class entities
	%% ------------------------------------------------------------------------------
	subgraph filter_class_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_class_entity_p-subscribe
	    comparator_class_entity_p-subscribe-->comparator_class_entity_p-processor
	    comparator_class_entity_p-processor-->comparator_class_entity_p-publish
	    comparator_class_entity_p-publish-->|Replace|comparator_class_entity_s-subject
	    comparator_class_entity_s-subject-->|AllRecordBatches|filter_class_entity_p-subscribe
	    filter_class_entity_p-subscribe-->filter_class_entity_p-processor
	    filter_class_entity_p-processor-->filter_class_entity_p-publish
	    filter_class_entity_p-publish-->|Replace|filter_class_entity_s-subject
	    filter_class_entity_s-subject-->|AllRecordBatches|select_class_entity_p-subscribe
	    select_class_entity_p-subscribe-->select_class_entity_p-processor
	    select_class_entity_p-processor-->select_class_entity_p-publish
	    select_class_entity_p-publish-->|Extend|select_class_entity_s-subject
	end
	extract_owl_r-rt-->filter_class_entity_t
	comparator_class_entity_p-processor@{shape: rect, label: Select}
	comparator_class_entity_p-publish@{shape: fork}
	comparator_class_entity_p-subscribe@{shape: diamond, label: All}
	comparator_class_entity_s-subject@{shape: doc, label: comparator_class_entity_s}
	filter_class_entity_p-processor@{shape: rect, label: Filter}
	filter_class_entity_p-publish@{shape: fork}
	filter_class_entity_p-subscribe@{shape: diamond, label: All}
	filter_class_entity_s-subject@{shape: doc, label: filter_class_entity_s}
	select_class_entity_p-processor@{shape: rect, label: Select}
	select_class_entity_p-publish@{shape: fork}
	select_class_entity_p-subscribe@{shape: diamond, label: All}
	select_class_entity_s-subject@{shape: doc, label: select_class_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:ObjectProperty entities
	%% ------------------------------------------------------------------------------
	subgraph filter_object_property_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_object_property_entity_p-subscribe
	    comparator_object_property_entity_p-subscribe-->comparator_object_property_entity_p-processor
	    comparator_object_property_entity_p-processor-->comparator_object_property_entity_p-publish
	    comparator_object_property_entity_p-publish-->|Replace|comparator_object_property_entity_s-subject
	    comparator_object_property_entity_s-subject-->|AllRecordBatches|filter_object_property_entity_p-subscribe
	    filter_object_property_entity_p-subscribe-->filter_object_property_entity_p-processor
	    filter_object_property_entity_p-processor-->filter_object_property_entity_p-publish
	    filter_object_property_entity_p-publish-->|Replace|filter_object_property_entity_s-subject
	    filter_object_property_entity_s-subject-->|AllRecordBatches|select_object_property_entity_p-subscribe
	    select_object_property_entity_p-subscribe-->select_object_property_entity_p-processor
	    select_object_property_entity_p-processor-->select_object_property_entity_p-publish
	    select_object_property_entity_p-publish-->|Extend|select_object_property_entity_s-subject
	end
	extract_owl_r-rt-->filter_object_property_entity_t
	comparator_object_property_entity_p-processor@{shape: rect, label: Select}
	comparator_object_property_entity_p-publish@{shape: fork}
	comparator_object_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_object_property_entity_s-subject@{shape: doc, label: comparator_object_property_entity_s}
	filter_object_property_entity_p-processor@{shape: rect, label: Filter}
	filter_object_property_entity_p-publish@{shape: fork}
	filter_object_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_object_property_entity_s-subject@{shape: doc, label: filter_object_property_entity_s}
	select_object_property_entity_p-processor@{shape: rect, label: Select}
	select_object_property_entity_p-publish@{shape: fork}
	select_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_object_property_entity_s-subject@{shape: doc, label: select_object_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:NamedIndividual entities
	%% ------------------------------------------------------------------------------
	subgraph filter_named_individual_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_named_individual_entity_p-subscribe
	    comparator_named_individual_entity_p-subscribe-->comparator_named_individual_entity_p-processor
	    comparator_named_individual_entity_p-processor-->comparator_named_individual_entity_p-publish
	    comparator_named_individual_entity_p-publish-->|Replace|comparator_named_individual_entity_s-subject
	    comparator_named_individual_entity_s-subject-->|AllRecordBatches|filter_named_individual_entity_p-subscribe
	    filter_named_individual_entity_p-subscribe-->filter_named_individual_entity_p-processor
	    filter_named_individual_entity_p-processor-->filter_named_individual_entity_p-publish
	    filter_named_individual_entity_p-publish-->|Replace|filter_named_individual_entity_s-subject
	    filter_named_individual_entity_s-subject-->|AllRecordBatches|select_named_individual_entity_p-subscribe
	    select_named_individual_entity_p-subscribe-->select_named_individual_entity_p-processor
	    select_named_individual_entity_p-processor-->select_named_individual_entity_p-publish
	    select_named_individual_entity_p-publish-->|Extend|select_named_individual_entity_s-subject
	end
	extract_owl_r-rt-->filter_named_individual_entity_t
	comparator_named_individual_entity_p-processor@{shape: rect, label: Select}
	comparator_named_individual_entity_p-publish@{shape: fork}
	comparator_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	comparator_named_individual_entity_s-subject@{shape: doc, label: comparator_named_individual_entity_s}
	filter_named_individual_entity_p-processor@{shape: rect, label: Filter}
	filter_named_individual_entity_p-publish@{shape: fork}
	filter_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	filter_named_individual_entity_s-subject@{shape: doc, label: filter_named_individual_entity_s}
	select_named_individual_entity_p-processor@{shape: rect, label: Select}
	select_named_individual_entity_p-publish@{shape: fork}
	select_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	select_named_individual_entity_s-subject@{shape: doc, label: select_named_individual_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter on Owl:Axiom entities
	%% ------------------------------------------------------------------------------
	subgraph filter_axiom_entity_t
	    ParseOwl-subject-.->|LastRecordBatch|comparator_axiom_entity_p-subscribe
	    comparator_axiom_entity_p-subscribe-->comparator_axiom_entity_p-processor
	    comparator_axiom_entity_p-processor-->comparator_axiom_entity_p-publish
	    comparator_axiom_entity_p-publish-->|Replace|comparator_axiom_entity_s-subject
	    comparator_axiom_entity_s-subject-->|AllRecordBatches|filter_axiom_entity_p-subscribe
	    filter_axiom_entity_p-subscribe-->filter_axiom_entity_p-processor
	    filter_axiom_entity_p-processor-->filter_axiom_entity_p-publish
	    filter_axiom_entity_p-publish-->|Replace|filter_axiom_entity_s-subject
	    filter_axiom_entity_s-subject-->|AllRecordBatches|select_axiom_entity_p-subscribe
	    select_axiom_entity_p-subscribe-->select_axiom_entity_p-processor
	    select_axiom_entity_p-processor-->select_axiom_entity_p-publish
	    select_axiom_entity_p-publish-->|Extend|select_axiom_entity_s-subject
	end
	extract_owl_r-rt-->filter_axiom_entity_t
	comparator_axiom_entity_p-processor@{shape: rect, label: Select}
	comparator_axiom_entity_p-publish@{shape: fork}
	comparator_axiom_entity_p-subscribe@{shape: diamond, label: All}
	comparator_axiom_entity_s-subject@{shape: doc, label: comparator_axiom_entity_s}
	filter_axiom_entity_p-processor@{shape: rect, label: Filter}
	filter_axiom_entity_p-publish@{shape: fork}
	filter_axiom_entity_p-subscribe@{shape: diamond, label: All}
	filter_axiom_entity_s-subject@{shape: doc, label: filter_axiom_entity_s}
	select_axiom_entity_p-processor@{shape: rect, label: Select}
	select_axiom_entity_p-publish@{shape: fork}
	select_axiom_entity_p-subscribe@{shape: diamond, label: All}
	select_axiom_entity_s-subject@{shape: doc, label: select_axiom_entity_s}
	%% ------------------------------------------------------------------------------
	%% Pivot Owl:AnnotationProperty on rdfs:label (or skos:prefLabel)
	%% ------------------------------------------------------------------------------
	subgraph pivot_annotation_property_t
	    select_annotation_property_entity_s-subject-.->|LastRecordBatch|coalesce_annotation_property_entity_p-subscribe
	    coalesce_annotation_property_entity_p-subscribe-->coalesce_annotation_property_entity_p-processor
	    coalesce_annotation_property_entity_p-processor-->coalesce_annotation_property_entity_p-publish
	    coalesce_annotation_property_entity_p-publish-->|Replace|coalesce_annotation_property_entity_s-subject
	    coalesce_annotation_property_entity_s-subject-->|AllRecordBatches|comparator_predicate_annotation_property_entity_p-subscribe
	    comparator_predicate_annotation_property_entity_p-subscribe-->comparator_predicate_annotation_property_entity_p-processor
	    comparator_predicate_annotation_property_entity_p-processor-->comparator_predicate_annotation_property_entity_p-publish
	    comparator_predicate_annotation_property_entity_p-publish-->|Replace|comparator_predicate_annotation_property_entity_s-subject
	    comparator_predicate_annotation_property_entity_s-subject-->|AllRecordBatches|filter_predicate_annotation_property_entity_p-subscribe
	    filter_predicate_annotation_property_entity_p-subscribe-->filter_predicate_annotation_property_entity_p-processor
	    filter_predicate_annotation_property_entity_p-processor-->filter_predicate_annotation_property_entity_p-publish
	    filter_predicate_annotation_property_entity_p-publish-->|Replace|filter_predicate_annotation_property_entity_s-subject
	    filter_predicate_annotation_property_entity_s-subject-->|AllRecordBatches|select_predicate_annotation_property_entity_p-subscribe
	    select_predicate_annotation_property_entity_p-subscribe-->select_predicate_annotation_property_entity_p-processor
	    select_predicate_annotation_property_entity_p-processor-->select_predicate_annotation_property_entity_p-publish
	    select_predicate_annotation_property_entity_p-publish-->|Replace|select_predicate_annotation_property_entity_s-subject
	    select_predicate_annotation_property_entity_s-subject-->|AllRecordBatches|pivot_annotation_property_entity_p-subscribe
	    pivot_annotation_property_entity_p-subscribe-->pivot_annotation_property_entity_p-processor
	    pivot_annotation_property_entity_p-processor-->pivot_annotation_property_entity_p-publish
	    pivot_annotation_property_entity_p-publish-->|Replace|pivot_annotation_property_entity_s-subject
	end
	extract_owl_r-rt-->pivot_annotation_property_t
	coalesce_annotation_property_entity_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_annotation_property_entity_p-publish@{shape: fork}
	coalesce_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	coalesce_annotation_property_entity_s-subject@{shape: doc, label: coalesce_annotation_property_entity_s}
	comparator_predicate_annotation_property_entity_p-processor@{shape: rect, label: Select}
	comparator_predicate_annotation_property_entity_p-publish@{shape: fork}
	comparator_predicate_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_predicate_annotation_property_entity_s-subject@{shape: doc, label: comparator_predicate_annotation_property_entity_s}
	filter_predicate_annotation_property_entity_p-processor@{shape: rect, label: Filter}
	filter_predicate_annotation_property_entity_p-publish@{shape: fork}
	filter_predicate_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_predicate_annotation_property_entity_s-subject@{shape: doc, label: filter_predicate_annotation_property_entity_s}
	select_predicate_annotation_property_entity_p-processor@{shape: rect, label: Select}
	select_predicate_annotation_property_entity_p-publish@{shape: fork}
	select_predicate_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	select_predicate_annotation_property_entity_s-subject@{shape: doc, label: select_predicate_annotation_property_entity_s}
	pivot_annotation_property_entity_p-processor@{shape: rect, label: Pivot}
	pivot_annotation_property_entity_p-publish@{shape: fork}
	pivot_annotation_property_entity_p-subscribe@{shape: diamond, label: All}
	pivot_annotation_property_entity_s-subject@{shape: doc, label: pivot_annotation_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Owl:AnnotationProperty post-pivot cleanup
	%% ------------------------------------------------------------------------------
	subgraph post_pivot_annotation_property_t
	    pivot_annotation_property_entity_s-subject-.->|AllRecordBatches|coalesce_annotation_property_pivot_p-subscribe
	    coalesce_annotation_property_pivot_p-subscribe-->coalesce_annotation_property_pivot_p-processor
	    coalesce_annotation_property_pivot_p-processor-->coalesce_annotation_property_pivot_p-publish
	    coalesce_annotation_property_pivot_p-publish-->|Replace|coalesce_annotation_property_pivot_s-subject
	    coalesce_annotation_property_pivot_s-subject-->|AllRecordBatches|group_by_annotation_property_pivot_p-subscribe
	    group_by_annotation_property_pivot_p-subscribe-->group_by_annotation_property_pivot_p-processor
	    group_by_annotation_property_pivot_p-processor-->group_by_annotation_property_pivot_p-publish
	    group_by_annotation_property_pivot_p-publish-->|Replace|group_by_annotation_property_pivot_s-subject
	    group_by_annotation_property_pivot_s-subject-->|AllRecordBatches|select_annotation_property_pivot_p-subscribe
	    select_annotation_property_pivot_p-subscribe-->select_annotation_property_pivot_p-processor
	    select_annotation_property_pivot_p-processor-->select_annotation_property_pivot_p-publish
	    select_annotation_property_pivot_p-publish-->|Replace|select_annotation_property_pivot_s-subject
	end
	extract_owl_r-rt-->post_pivot_annotation_property_t
	coalesce_annotation_property_pivot_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_annotation_property_pivot_p-publish@{shape: fork}
	coalesce_annotation_property_pivot_p-subscribe@{shape: diamond, label: All}
	coalesce_annotation_property_pivot_s-subject@{shape: doc, label: coalesce_annotation_property_pivot_s}
	group_by_annotation_property_pivot_p-processor@{shape: rect, label: GroupBy}
	group_by_annotation_property_pivot_p-publish@{shape: fork}
	group_by_annotation_property_pivot_p-subscribe@{shape: diamond, label: All}
	group_by_annotation_property_pivot_s-subject@{shape: doc, label: group_by_annotation_property_pivot_s}
	select_annotation_property_pivot_p-processor@{shape: rect, label: Select}
	select_annotation_property_pivot_p-publish@{shape: fork}
	select_annotation_property_pivot_p-subscribe@{shape: diamond, label: All}
	select_annotation_property_pivot_s-subject@{shape: doc, label: select_annotation_property_pivot_s}
	%% ------------------------------------------------------------------------------
	%% Pivot Owl:Class on rdfs:label (or skos:prefLabel)
	%% ------------------------------------------------------------------------------
	subgraph pivot_class_entity_t
	    select_class_entity_s-subject-.->|LastRecordBatch|coalesce_class_entity_p-subscribe
	    coalesce_class_entity_p-subscribe-->coalesce_class_entity_p-processor
	    coalesce_class_entity_p-processor-->coalesce_class_entity_p-publish
	    coalesce_class_entity_p-publish-->|Replace|coalesce_class_entity_s-subject
	    coalesce_class_entity_s-subject-->|AllRecordBatches|comparator_predicate_class_entity_p-subscribe
	    comparator_predicate_class_entity_p-subscribe-->comparator_predicate_class_entity_p-processor
	    comparator_predicate_class_entity_p-processor-->comparator_predicate_class_entity_p-publish
	    comparator_predicate_class_entity_p-publish-->|Replace|comparator_predicate_class_entity_s-subject
	    comparator_predicate_class_entity_s-subject-->|AllRecordBatches|filter_predicate_class_entity_p-subscribe
	    filter_predicate_class_entity_p-subscribe-->filter_predicate_class_entity_p-processor
	    filter_predicate_class_entity_p-processor-->filter_predicate_class_entity_p-publish
	    filter_predicate_class_entity_p-publish-->|Replace|filter_predicate_class_entity_s-subject
	    filter_predicate_class_entity_s-subject-->|AllRecordBatches|select_predicate_class_entity_p-subscribe
	    select_predicate_class_entity_p-subscribe-->select_predicate_class_entity_p-processor
	    select_predicate_class_entity_p-processor-->select_predicate_class_entity_p-publish
	    select_predicate_class_entity_p-publish-->|Replace|select_predicate_class_entity_s-subject
	    select_predicate_class_entity_s-subject-->|AllRecordBatches|pivot_class_entity_p-subscribe
	    pivot_class_entity_p-subscribe-->pivot_class_entity_p-processor
	    pivot_class_entity_p-processor-->pivot_class_entity_p-publish
	    pivot_class_entity_p-publish-->|Replace|pivot_class_entity_s-subject
	end
	extract_owl_r-rt-->pivot_class_entity_t
	coalesce_class_entity_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_class_entity_p-publish@{shape: fork}
	coalesce_class_entity_p-subscribe@{shape: diamond, label: All}
	coalesce_class_entity_s-subject@{shape: doc, label: coalesce_class_entity_s}
	comparator_predicate_class_entity_p-processor@{shape: rect, label: Select}
	comparator_predicate_class_entity_p-publish@{shape: fork}
	comparator_predicate_class_entity_p-subscribe@{shape: diamond, label: All}
	comparator_predicate_class_entity_s-subject@{shape: doc, label: comparator_predicate_class_entity_s}
	filter_predicate_class_entity_p-processor@{shape: rect, label: Filter}
	filter_predicate_class_entity_p-publish@{shape: fork}
	filter_predicate_class_entity_p-subscribe@{shape: diamond, label: All}
	filter_predicate_class_entity_s-subject@{shape: doc, label: filter_predicate_class_entity_s}
	select_predicate_class_entity_p-processor@{shape: rect, label: Select}
	select_predicate_class_entity_p-publish@{shape: fork}
	select_predicate_class_entity_p-subscribe@{shape: diamond, label: All}
	select_predicate_class_entity_s-subject@{shape: doc, label: select_predicate_class_entity_s}
	pivot_class_entity_p-processor@{shape: rect, label: Pivot}
	pivot_class_entity_p-publish@{shape: fork}
	pivot_class_entity_p-subscribe@{shape: diamond, label: All}
	pivot_class_entity_s-subject@{shape: doc, label: pivot_class_entity_s}
	%% ------------------------------------------------------------------------------
	%% Owl:Class post-pivot cleanup
	%% ------------------------------------------------------------------------------
	subgraph post_pivot_class_entity_t
	    pivot_class_entity_s-subject-.->|AllRecordBatches|coalesce_class_pivot_p-subscribe
	    coalesce_class_pivot_p-subscribe-->coalesce_class_pivot_p-processor
	    coalesce_class_pivot_p-processor-->coalesce_class_pivot_p-publish
	    coalesce_class_pivot_p-publish-->|Replace|coalesce_class_pivot_s-subject
	    coalesce_class_pivot_s-subject-->|AllRecordBatches|group_by_class_pivot_p-subscribe
	    group_by_class_pivot_p-subscribe-->group_by_class_pivot_p-processor
	    group_by_class_pivot_p-processor-->group_by_class_pivot_p-publish
	    group_by_class_pivot_p-publish-->|Replace|group_by_class_pivot_s-subject
	    group_by_class_pivot_s-subject-->|AllRecordBatches|select_class_pivot_p-subscribe
	    select_class_pivot_p-subscribe-->select_class_pivot_p-processor
	    select_class_pivot_p-processor-->select_class_pivot_p-publish
	    select_class_pivot_p-publish-->|Extend|merge_object_property_class_named_individual_pivot_s-subject
	end
	extract_owl_r-rt-->post_pivot_class_entity_t
	coalesce_class_pivot_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_class_pivot_p-publish@{shape: fork}
	coalesce_class_pivot_p-subscribe@{shape: diamond, label: All}
	coalesce_class_pivot_s-subject@{shape: doc, label: coalesce_class_pivot_s}
	group_by_class_pivot_p-processor@{shape: rect, label: GroupBy}
	group_by_class_pivot_p-publish@{shape: fork}
	group_by_class_pivot_p-subscribe@{shape: diamond, label: All}
	group_by_class_pivot_s-subject@{shape: doc, label: group_by_class_pivot_s}
	select_class_pivot_p-processor@{shape: rect, label: Select}
	select_class_pivot_p-publish@{shape: fork}
	select_class_pivot_p-subscribe@{shape: diamond, label: All}
	merge_object_property_class_named_individual_pivot_s-subject@{shape: doc, label: merge_object_property_class_named_individual_pivot_s}
	%% ------------------------------------------------------------------------------
	%% Pivot Owl:ObjectProperty on rdfs:label (or skos:prefLabel)
	%% ------------------------------------------------------------------------------
	subgraph pivot_object_property_t
	    select_object_property_entity_s-subject-.->|LastRecordBatch|coalesce_object_property_entity_p-subscribe
	    coalesce_object_property_entity_p-subscribe-->coalesce_object_property_entity_p-processor
	    coalesce_object_property_entity_p-processor-->coalesce_object_property_entity_p-publish
	    coalesce_object_property_entity_p-publish-->|Replace|coalesce_object_property_entity_s-subject
	    coalesce_object_property_entity_s-subject-->|AllRecordBatches|comparator_predicate_object_property_entity_p-subscribe
	    comparator_predicate_object_property_entity_p-subscribe-->comparator_predicate_object_property_entity_p-processor
	    comparator_predicate_object_property_entity_p-processor-->comparator_predicate_object_property_entity_p-publish
	    comparator_predicate_object_property_entity_p-publish-->|Replace|comparator_predicate_object_property_entity_s-subject
	    comparator_predicate_object_property_entity_s-subject-->|AllRecordBatches|filter_predicate_object_property_entity_p-subscribe
	    filter_predicate_object_property_entity_p-subscribe-->filter_predicate_object_property_entity_p-processor
	    filter_predicate_object_property_entity_p-processor-->filter_predicate_object_property_entity_p-publish
	    filter_predicate_object_property_entity_p-publish-->|Replace|filter_predicate_object_property_entity_s-subject
	    filter_predicate_object_property_entity_s-subject-->|AllRecordBatches|select_predicate_object_property_entity_p-subscribe
	    select_predicate_object_property_entity_p-subscribe-->select_predicate_object_property_entity_p-processor
	    select_predicate_object_property_entity_p-processor-->select_predicate_object_property_entity_p-publish
	    select_predicate_object_property_entity_p-publish-->|Replace|select_predicate_object_property_entity_s-subject
	    select_predicate_object_property_entity_s-subject-->|AllRecordBatches|pivot_object_property_entity_p-subscribe
	    pivot_object_property_entity_p-subscribe-->pivot_object_property_entity_p-processor
	    pivot_object_property_entity_p-processor-->pivot_object_property_entity_p-publish
	    pivot_object_property_entity_p-publish-->|Replace|pivot_object_property_entity_s-subject
	end
	extract_owl_r-rt-->pivot_object_property_t
	coalesce_object_property_entity_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_object_property_entity_p-publish@{shape: fork}
	coalesce_object_property_entity_p-subscribe@{shape: diamond, label: All}
	coalesce_object_property_entity_s-subject@{shape: doc, label: coalesce_object_property_entity_s}
	comparator_predicate_object_property_entity_p-processor@{shape: rect, label: Select}
	comparator_predicate_object_property_entity_p-publish@{shape: fork}
	comparator_predicate_object_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_predicate_object_property_entity_s-subject@{shape: doc, label: comparator_predicate_object_property_entity_s}
	filter_predicate_object_property_entity_p-processor@{shape: rect, label: Filter}
	filter_predicate_object_property_entity_p-publish@{shape: fork}
	filter_predicate_object_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_predicate_object_property_entity_s-subject@{shape: doc, label: filter_predicate_object_property_entity_s}
	select_predicate_object_property_entity_p-processor@{shape: rect, label: Select}
	select_predicate_object_property_entity_p-publish@{shape: fork}
	select_predicate_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_predicate_object_property_entity_s-subject@{shape: doc, label: select_predicate_object_property_entity_s}
	pivot_object_property_entity_p-processor@{shape: rect, label: Pivot}
	pivot_object_property_entity_p-publish@{shape: fork}
	pivot_object_property_entity_p-subscribe@{shape: diamond, label: All}
	pivot_object_property_entity_s-subject@{shape: doc, label: pivot_object_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Owl:ObjectProperty post-pivot cleanup
	%% ------------------------------------------------------------------------------
	subgraph post_pivot_object_property_t
	    pivot_object_property_entity_s-subject-.->|AllRecordBatches|coalesce_object_property_pivot_p-subscribe
	    coalesce_object_property_pivot_p-subscribe-->coalesce_object_property_pivot_p-processor
	    coalesce_object_property_pivot_p-processor-->coalesce_object_property_pivot_p-publish
	    coalesce_object_property_pivot_p-publish-->|Replace|coalesce_object_property_pivot_s-subject
	    coalesce_object_property_pivot_s-subject-->|AllRecordBatches|group_by_object_property_pivot_p-subscribe
	    group_by_object_property_pivot_p-subscribe-->group_by_object_property_pivot_p-processor
	    group_by_object_property_pivot_p-processor-->group_by_object_property_pivot_p-publish
	    group_by_object_property_pivot_p-publish-->|Replace|group_by_object_property_pivot_s-subject
	    group_by_object_property_pivot_s-subject-->|AllRecordBatches|select_object_property_pivot_p-subscribe
	    select_object_property_pivot_p-subscribe-->select_object_property_pivot_p-processor
	    select_object_property_pivot_p-processor-->select_object_property_pivot_p-publish
	    select_object_property_pivot_p-publish-->|Extend|merge_object_property_class_named_individual_pivot_s-subject
	end
	extract_owl_r-rt-->post_pivot_object_property_t
	coalesce_object_property_pivot_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_object_property_pivot_p-publish@{shape: fork}
	coalesce_object_property_pivot_p-subscribe@{shape: diamond, label: All}
	coalesce_object_property_pivot_s-subject@{shape: doc, label: coalesce_object_property_pivot_s}
	group_by_object_property_pivot_p-processor@{shape: rect, label: GroupBy}
	group_by_object_property_pivot_p-publish@{shape: fork}
	group_by_object_property_pivot_p-subscribe@{shape: diamond, label: All}
	group_by_object_property_pivot_s-subject@{shape: doc, label: group_by_object_property_pivot_s}
	select_object_property_pivot_p-processor@{shape: rect, label: Select}
	select_object_property_pivot_p-publish@{shape: fork}
	select_object_property_pivot_p-subscribe@{shape: diamond, label: All}
	%% ------------------------------------------------------------------------------
	%% Pivot Owl:NamedIndividual on rdfs:label (or skos:prefLabel)
	%% ------------------------------------------------------------------------------
	subgraph pivot_named_individual_t
	    select_named_individual_entity_s-subject-.->|LastRecordBatch|coalesce_named_individual_entity_p-subscribe
	    coalesce_named_individual_entity_p-subscribe-->coalesce_named_individual_entity_p-processor
	    coalesce_named_individual_entity_p-processor-->coalesce_named_individual_entity_p-publish
	    coalesce_named_individual_entity_p-publish-->|Replace|coalesce_named_individual_entity_s-subject
	    coalesce_named_individual_entity_s-subject-->|AllRecordBatches|comparator_predicate_named_individual_entity_p-subscribe
	    comparator_predicate_named_individual_entity_p-subscribe-->comparator_predicate_named_individual_entity_p-processor
	    comparator_predicate_named_individual_entity_p-processor-->comparator_predicate_named_individual_entity_p-publish
	    comparator_predicate_named_individual_entity_p-publish-->|Replace|comparator_predicate_named_individual_entity_s-subject
	    comparator_predicate_named_individual_entity_s-subject-->|AllRecordBatches|filter_predicate_named_individual_entity_p-subscribe
	    filter_predicate_named_individual_entity_p-subscribe-->filter_predicate_named_individual_entity_p-processor
	    filter_predicate_named_individual_entity_p-processor-->filter_predicate_named_individual_entity_p-publish
	    filter_predicate_named_individual_entity_p-publish-->|Replace|filter_predicate_named_individual_entity_s-subject
	    filter_predicate_named_individual_entity_s-subject-->|AllRecordBatches|select_predicate_named_individual_entity_p-subscribe
	    select_predicate_named_individual_entity_p-subscribe-->select_predicate_named_individual_entity_p-processor
	    select_predicate_named_individual_entity_p-processor-->select_predicate_named_individual_entity_p-publish
	    select_predicate_named_individual_entity_p-publish-->|Replace|select_predicate_named_individual_entity_s-subject
	    select_predicate_named_individual_entity_s-subject-->|AllRecordBatches|pivot_named_individual_entity_p-subscribe
	    pivot_named_individual_entity_p-subscribe-->pivot_named_individual_entity_p-processor
	    pivot_named_individual_entity_p-processor-->pivot_named_individual_entity_p-publish
	    pivot_named_individual_entity_p-publish-->|Replace|pivot_named_individual_entity_s-subject
	end
	extract_owl_r-rt-->pivot_named_individual_t
	coalesce_named_individual_entity_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_named_individual_entity_p-publish@{shape: fork}
	coalesce_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	coalesce_named_individual_entity_s-subject@{shape: doc, label: coalesce_named_individual_entity_s}
	comparator_predicate_named_individual_entity_p-processor@{shape: rect, label: Select}
	comparator_predicate_named_individual_entity_p-publish@{shape: fork}
	comparator_predicate_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	comparator_predicate_named_individual_entity_s-subject@{shape: doc, label: comparator_predicate_named_individual_entity_s}
	filter_predicate_named_individual_entity_p-processor@{shape: rect, label: Filter}
	filter_predicate_named_individual_entity_p-publish@{shape: fork}
	filter_predicate_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	filter_predicate_named_individual_entity_s-subject@{shape: doc, label: filter_predicate_named_individual_entity_s}
	select_predicate_named_individual_entity_p-processor@{shape: rect, label: Select}
	select_predicate_named_individual_entity_p-publish@{shape: fork}
	select_predicate_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	select_predicate_named_individual_entity_s-subject@{shape: doc, label: select_predicate_named_individual_entity_s}
	pivot_named_individual_entity_p-processor@{shape: rect, label: Pivot}
	pivot_named_individual_entity_p-publish@{shape: fork}
	pivot_named_individual_entity_p-subscribe@{shape: diamond, label: All}
	pivot_named_individual_entity_s-subject@{shape: doc, label: pivot_named_individual_entity_s}
	%% ------------------------------------------------------------------------------
	%% Owl:NamedIndividual post-pivot cleanup
	%% ------------------------------------------------------------------------------
	subgraph post_pivot_named_individual_t
	    pivot_named_individual_entity_s-subject-.->|AllRecordBatches|coalesce_named_individual_pivot_p-subscribe
	    coalesce_named_individual_pivot_p-subscribe-->coalesce_named_individual_pivot_p-processor
	    coalesce_named_individual_pivot_p-processor-->coalesce_named_individual_pivot_p-publish
	    coalesce_named_individual_pivot_p-publish-->|Replace|coalesce_named_individual_pivot_s-subject
	    coalesce_named_individual_pivot_s-subject-->|AllRecordBatches|group_by_named_individual_pivot_p-subscribe
	    group_by_named_individual_pivot_p-subscribe-->group_by_named_individual_pivot_p-processor
	    group_by_named_individual_pivot_p-processor-->group_by_named_individual_pivot_p-publish
	    group_by_named_individual_pivot_p-publish-->|Replace|group_by_named_individual_pivot_s-subject
	    group_by_named_individual_pivot_s-subject-->|AllRecordBatches|select_named_individual_pivot_p-subscribe
	    select_named_individual_pivot_p-subscribe-->select_named_individual_pivot_p-processor
	    select_named_individual_pivot_p-processor-->select_named_individual_pivot_p-publish
	    select_named_individual_pivot_p-publish-->|Extend|merge_object_property_class_named_individual_pivot_s-subject
	end
	extract_owl_r-rt-->post_pivot_named_individual_t
	coalesce_named_individual_pivot_p-processor@{shape: rect, label: CoalesceProcessor}
	coalesce_named_individual_pivot_p-publish@{shape: fork}
	coalesce_named_individual_pivot_p-subscribe@{shape: diamond, label: All}
	coalesce_named_individual_pivot_s-subject@{shape: doc, label: coalesce_named_individual_pivot_s}
	group_by_named_individual_pivot_p-processor@{shape: rect, label: GroupBy}
	group_by_named_individual_pivot_p-publish@{shape: fork}
	group_by_named_individual_pivot_p-subscribe@{shape: diamond, label: All}
	group_by_named_individual_pivot_s-subject@{shape: doc, label: group_by_named_individual_pivot_s}
	select_named_individual_pivot_p-processor@{shape: rect, label: Select}
	select_named_individual_pivot_p-publish@{shape: fork}
	select_named_individual_pivot_p-subscribe@{shape: diamond, label: All}
	%% ------------------------------------------------------------------------------
	%% Join Owl:AnnotationProperty with Owl:Class on predicates
	%% ------------------------------------------------------------------------------
	subgraph join_class_on_predicates_t
	    select_class_entity_s-subject-.->|AllRecordBatches|join_predicates_class_entity_p-subscribe
	    select_annotation_property_pivot_s-subject-.->|AllRecordBatches|join_predicates_class_entity_p-subscribe
	    join_predicates_class_entity_p-subscribe-->join_predicates_class_entity_p-processor
	    join_predicates_class_entity_p-processor-->join_predicates_class_entity_p-publish
	    join_predicates_class_entity_p-publish-->|Replace|join_predicates_class_entity_s-subject
	    join_predicates_class_entity_s-subject-->|AllRecordBatches|select_predicates_class_entity_p-subscribe
	    select_predicates_class_entity_p-subscribe-->select_predicates_class_entity_p-processor
	    select_predicates_class_entity_p-processor-->select_predicates_class_entity_p-publish
	    select_predicates_class_entity_p-publish-->|Replace|select_predicates_class_entity_s-subject
	end
	extract_owl_r-rt-->join_class_on_predicates_t
	join_predicates_class_entity_p-processor@{shape: rect, label: Join}
	join_predicates_class_entity_p-publish@{shape: fork}
	join_predicates_class_entity_p-subscribe@{shape: diamond, label: All}
	join_predicates_class_entity_s-subject@{shape: doc, label: join_predicates_class_entity_s}
	select_predicates_class_entity_p-processor@{shape: rect, label: Select}
	select_predicates_class_entity_p-publish@{shape: fork}
	select_predicates_class_entity_p-subscribe@{shape: diamond, label: All}
	select_predicates_class_entity_s-subject@{shape: doc, label: select_predicates_class_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter Owl:Class for rdf:resource objects
	%% ------------------------------------------------------------------------------
	subgraph filter_resources_class_entity_t
	    select_predicates_class_entity_s-subject-.->|AllRecordBatches|comparator_resource_class_entity_p-subscribe
	    comparator_resource_class_entity_p-subscribe-->comparator_resource_class_entity_p-processor
	    comparator_resource_class_entity_p-processor-->comparator_resource_class_entity_p-publish
	    comparator_resource_class_entity_p-publish-->|Replace|comparator_resource_class_entity_s-subject
	    comparator_resource_class_entity_s-subject-->|AllRecordBatches|filter_resource_class_entity_p-subscribe
	    filter_resource_class_entity_p-subscribe-->filter_resource_class_entity_p-processor
	    filter_resource_class_entity_p-processor-->filter_resource_class_entity_p-publish
	    filter_resource_class_entity_p-publish-->|Replace|filter_resource_class_entity_s-subject
	    filter_resource_class_entity_s-subject-->|AllRecordBatches|select_resource_class_entity_p-subscribe
	    select_resource_class_entity_p-subscribe-->select_resource_class_entity_p-processor
	    select_resource_class_entity_p-processor-->select_resource_class_entity_p-publish
	    select_resource_class_entity_p-publish-->|Replace|select_resource_class_entity_s-subject
	end
	extract_owl_r-rt-->filter_resources_class_entity_t
	comparator_resource_class_entity_p-processor@{shape: rect, label: Select}
	comparator_resource_class_entity_p-publish@{shape: fork}
	comparator_resource_class_entity_p-subscribe@{shape: diamond, label: All}
	comparator_resource_class_entity_s-subject@{shape: doc, label: comparator_resource_class_entity_s}
	filter_resource_class_entity_p-processor@{shape: rect, label: Filter}
	filter_resource_class_entity_p-publish@{shape: fork}
	filter_resource_class_entity_p-subscribe@{shape: diamond, label: All}
	filter_resource_class_entity_s-subject@{shape: doc, label: filter_resource_class_entity_s}
	select_resource_class_entity_p-processor@{shape: rect, label: Select}
	select_resource_class_entity_p-publish@{shape: fork}
	select_resource_class_entity_p-subscribe@{shape: diamond, label: All}
	select_resource_class_entity_s-subject@{shape: doc, label: select_resource_class_entity_s}
	%% ------------------------------------------------------------------------------
	%% Join Owl:Class with Owl:Class on objects
	%% ------------------------------------------------------------------------------
	subgraph join_class_on_objects_t
	    select_resource_class_entity_s-subject-.->|AllRecordBatches|join_objects_class_entity_p-subscribe
	    merge_object_property_class_named_individual_pivot_s-subject-.->|AllRecordBatches|join_objects_class_entity_p-subscribe
	    join_objects_class_entity_p-subscribe-->join_objects_class_entity_p-processor
	    join_objects_class_entity_p-processor-->join_objects_class_entity_p-publish
	    join_objects_class_entity_p-publish-->|Replace|join_objects_class_entity_s-subject
	    join_objects_class_entity_s-subject-->|AllRecordBatches|select_objects_class_entity_p-subscribe
	    select_objects_class_entity_p-subscribe-->select_objects_class_entity_p-processor
	    select_objects_class_entity_p-processor-->select_objects_class_entity_p-publish
	    select_objects_class_entity_p-publish-->|Extend|select_objects_class_entity_s-subject
	end
	extract_owl_r-rt-->join_class_on_objects_t
	join_objects_class_entity_p-processor@{shape: rect, label: Join}
	join_objects_class_entity_p-publish@{shape: fork}
	join_objects_class_entity_p-subscribe@{shape: diamond, label: All}
	join_objects_class_entity_s-subject@{shape: doc, label: join_objects_class_entity_s}
	select_objects_class_entity_p-processor@{shape: rect, label: Select}
	select_objects_class_entity_p-publish@{shape: fork}
	select_objects_class_entity_p-subscribe@{shape: diamond, label: All}
	select_objects_class_entity_s-subject@{shape: doc, label: select_objects_class_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter Owl:Class for rdf:literal objects
	%% ------------------------------------------------------------------------------
	subgraph filter_literals_class_entity_t
	    select_resource_class_entity_s-subject-.->|Empty|pause_filter_literals_class_entity_p-subscribe
	    select_resource_object_property_entity_s-subject-.->|Empty|pause_filter_literals_class_entity_p-subscribe
	    pause_filter_literals_class_entity_p-subscribe-->pause_filter_literals_class_entity_p-processor
	    pause_filter_literals_class_entity_p-processor-->pause_filter_literals_class_entity_p-publish
	    pause_filter_literals_class_entity_p-publish-->|None|pause_filter_literals_class_entity_s-subject
	    select_predicates_class_entity_s-subject-.->|AllRecordBatches|comparator_literal_class_entity_p-subscribe
	    comparator_literal_class_entity_p-subscribe-->comparator_literal_class_entity_p-processor
	    comparator_literal_class_entity_p-processor-->comparator_literal_class_entity_p-publish
	    comparator_literal_class_entity_p-publish-->|Replace|comparator_literal_class_entity_s-subject
	    comparator_literal_class_entity_s-subject-->|AllRecordBatches|filter_literal_class_entity_p-subscribe
	    filter_literal_class_entity_p-subscribe-->filter_literal_class_entity_p-processor
	    filter_literal_class_entity_p-processor-->filter_literal_class_entity_p-publish
	    filter_literal_class_entity_p-publish-->|Replace|filter_literal_class_entity_s-subject
	    filter_literal_class_entity_s-subject-->|AllRecordBatches|select_literal_class_entity_p-subscribe
	    select_literal_class_entity_p-subscribe-->select_literal_class_entity_p-processor
	    select_literal_class_entity_p-processor-->select_literal_class_entity_p-publish
	    select_literal_class_entity_p-publish-->|Extend|select_objects_class_entity_s-subject
	end
	extract_owl_r-rt-->filter_literals_class_entity_t
	pause_filter_literals_class_entity_p-processor@{shape: rect, label: ProcessorEcho}
	pause_filter_literals_class_entity_p-publish@{shape: fork}
	pause_filter_literals_class_entity_p-subscribe@{shape: diamond, label: Any}
	pause_filter_literals_class_entity_s-subject@{shape: doc, label: pause_filter_literals_class_entity_s}
	comparator_literal_class_entity_p-processor@{shape: rect, label: Select}
	comparator_literal_class_entity_p-publish@{shape: fork}
	comparator_literal_class_entity_p-subscribe@{shape: diamond, label: All}
	comparator_literal_class_entity_s-subject@{shape: doc, label: comparator_literal_class_entity_s}
	filter_literal_class_entity_p-processor@{shape: rect, label: Filter}
	filter_literal_class_entity_p-publish@{shape: fork}
	filter_literal_class_entity_p-subscribe@{shape: diamond, label: All}
	filter_literal_class_entity_s-subject@{shape: doc, label: filter_literal_class_entity_s}
	select_literal_class_entity_p-processor@{shape: rect, label: Select}
	select_literal_class_entity_p-publish@{shape: fork}
	select_literal_class_entity_p-subscribe@{shape: diamond, label: All}
	%% ------------------------------------------------------------------------------
	%% Join Owl:AnnotationProperty with Owl:ObjectProperty on predicates
	%% ------------------------------------------------------------------------------
	subgraph join_object_property_on_predicates_t
	    select_object_property_entity_s-subject-.->|AllRecordBatches|join_predicates_object_property_entity_p-subscribe
	    select_annotation_property_pivot_s-subject-.->|AllRecordBatches|join_predicates_object_property_entity_p-subscribe
	    join_predicates_object_property_entity_p-subscribe-->join_predicates_object_property_entity_p-processor
	    join_predicates_object_property_entity_p-processor-->join_predicates_object_property_entity_p-publish
	    join_predicates_object_property_entity_p-publish-->|Replace|join_predicates_object_property_entity_s-subject
	    join_predicates_object_property_entity_s-subject-->|AllRecordBatches|select_predicates_object_property_entity_p-subscribe
	    select_predicates_object_property_entity_p-subscribe-->select_predicates_object_property_entity_p-processor
	    select_predicates_object_property_entity_p-processor-->select_predicates_object_property_entity_p-publish
	    select_predicates_object_property_entity_p-publish-->|Replace|select_predicates_object_property_entity_s-subject
	end
	extract_owl_r-rt-->join_object_property_on_predicates_t
	join_predicates_object_property_entity_p-processor@{shape: rect, label: Join}
	join_predicates_object_property_entity_p-publish@{shape: fork}
	join_predicates_object_property_entity_p-subscribe@{shape: diamond, label: All}
	join_predicates_object_property_entity_s-subject@{shape: doc, label: join_predicates_object_property_entity_s}
	select_predicates_object_property_entity_p-processor@{shape: rect, label: Select}
	select_predicates_object_property_entity_p-publish@{shape: fork}
	select_predicates_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_predicates_object_property_entity_s-subject@{shape: doc, label: select_predicates_object_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Filter Owl:ObjectProperty for rdf:resource objects
	%% ------------------------------------------------------------------------------
	subgraph filter_resources_object_property_entity_t
	    select_predicates_object_property_entity_s-subject-.->|AllRecordBatches|comparator_resource_object_property_entity_p-subscribe
	    comparator_resource_object_property_entity_p-subscribe-->comparator_resource_object_property_entity_p-processor
	    comparator_resource_object_property_entity_p-processor-->comparator_resource_object_property_entity_p-publish
	    comparator_resource_object_property_entity_p-publish-->|Replace|comparator_resource_object_property_entity_s-subject
	    comparator_resource_object_property_entity_s-subject-->|AllRecordBatches|filter_resource_object_property_entity_p-subscribe
	    filter_resource_object_property_entity_p-subscribe-->filter_resource_object_property_entity_p-processor
	    filter_resource_object_property_entity_p-processor-->filter_resource_object_property_entity_p-publish
	    filter_resource_object_property_entity_p-publish-->|Replace|filter_resource_object_property_entity_s-subject
	    filter_resource_object_property_entity_s-subject-->|AllRecordBatches|select_resource_object_property_entity_p-subscribe
	    select_resource_object_property_entity_p-subscribe-->select_resource_object_property_entity_p-processor
	    select_resource_object_property_entity_p-processor-->select_resource_object_property_entity_p-publish
	    select_resource_object_property_entity_p-publish-->|Replace|select_resource_object_property_entity_s-subject
	end
	extract_owl_r-rt-->filter_resources_object_property_entity_t
	comparator_resource_object_property_entity_p-processor@{shape: rect, label: Select}
	comparator_resource_object_property_entity_p-publish@{shape: fork}
	comparator_resource_object_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_resource_object_property_entity_s-subject@{shape: doc, label: comparator_resource_object_property_entity_s}
	filter_resource_object_property_entity_p-processor@{shape: rect, label: Filter}
	filter_resource_object_property_entity_p-publish@{shape: fork}
	filter_resource_object_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_resource_object_property_entity_s-subject@{shape: doc, label: filter_resource_object_property_entity_s}
	select_resource_object_property_entity_p-processor@{shape: rect, label: Select}
	select_resource_object_property_entity_p-publish@{shape: fork}
	select_resource_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_resource_object_property_entity_s-subject@{shape: doc, label: select_resource_object_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Join Owl:Class with Owl:ObjectProperty on objects
	%% ------------------------------------------------------------------------------
	subgraph join_object_property_on_objects_t
	    select_resource_object_property_entity_s-subject-.->|AllRecordBatches|join_objects_object_property_entity_p-subscribe
	    merge_object_property_class_named_individual_pivot_s-subject-.->|AllRecordBatches|join_objects_object_property_entity_p-subscribe
	    join_objects_object_property_entity_p-subscribe-->join_objects_object_property_entity_p-processor
	    join_objects_object_property_entity_p-processor-->join_objects_object_property_entity_p-publish
	    join_objects_object_property_entity_p-publish-->|Replace|join_objects_object_property_entity_s-subject
	    join_objects_object_property_entity_s-subject-->|AllRecordBatches|select_objects_object_property_entity_p-subscribe
	    select_objects_object_property_entity_p-subscribe-->select_objects_object_property_entity_p-processor
	    select_objects_object_property_entity_p-processor-->select_objects_object_property_entity_p-publish
	    select_objects_object_property_entity_p-publish-->|Extend|select_objects_object_property_entity_s-subject
	end
	extract_owl_r-rt-->join_object_property_on_objects_t
	join_objects_object_property_entity_p-processor@{shape: rect, label: Join}
	join_objects_object_property_entity_p-publish@{shape: fork}
	join_objects_object_property_entity_p-subscribe@{shape: diamond, label: All}
	join_objects_object_property_entity_s-subject@{shape: doc, label: join_objects_object_property_entity_s}
	select_objects_object_property_entity_p-processor@{shape: rect, label: Select}
	select_objects_object_property_entity_p-publish@{shape: fork}
	select_objects_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_objects_object_property_entity_s-subject@{shape: doc, label: select_objects_object_property_entity_s}
	%% ------------------------------------------------------------------------------
	%% Merge Owl:OjectProperty rdf:literal and rdf:resource tables
	%% ------------------------------------------------------------------------------
	subgraph filter_literals_object_property_entity_t
	    select_resource_class_entity_s-subject-.->|Empty|pause_filter_literals_object_property_entity_p-subscribe
	    select_resource_object_property_entity_s-subject-.->|Empty|pause_filter_literals_object_property_entity_p-subscribe
	    pause_filter_literals_object_property_entity_p-subscribe-->pause_filter_literals_object_property_entity_p-processor
	    pause_filter_literals_object_property_entity_p-processor-->pause_filter_literals_object_property_entity_p-publish
	    pause_filter_literals_object_property_entity_p-publish-->|None|pause_filter_literals_object_property_entity_s-subject
	    select_predicates_object_property_entity_s-subject-.->|AllRecordBatches|comparator_literal_object_property_entity_p-subscribe
	    comparator_literal_object_property_entity_p-subscribe-->comparator_literal_object_property_entity_p-processor
	    comparator_literal_object_property_entity_p-processor-->comparator_literal_object_property_entity_p-publish
	    comparator_literal_object_property_entity_p-publish-->|Replace|comparator_literal_object_property_entity_s-subject
	    comparator_literal_object_property_entity_s-subject-->|AllRecordBatches|filter_literal_object_property_entity_p-subscribe
	    filter_literal_object_property_entity_p-subscribe-->filter_literal_object_property_entity_p-processor
	    filter_literal_object_property_entity_p-processor-->filter_literal_object_property_entity_p-publish
	    filter_literal_object_property_entity_p-publish-->|Replace|filter_literal_object_property_entity_s-subject
	    filter_literal_object_property_entity_s-subject-->|AllRecordBatches|select_literal_object_property_entity_p-subscribe
	    select_literal_object_property_entity_p-subscribe-->select_literal_object_property_entity_p-processor
	    select_literal_object_property_entity_p-processor-->select_literal_object_property_entity_p-publish
	    select_literal_object_property_entity_p-publish-->|Extend|select_objects_object_property_entity_s-subject
	end
	extract_owl_r-rt-->filter_literals_object_property_entity_t
	pause_filter_literals_object_property_entity_p-processor@{shape: rect, label: ProcessorEcho}
	pause_filter_literals_object_property_entity_p-publish@{shape: fork}
	pause_filter_literals_object_property_entity_p-subscribe@{shape: diamond, label: Any}
	pause_filter_literals_object_property_entity_s-subject@{shape: doc, label: pause_filter_literals_object_property_entity_s}
	comparator_literal_object_property_entity_p-processor@{shape: rect, label: Select}
	comparator_literal_object_property_entity_p-publish@{shape: fork}
	comparator_literal_object_property_entity_p-subscribe@{shape: diamond, label: All}
	comparator_literal_object_property_entity_s-subject@{shape: doc, label: comparator_literal_object_property_entity_s}
	filter_literal_object_property_entity_p-processor@{shape: rect, label: Filter}
	filter_literal_object_property_entity_p-publish@{shape: fork}
	filter_literal_object_property_entity_p-subscribe@{shape: diamond, label: All}
	filter_literal_object_property_entity_s-subject@{shape: doc, label: filter_literal_object_property_entity_s}
	select_literal_object_property_entity_p-processor@{shape: rect, label: Select}
	select_literal_object_property_entity_p-publish@{shape: fork}
	select_literal_object_property_entity_p-subscribe@{shape: diamond, label: All}
	%% ------------------------------------------------------------------------------
	%% Apply embedding template to Owl:Class
	%% ------------------------------------------------------------------------------
	subgraph apply_embedding_template_class_t
	    select_objects_class_entity_s-subject-.->|AllRecordBatches|concat_cols_class_entity_p-subscribe
	    concat_cols_class_entity_p-subscribe-->concat_cols_class_entity_p-processor
	    concat_cols_class_entity_p-processor-->concat_cols_class_entity_p-publish
	    concat_cols_class_entity_p-publish-->|Replace|concat_cols_class_entity_s-subject
	    concat_cols_class_entity_s-subject-->|AllRecordBatches|select_cols_class_entity_p-subscribe
	    select_cols_class_entity_p-subscribe-->select_cols_class_entity_p-processor
	    select_cols_class_entity_p-processor-->select_cols_class_entity_p-publish
	    select_cols_class_entity_p-publish-->|Replace|select_cols_class_entity_s-subject
	    select_cols_class_entity_s-subject-->|AllRecordBatches|list_rows_class_entity_p-subscribe
	    list_rows_class_entity_p-subscribe-->list_rows_class_entity_p-processor
	    list_rows_class_entity_p-processor-->list_rows_class_entity_p-publish
	    list_rows_class_entity_p-publish-->|Replace|list_rows_class_entity_s-subject
	    list_rows_class_entity_s-subject-->|AllRecordBatches|select_rows_class_entity_p-subscribe
	    select_rows_class_entity_p-subscribe-->select_rows_class_entity_p-processor
	    select_rows_class_entity_p-processor-->select_rows_class_entity_p-publish
	    select_rows_class_entity_p-publish-->|Replace|select_rows_class_entity_s-subject
	    select_rows_class_entity_s-subject-->|AllRecordBatches|apply_template_class_entity_p-subscribe
	    apply_template_class_entity_p-subscribe-->apply_template_class_entity_p-processor
	    apply_template_class_entity_p-processor-->apply_template_class_entity_p-publish
	    apply_template_class_entity_p-publish-->|Extend|Documents-subject
	end
	extract_owl_r-rt-->apply_embedding_template_class_t
	concat_cols_class_entity_p-processor@{shape: rect, label: Select}
	concat_cols_class_entity_p-publish@{shape: fork}
	concat_cols_class_entity_p-subscribe@{shape: diamond, label: All}
	concat_cols_class_entity_s-subject@{shape: doc, label: concat_cols_class_entity_s}
	select_cols_class_entity_p-processor@{shape: rect, label: Select}
	select_cols_class_entity_p-publish@{shape: fork}
	select_cols_class_entity_p-subscribe@{shape: diamond, label: All}
	select_cols_class_entity_s-subject@{shape: doc, label: select_cols_class_entity_s}
	list_rows_class_entity_p-processor@{shape: rect, label: GroupBy}
	list_rows_class_entity_p-publish@{shape: fork}
	list_rows_class_entity_p-subscribe@{shape: diamond, label: All}
	list_rows_class_entity_s-subject@{shape: doc, label: list_rows_class_entity_s}
	select_rows_class_entity_p-processor@{shape: rect, label: Select}
	select_rows_class_entity_p-publish@{shape: fork}
	select_rows_class_entity_p-subscribe@{shape: diamond, label: All}
	select_rows_class_entity_s-subject@{shape: doc, label: select_rows_class_entity_s}
	apply_template_class_entity_p-processor@{shape: rect, label: Select}
	apply_template_class_entity_p-publish@{shape: fork}
	apply_template_class_entity_p-subscribe@{shape: diamond, label: All}
	Documents-subject@{shape: doc, label: Documents}
	%% ------------------------------------------------------------------------------
	%% Apply embedding template to Owl:ObjectProperty 
	%% ------------------------------------------------------------------------------
	subgraph apply_embedding_template_object_property_t
	    select_objects_object_property_entity_s-subject-.->|AllRecordBatches|concat_cols_object_property_entity_p-subscribe
	    concat_cols_object_property_entity_p-subscribe-->concat_cols_object_property_entity_p-processor
	    concat_cols_object_property_entity_p-processor-->concat_cols_object_property_entity_p-publish
	    concat_cols_object_property_entity_p-publish-->|Replace|concat_cols_object_property_entity_s-subject
	    concat_cols_object_property_entity_s-subject-->|AllRecordBatches|select_cols_object_property_entity_p-subscribe
	    select_cols_object_property_entity_p-subscribe-->select_cols_object_property_entity_p-processor
	    select_cols_object_property_entity_p-processor-->select_cols_object_property_entity_p-publish
	    select_cols_object_property_entity_p-publish-->|Replace|select_cols_object_property_entity_s-subject
	    select_cols_object_property_entity_s-subject-->|AllRecordBatches|list_rows_object_property_entity_p-subscribe
	    list_rows_object_property_entity_p-subscribe-->list_rows_object_property_entity_p-processor
	    list_rows_object_property_entity_p-processor-->list_rows_object_property_entity_p-publish
	    list_rows_object_property_entity_p-publish-->|Replace|list_rows_object_property_entity_s-subject
	    list_rows_object_property_entity_s-subject-->|AllRecordBatches|select_rows_object_property_entity_p-subscribe
	    select_rows_object_property_entity_p-subscribe-->select_rows_object_property_entity_p-processor
	    select_rows_object_property_entity_p-processor-->select_rows_object_property_entity_p-publish
	    select_rows_object_property_entity_p-publish-->|Replace|select_rows_object_property_entity_s-subject
	    select_rows_object_property_entity_s-subject-->|AllRecordBatches|apply_template_object_property_entity_p-subscribe
	    apply_template_object_property_entity_p-subscribe-->apply_template_object_property_entity_p-processor
	    apply_template_object_property_entity_p-processor-->apply_template_object_property_entity_p-publish
	    apply_template_object_property_entity_p-publish-->|Extend|Documents-subject
	end
	extract_owl_r-rt-->apply_embedding_template_object_property_t
	concat_cols_object_property_entity_p-processor@{shape: rect, label: Select}
	concat_cols_object_property_entity_p-publish@{shape: fork}
	concat_cols_object_property_entity_p-subscribe@{shape: diamond, label: All}
	concat_cols_object_property_entity_s-subject@{shape: doc, label: concat_cols_object_property_entity_s}
	select_cols_object_property_entity_p-processor@{shape: rect, label: Select}
	select_cols_object_property_entity_p-publish@{shape: fork}
	select_cols_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_cols_object_property_entity_s-subject@{shape: doc, label: select_cols_object_property_entity_s}
	list_rows_object_property_entity_p-processor@{shape: rect, label: GroupBy}
	list_rows_object_property_entity_p-publish@{shape: fork}
	list_rows_object_property_entity_p-subscribe@{shape: diamond, label: All}
	list_rows_object_property_entity_s-subject@{shape: doc, label: list_rows_object_property_entity_s}
	select_rows_object_property_entity_p-processor@{shape: rect, label: Select}
	select_rows_object_property_entity_p-publish@{shape: fork}
	select_rows_object_property_entity_p-subscribe@{shape: diamond, label: All}
	select_rows_object_property_entity_s-subject@{shape: doc, label: select_rows_object_property_entity_s}
	apply_template_object_property_entity_p-processor@{shape: rect, label: Select}
	apply_template_object_property_entity_p-publish@{shape: fork}
	apply_template_object_property_entity_p-subscribe@{shape: diamond, label: All}
	%% ------------------------------------------------------------------------------"#
    }
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    UserScript["UserScript"] {
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }
    ParseOwl["ParseOwl"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    extract_owl_p["extract_owl_p"] {
        Boolean cpu "false"
        Utf8 format "Owl"
        Utf8 lhs_name "UserScript"
        List-Utf8 lhs_values "['bytes']"
        Utf8 operator "ExtractXML"
        Utf8 lhs_stream "Accumulate"
    }
    comparator_ontology_entity_p["comparator_ontology_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#Ontology']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_ontology_entity_p["filter_ontology_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_ontology_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_ontology_entity_p["select_ontology_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_ontology_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_ontology_entity_s["select_ontology_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    comparator_annotation_property_entity_p["comparator_annotation_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#AnnotationProperty']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_annotation_property_entity_p["filter_annotation_property_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_annotation_property_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_annotation_property_entity_p["select_annotation_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_annotation_property_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_annotation_property_entity_s["select_annotation_property_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    comparator_datatype_property_entity_p["comparator_datatype_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#DatatypeProperty']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_datatype_property_entity_p["filter_datatype_property_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_datatype_property_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_datatype_property_entity_p["select_datatype_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_datatype_property_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_datatype_property_entity_s["select_datatype_property_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    comparator_class_entity_p["comparator_class_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#Class']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_class_entity_p["filter_class_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_class_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_class_entity_p["select_class_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_class_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_class_entity_s["select_class_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    comparator_object_property_entity_p["comparator_object_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#ObjectProperty']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_object_property_entity_p["filter_object_property_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_object_property_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_object_property_entity_p["select_object_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_object_property_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_object_property_entity_s["select_object_property_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    comparator_named_individual_entity_p["comparator_named_individual_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#NamedIndividual']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_named_individual_entity_p["filter_named_individual_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_named_individual_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_named_individual_entity_p["select_named_individual_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_named_individual_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_named_individual_entity_s["select_named_individual_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    comparator_axiom_entity_p["comparator_axiom_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','cmp']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2002/07/owl#Axiom']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','cmp']"
        Boolean cpu "false"
        Utf8 lhs_name "ParseOwl"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_axiom_entity_p["filter_axiom_entity_p"] {
        List-Utf8 cmp_columns "['cmp']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_axiom_entity_s"
        List-Utf8 lhs_values "['entity']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_axiom_entity_p["select_axiom_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_axiom_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_axiom_entity_s["select_axiom_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
    }
    coalesce_annotation_property_entity_p["coalesce_annotation_property_entity_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
    comparator_predicate_annotation_property_entity_p["comparator_predicate_annotation_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','rdfs_label','obo_IAO_0000115']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2000/01/rdf-schema#label','http://purl.obolibrary.org/obo/IAO_0000115']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','rdfs_label','obo_IAO_0000115']"
        Boolean cpu "false"
        Utf8 lhs_name "coalesce_annotation_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_predicate_annotation_property_entity_p["filter_predicate_annotation_property_entity_p"] {
        List-Utf8 cmp_columns "['rdfs_label','obo_IAO_0000115']"
        List-Utf8 cmp_operators "['Like','Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_predicate_annotation_property_entity_s"
        List-Utf8 lhs_values "['predicate','predicate']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_predicate_annotation_property_entity_p["select_predicate_annotation_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_predicate_annotation_property_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	pivot_annotation_property_entity_p["pivot_annotation_property_entity_p"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "select_predicate_annotation_property_entity_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	pivot_annotation_property_entity_s["pivot_annotation_property_entity_s"] {
	    Utf8 subject
	    Utf8 http://purl.obolibrary.org/obo/IAO_0000115-object-First
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First
	}
    coalesce_annotation_property_pivot_p["coalesce_annotation_property_pivot_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
	group_by_annotation_property_pivot_p["group_by_annotation_property_pivot_p"] {
	    List-Utf8 agg_columns "['http://www.w3.org/2000/01/rdf-schema#label-object-First','http://purl.obolibrary.org/obo/IAO_0000115-object-First']"
	    List-Utf8 agg_operators "['First','First']"
	    Boolean cpu "false"
	    Utf8 lhs_name "coalesce_annotation_property_pivot_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "GroupBy"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	group_by_annotation_property_pivot_s["group_by_annotation_property_pivot_s"] {
	    Utf8 subject
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First-First
	    Utf8 http://purl.obolibrary.org/obo/IAO_0000115-object-First-First
	}
    select_annotation_property_pivot_p["select_annotation_property_pivot_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "group_by_annotation_property_pivot_s"
        List-Utf8 as_columns "['uri','rdfs_label','obo_IAO_0000115']"
        List-Utf8 lhs_values "['subject','http://www.w3.org/2000/01/rdf-schema#label-object-First-First','http://purl.obolibrary.org/obo/IAO_0000115-object-First-First']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	select_annotation_property_pivot_s["select_annotation_property_pivot_s"] {
	    Utf8 uri
	    Utf8 rdfs_label
	    Utf8 obo_IAO_0000115
	}
    comparator_predicate_class_entity_p["comparator_predicate_class_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','rdfs_label','obo_IAO_0000115']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2000/01/rdf-schema#label','http://purl.obolibrary.org/obo/IAO_0000115']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','rdfs_label','obo_IAO_0000115']"
        Boolean cpu "false"
        Utf8 lhs_name "coalesce_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_predicate_class_entity_p["filter_predicate_class_entity_p"] {
        List-Utf8 cmp_columns "['rdfs_label','obo_IAO_0000115']"
        List-Utf8 cmp_operators "['Like','Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_predicate_class_entity_s"
        List-Utf8 lhs_values "['predicate','predicate']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_predicate_class_entity_p["select_predicate_class_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_predicate_class_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	pivot_class_entity_p["pivot_class_entity_p"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "select_predicate_class_entity_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	pivot_class_entity_s["pivot_class_entity_s"] {
	    Utf8 subject
	    Utf8 http://purl.obolibrary.org/obo/IAO_0000115-object-First
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First
	}
    coalesce_class_pivot_p["coalesce_class_pivot_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
	group_by_class_pivot_p["group_by_class_pivot_p"] {
	    List-Utf8 agg_columns "['http://www.w3.org/2000/01/rdf-schema#label-object-First','http://purl.obolibrary.org/obo/IAO_0000115-object-First']"
	    List-Utf8 agg_operators "['First','First']"
	    Boolean cpu "false"
	    Utf8 lhs_name "coalesce_class_pivot_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "GroupBy"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	group_by_class_pivot_s["group_by_class_pivot_s"] {
	    Utf8 subject
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First-First
	    Utf8 http://purl.obolibrary.org/obo/IAO_0000115-object-First-First
	}
    select_class_pivot_p["select_class_pivot_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "group_by_class_pivot_s"
        List-Utf8 as_columns "['uri','rdfs_label','obo_IAO_0000115']"
        List-Utf8 lhs_values "['subject','http://www.w3.org/2000/01/rdf-schema#label-object-First-First','http://purl.obolibrary.org/obo/IAO_0000115-object-First-First']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    coalesce_object_property_entity_p["coalesce_object_property_entity_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
    comparator_predicate_object_property_entity_p["comparator_predicate_object_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','rdfs_label','obo_IAO_0000115']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2000/01/rdf-schema#label','http://purl.obolibrary.org/obo/IAO_0000115']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','rdfs_label','obo_IAO_0000115']"
        Boolean cpu "false"
        Utf8 lhs_name "coalesce_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_predicate_object_property_entity_p["filter_predicate_object_property_entity_p"] {
        List-Utf8 cmp_columns "['rdfs_label','obo_IAO_0000115']"
        List-Utf8 cmp_operators "['Like','Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_predicate_object_property_entity_s"
        List-Utf8 lhs_values "['predicate','predicate']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_predicate_object_property_entity_p["select_predicate_object_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_predicate_object_property_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	pivot_object_property_entity_p["pivot_object_property_entity_p"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "select_predicate_object_property_entity_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	pivot_object_property_entity_s["pivot_object_property_entity_s"] {
	    Utf8 subject
	    Utf8 http://purl.obolibrary.org/obo/IAO_0000115-object-First
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First
	}
    coalesce_object_property_pivot_p["coalesce_object_property_pivot_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
	group_by_object_property_pivot_p["group_by_object_property_pivot_p"] {
	    List-Utf8 agg_columns "['http://www.w3.org/2000/01/rdf-schema#label-object-First','http://purl.obolibrary.org/obo/IAO_0000115-object-First']"
	    List-Utf8 agg_operators "['First','First']"
	    Boolean cpu "false"
	    Utf8 lhs_name "coalesce_object_property_pivot_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "GroupBy"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	group_by_object_property_pivot_s["group_by_object_property_pivot_s"] {
	    Utf8 subject
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First-First
	    Utf8 http://purl.obolibrary.org/obo/IAO_0000115-object-First-First
	}
    select_object_property_pivot_p["select_object_property_pivot_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "group_by_object_property_pivot_s"
        List-Utf8 as_columns "['uri','rdfs_label','obo_IAO_0000115']"
        List-Utf8 lhs_values "['subject','http://www.w3.org/2000/01/rdf-schema#label-object-First-First','http://purl.obolibrary.org/obo/IAO_0000115-object-First-First']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    coalesce_named_individual_entity_p["coalesce_named_individual_entity_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
    comparator_predicate_named_individual_entity_p["comparator_predicate_named_individual_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','rdfs_label']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','http://www.w3.org/2000/01/rdf-schema#label']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','rdfs_label']"
        Boolean cpu "false"
        Utf8 lhs_name "coalesce_named_individual_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_predicate_named_individual_entity_p["filter_predicate_named_individual_entity_p"] {
        List-Utf8 cmp_columns "['rdfs_label']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_predicate_named_individual_entity_s"
        List-Utf8 lhs_values "['predicate']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_predicate_named_individual_entity_p["select_predicate_named_individual_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_predicate_named_individual_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	pivot_named_individual_entity_p["pivot_named_individual_entity_p"] {
	    List-Utf8 agg_columns "['object']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "false"
	    List-Utf8 default_values "['']"
	    Utf8 lhs_name "select_predicate_named_individual_entity_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "Pivot"
	    List-Utf8 pvt_columns "['predicate']"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	pivot_named_individual_entity_s["pivot_named_individual_entity_s"] {
	    Utf8 subject
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First
	}
    coalesce_named_individual_pivot_p["coalesce_named_individual_pivot_p"] {
        Int64 fetch "512"
        Utf8 summary_format "None"
    }
	group_by_named_individual_pivot_p["group_by_named_individual_pivot_p"] {
	    List-Utf8 agg_columns "['http://www.w3.org/2000/01/rdf-schema#label-object-First']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "false"
	    Utf8 lhs_name "coalesce_named_individual_pivot_s"
	    List-Utf8 lhs_values "['subject']"
	    Utf8 operator "GroupBy"
	    Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
	}
	group_by_named_individual_pivot_s["group_by_named_individual_pivot_s"] {
	    Utf8 subject
	    Utf8 http://www.w3.org/2000/01/rdf-schema#label-object-First-First
	}
    select_named_individual_pivot_p["select_named_individual_pivot_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "group_by_named_individual_pivot_s"
	    List-Utf8 as_columns "['uri','rdfs_label','obo_IAO_0000115']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8']"
	    List-Utf8 column_operators "['None','None','String']"
        List-Utf8 lhs_values "['subject','http://www.w3.org/2000/01/rdf-schema#label-object-First-First','obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	merge_object_property_class_named_individual_pivot_s["merge_object_property_class_named_individual_pivot_s"] {
	    Utf8 uri
	    Utf8 rdfs_label
	    Utf8 obo_IAO_0000115
	}
	join_predicates_class_entity_p["join_predicates_class_entity_p"] {
	    Boolean cpu "false"
	    Utf8 operator "Join"
	    Utf8 lhs_name "select_class_entity_s"
	    Utf8 lhs_fk "predicate"
	    Utf8 lhs_pk "predicate"
	    Utf8 rhs_name "select_annotation_property_pivot_s"
	    Utf8 rhs_fk "uri"
	    Utf8 rhs_pk "uri"
	    Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
	}
    select_predicates_class_entity_p["select_predicates_class_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "join_predicates_class_entity_s"
        List-Utf8 as_columns "['','','','','','','predicate_rdfs_label','predicate_obo_IAO_0000115']"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','rdfs_label','obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	select_predicates_class_entity_s["select_predicates_class_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
	    Utf8 predicate_rdfs_label
	    Utf8 predicate_obo_IAO_0000115
	}
    comparator_resource_class_entity_p["comparator_resource_class_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','','','resource']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','','','http://%']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','resource']"
        Boolean cpu "false"
        Utf8 lhs_name "select_predicates_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_resource_class_entity_p["filter_resource_class_entity_p"] {
        List-Utf8 cmp_columns "['resource']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_resource_class_entity_s"
        List-Utf8 lhs_values "['object']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_resource_class_entity_p["select_resource_class_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_resource_class_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_resource_class_entity_s["select_resource_class_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
	    Utf8 predicate_rdfs_label
	    Utf8 predicate_obo_IAO_0000115
	}
	join_objects_class_entity_p["join_objects_class_entity_p"] {
	    Boolean cpu "false"
	    Utf8 operator "Join"
	    Utf8 lhs_name "select_resource_class_entity_s"
	    Utf8 lhs_fk "object"
	    Utf8 lhs_pk "object"
	    Utf8 rhs_name "merge_object_property_class_named_individual_pivot_s"
	    Utf8 rhs_fk "uri"
	    Utf8 rhs_pk "uri"
	    Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
	}
    select_objects_class_entity_p["select_objects_class_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "join_objects_class_entity_s"
        List-Utf8 as_columns "['','','','','','','object_rdfs_label','object_obo_IAO_0000115']"
        List-Utf8 lhs_values "['entity','subject','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','rdfs_label','obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	select_objects_class_entity_s["select_objects_class_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 graph
        Utf8 dataset
	    Utf8 predicate_rdfs_label
	    Utf8 predicate_obo_IAO_0000115
	    Utf8 object_rdfs_label
	    Utf8 object_obo_IAO_0000115
	}
    comparator_literal_class_entity_p["comparator_literal_class_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','','','resource']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','','','http://%']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','resource']"
        Boolean cpu "false"
        Utf8 lhs_name "select_predicates_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_literal_class_entity_p["filter_literal_class_entity_p"] {
        List-Utf8 cmp_columns "['resource']"
        List-Utf8 cmp_operators "['NotLike']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_literal_class_entity_s"
        List-Utf8 lhs_values "['object']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_literal_class_entity_p["select_literal_class_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_literal_class_entity_s"
        List-Utf8 as_columns "['','','','','','','object_rdfs_label','']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','String']"
        List-Utf8 lhs_values "['entity','subject','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','object','object_obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	join_predicates_object_property_entity_p["join_predicates_object_property_entity_p"] {
	    Boolean cpu "false"
	    Utf8 operator "Join"
	    Utf8 lhs_name "select_object_property_entity_s"
	    Utf8 lhs_fk "predicate"
	    Utf8 lhs_pk "predicate"
	    Utf8 rhs_name "select_annotation_property_pivot_s"
	    Utf8 rhs_fk "uri"
	    Utf8 rhs_pk "uri"
	    Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
	}
    select_predicates_object_property_entity_p["select_predicates_object_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "join_predicates_object_property_entity_s"
        List-Utf8 as_columns "['','','','','','','predicate_rdfs_label','predicate_obo_IAO_0000115']"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','rdfs_label','obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	select_predicates_object_property_entity_s["select_predicates_object_property_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
	    Utf8 predicate_rdfs_label
	    Utf8 predicate_obo_IAO_0000115
	}
    comparator_resource_object_property_entity_p["comparator_resource_object_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','','','resource']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','','','http://%']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','resource']"
        Boolean cpu "false"
        Utf8 lhs_name "select_predicates_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_resource_object_property_entity_p["filter_resource_object_property_entity_p"] {
        List-Utf8 cmp_columns "['resource']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_resource_object_property_entity_s"
        List-Utf8 lhs_values "['object']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_resource_object_property_entity_p["select_resource_object_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_resource_object_property_entity_s"
        List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_resource_object_property_entity_s["select_resource_object_property_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 predicate
        Utf8 object
        Utf8 graph
        Utf8 dataset
	    Utf8 predicate_rdfs_label
	    Utf8 predicate_obo_IAO_0000115
	}
	join_objects_object_property_entity_p["join_objects_object_property_entity_p"] {
	    Boolean cpu "false"
	    Utf8 operator "Join"
	    Utf8 lhs_name "select_resource_object_property_entity_s"
	    Utf8 lhs_fk "object"
	    Utf8 lhs_pk "object"
	    Utf8 rhs_name "merge_object_property_class_named_individual_pivot_s"
	    Utf8 rhs_fk "uri"
	    Utf8 rhs_pk "uri"
	    Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
	}
    select_objects_object_property_entity_p["select_objects_object_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "join_objects_object_property_entity_s"
        List-Utf8 as_columns "['','','','','','','object_rdfs_label','object_obo_IAO_0000115']"
        List-Utf8 lhs_values "['entity','subject','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','rdfs_label','obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	select_objects_object_property_entity_s["select_objects_object_property_entity_s"] {
        Utf8 entity
        Utf8 subject
        Utf8 graph
        Utf8 dataset
	    Utf8 predicate_rdfs_label
	    Utf8 predicate_obo_IAO_0000115
	    Utf8 object_rdfs_label
	    Utf8 object_obo_IAO_0000115
	}
    comparator_literal_object_property_entity_p["comparator_literal_object_property_entity_p"] {
	    List-Utf8 as_columns "['','','','','','','','','resource']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','','','','','','http://%']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Value']"
	    List-Utf8 lhs_values "['entity','subject','predicate','object','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','resource']"
        Boolean cpu "false"
        Utf8 lhs_name "select_predicates_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    filter_literal_object_property_entity_p["filter_literal_object_property_entity_p"] {
        List-Utf8 cmp_columns "['resource']"
        List-Utf8 cmp_operators "['NotLike']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "comparator_literal_object_property_entity_s"
        List-Utf8 lhs_values "['object']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_literal_object_property_entity_p["select_literal_object_property_entity_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_literal_object_property_entity_s"
        List-Utf8 as_columns "['','','','','','','object_rdfs_label','']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','String']"
        List-Utf8 lhs_values "['entity','subject','graph','dataset','predicate_rdfs_label','predicate_obo_IAO_0000115','object','object_obo_IAO_0000115']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    concat_cols_class_entity_p["concat_cols_class_entity_p"] {
        List-Utf8 as_columns "['','','','','predicate_rdfs_label-Cast','object_obo_IAO_0000115-Cast','object-Concat','text']"
        List-Utf8 cast_templates "['','','','','**{{ predicate_rdfs_label }}** ','{% if object_obo_IAO_0000115 %} with definition {{ object_obo_IAO_0000115 }}{% endif %}','','']"
        List-Utf8 column_operators "['None','None','None','None','None','None','Concat','Concat']"
        List-Utf8 rhs_values "['','','','','','','object_obo_IAO_0000115-Cast','object-Concat']"
        List-Utf8 lhs_values "['entity','subject','graph','dataset','predicate_rdfs_label','object_obo_IAO_0000115','object_rdfs_label','predicate_rdfs_label-Cast']"
        Boolean cpu "false"
        Utf8 lhs_name "select_objects_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_cols_class_entity_p["select_cols_class_entity_p"] {
        List-Utf8 lhs_values "['entity','subject','graph','dataset','text']"
        Boolean cpu "false"
        Utf8 lhs_name "concat_cols_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    list_rows_class_entity_p["list_rows_class_entity_p"] {
	    List-Utf8 agg_columns "['text']"
	    List-Utf8 agg_operators "['List']"
	    List-Utf8 lhs_values "['subject','dataset']"
        Boolean cpu "false"
        Utf8 lhs_name "select_cols_class_entity_s"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    select_rows_class_entity_p["select_rows_class_entity_p"] {
        List-Utf8 as_columns "['subject','dataset','text_List']"
        List-Utf8 lhs_values "['subject','dataset','text-List']"
        Boolean cpu "false"
        Utf8 lhs_name "list_rows_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    apply_template_class_entity_p["apply_template_class_entity_p"] {
        List-Utf8 as_columns "['chunk_id','document_id','text']"
        List-Utf8 cast_templates "['','','{% for item in text_List %}{{ item }}{% if not loop.last %}\n{% endif %}{% endfor %}']"
        List-Utf8 lhs_values "['subject','dataset','text_List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_rows_class_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    concat_cols_object_property_entity_p["concat_cols_object_property_entity_p"] {
        List-Utf8 as_columns "['','','','','predicate_rdfs_label-Cast','object_obo_IAO_0000115-Cast','object-Concat','text']"
        List-Utf8 cast_templates "['','','','','**{{ predicate_rdfs_label }}** ','{% if object_obo_IAO_0000115 %} with definition {{ object_obo_IAO_0000115 }}{% endif %}','','']"
        List-Utf8 column_operators "['None','None','None','None','None','None','Concat','Concat']"
        List-Utf8 rhs_values "['','','','','','','object_obo_IAO_0000115-Cast','object-Concat']"
        List-Utf8 lhs_values "['entity','subject','graph','dataset','predicate_rdfs_label','object_obo_IAO_0000115','object_rdfs_label','predicate_rdfs_label-Cast']"
        Boolean cpu "false"
        Utf8 lhs_name "select_objects_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    select_cols_object_property_entity_p["select_cols_object_property_entity_p"] {
        List-Utf8 lhs_values "['entity','subject','graph','dataset','text']"
        Boolean cpu "false"
        Utf8 lhs_name "concat_cols_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    list_rows_object_property_entity_p["list_rows_object_property_entity_p"] {
	    List-Utf8 agg_columns "['text']"
	    List-Utf8 agg_operators "['List']"
	    List-Utf8 lhs_values "['subject','dataset']"
        Boolean cpu "false"
        Utf8 lhs_name "select_cols_object_property_entity_s"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    select_rows_object_property_entity_p["select_rows_object_property_entity_p"] {
        List-Utf8 as_columns "['subject','dataset','text_List']"
        List-Utf8 lhs_values "['subject','dataset','text-List']"
        Boolean cpu "false"
        Utf8 lhs_name "list_rows_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
    apply_template_object_property_entity_p["apply_template_object_property_entity_p"] {
        List-Utf8 as_columns "['chunk_id','document_id','text']"
        List-Utf8 cast_templates "['','','{% for item in text_List %}{{ item }}{% if not loop.last %}\n{% endif %}{% endfor %}']"
        List-Utf8 lhs_values "['subject','dataset','text_List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_rows_object_property_entity_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Stream"
        Utf8 rhs_stream "Stream"
    }
	Documents["Documents"] {
        Utf8 chunk_id
        Utf8 document_id
        Utf8 text
	}"#
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
    use phymes_diagnostics::{HashMap, create_timestamp_micros};
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, NetworkStreamStep, NetworkStreamStepTrait,
    };
    use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjects, create_attachments_batch};
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_extract_ontology_network() -> Result<()> {
        // Initialize the session
        let extract_onto_session = ExtractOntologyNetworkBuilder::default();
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            extract_onto_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            extract_onto_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(extract_onto_session.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://purl.obolibrary.org/obo/ro.owl#"
     xml:base="http://purl.obolibrary.org/obo/ro.owl"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:obo="http://purl.obolibrary.org/obo/"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:terms="http://purl.org/dc/terms/"
     xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
    <owl:Ontology rdf:about="http://purl.obolibrary.org/obo/ro.owl">
        <terms:description xml:lang="en">The OBO Relations Ontology (RO) is a collection of OWL relations (ObjectProperties) intended for use across a wide variety of biological ontologies.</terms:description>
        <terms:title xml:lang="en">OBO Relations Ontology</terms:title>
    </owl:Ontology>

    <owl:AnnotationProperty rdf:about="http://purl.org/dc/terms/description">
        <terms:description>An account of the resource.</terms:description>
        <rdfs:label xml:lang="en">Description</rdfs:label>
    </owl:AnnotationProperty>
	
    <owl:AnnotationProperty rdf:about="http://purl.org/dc/terms/title">
        <terms:description>A name given to the resource.</terms:description>
        <rdfs:label xml:lang="en">Title</rdfs:label>
    </owl:AnnotationProperty>
	
    <owl:AnnotationProperty rdf:about="http://www.w3.org/2000/01/rdf-schema#label">
        <rdfs:comment>A human-readable name for the subject.</rdfs:comment>
        <rdfs:label>label</rdfs:label>
    </owl:AnnotationProperty>

    <owl:AnnotationProperty rdf:about="http://www.w3.org/2000/01/rdf-schema#subPropertyOf">
        <rdfs:comment>The subject is a subclass of a class.</rdfs:comment>
        <rdfs:label>subPropertyOf</rdfs:label>
    </owl:AnnotationProperty>

    <owl:AnnotationProperty rdf:about="http://www.w3.org/2000/01/rdf-schema#comment">
        <rdfs:comment>A description of the subject resource.</rdfs:comment>
        <rdfs:label>comment</rdfs:label>
    </owl:AnnotationProperty>

    <owl:AnnotationProperty rdf:about="http://purl.obolibrary.org/obo/IAO_0000115">
        <obo:IAO_0000115 xml:lang="en">The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions.</obo:IAO_0000115>
        <rdfs:label>definition</rdfs:label>
        <rdfs:label xml:lang="en">definition</rdfs:label>
    </owl:AnnotationProperty>

    <owl:AnnotationProperty rdf:about="http://www.geneontology.org/formats/oboInOwl#hasExactSynonym">
        <obo:IAO_0000115>An alternative label for a class or property which has the exact same meaning than the preferred name/primary label.</obo:IAO_0000115>
        <rdfs:label xml:lang="en">has exact synonym</rdfs:label>
        <rdfs:label>has_exact_synonym</rdfs:label>
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/IAO_0000118"/>
    </owl:AnnotationProperty>

    <owl:ObjectProperty rdf:about="http://purl.obolibrary.org/obo/BFO_0000050">
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/RO_0002131"/>
        <obo:IAO_0000115 xml:lang="en">a core relation that holds between a part and its whole</obo:IAO_0000115>
        <rdfs:label xml:lang="en">part of</rdfs:label>
    </owl:ObjectProperty>

    <owl:ObjectProperty rdf:about="http://purl.obolibrary.org/obo/RO_0002131">
        <obo:IAO_0000115>x overlaps y if and only if there exists some z such that x has part z and z part of y</obo:IAO_0000115>
        <rdfs:label xml:lang="en">overlaps</rdfs:label>
    </owl:ObjectProperty>

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/BFO_0000003">
        <obo:IAO_0000115 xml:lang="en">An entity that has temporal parts and that happens, unfolds or develops through time.</obo:IAO_0000115>
        <rdfs:label xml:lang="en">occurrent</rdfs:label>
        <oboInOwl:hasExactSynonym>through time</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>has temporal part</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>unfolds in time</oboInOwl:hasExactSynonym>
    </owl:Class>

    <owl:NamedIndividual rdf:about="http://purl.obolibrary.org/obo/ENVO_01001569">
        <rdfs:label xml:lang="en">Western Australian Mulga Shrublands Ecoregion</rdfs:label>
    </owl:NamedIndividual>

</rdf:RDF>"#;

        // Make the test data
        let batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["owl".to_string()],
            vec![owl.into()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;
        let table = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserScript.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let owl_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableInterfaceSubjects::UserScript.to_string().as_str())
            .with_update(&Publication::Replace {
                subject_name: AvailableInterfaceSubjects::UserScript.to_string(),
            })
            .with_publisher(extract_onto_session.network_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![owl_message]);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Run the first superstep
        let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), message_map)
            .await?
            .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::ParseOwl.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::ParseOwl.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 30);
            let column = subject.get_column_as_vec_str("entity");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2002/07/owl#AnnotationProperty"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2002/07/owl#Ontology"
            );
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/ro.owl"
            );
            let column = subject.get_column_as_vec_str("predicate");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(column.last().unwrap(), &"http://purl.org/dc/terms/title");
            let column = subject.get_column_as_vec_str("object");
            assert_eq!(
                column.first().unwrap(),
                &"The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions."
            );
            assert_eq!(column.last().unwrap(), &"OBO Relations Ontology");
            let column = subject.get_column_as_vec_str("graph");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115-http://purl.obolibrary.org/obo/IAO_0000115-The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions."
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/ro.owl-http://purl.org/dc/terms/title-OBO Relations Ontology"
            );
            let column = subject.get_column_as_vec_str("dataset");
            assert_eq!(column.first().unwrap(), &"UserScript");
            assert_eq!(column.last().unwrap(), &"UserScript");
        }

        // Run the second superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_ontology_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_ontology_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 2);
            let column = subject.get_column_as_vec_str("entity");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2002/07/owl#Ontology"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2002/07/owl#Ontology"
            );
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/ro.owl"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/ro.owl"
            );
            let column = subject.get_column_as_vec_str("predicate");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.org/dc/terms/description"
            );
            assert_eq!(column.last().unwrap(), &"http://purl.org/dc/terms/title");
            let column = subject.get_column_as_vec_str("object");
            assert_eq!(
                column.first().unwrap(),
                &"The OBO Relations Ontology (RO) is a collection of OWL relations (ObjectProperties) intended for use across a wide variety of biological ontologies."
            );
            assert_eq!(column.last().unwrap(), &"OBO Relations Ontology");
            let column = subject.get_column_as_vec_str("graph");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/ro.owl-http://purl.org/dc/terms/description-The OBO Relations Ontology (RO) is a collection of OWL relations (ObjectProperties) intended for use across a wide variety of biological ontologies."
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/ro.owl-http://purl.org/dc/terms/title-OBO Relations Ontology"
            );
            let column = subject.get_column_as_vec_str("dataset");
            assert_eq!(column.first().unwrap(), &"UserScript");
            assert_eq!(column.last().unwrap(), &"UserScript");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_annotation_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_annotation_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 17);
            let column = subject.get_column_as_vec_str("entity");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2002/07/owl#AnnotationProperty"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2002/07/owl#AnnotationProperty"
            );
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
            );
            let column = subject.get_column_as_vec_str("predicate");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#label"
            );
            let column = subject.get_column_as_vec_str("object");
            assert_eq!(
                column.first().unwrap(),
                &"The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions."
            );
            assert_eq!(column.last().unwrap(), &"subPropertyOf");
            let column = subject.get_column_as_vec_str("graph");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115-http://purl.obolibrary.org/obo/IAO_0000115-The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions."
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#subPropertyOf-http://www.w3.org/2000/01/rdf-schema#label-subPropertyOf"
            );
            let column = subject.get_column_as_vec_str("dataset");
            assert_eq!(column.first().unwrap(), &"UserScript");
            assert_eq!(column.last().unwrap(), &"UserScript");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_datatype_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert_eq!(batches.len(), 0);
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_class_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_class_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 5);
            let column = subject.get_column_as_vec_str("entity");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2002/07/owl#Class"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2002/07/owl#Class"
            );
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000003"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000003"
            );
            let column = subject.get_column_as_vec_str("predicate");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#label"
            );
            let column = subject.get_column_as_vec_str("object");
            assert_eq!(
                column.first().unwrap(),
                &"An entity that has temporal parts and that happens, unfolds or develops through time."
            );
            assert_eq!(column.last().unwrap(), &"occurrent");
            let column = subject.get_column_as_vec_str("graph");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000003-http://purl.obolibrary.org/obo/IAO_0000115-An entity that has temporal parts and that happens, unfolds or develops through time."
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000003-http://www.w3.org/2000/01/rdf-schema#label-occurrent"
            );
            let column = subject.get_column_as_vec_str("dataset");
            assert_eq!(column.first().unwrap(), &"UserScript");
            assert_eq!(column.last().unwrap(), &"UserScript");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_object_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_object_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 5);
            let column = subject.get_column_as_vec_str("entity");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2002/07/owl#ObjectProperty"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2002/07/owl#ObjectProperty"
            );
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000050"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/RO_0002131"
            );
            let column = subject.get_column_as_vec_str("predicate");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#label"
            );
            let column = subject.get_column_as_vec_str("object");
            assert_eq!(
                column.first().unwrap(),
                &"a core relation that holds between a part and its whole"
            );
            assert_eq!(column.last().unwrap(), &"overlaps");
            let column = subject.get_column_as_vec_str("graph");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000050-http://purl.obolibrary.org/obo/IAO_0000115-a core relation that holds between a part and its whole"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/RO_0002131-http://www.w3.org/2000/01/rdf-schema#label-overlaps"
            );
            let column = subject.get_column_as_vec_str("dataset");
            assert_eq!(column.first().unwrap(), &"UserScript");
            assert_eq!(column.last().unwrap(), &"UserScript");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_named_individual_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_named_individual_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);
            let column = subject.get_column_as_vec_str("entity");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2002/07/owl#NamedIndividual"
            );
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/ENVO_01001569"
            );
            let column = subject.get_column_as_vec_str("predicate");
            assert_eq!(
                column.first().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#label"
            );
            let column = subject.get_column_as_vec_str("object");
            assert_eq!(
                column.first().unwrap(),
                &"Western Australian Mulga Shrublands Ecoregion"
            );
            let column = subject.get_column_as_vec_str("graph");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/ENVO_01001569-http://www.w3.org/2000/01/rdf-schema#label-Western Australian Mulga Shrublands Ecoregion"
            );
            let column = subject.get_column_as_vec_str("dataset");
            assert_eq!(column.first().unwrap(), &"UserScript");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_axiom_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert_eq!(batches.len(), 0);
        }

        // Run the third superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "pivot_annotation_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("pivot_annotation_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 7);
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
            );
            let column = subject
                .get_column_as_vec_str("http://www.w3.org/2000/01/rdf-schema#label-object-First");
            assert_eq!(column.first().unwrap(), &"definition");
            assert_eq!(column.last().unwrap(), &"subPropertyOf");
            let column = subject
                .get_column_as_vec_str("http://purl.obolibrary.org/obo/IAO_0000115-object-First");
            assert_eq!(
                column.first().unwrap(),
                &"The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions."
            );
            assert_eq!(column.last().unwrap(), &"");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "pivot_class_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("pivot_class_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000003"
            );
            let column = subject
                .get_column_as_vec_str("http://www.w3.org/2000/01/rdf-schema#label-object-First");
            assert_eq!(column.first().unwrap(), &"occurrent");
            let column = subject
                .get_column_as_vec_str("http://purl.obolibrary.org/obo/IAO_0000115-object-First");
            assert_eq!(
                column.first().unwrap(),
                &"An entity that has temporal parts and that happens, unfolds or develops through time."
            );
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "pivot_object_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("pivot_object_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 2);
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000050"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/RO_0002131"
            );
            let column = subject
                .get_column_as_vec_str("http://www.w3.org/2000/01/rdf-schema#label-object-First");
            assert_eq!(column.first().unwrap(), &"part of");
            assert_eq!(column.last().unwrap(), &"overlaps");
            let column = subject
                .get_column_as_vec_str("http://purl.obolibrary.org/obo/IAO_0000115-object-First");
            assert_eq!(
                column.first().unwrap(),
                &"a core relation that holds between a part and its whole"
            );
            assert_eq!(
                column.last().unwrap(),
                &"x overlaps y if and only if there exists some z such that x has part z and z part of y"
            );
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "pivot_named_individual_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("pivot_named_individual_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);
            let column = subject.get_column_as_vec_str("subject");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/ENVO_01001569"
            );
            let column = subject
                .get_column_as_vec_str("http://www.w3.org/2000/01/rdf-schema#label-object-First");
            assert_eq!(
                column.first().unwrap(),
                &"Western Australian Mulga Shrublands Ecoregion"
            );
        }

        // Run the fourth superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_annotation_property_pivot_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_annotation_property_pivot_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 7);
            let column = subject.get_column_as_vec_str("uri");
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/IAO_0000115"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
            );
            let column = subject.get_column_as_vec_str("rdfs_label");
            assert_eq!(column.first().unwrap(), &"definition");
            assert_eq!(column.last().unwrap(), &"subPropertyOf");
            let column = subject.get_column_as_vec_str("obo_IAO_0000115");
            assert_eq!(
                column.first().unwrap(),
                &"The official definition, explaining the meaning of a class or property. Shall be Aristotelian, formalized and normalized. Can be augmented with colloquial definitions."
            );
            assert_eq!(column.last().unwrap(), &"");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "merge_object_property_class_named_individual_pivot_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("merge_object_property_class_named_individual_pivot_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 4);
            // DM: the ordering of results is variable
            // let column = subject.get_column_as_vec_str("uri");
            // assert_eq!(column.first().unwrap(), &"http://purl.obolibrary.org/obo/BFO_0000003");
            // assert_eq!(column.last().unwrap(), &"http://purl.obolibrary.org/obo/ENVO_01001569");
            // let column = subject.get_column_as_vec_str("rdfs_label");
            // assert_eq!(column.first().unwrap(), &"part of");
            // assert_eq!(column.last().unwrap(), &"Western Australian Mulga Shrublands Ecoregion");
            // let column = subject.get_column_as_vec_str("obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"a core relation that holds between a part and its whole");
            // assert_eq!(column.last().unwrap(), &"");
        }

        // Run the fifth superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_predicates_object_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_predicates_object_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 5);
            // let column = subject.get_column_as_vec_str("entity");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("subject");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("graph");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("dataset");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_predicates_class_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_predicates_class_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 5);
            // let column = subject.get_column_as_vec_str("entity");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("subject");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("graph");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("dataset");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
        }

        // Run the sixth superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_resource_class_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert_eq!(batches.len(), 0);
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_resource_object_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_resource_object_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);
            // let column = subject.get_column_as_vec_str("entity");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("subject");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("graph");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("dataset");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
        }

        // Run the seventh superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_objects_class_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_objects_class_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 5);
            // let column = subject.get_column_as_vec_str("entity");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("subject");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("graph");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("dataset");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_objects_object_property_entity_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_objects_object_property_entity_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 5);
            // let column = subject.get_column_as_vec_str("entity");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("subject");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("graph");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("dataset");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("predicate_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object_rdfs_label");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
            // let column = subject.get_column_as_vec_str("object_obo_IAO_0000115");
            // assert_eq!(column.first().unwrap(), &"");
            // assert_eq!(column.last().unwrap(), &"");
        }

        // Run the eigth superstep
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::Documents.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::Documents.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 3);
            let mut column = subject.get_column_as_vec_str("chunk_id");
            column.sort();
            assert_eq!(
                column.first().unwrap(),
                &"http://purl.obolibrary.org/obo/BFO_0000003"
            );
            assert_eq!(
                column.last().unwrap(),
                &"http://purl.obolibrary.org/obo/RO_0002131"
            );
            let mut column = subject.get_column_as_vec_str("document_id");
            column.sort();
            assert_eq!(column.first().unwrap(), &"UserScript");
            assert_eq!(column.last().unwrap(), &"UserScript");
            let mut column = subject.get_column_as_vec_str("text");
            column.sort();
            assert_eq!(
                column.first().unwrap(),
                &"**definition** An entity that has temporal parts and that happens, unfolds or develops through time.\n**has exact synonym** has temporal part\n**has exact synonym** through time\n**has exact synonym** unfolds in time\n**label** occurrent"
            );
        }
        Ok(())
    }
}
