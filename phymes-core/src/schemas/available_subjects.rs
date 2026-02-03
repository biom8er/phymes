use crate::{
    runtime_env::{BuildableTrait, BuilderTrait},
    schemas::{
        blob::create_blob_fields,
        chat::create_chat_fields,
        diagnostics::{
            create_events_fields, create_metrics_fields, create_metrics_mermaid_gantt_fields,
            create_metrics_pivot_fields, create_metrics_pivot_norm_time_fields,
            create_traces_fields,
        },
        graph::{
            create_n_quads_fields, create_n_triples_fields, create_parse_owl_fields,
            create_parse_xml_fields,
        },
        mermaid::{
            create_mermaid_content_template_fields,
            create_mermaid_er_diagram_entities_template_fields,
            create_mermaid_er_diagram_relations_template_fields,
            create_mermaid_flowchart_links_template_fields,
            create_mermaid_flowchart_nodes_template_fields, create_mermaid_gantt_template_fields,
            create_mermaid_kanban_template_fields,
            create_mermaid_sequence_diagram_messages_template_fields,
            create_mermaid_sequence_diagram_participants_template_fields,
            create_mermaid_visualization_fields, create_mermaid_xychart_template_fields,
            create_session_mermaid_fields,
        },
        queries::create_queries_fields,
        session::{
            create_session_processors_fields, create_session_runtime_envs_fields,
            create_session_subjects_fields, create_session_superstep_max_fields,
            create_session_supersteps_fields, create_session_tasks_check_fields,
            create_session_tasks_fields, create_session_tasks_publish_aggregate_fields,
            create_session_tasks_publish_fields, create_session_tasks_run_log_fields,
            create_session_tasks_subscribe_aggregate_fields, create_session_tasks_subscribe_fields,
            create_session_tasks_subscribe_publish_fields,
        },
        subjects::{create_subjects_change_log_fields, create_subjects_num_rows_fields},
        user::{
            create_join_user_inbox_session_contexts_fields,
            create_join_user_inbox_session_contexts_mermaid_diagrams_fields, create_user_fields,
            create_user_inbox_fields, create_user_session_contexts_fields,
        },
    },
    table::{Table, TableBuilder, TableBuilderTrait},
};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Float32Builder, ListBuilder, StringArray},
    datatypes::{DataType, Field, Fields, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};

pub fn create_schema_from_fields(f: &dyn Fn() -> Fields) -> SchemaRef {
    Arc::new(Schema::new(f()))
}

pub fn create_values_fields() -> Fields {
    let field_names = ["name", "publisher", "subject", "values"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_values_record_batch(
    names: Vec<String>,
    publishers: Vec<String>,
    subjects: Vec<String>,
    values: Vec<String>,
) -> Result<RecordBatch> {
    let names: ArrayRef = Arc::new(StringArray::from(names));
    let publishers: ArrayRef = Arc::new(StringArray::from(publishers));
    let subjects: ArrayRef = Arc::new(StringArray::from(subjects));
    let values: ArrayRef = Arc::new(StringArray::from(values));
    let batch = RecordBatch::try_from_iter(vec![
        ("name", names),
        ("publisher", publishers),
        ("subject", subjects),
        ("values", values),
    ])?;
    Ok(batch)
}

pub fn create_config_fields() -> Fields {
    let field_names = ["values"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_tools_fields() -> Fields {
    let field_names = ["tool_id", "tool"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_tools_record_batch(tool_ids: Vec<String>, tools: Vec<String>) -> Result<RecordBatch> {
    let tool_ids: ArrayRef = Arc::new(StringArray::from(tool_ids));
    let tools: ArrayRef = Arc::new(StringArray::from(tools));
    let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_ids), ("tool", tools)])?;
    Ok(batch)
}

pub fn create_documents_fields() -> Fields {
    let field_names = ["chunk_id", "document_id", "text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_documents_batch(
    chunk_id: Vec<String>,
    document_id: Vec<String>,
    text: Vec<String>,
) -> Result<RecordBatch> {
    let chunk_id: ArrayRef = Arc::new(StringArray::from(chunk_id));
    let document_id: ArrayRef = Arc::new(StringArray::from(document_id));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let batch = RecordBatch::try_from_iter(vec![
        ("chunk_id", chunk_id),
        ("document_id", document_id),
        ("text", text),
    ])?;
    Ok(batch)
}

pub fn create_document_embeddings_fields() -> Fields {
    let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
    let document_id = Field::new("document_id", DataType::Utf8, false);
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)));
    let embedding = Field::new("embedding", list_data_type, false);
    Fields::from(vec![chunk_id, document_id, embedding])
}

pub fn create_documents_embeddings_batch(
    chunk_id: Vec<String>,
    document_id: Vec<String>,
    embedding: Vec<Vec<f32>>,
) -> Result<RecordBatch> {
    let chunk_id: ArrayRef = Arc::new(StringArray::from(chunk_id));
    let document_id: ArrayRef = Arc::new(StringArray::from(document_id));
    let value_builder = Float32Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Float32, false));
    for values in embedding.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let embedding: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("chunk_id", chunk_id),
        ("document_id", document_id),
        ("embedding", embedding),
    ])?;
    Ok(batch)
}

pub fn create_query_embeddings_fields() -> Fields {
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)));
    let embedding = Field::new("embedding", list_data_type, false);
    Fields::from(vec![query_id, embedding])
}

pub fn create_query_embeddings_batch(
    query_id: Vec<String>,
    embedding: Vec<Vec<f32>>,
) -> Result<RecordBatch> {
    let query_id: ArrayRef = Arc::new(StringArray::from(query_id));
    let value_builder = Float32Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Float32, false));
    for values in embedding.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let embedding: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("query_id", query_id),
        ("embedding", embedding),
    ])?;
    Ok(batch)
}

pub fn create_embeddings_scores_fields() -> Fields {
    let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let score = Field::new("score", DataType::Float32, false);
    Fields::from(vec![chunk_id, query_id, score])
}

pub fn create_join_chunks_scores_fields() -> Fields {
    let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let score = Field::new("score", DataType::Float32, false);
    let document_id = Field::new("document_id", DataType::Utf8, false);
    let text = Field::new("text", DataType::Utf8, false);
    Fields::from(vec![chunk_id, query_id, score, document_id, text])
}

pub trait AvailableSubjectsTrait {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Table>;
    fn to_table_builder(&self, name: Option<&str>) -> TableBuilder;
    fn to_schema(&self) -> SchemaRef;
}

/// The available subject schmeas
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableSubjects {
    #[value(name = "Empty")]
    Empty,
    #[value(name = "Messages")]
    Messages,
    #[default]
    #[value(name = "Values")]
    Values,
    #[value(name = "Configs")]
    Configs,
    #[value(name = "Tools")]
    Tools,
    #[value(name = "Documents")]
    Documents,
    #[value(name = "Queries")]
    Queries,
    #[value(name = "DocumentEmbeddings")]
    DocumentEmbeddings,
    #[value(name = "QueryEmbeddings")]
    QueryEmbeddings,
    #[value(name = "EmbeddingScores")]
    EmbeddingScores,
    #[value(name = "JoinChunksScores")]
    JoinChunksScores,
    #[value(name = "Blob")]
    Blob,
    #[value(name = "User")]
    User,
    #[value(name = "UserSessionContexts")]
    UserSessionContexts,
    #[value(name = "UserInbox")]
    UserInbox,
    #[value(name = "JoinUserInboxSessionContexts")]
    JoinUserInboxSessionContexts,
    #[value(name = "JoinUserInboxSessionContextsMermaid")]
    JoinUserInboxSessionContextsMermaid,
    #[value(name = "SessionMermaid")]
    SessionMermaid,
    #[value(name = "BuilderMermaid")]
    BuilderMermaid,
    #[value(name = "Errors")]
    SessionErrors,
    #[value(name = "Metrics")]
    SessionMetrics,
    #[value(name = "MetricMermaidGantt")]
    MetricMermaidGantt,
    #[value(name = "Traces")]
    SessionTraces,
    #[value(name = "Events")]
    SessionEvents,
    #[value(name = "MetricPivot")]
    MetricPivot,
    #[value(name = "MetricPivotNormTime")]
    MetricPivotNormTime,
    #[value(name = "SessionSubjects")]
    SessionSubjects,
    #[value(name = "SessionTasks")]
    SessionTasks,
    #[value(name = "SessionProcessors")]
    SessionProcessors,
    #[value(name = "SessionRuntimeEnvs")]
    SessionRuntimeEnvs,
    #[value(name = "SessionTasksRunLog")]
    SessionTasksRunLog,
    #[value(name = "SubjectsNumRows")]
    SubjectsNumRows,
    #[value(name = "SubjectsChangeLog")]
    SubjectsChangeLog,
    #[value(name = "MermaidContentTemplate")]
    MermaidContentTemplate,
    #[value(name = "MermaidXYChart")]
    MermaidXYChart,
    #[value(name = "MermaidGanttTemplate")]
    MermaidGanttTemplate,
    #[value(name = "MermaidFlowchartNodesTemplate")]
    MermaidFlowchartNodesTemplate,
    #[value(name = "MermaidFlowchartLinksTemplate")]
    MermaidFlowchartLinksTemplate,
    #[value(name = "MermaidSequenceDiagramParticipantsTemplate")]
    MermaidSequenceDiagramParticipantsTemplate,
    #[value(name = "MermaidSequenceDiagramMessagesTemplate")]
    MermaidSequenceDiagramMessagesTemplate,
    #[value(name = "MermaidKanbanTemplate")]
    MermaidKanbanTemplate,
    #[value(name = "MermaidVisualization")]
    MermaidVisualization,
    #[value(name = "AnalyticsErrors")]
    AnalyticsErrors,
    #[value(name = "AnalyticsMetrics")]
    AnalyticsMetrics,
    #[value(name = "AnalyticsTraces")]
    AnalyticsTraces,
    #[value(name = "AnalyticsEvents")]
    AnalyticsEvents,
    #[value(name = "AnalyticsTasks")]
    AnalyticsTasks,
    #[value(name = "MermaidERDiagramEntitiesTemplate")]
    MermaidERDiagramEntitiesTemplate,
    #[value(name = "MermaidERDiagramRelationsTemplate")]
    MermaidERDiagramRelationsTemplate,
    #[value(name = "ParseXml")]
    ParseXml,
    #[value(name = "ParseOwl")]
    ParseOwl,
    #[value(name = "NTriples")]
    NTriples,
    #[value(name = "NQuads")]
    NQuads,
    #[value(name = "SessionTasksCheck")]
    SessionTasksCheck,
    #[value(name = "SessionTasksSubscribe")]
    SessionTasksSubscribe,
    #[value(name = "SessionTasksPublish")]
    SessionTasksPublish,
    #[value(name = "SessionTasksSubscribeAggregate")]
    SessionTasksSubscribeAggregate,
    #[value(name = "SessionTasksPublishAggregate")]
    SessionTasksPublishAggregate,
    #[value(name = "SessionTasksSubscribePublish")]
    SessionTasksSubscribePublish,
    #[value(name = "SessionSupersteps")]
    SessionSupersteps,
    #[value(name = "SessionSuperstepMax")]
    SessionSuperstepMax,
}

impl Display for AvailableSubjects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AvailableSubjects::Empty => write!(f, "Empty"),
            AvailableSubjects::Messages => write!(f, "Messages"),
            AvailableSubjects::Values => write!(f, "Values"),
            AvailableSubjects::Configs => write!(f, "Configs"),
            AvailableSubjects::Tools => write!(f, "Tools"),
            AvailableSubjects::Documents => write!(f, "Documents"),
            AvailableSubjects::Queries => write!(f, "Queries"),
            AvailableSubjects::DocumentEmbeddings => write!(f, "DocumentEmbeddings"),
            AvailableSubjects::QueryEmbeddings => write!(f, "QueryEmbeddings"),
            AvailableSubjects::EmbeddingScores => write!(f, "EmbeddingScores"),
            AvailableSubjects::JoinChunksScores => write!(f, "JoinChunksScores"),
            AvailableSubjects::Blob => write!(f, "Blob"),
            AvailableSubjects::User => write!(f, "User"),
            AvailableSubjects::UserSessionContexts => write!(f, "UserSessionContexts"),
            AvailableSubjects::UserInbox => write!(f, "UserInbox"),
            AvailableSubjects::JoinUserInboxSessionContexts => {
                write!(f, "JoinUserInboxSessionContexts")
            }
            AvailableSubjects::JoinUserInboxSessionContextsMermaid => {
                write!(f, "JoinUserInboxSessionContextsMermaid")
            }
            AvailableSubjects::SessionMermaid => write!(f, "SessionMermaid"),
            AvailableSubjects::BuilderMermaid => write!(f, "BuilderMermaid"),
            AvailableSubjects::SessionErrors => write!(f, "SessionErrors"),
            AvailableSubjects::SessionMetrics => write!(f, "SessionMetrics"),
            AvailableSubjects::MetricMermaidGantt => write!(f, "MetricMermaidGantt"),
            AvailableSubjects::SessionTraces => write!(f, "SessionTraces"),
            AvailableSubjects::SessionEvents => write!(f, "SessionEvents"),
            AvailableSubjects::MetricPivot => write!(f, "MetricPivot"),
            AvailableSubjects::MetricPivotNormTime => write!(f, "MetricPivotNormTime"),
            AvailableSubjects::SessionSubjects => write!(f, "SessionSubjects"),
            AvailableSubjects::SessionTasks => write!(f, "SessionTasks"),
            AvailableSubjects::SessionProcessors => write!(f, "SessionProcessors"),
            AvailableSubjects::SessionRuntimeEnvs => write!(f, "SessionRuntimeEnvs"),
            AvailableSubjects::SessionTasksRunLog => write!(f, "SessionTasksRunLog"),
            AvailableSubjects::SubjectsNumRows => write!(f, "SubjectsNumRows"),
            AvailableSubjects::SubjectsChangeLog => write!(f, "SubjectsChangeLog"),
            AvailableSubjects::MermaidContentTemplate => write!(f, "MermaidContentTemplate"),
            AvailableSubjects::MermaidGanttTemplate => write!(f, "MermaidGanttTemplate"),
            AvailableSubjects::MermaidFlowchartNodesTemplate => {
                write!(f, "MermaidFlowchartNodesTemplate")
            }
            AvailableSubjects::MermaidFlowchartLinksTemplate => {
                write!(f, "MermaidFlowchartLinksTemplate")
            }
            AvailableSubjects::MermaidSequenceDiagramParticipantsTemplate => {
                write!(f, "MermaidSequenceDiagramParticipantsTemplate")
            }
            AvailableSubjects::MermaidSequenceDiagramMessagesTemplate => {
                write!(f, "MermaidSequenceDiagramMessagesTemplate")
            }
            AvailableSubjects::MermaidKanbanTemplate => write!(f, "MermaidKanbanTemplate"),
            AvailableSubjects::MermaidVisualization => write!(f, "MermaidVisualization"),
            AvailableSubjects::MermaidXYChart => write!(f, "MermaidXYChart"),
            AvailableSubjects::AnalyticsErrors => write!(f, "AnalyticsErrors"),
            AvailableSubjects::AnalyticsMetrics => write!(f, "AnalyticsMetrics"),
            AvailableSubjects::AnalyticsTraces => write!(f, "AnalyticsTraces"),
            AvailableSubjects::AnalyticsEvents => write!(f, "AnalyticsEvents"),
            AvailableSubjects::AnalyticsTasks => write!(f, "AnalyticsTasks"),
            AvailableSubjects::MermaidERDiagramEntitiesTemplate => {
                write!(f, "MermaidERDiagramEntitiesTemplate")
            }
            AvailableSubjects::MermaidERDiagramRelationsTemplate => {
                write!(f, "MermaidERDiagramRelationsTemplate")
            }
            AvailableSubjects::ParseXml => write!(f, "ParseXml"),
            AvailableSubjects::ParseOwl => write!(f, "ParseOwl"),
            AvailableSubjects::NTriples => write!(f, "NTriples"),
            AvailableSubjects::NQuads => write!(f, "NQuads"),
            AvailableSubjects::SessionTasksCheck => write!(f, "SessionTasksCheck"),
            AvailableSubjects::SessionTasksSubscribe => write!(f, "SessionTasksSubscribe"),
            AvailableSubjects::SessionTasksPublish => write!(f, "SessionTasksPublish"),
            AvailableSubjects::SessionTasksSubscribeAggregate => {
                write!(f, "SessionTasksSubscribeAggregate")
            }
            AvailableSubjects::SessionTasksPublishAggregate => {
                write!(f, "SessionTasksPublishAggregate")
            }
            AvailableSubjects::SessionTasksSubscribePublish => {
                write!(f, "SessionTasksSubscribePublish")
            }
            AvailableSubjects::SessionSupersteps => write!(f, "SessionSupersteps"),
            AvailableSubjects::SessionSuperstepMax => write!(f, "SessionSuperstepMax"),
        }
    }
}

impl AvailableSubjectsTrait for AvailableSubjects {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Table> {
        let builder = self.to_table_builder(name);
        let batches = batches.unwrap_or_default();
        builder.with_record_batches(batches)?.build()
    }
    fn to_table_builder(&self, name: Option<&str>) -> TableBuilder {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        Table::get_builder()
            .with_name(&name)
            .with_schema(self.to_schema())
    }
    fn to_schema(&self) -> SchemaRef {
        match self {
            AvailableSubjects::Empty => Arc::new(Schema::empty()),
            AvailableSubjects::Messages => create_schema_from_fields(&create_chat_fields),
            AvailableSubjects::Values => create_schema_from_fields(&create_values_fields),
            AvailableSubjects::Configs => create_schema_from_fields(&create_config_fields),
            AvailableSubjects::Tools => create_schema_from_fields(&create_tools_fields),
            AvailableSubjects::Documents => create_schema_from_fields(&create_documents_fields),
            AvailableSubjects::Queries => create_schema_from_fields(&create_queries_fields),
            AvailableSubjects::DocumentEmbeddings => {
                create_schema_from_fields(&create_document_embeddings_fields)
            }
            AvailableSubjects::QueryEmbeddings => {
                create_schema_from_fields(&create_query_embeddings_fields)
            }
            AvailableSubjects::EmbeddingScores => {
                create_schema_from_fields(&create_embeddings_scores_fields)
            }
            AvailableSubjects::JoinChunksScores => {
                create_schema_from_fields(&create_join_chunks_scores_fields)
            }
            AvailableSubjects::Blob => create_schema_from_fields(&create_blob_fields),
            AvailableSubjects::User => create_schema_from_fields(&create_user_fields),
            AvailableSubjects::UserSessionContexts => {
                create_schema_from_fields(&create_user_session_contexts_fields)
            }
            AvailableSubjects::UserInbox => create_schema_from_fields(&create_user_inbox_fields),
            AvailableSubjects::JoinUserInboxSessionContexts => {
                create_schema_from_fields(&create_join_user_inbox_session_contexts_fields)
            }
            AvailableSubjects::JoinUserInboxSessionContextsMermaid => create_schema_from_fields(
                &create_join_user_inbox_session_contexts_mermaid_diagrams_fields,
            ),
            AvailableSubjects::SessionMermaid => {
                create_schema_from_fields(&create_session_mermaid_fields)
            }
            AvailableSubjects::BuilderMermaid => {
                create_schema_from_fields(&create_session_mermaid_fields)
            }
            AvailableSubjects::SessionErrors => create_schema_from_fields(&create_chat_fields),
            AvailableSubjects::SessionMetrics => create_schema_from_fields(&create_metrics_fields),
            AvailableSubjects::MetricMermaidGantt => {
                create_schema_from_fields(&create_metrics_mermaid_gantt_fields)
            }
            AvailableSubjects::SessionTraces => create_schema_from_fields(&create_traces_fields),
            AvailableSubjects::SessionEvents => create_schema_from_fields(&create_events_fields),
            AvailableSubjects::MetricPivot => {
                create_schema_from_fields(&create_metrics_pivot_fields)
            }
            AvailableSubjects::MetricPivotNormTime => {
                create_schema_from_fields(&create_metrics_pivot_norm_time_fields)
            }
            AvailableSubjects::SessionSubjects => {
                create_schema_from_fields(&create_session_subjects_fields)
            }
            AvailableSubjects::SessionTasks => {
                create_schema_from_fields(&create_session_tasks_fields)
            }
            AvailableSubjects::SessionProcessors => {
                create_schema_from_fields(&create_session_processors_fields)
            }
            AvailableSubjects::SessionRuntimeEnvs => {
                create_schema_from_fields(&create_session_runtime_envs_fields)
            }
            AvailableSubjects::SessionTasksRunLog => {
                create_schema_from_fields(&create_session_tasks_run_log_fields)
            }
            AvailableSubjects::SubjectsNumRows => {
                create_schema_from_fields(&create_subjects_num_rows_fields)
            }
            AvailableSubjects::SubjectsChangeLog => {
                create_schema_from_fields(&create_subjects_change_log_fields)
            }
            AvailableSubjects::MermaidContentTemplate => {
                create_schema_from_fields(&create_mermaid_content_template_fields)
            }
            AvailableSubjects::MermaidGanttTemplate => {
                create_schema_from_fields(&create_mermaid_gantt_template_fields)
            }
            AvailableSubjects::MermaidFlowchartNodesTemplate => {
                create_schema_from_fields(&create_mermaid_flowchart_nodes_template_fields)
            }
            AvailableSubjects::MermaidFlowchartLinksTemplate => {
                create_schema_from_fields(&create_mermaid_flowchart_links_template_fields)
            }
            AvailableSubjects::MermaidSequenceDiagramParticipantsTemplate => {
                create_schema_from_fields(
                    &create_mermaid_sequence_diagram_participants_template_fields,
                )
            }
            AvailableSubjects::MermaidSequenceDiagramMessagesTemplate => {
                create_schema_from_fields(&create_mermaid_sequence_diagram_messages_template_fields)
            }
            AvailableSubjects::MermaidKanbanTemplate => {
                create_schema_from_fields(&create_mermaid_kanban_template_fields)
            }
            AvailableSubjects::MermaidVisualization => {
                create_schema_from_fields(&create_mermaid_visualization_fields)
            }
            AvailableSubjects::MermaidXYChart => {
                create_schema_from_fields(&create_mermaid_xychart_template_fields)
            }
            AvailableSubjects::AnalyticsErrors => create_schema_from_fields(&create_chat_fields),
            AvailableSubjects::AnalyticsMetrics => {
                create_schema_from_fields(&create_metrics_fields)
            }
            AvailableSubjects::AnalyticsTraces => create_schema_from_fields(&create_traces_fields),
            AvailableSubjects::AnalyticsEvents => create_schema_from_fields(&create_events_fields),
            AvailableSubjects::AnalyticsTasks => {
                create_schema_from_fields(&create_session_tasks_fields)
            }
            AvailableSubjects::MermaidERDiagramEntitiesTemplate => {
                create_schema_from_fields(&create_mermaid_er_diagram_entities_template_fields)
            }
            AvailableSubjects::MermaidERDiagramRelationsTemplate => {
                create_schema_from_fields(&create_mermaid_er_diagram_relations_template_fields)
            }
            AvailableSubjects::ParseXml => create_schema_from_fields(&create_parse_xml_fields),
            AvailableSubjects::ParseOwl => create_schema_from_fields(&create_parse_owl_fields),
            AvailableSubjects::NTriples => create_schema_from_fields(&create_n_triples_fields),
            AvailableSubjects::NQuads => create_schema_from_fields(&create_n_quads_fields),
            AvailableSubjects::SessionTasksCheck => {
                create_schema_from_fields(&create_session_tasks_check_fields)
            }
            AvailableSubjects::SessionTasksSubscribe => {
                create_schema_from_fields(&create_session_tasks_subscribe_fields)
            }
            AvailableSubjects::SessionTasksPublish => {
                create_schema_from_fields(&create_session_tasks_publish_fields)
            }
            AvailableSubjects::SessionTasksSubscribeAggregate => {
                create_schema_from_fields(&create_session_tasks_subscribe_aggregate_fields)
            }
            AvailableSubjects::SessionTasksPublishAggregate => {
                create_schema_from_fields(&create_session_tasks_publish_aggregate_fields)
            }
            AvailableSubjects::SessionTasksSubscribePublish => {
                create_schema_from_fields(&create_session_tasks_subscribe_publish_fields)
            }
            AvailableSubjects::SessionSupersteps => {
                create_schema_from_fields(&create_session_supersteps_fields)
            }
            AvailableSubjects::SessionSuperstepMax => {
                create_schema_from_fields(&create_session_superstep_max_fields)
            }
        }
    }
}
