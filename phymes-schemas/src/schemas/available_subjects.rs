use crate::{
    AvailableSchemaTrait, AvailableSubjectsTrait, PdfDocumentSubject, PdfGraphicsSubject, PdfPageSubject, PdfTextSubject, chat::{create_route_bytes_fields, create_tools_fields, create_values_fields}, core::{
        create_events_fields, create_join_user_inbox_networks_fields,
        create_join_user_inbox_networks_mermaid_diagrams_fields,
        create_mermaid_content_template_fields, create_mermaid_er_diagram_entities_template_fields,
        create_mermaid_er_diagram_relations_template_fields,
        create_mermaid_flowchart_links_template_fields,
        create_mermaid_flowchart_nodes_template_fields, create_mermaid_gantt_template_fields,
        create_mermaid_kanban_template_fields,
        create_mermaid_sequence_diagram_messages_template_fields,
        create_mermaid_sequence_diagram_participants_template_fields,
        create_mermaid_visualization_fields, create_mermaid_xychart_template_fields,
        create_metrics_fields, create_metrics_mermaid_gantt_fields, create_metrics_pivot_fields,
        create_metrics_pivot_norm_time_fields, create_network_mermaid_fields,
        create_session_processors_fields, create_session_runtime_envs_fields,
        create_session_subject_schemas_fields, create_session_superstep_max_fields,
        create_session_supersteps_fields, create_session_tasks_check_fields,
        create_session_tasks_fields, create_session_tasks_publish_aggregate_fields,
        create_session_tasks_publish_fields, create_session_tasks_run_log_fields,
        create_session_tasks_subscribe_aggregate_fields, create_session_tasks_subscribe_fields,
        create_session_tasks_subscribe_publish_fields, create_subjects_change_log_fields,
        create_subjects_num_rows_fields, create_subjects_object_store_meta_fields,
        create_traces_fields, create_user_fields, create_user_inbox_fields,
        create_user_networks_fields,
    }, create_bytes_fields, create_chat_fields, create_repository_fields, create_repository_patch_fields, create_workspace_fields, create_workspace_patch_fields, embed::{
        create_document_embeddings_fields, create_documents_fields,
        create_embeddings_scores_fields, create_join_chunks_scores_fields, create_queries_fields,
        create_query_embeddings_fields,
    }, storage::{
        create_attachments_fields, create_n_quads_fields, create_n_triples_fields,
        create_object_store_fields, create_object_store_meta_fields, create_parse_owl_fields,
        create_parse_xml_fields,
    },
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};

use anyhow::Result;
use arrow::{
    datatypes::{Fields, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};

pub fn create_schema_from_fields(f: &dyn Fn() -> Fields) -> SchemaRef {
    Arc::new(Schema::new(f()))
}

/// The available subject schmeas
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableSubjects {
    #[value(name = "None")]
    None,
    #[value(name = "Empty")]
    Empty,
    #[value(name = "Messages")]
    Messages,
    #[default]
    #[value(name = "Values")]
    Values,
    #[value(name = "RouteBytes")]
    RouteBytes,
    #[value(name = "Bytes")]
    Bytes,
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
    #[value(name = "Attachments")]
    Attachments,
    #[value(name = "ObjectStore")]
    ObjectStore,
    #[value(name = "ObjectStoreMeta")]
    ObjectStoreMeta,
    #[value(name = "Workspace")]
    Workspace,
    #[value(name = "Repository")]
    Repository,
    #[value(name = "WorkspacePatch")]
    WorkspacePatch,
    #[value(name = "RepositoryPatch")]
    RepositoryPatch,
    #[value(name = "User")]
    User,
    #[value(name = "UserNetworks")]
    UserNetworks,
    #[value(name = "UserInbox")]
    UserInbox,
    #[value(name = "JoinUserInboxNetworks")]
    JoinUserInboxNetworks,
    #[value(name = "JoinUserInboxNetworksMermaid")]
    JoinUserInboxNetworksMermaid,
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
    #[value(name = "SessionSubjectSchemas")]
    SessionSubjectSchemas,
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
    #[value(name = "SubjectsObjectStoreMeta")]
    SubjectsObjectStoreMeta,
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
    #[value(name = "OpenAlexResponseWorks")]
    OpenAlexResponseWorks,
    #[value(name = "OpenAlexResponseAuthors")]
    OpenAlexResponseAuthors,
    #[value(name = "OpenAlexResponseInstitutions")]
    OpenAlexResponseInstitutions,
    #[value(name = "OpenAlexResponseTopics")]
    OpenAlexResponseTopics,
    #[value(name = "OpenAlexResponseAwards")]
    OpenAlexResponseAwards,
    #[value(name = "OpenAlexResponseFunders")]
    OpenAlexResponseFunders,
    #[value(name = "OpenAlexResponsePublishers")]
    OpenAlexResponsePublishers,
    #[value(name = "OpenAlexResponseSources")]
    OpenAlexResponseSources,
    #[value(name = "PdfTextSubject")]
    PdfTextSubject,
    #[value(name = "PdfGraphicsSubject")]
    PdfGraphicsSubject,
    #[value(name = "PdfPageSubject")]
    PdfPageSubject,
    #[value(name = "PdfDocumentSubject")]
    PdfDocumentSubject,
}

impl Display for AvailableSubjects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AvailableSubjects::None => write!(f, "None"),
            AvailableSubjects::Empty => write!(f, "Empty"),
            AvailableSubjects::Messages => write!(f, "Messages"),
            AvailableSubjects::Values => write!(f, "Values"),
            AvailableSubjects::RouteBytes => write!(f, "RouteBytes"),
            AvailableSubjects::Bytes => write!(f, "Bytes"),
            AvailableSubjects::Tools => write!(f, "Tools"),
            AvailableSubjects::Documents => write!(f, "Documents"),
            AvailableSubjects::Queries => write!(f, "Queries"),
            AvailableSubjects::DocumentEmbeddings => write!(f, "DocumentEmbeddings"),
            AvailableSubjects::QueryEmbeddings => write!(f, "QueryEmbeddings"),
            AvailableSubjects::EmbeddingScores => write!(f, "EmbeddingScores"),
            AvailableSubjects::JoinChunksScores => write!(f, "JoinChunksScores"),
            AvailableSubjects::Attachments => write!(f, "Attachments"),
            AvailableSubjects::ObjectStore => write!(f, "ObjectStore"),
            AvailableSubjects::ObjectStoreMeta => write!(f, "ObjectStoreMeta"),
            AvailableSubjects::Workspace => write!(f, "Workspace"),
            AvailableSubjects::Repository => write!(f, "Repository"),
            AvailableSubjects::WorkspacePatch => write!(f, "WorkspacePatch"),
            AvailableSubjects::RepositoryPatch => write!(f, "RepositoryPatch"),
            AvailableSubjects::User => write!(f, "User"),
            AvailableSubjects::UserNetworks => write!(f, "UserNetworks"),
            AvailableSubjects::UserInbox => write!(f, "UserInbox"),
            AvailableSubjects::JoinUserInboxNetworks => {
                write!(f, "JoinUserInboxNetworks")
            }
            AvailableSubjects::JoinUserInboxNetworksMermaid => {
                write!(f, "JoinUserInboxNetworksMermaid")
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
            AvailableSubjects::SessionSubjectSchemas => write!(f, "SessionSubjectSchemas"),
            AvailableSubjects::SessionTasks => write!(f, "SessionTasks"),
            AvailableSubjects::SessionProcessors => write!(f, "SessionProcessors"),
            AvailableSubjects::SessionRuntimeEnvs => write!(f, "SessionRuntimeEnvs"),
            AvailableSubjects::SessionTasksRunLog => write!(f, "SessionTasksRunLog"),
            AvailableSubjects::SubjectsNumRows => write!(f, "SubjectsNumRows"),
            AvailableSubjects::SubjectsChangeLog => write!(f, "SubjectsChangeLog"),
            AvailableSubjects::SubjectsObjectStoreMeta => write!(f, "SubjectsObjectStoreMeta"),
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
            AvailableSubjects::OpenAlexResponseWorks => write!(f, "OpenAlexResponseWorks"),
            AvailableSubjects::OpenAlexResponseAuthors => write!(f, "OpenAlexResponseAuthors"),
            AvailableSubjects::OpenAlexResponseInstitutions => {
                write!(f, "OpenAlexResponseInstitutions")
            }
            AvailableSubjects::OpenAlexResponseTopics => write!(f, "OpenAlexResponseTopics"),
            AvailableSubjects::OpenAlexResponseAwards => write!(f, "OpenAlexResponseAwards"),
            AvailableSubjects::OpenAlexResponseFunders => write!(f, "OpenAlexResponseFunders"),
            AvailableSubjects::OpenAlexResponsePublishers => {
                write!(f, "OpenAlexResponsePublishers")
            }
            AvailableSubjects::OpenAlexResponseSources => write!(f, "OpenAlexResponseSources"),
            AvailableSubjects::PdfTextSubject => write!(f, "PdfTextSubject"),
            AvailableSubjects::PdfGraphicsSubject => write!(f, "PdfGraphicsSubject"),
            AvailableSubjects::PdfPageSubject => write!(f, "PdfPageSubject"),
            AvailableSubjects::PdfDocumentSubject => write!(f, "PdfDocumentSubject"),
        }
    }
}

impl AvailableSubjectsTrait for AvailableSubjects {
    fn to_subject(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Subject> {
        let builder = self.to_subject_builder(name);
        let batches = batches.unwrap_or_default();
        builder.with_record_batches(batches)?.build()
    }

    fn to_subject_builder(&self, name: Option<&str>) -> SubjectBuilder {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        Subject::get_builder()
            .with_name(&name)
            .with_schema(self.to_schema())
    }

    fn to_subject_plan(
        &self,
        name: Option<&str>,
        batches: Option<Vec<RecordBatch>>,
    ) -> Result<SubjectPlan> {
        let subject = self.to_subject(name, batches)?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }
}

impl AvailableSchemaTrait for AvailableSubjects {
    fn to_schema(&self) -> SchemaRef {
        match self {
            AvailableSubjects::Empty | AvailableSubjects::None => Arc::new(Schema::empty()),
            AvailableSubjects::Messages => create_schema_from_fields(&create_chat_fields),
            AvailableSubjects::Values => create_schema_from_fields(&create_values_fields),
            AvailableSubjects::RouteBytes => create_schema_from_fields(&create_route_bytes_fields),
            AvailableSubjects::Bytes => create_schema_from_fields(&create_bytes_fields),
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
            AvailableSubjects::Attachments => create_schema_from_fields(&create_attachments_fields),
            AvailableSubjects::ObjectStore => {
                create_schema_from_fields(&create_object_store_fields)
            }
            AvailableSubjects::ObjectStoreMeta => {
                create_schema_from_fields(&create_object_store_meta_fields)
            }
            AvailableSubjects::Workspace => create_schema_from_fields(&create_workspace_fields),
            AvailableSubjects::Repository => create_schema_from_fields(&create_repository_fields),
            AvailableSubjects::WorkspacePatch => {
                create_schema_from_fields(&create_workspace_patch_fields)
            }
            AvailableSubjects::RepositoryPatch => {
                create_schema_from_fields(&create_repository_patch_fields)
            }
            AvailableSubjects::User => create_schema_from_fields(&create_user_fields),
            AvailableSubjects::UserNetworks => {
                create_schema_from_fields(&create_user_networks_fields)
            }
            AvailableSubjects::UserInbox => create_schema_from_fields(&create_user_inbox_fields),
            AvailableSubjects::JoinUserInboxNetworks => {
                create_schema_from_fields(&create_join_user_inbox_networks_fields)
            }
            AvailableSubjects::JoinUserInboxNetworksMermaid => {
                create_schema_from_fields(&create_join_user_inbox_networks_mermaid_diagrams_fields)
            }
            AvailableSubjects::SessionMermaid => {
                create_schema_from_fields(&create_network_mermaid_fields)
            }
            AvailableSubjects::BuilderMermaid => {
                create_schema_from_fields(&create_network_mermaid_fields)
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
            AvailableSubjects::SessionSubjectSchemas => {
                create_schema_from_fields(&create_session_subject_schemas_fields)
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
            AvailableSubjects::SubjectsObjectStoreMeta => {
                create_schema_from_fields(&create_subjects_object_store_meta_fields)
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
            AvailableSubjects::OpenAlexResponseWorks => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponseAuthors => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponseInstitutions => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponseTopics => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponseAwards => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponseFunders => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponsePublishers => Arc::new(Schema::empty()),
            AvailableSubjects::OpenAlexResponseSources => Arc::new(Schema::empty()),
            AvailableSubjects::PdfTextSubject => {
                create_schema_from_fields(&PdfTextSubject::to_fields)
            }
            AvailableSubjects::PdfGraphicsSubject => {
                create_schema_from_fields(&PdfGraphicsSubject::to_fields)
            }
            AvailableSubjects::PdfPageSubject => {
                create_schema_from_fields(&PdfPageSubject::to_fields)
            }
            AvailableSubjects::PdfDocumentSubject => {
                create_schema_from_fields(&PdfDocumentSubject::to_fields)
            }
        }
    }
}
