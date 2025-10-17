use std::sync::Arc;
use anyhow::Result;

use phymes_core::{
    schemas::{available_subjects::{AvailableSubjects, AvailableSubjectsTrait}, user::{create_user_batch, create_user_session_contexts_batch}}, session::{
        common_traits::BuilderTrait,
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::TaskPlan,
    }, table::{
        data_format::DataFormat, table_trait::{Table, TableBuilder, TableBuilderTrait}, table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, AnyTableNameSubscribe, SubscribeTrait, TableSubscribe}
    }, task::processor::{ProcessorEcho, ProcessorTrait}
};
use phymes_data::{candle_data::{data_config::DataConfig, data_processor::CandleDataProcessor, summary_config::DataSummaryConfig, summary_processor::DataSummaryProcessor}, candle_operators::available_candle_operators::AvailableCandleOperators};
use phymes_diagnostics::create_timestamp_micros;

use crate::{session_plans::{available_interface_subjects::AvailableInterfaceSubjects, available_session_plans::AvailableSessionPlans, builder_session::make_example_mermaid_table}, session_traits::agents::CustomAgentsBuilderTrait};

/// A session for gathering analytics based on the session metrics
/// 
/// # Notes
/// 
/// Supported tasks include the following:
/// 
/// 1. Creating a pivot table for metrics and creating a gantt view of the metrics based on the pivot table
/// 2. Creating a sequence diagram view of the traces
/// 3. Joining the Errors with Events and creating a kanban view of the errors and events
/// 
/// An inbox and outbox for each support task are provided
///   that trigger the task
pub struct MetricSession<'a> {
    /// Extract data from inbox subtask
    pub filter_metrics_inbox_task_name: &'a str,
    pub filter_traces_inbox_task_name: &'a str,
    pub filter_events_inbox_task_name: &'a str,
    pub filter_and_join_events_and_errors_inbox_task_name: &'a str,
    pub filter_metrics_inbox_processor_name: &'a str,

    /// Make outbox attachment subtask
    pub metrics_mermaid_outbox_task_name: &'a str,
    pub traces_mermaid_outbox_task_name: &'a str,
    pub events_mermaid_outbox_task_name: &'a str,
    pub metrics_mermaid_outbox_processor_name: &'a str,

    /// TODO
    /// make the normalized start and end time for the metrics
    /// select and cast metrics to gantt
    /// make the processor traces gantt
    /// make the elapsed compute gantt
    /// make the output rows gantt
    
    /// TODO
    /// select and cast traces to sequence diagram participants
    /// apply MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE
    /// select and cast traces to sequence diagram messages
    /// apply MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE
    /// apply MERMAID_SEQUENCE_DIAGRAM_TEMPLATE
    
    /// TODO
    /// 
    /// select and cast events and errors to kanban
    /// apply MERMAID_KANBAN_TEMPLATE

    /// Filter session contexts by email subtask
    pub filter_session_contexts_by_email_runtime_env_name: &'a str,
    pub filter_session_contexts_by_email_task_name: &'a str,
    pub filter_session_contexts_by_email_processor_name: &'a str,

    /// Join session contexts by email subtask
    pub join_session_contexts_with_mermaid_diagrams_runtime_env_name: &'a str,
    pub join_session_contexts_with_mermaid_diagrams_task_name: &'a str,
    pub join_session_contexts_with_mermaid_diagrams_processor_name: &'a str,

    /// Filter user info by email subtask
    pub filter_user_info_by_email_runtime_env_name: &'a str,
    pub filter_user_info_by_email_task_name: &'a str,
    pub filter_user_info_by_email_processor_name: &'a str,
    pub filter_user_info_by_email_table_name: &'a str,

    /// Session
    pub session_context_name: &'a str,
}