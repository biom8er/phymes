use std::sync::Arc;

use arrow::datatypes::DataType;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableTableSubscribePolicies, BuilderTrait,
    DataFormat, DiagnosticsVisualizations, ProcessorTrait, RuntimeEnv, RuntimeEnvTrait, Table,
    TableBuilder, TableBuilderTrait, TablePublication, TableSubscription,
};
use phymes_data::{
    AvailableCandleOperators, AvailableJinja2Templates, DataAggregatorOperator, DataCastOperator,
    DataColumnOperator, DataConfig,
};
use serde_json::json;

use crate::{AvailableInterfaceSubjects, AvailableProcessors, CustomAgentsBuilderTrait, TaskPlan};

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
pub struct DiagnosticSession<'a> {
    // /// Extract data from inbox subtask
    // pub metrics_from_inbox_task_name: &'a str,
    // pub traces_from_inbox_task_name: &'a str,
    // pub events_from_inbox_task_name: &'a str,
    // pub metrics_from_inbox_processor_name: &'a str,
    // pub traces_from_inbox_processor_name: &'a str,
    // pub events_from_inbox_processor_name: &'a str,

    // /// Make outbox attachment subtask
    // pub metrics_mermaid_outbox_task_name: &'a str,
    // pub traces_mermaid_outbox_task_name: &'a str,
    // pub events_mermaid_outbox_task_name: &'a str,
    // pub metrics_mermaid_outbox_processor_name: &'a str,
    // pub traces_mermaid_outbox_processor_name: &'a str,
    // pub events_mermaid_outbox_processor_name: &'a str,
    /// Metrics analytics
    pub metrics_pivot_task_name: &'a str,
    pub metrics_pivot_processor_name: &'a str,
    pub metrics_normalize_time_task_name: &'a str,
    pub metrics_normalize_time_processor_name: &'a str,
    pub metrics_processors_traces_select_and_cast_to_gantt_task_name: &'a str,
    pub metrics_processors_traces_select_and_cast_to_gantt_processor_name: &'a str,
    pub metrics_elapsed_compute_select_and_cast_to_gantt_task_name: &'a str,
    pub metrics_elapsed_compute_select_and_cast_to_gantt_processor_name: &'a str,
    pub metrics_output_rows_select_and_cast_to_gantt_task_name: &'a str,
    pub metrics_output_rows_select_and_cast_to_gantt_processor_name: &'a str,
    pub metrics_processors_traces_apply_gantt_task_name: &'a str,
    pub metrics_processors_traces_apply_gantt_processor_name: &'a str,
    pub metrics_elapsed_compute_apply_gantt_task_name: &'a str,
    pub metrics_elapsed_compute_apply_gantt_processor_name: &'a str,
    pub metrics_output_rows_apply_gantt_task_name: &'a str,
    pub metrics_output_rows_apply_gantt_processor_name: &'a str,
    pub metrics_runtime_env_name: &'a str,
    pub metrics_processors_traces_runtime_env_name: &'a str,
    pub metrics_elapsed_compute_runtime_env_name: &'a str,
    pub metrics_output_rows_runtime_env_name: &'a str,

    /// Traces analytics
    pub traces_to_sequence_diagram_messages_task_name: &'a str,
    pub traces_to_sequence_diagram_messages_processor_name: &'a str,
    pub apply_sequence_diagram_messages_task_name: &'a str,
    pub apply_sequence_diagram_messages_processor_name: &'a str,
    pub select_sequence_diagram_messages_task_name: &'a str,
    pub select_sequence_diagram_messages_processor_name: &'a str,
    pub session_tasks_to_sequence_diagram_participants_task_name: &'a str,
    pub session_tasks_to_sequence_diagram_participants_processor_name: &'a str,
    pub apply_sequence_diagram_participants_task_name: &'a str,
    pub apply_sequence_diagram_participants_processor_name: &'a str,
    pub select_sequence_diagram_participants_task_name: &'a str,
    pub select_sequence_diagram_participants_processor_name: &'a str,
    pub traces_aggregate_sequence_diagram_content_task_name: &'a str,
    pub traces_aggregate_sequence_diagram_content_processor_name: &'a str,
    pub apply_sequence_diagram_task_name: &'a str,
    pub apply_sequence_diagram_processor_name: &'a str,
    pub traces_runtime_env_name: &'a str,

    /// Errors analytics
    pub errors_select_and_cast_to_kanban_task_name: &'a str,
    pub errors_select_and_cast_to_kanban_processor_name: &'a str,
    pub errors_runtime_env_name: &'a str,
    pub errors_apply_kanban_task_name: &'a str,
    pub errors_apply_kanban_processor_name: &'a str,

    /// Events analytics
    pub events_select_and_cast_to_kanban_task_name: &'a str,
    pub events_select_and_cast_to_kanban_processor_1_name: &'a str,
    pub events_select_and_cast_to_kanban_processor_2_name: &'a str,
    pub events_select_and_cast_tmp: &'a str,
    pub events_apply_kanban_task_name: &'a str,
    pub events_apply_kanban_processor_name: &'a str,
    pub events_runtime_env_name: &'a str,

    /// Outbox
    pub aggregate_visualizations_task_name: &'a str,
    pub aggregate_visualizations_processor_name: &'a str,

    /// Session
    pub session_context_name: &'a str,
}

impl<'a> DiagnosticSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        DiagnosticSession {
            session_context_name,
            ..Default::default()
        }
    }
}

impl Default for DiagnosticSession<'_> {
    fn default() -> Self {
        DiagnosticSession {
            session_context_name: "session_context_name",

            // Metrics analytics
            metrics_pivot_task_name: "metrics_pivot_task_name",
            metrics_pivot_processor_name: "metrics_pivot_processor_name",
            metrics_normalize_time_task_name: "metrics_normalize_time_task_name",
            metrics_normalize_time_processor_name: "metrics_normalize_time_processor_name",
            metrics_processors_traces_select_and_cast_to_gantt_task_name: "metrics_processors_traces_select_and_cast_to_gantt_task_name",
            metrics_processors_traces_select_and_cast_to_gantt_processor_name: "metrics_processors_traces_select_and_cast_to_gantt_processor_name",
            metrics_elapsed_compute_select_and_cast_to_gantt_task_name: "metrics_elapsed_compute_select_and_cast_to_gantt_task_name",
            metrics_elapsed_compute_select_and_cast_to_gantt_processor_name: "metrics_elapsed_compute_select_and_cast_to_gantt_processor_name",
            metrics_output_rows_select_and_cast_to_gantt_task_name: "metrics_output_rows_select_and_cast_to_gantt_task_name",
            metrics_output_rows_select_and_cast_to_gantt_processor_name: "metrics_output_rows_select_and_cast_to_gantt_processor_name",
            metrics_processors_traces_apply_gantt_task_name: "Processor traces gantt",
            metrics_processors_traces_apply_gantt_processor_name: "metrics_processors_traces_apply_gantt_processor_name",
            metrics_elapsed_compute_apply_gantt_task_name: "Elapsed compute barplot",
            metrics_elapsed_compute_apply_gantt_processor_name: "metrics_elapsed_compute_apply_gantt_processor_name",
            metrics_output_rows_apply_gantt_task_name: "Output rows barplot",
            metrics_output_rows_apply_gantt_processor_name: "metrics_output_rows_apply_gantt_processor_name",
            metrics_runtime_env_name: "metrics_runtime_env_name",
            metrics_processors_traces_runtime_env_name: "metrics_processors_traces_runtime_env_name",
            metrics_elapsed_compute_runtime_env_name: "metrics_elapsed_compute_runtime_env_name",
            metrics_output_rows_runtime_env_name: "metrics_output_rows_runtime_env_name",

            // Traces analytics
            traces_to_sequence_diagram_messages_task_name: "traces_to_sequence_diagram_messages_task_name",
            traces_to_sequence_diagram_messages_processor_name: "traces_to_sequence_diagram_messages_processor_name",
            apply_sequence_diagram_messages_task_name: "apply_sequence_diagram_messages_task_name",
            apply_sequence_diagram_messages_processor_name: "apply_sequence_diagram_messages_processor_name",
            select_sequence_diagram_messages_task_name: "select_sequence_diagram_messages_task_name",
            select_sequence_diagram_messages_processor_name: "select_sequence_diagram_messages_processor_name",
            session_tasks_to_sequence_diagram_participants_task_name: "session_tasks_to_sequence_diagram_participants_task_name",
            session_tasks_to_sequence_diagram_participants_processor_name: "session_tasks_to_sequence_diagram_participants_processor_name",
            apply_sequence_diagram_participants_task_name: "apply_sequence_diagram_participants_task_name",
            apply_sequence_diagram_participants_processor_name: "apply_sequence_diagram_participants_processor_name",
            select_sequence_diagram_participants_task_name: "select_sequence_diagram_participants_task_name",
            select_sequence_diagram_participants_processor_name: "select_sequence_diagram_participants_processor_name",
            traces_aggregate_sequence_diagram_content_task_name: "traces_aggregate_sequence_diagram_content_task_name",
            traces_aggregate_sequence_diagram_content_processor_name: "traces_aggregate_sequence_diagram_content_processor_name",
            apply_sequence_diagram_task_name: "apply_sequence_diagram_task_name",
            apply_sequence_diagram_processor_name: "apply_sequence_diagram_processor_name",
            traces_runtime_env_name: "traces_runtime_env_name",

            // Errors analytics
            errors_select_and_cast_to_kanban_task_name: "errors_select_and_cast_to_kanban_task_name",
            errors_select_and_cast_to_kanban_processor_name: "errors_select_and_cast_to_kanban_processor_name",
            errors_runtime_env_name: "errors_runtime_env_name",
            errors_apply_kanban_task_name: "errors_apply_kanban_task_name",
            errors_apply_kanban_processor_name: "errors_apply_kanban_processor_name",

            // Events analytics
            events_select_and_cast_to_kanban_task_name: "events_select_and_cast_to_kanban_task_name",
            events_select_and_cast_to_kanban_processor_1_name: "events_select_and_cast_to_kanban_processor_1_name",
            events_select_and_cast_to_kanban_processor_2_name: "events_select_and_cast_to_kanban_processor_2_name",
            events_select_and_cast_tmp: "events_select_and_cast_tmp",
            events_apply_kanban_task_name: "events_apply_kanban_task_name",
            events_apply_kanban_processor_name: "events_apply_kanban_processor_name",
            events_runtime_env_name: "events_runtime_env_name",

            // Outbox
            aggregate_visualizations_task_name: "aggregate_visualizations_task_name",
            aggregate_visualizations_processor_name: "aggregate_visualizations_processor_name",
        }
    }
}

impl CustomAgentsBuilderTrait for DiagnosticSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![
            TaskPlan {
                task_name: self.metrics_pivot_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_pivot_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_normalize_time_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_normalize_time_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self
                    .metrics_processors_traces_select_and_cast_to_gantt_task_name
                    .to_string(),
                runtime_env_name: self.metrics_processors_traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.metrics_processors_traces_select_and_cast_to_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                    .to_string(),
                runtime_env_name: self.metrics_elapsed_compute_runtime_env_name.to_string(),
                processor_names: vec![
                    self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_output_rows_select_and_cast_to_gantt_task_name
                    .to_string(),
                runtime_env_name: self.metrics_output_rows_runtime_env_name.to_string(),
                processor_names: vec![
                    self.metrics_output_rows_select_and_cast_to_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_processors_traces_apply_gantt_task_name
                    .to_string(),
                runtime_env_name: self.metrics_processors_traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.metrics_processors_traces_apply_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_elapsed_compute_apply_gantt_task_name
                    .to_string(),
                runtime_env_name: self.metrics_elapsed_compute_runtime_env_name.to_string(),
                processor_names: vec![
                    self.metrics_elapsed_compute_apply_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.metrics_output_rows_apply_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_output_rows_runtime_env_name.to_string(),
                processor_names: vec![
                    self.metrics_output_rows_apply_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .traces_to_sequence_diagram_messages_task_name
                    .to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.traces_to_sequence_diagram_messages_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.apply_sequence_diagram_messages_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.select_sequence_diagram_messages_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.select_sequence_diagram_messages_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .session_tasks_to_sequence_diagram_participants_task_name
                    .to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.session_tasks_to_sequence_diagram_participants_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .apply_sequence_diagram_participants_task_name
                    .to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.apply_sequence_diagram_participants_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .select_sequence_diagram_participants_task_name
                    .to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.select_sequence_diagram_participants_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .traces_aggregate_sequence_diagram_content_task_name
                    .to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![
                    self.traces_aggregate_sequence_diagram_content_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.errors_select_and_cast_to_kanban_task_name.to_string(),
                runtime_env_name: self.errors_runtime_env_name.to_string(),
                processor_names: vec![
                    self.errors_select_and_cast_to_kanban_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.errors_apply_kanban_task_name.to_string(),
                runtime_env_name: self.errors_runtime_env_name.to_string(),
                processor_names: vec![self.errors_apply_kanban_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![
                    self.events_select_and_cast_to_kanban_processor_1_name
                        .to_string(),
                    self.events_select_and_cast_to_kanban_processor_2_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.events_apply_kanban_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![self.events_apply_kanban_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.aggregate_visualizations_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.aggregate_visualizations_processor_name.to_string()],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            AvailableProcessors::Pivot.build_arc(
                self.metrics_pivot_processor_name,
                &[TablePublication::Replace {
                    table_name: AvailableSubjects::MetricPivot.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::AnalyticsMetrics.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.metrics_pivot_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::NormalizeTime.build_arc(
                self.metrics_normalize_time_processor_name,
                &[TablePublication::Replace {
                    table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivot.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.metrics_normalize_time_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.metrics_processors_traces_select_and_cast_to_gantt_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .metrics_processors_traces_select_and_cast_to_gantt_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .metrics_processors_traces_select_and_cast_to_gantt_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .metrics_elapsed_compute_select_and_cast_to_gantt_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.metrics_output_rows_select_and_cast_to_gantt_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .metrics_output_rows_select_and_cast_to_gantt_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .metrics_output_rows_select_and_cast_to_gantt_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.metrics_processors_traces_apply_gantt_processor_name,
                &[TablePublication::Replace {
                    table_name: DiagnosticsVisualizations::MetricProcessorTracesGantt.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .metrics_processors_traces_select_and_cast_to_gantt_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .metrics_processors_traces_apply_gantt_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.metrics_elapsed_compute_apply_gantt_processor_name,
                &[TablePublication::Replace {
                    table_name: DiagnosticsVisualizations::MetricElapsedComputeGantt.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .metrics_elapsed_compute_apply_gantt_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.metrics_output_rows_apply_gantt_processor_name,
                &[TablePublication::Replace {
                    table_name: DiagnosticsVisualizations::MetricOutputRowsGantt.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .metrics_output_rows_select_and_cast_to_gantt_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .metrics_output_rows_apply_gantt_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::CandleDataProcessor.build_arc(
                self.traces_to_sequence_diagram_messages_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .traces_to_sequence_diagram_messages_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::AnalyticsTraces.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::AnalyticsTasks.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .traces_to_sequence_diagram_messages_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.apply_sequence_diagram_messages_processor_name,
                &[TablePublication::Replace {
                    table_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .traces_to_sequence_diagram_messages_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .apply_sequence_diagram_messages_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_sequence_diagram_messages_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_sequence_diagram_messages_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .select_sequence_diagram_messages_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::CandleDataProcessor.build_arc(
                self.session_tasks_to_sequence_diagram_participants_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .session_tasks_to_sequence_diagram_participants_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::AnalyticsTasks.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .traces_to_sequence_diagram_messages_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .session_tasks_to_sequence_diagram_participants_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.apply_sequence_diagram_participants_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .apply_sequence_diagram_participants_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .session_tasks_to_sequence_diagram_participants_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .apply_sequence_diagram_participants_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_sequence_diagram_participants_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .select_sequence_diagram_participants_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .apply_sequence_diagram_participants_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .select_sequence_diagram_participants_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::MessageAggregatorProcessor.build_arc(
                self.traces_aggregate_sequence_diagram_content_processor_name,
                &[TablePublication::Replace {
                    table_name: self
                        .traces_aggregate_sequence_diagram_content_task_name
                        .to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .select_sequence_diagram_participants_task_name
                            .to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.select_sequence_diagram_messages_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .traces_aggregate_sequence_diagram_content_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.apply_sequence_diagram_processor_name,
                &[TablePublication::Replace {
                    table_name: DiagnosticsVisualizations::TraceSequenceDiagram.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self
                            .traces_aggregate_sequence_diagram_content_task_name
                            .to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.apply_sequence_diagram_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.errors_select_and_cast_to_kanban_processor_name,
                &[TablePublication::Replace {
                    table_name: self.errors_select_and_cast_to_kanban_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::AnalyticsErrors.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .errors_select_and_cast_to_kanban_processor_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.errors_apply_kanban_processor_name,
                &[TablePublication::Replace {
                    table_name: DiagnosticsVisualizations::ErrorKanban.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.errors_select_and_cast_to_kanban_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.errors_apply_kanban_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.events_select_and_cast_to_kanban_processor_1_name,
                &[TablePublication::Replace {
                    table_name: self.events_select_and_cast_tmp.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::AnalyticsEvents.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .events_select_and_cast_to_kanban_processor_1_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.events_select_and_cast_to_kanban_processor_2_name,
                &[TablePublication::Replace {
                    table_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.events_select_and_cast_tmp.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self
                            .events_select_and_cast_to_kanban_processor_2_name
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::ApplyTemplate.build_arc(
                self.events_apply_kanban_processor_name,
                &[TablePublication::Replace {
                    table_name: DiagnosticsVisualizations::EventKanban.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.events_apply_kanban_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::AttachmentAggregatorProcessor.build_arc(
                self.aggregate_visualizations_processor_name,
                &[TablePublication::Extend {
                    table_name: AvailableInterfaceSubjects::AggregatedAttachments.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: DiagnosticsVisualizations::MetricProcessorTracesGantt
                            .to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: DiagnosticsVisualizations::MetricElapsedComputeGantt
                            .to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: DiagnosticsVisualizations::MetricOutputRowsGantt.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: DiagnosticsVisualizations::TraceSequenceDiagram.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: DiagnosticsVisualizations::EventKanban.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: DiagnosticsVisualizations::ErrorKanban.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.aggregate_visualizations_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AnyTableNameSubscribe.build(),
                // AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(), // DM: Was Replace Publish, but changed to Extend
            ),
        ];

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name(self.metrics_runtime_env_name),
            RuntimeEnv::new().with_name(self.metrics_processors_traces_runtime_env_name),
            RuntimeEnv::new().with_name(self.metrics_elapsed_compute_runtime_env_name),
            RuntimeEnv::new().with_name(self.metrics_output_rows_runtime_env_name),
            RuntimeEnv::new().with_name(self.traces_runtime_env_name),
            RuntimeEnv::new().with_name(self.events_runtime_env_name),
            RuntimeEnv::new().with_name(self.errors_runtime_env_name),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Metrics pivot
        let metrics_pivot_config = DataConfig {
            lhs_name: Some(AvailableSubjects::AnalyticsMetrics.to_string()),
            lhs_values: Some(vec![
                "span_name".to_string(),
                "span_id".to_string(),
                "parent_name".to_string(),
                "parent_id".to_string(),
            ]),
            agg_columns: Some(vec!["metric_value".to_string()]),
            agg_operators: Some(vec![DataAggregatorOperator::Sum]),
            default_values: Some(vec!["0".to_string()]),
            pvt_columns: Some(vec!["metric_name".to_string()]),
            operator: AvailableCandleOperators::Pivot,
            cpu: true,
            ..Default::default()
        };
        let metrics_pivot_config_json = serde_json::to_vec(&metrics_pivot_config).unwrap();
        let metrics_pivot_config_1_state = TableBuilder::new()
            .with_name(self.metrics_pivot_processor_name)
            .with_json(&metrics_pivot_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics normalize time
        let metrics_normalize_time_config = DataConfig {
            lhs_name: Some(AvailableSubjects::MetricPivot.to_string()),
            lhs_values: Some(vec![
                "start_timestamp-metric_value-Sum".to_string(),
                "end_timestamp-metric_value-Sum".to_string(),
            ]),
            operator: AvailableCandleOperators::NormalizeTime,
            ..Default::default()
        };
        let metrics_normalize_time_config_json =
            serde_json::to_vec(&metrics_normalize_time_config).unwrap();
        let metrics_normalize_time_config_1_state = TableBuilder::new()
            .with_name(self.metrics_normalize_time_processor_name)
            .with_json(&metrics_normalize_time_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics processor traces select and cast
        let metrics_processors_traces_select_and_cast_to_gantt_config = DataConfig {
            lhs_name: Some(AvailableSubjects::MetricPivotNormTime.to_string()),
            lhs_values: Some(vec![
                "span_name".to_string(),
                "parent_name".to_string(),
                "start_timestamp-metric_value-Sum-normalized".to_string(),
                "end_timestamp-metric_value-Sum-normalized".to_string(),
            ]),
            rhs_values: Some(vec![
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            as_columns: Some(vec![
                "section".to_string(),
                "task".to_string(),
                "start".to_string(),
                "end".to_string(),
            ]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
                DataCastOperator::Cast,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
            ]),
            cast_templates: Some(vec![
                "Traces[ns]".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let metrics_processors_traces_select_and_cast_to_gantt_config_json =
            serde_json::to_vec(&metrics_processors_traces_select_and_cast_to_gantt_config).unwrap();
        let metrics_processors_traces_select_and_cast_to_gantt_config_1_state = TableBuilder::new()
            .with_name(self.metrics_processors_traces_select_and_cast_to_gantt_processor_name)
            .with_json(
                &metrics_processors_traces_select_and_cast_to_gantt_config_json.clone(),
                1,
            )
            .unwrap()
            .build()
            .unwrap();

        // Metrics processor traces select and cast
        let metrics_elapsed_compute_select_and_cast_to_gantt_config = DataConfig {
            lhs_name: Some(AvailableSubjects::MetricPivotNormTime.to_string()),
            lhs_values: Some(vec![
                "span_name".to_string(),
                "parent_name".to_string(),
                "span_name".to_string(),
                "elapsed_compute-metric_value-Sum".to_string(),
            ]),
            rhs_values: Some(vec![
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            as_columns: Some(vec![
                "section".to_string(),
                "task".to_string(),
                "start".to_string(),
                "end".to_string(),
            ]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
            ]),
            cast_templates: Some(vec![
                "Time[ns]".to_string(),
                "".to_string(),
                "0".to_string(),
                "".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let metrics_elapsed_compute_select_and_cast_to_gantt_config_json =
            serde_json::to_vec(&metrics_elapsed_compute_select_and_cast_to_gantt_config).unwrap();
        let metrics_elapsed_compute_select_and_cast_to_gantt_config_1_state = TableBuilder::new()
            .with_name(self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name)
            .with_json(
                &metrics_elapsed_compute_select_and_cast_to_gantt_config_json.clone(),
                1,
            )
            .unwrap()
            .build()
            .unwrap();

        // Metrics output rows select and cast
        let metrics_output_rows_select_and_cast_to_gantt_config = DataConfig {
            lhs_name: Some(AvailableSubjects::MetricPivotNormTime.to_string()),
            lhs_values: Some(vec![
                "span_name".to_string(),
                "parent_name".to_string(),
                "span_name".to_string(),
                "output_rows-metric_value-Sum".to_string(),
            ]),
            rhs_values: Some(vec![
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            as_columns: Some(vec![
                "section".to_string(),
                "task".to_string(),
                "start".to_string(),
                "end".to_string(),
            ]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
            ]),
            cast_templates: Some(vec![
                "Counts".to_string(),
                "".to_string(),
                "0".to_string(),
                "".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let metrics_output_rows_select_and_cast_to_gantt_config_json =
            serde_json::to_vec(&metrics_output_rows_select_and_cast_to_gantt_config).unwrap();
        let metrics_output_rows_select_and_cast_to_gantt_config_1_state = TableBuilder::new()
            .with_name(self.metrics_output_rows_select_and_cast_to_gantt_processor_name)
            .with_json(
                &metrics_output_rows_select_and_cast_to_gantt_config_json.clone(),
                1,
            )
            .unwrap()
            .build()
            .unwrap();

        // Metrics processor traces apply gantt
        let metrics_processors_traces_apply_gantt_config = DataConfig {
            lhs_name: Some(
                self.metrics_processors_traces_select_and_cast_to_gantt_task_name
                    .to_string(),
            ),
            doc_template: Some(AvailableJinja2Templates::MermaidGanttTemplate),
            doc_name: Some(DiagnosticsVisualizations::MetricProcessorTracesGantt.to_string()),
            doc_input: Some(
                serde_json::to_string(&json!({
                "title": self.metrics_processors_traces_apply_gantt_task_name,
                "dateFormat": "x",
                "axisFormat": "%s"}))
                .unwrap(),
            ),
            format: Some(DataFormat::Txt),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let metrics_processors_traces_apply_gantt_config_json =
            serde_json::to_vec(&metrics_processors_traces_apply_gantt_config).unwrap();
        let metrics_processors_traces_apply_gantt_config_state = TableBuilder::new()
            .with_name(self.metrics_processors_traces_apply_gantt_processor_name)
            .with_json(&metrics_processors_traces_apply_gantt_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics elapsed compute apply gantt
        let metrics_elapsed_compute_apply_gantt_config = DataConfig {
            lhs_name: Some(
                self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                    .to_string(),
            ),
            doc_template: Some(AvailableJinja2Templates::MermaidGanttTemplate),
            doc_name: Some(DiagnosticsVisualizations::MetricElapsedComputeGantt.to_string()),
            doc_input: Some(
                serde_json::to_string(&json!({
                "title": self.metrics_elapsed_compute_apply_gantt_task_name,
                "dateFormat": "X",
                "axisFormat": "%s"}))
                .unwrap(),
            ),
            format: Some(DataFormat::Txt),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let metrics_elapsed_compute_apply_gantt_config_json =
            serde_json::to_vec(&metrics_elapsed_compute_apply_gantt_config).unwrap();
        let metrics_elapsed_compute_apply_gantt_config_state = TableBuilder::new()
            .with_name(self.metrics_elapsed_compute_apply_gantt_processor_name)
            .with_json(&metrics_elapsed_compute_apply_gantt_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics output rows apply gantt
        let metrics_output_rows_apply_gantt_config = DataConfig {
            lhs_name: Some(
                self.metrics_output_rows_select_and_cast_to_gantt_task_name
                    .to_string(),
            ),
            doc_template: Some(AvailableJinja2Templates::MermaidGanttTemplate),
            doc_name: Some(DiagnosticsVisualizations::MetricOutputRowsGantt.to_string()),
            doc_input: Some(
                serde_json::to_string(&json!({
                "title": self.metrics_output_rows_apply_gantt_task_name,
                "dateFormat": "X",
                "axisFormat": "%s"}))
                .unwrap(),
            ),
            format: Some(DataFormat::Txt),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let metrics_output_rows_apply_gantt_config_json =
            serde_json::to_vec(&metrics_output_rows_apply_gantt_config).unwrap();
        let metrics_output_rows_apply_gantt_config_state = TableBuilder::new()
            .with_name(self.metrics_output_rows_apply_gantt_processor_name)
            .with_json(&metrics_output_rows_apply_gantt_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Traces to sequence diagram messages
        let traces_to_sequence_diagram_messages_config = DataConfig {
            lhs_name: Some(AvailableSubjects::AnalyticsTraces.to_string()),
            rhs_name: Some(AvailableSubjects::AnalyticsTasks.to_string()),
            operator: AvailableCandleOperators::FromTracesToMessages,
            ..Default::default()
        };
        let traces_to_sequence_diagram_messages_config_json =
            serde_json::to_vec(&traces_to_sequence_diagram_messages_config).unwrap();
        let traces_to_sequence_diagram_messages_config_state = TableBuilder::new()
            .with_name(self.traces_to_sequence_diagram_messages_processor_name)
            .with_json(&traces_to_sequence_diagram_messages_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Traces apply sequence diagram messages
        let apply_sequence_diagram_messages_config = DataConfig {
            lhs_name: Some(
                self.traces_to_sequence_diagram_messages_task_name
                    .to_string(),
            ),
            doc_template: Some(AvailableJinja2Templates::MermaidSequenceDiagramMessagesTemplate),
            doc_name: Some(self.apply_sequence_diagram_messages_task_name.to_string()),
            doc_input: Some(serde_json::to_string(&json!({})).unwrap()),
            format: Some(DataFormat::None),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let apply_sequence_diagram_messages_config_json =
            serde_json::to_vec(&apply_sequence_diagram_messages_config).unwrap();
        let apply_sequence_diagram_messages_config_state = TableBuilder::new()
            .with_name(self.apply_sequence_diagram_messages_processor_name)
            .with_json(&apply_sequence_diagram_messages_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Traces select sequence diagram messages
        let select_sequence_diagram_messages_config = DataConfig {
            lhs_name: Some(self.apply_sequence_diagram_messages_task_name.to_string()),
            lhs_values: Some(vec![
                "role".to_string(),
                "content".to_string(),
                "timestamp".to_string(),
            ]),
            rhs_values: Some(vec!["".to_string(), "".to_string(), "".to_string()]),
            as_columns: Some(vec!["".to_string(), "".to_string(), "".to_string()]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Int64.to_string(),
            ]),
            cast_templates: Some(vec![
                "1".to_string(), // Needs to come second in the aggregation
                "".to_string(),
                "".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let select_sequence_diagram_messages_config_json =
            serde_json::to_vec(&select_sequence_diagram_messages_config).unwrap();
        let select_sequence_diagram_messages_config_state = TableBuilder::new()
            .with_name(self.select_sequence_diagram_messages_processor_name)
            .with_json(&select_sequence_diagram_messages_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Traces to sequence diagram participants
        let session_tasks_to_sequence_diagram_participants_config = DataConfig {
            lhs_name: Some(AvailableSubjects::AnalyticsTasks.to_string()),
            rhs_name: Some(
                self.traces_to_sequence_diagram_messages_task_name
                    .to_string(),
            ),
            operator: AvailableCandleOperators::FromTasksToParticipants,
            ..Default::default()
        };
        let session_tasks_to_sequence_diagram_participants_config_json =
            serde_json::to_vec(&session_tasks_to_sequence_diagram_participants_config).unwrap();
        let session_tasks_to_sequence_diagram_participants_config_state = TableBuilder::new()
            .with_name(self.session_tasks_to_sequence_diagram_participants_processor_name)
            .with_json(
                &session_tasks_to_sequence_diagram_participants_config_json,
                1,
            )
            .unwrap()
            .build()
            .unwrap();

        // Traces apply sequence diagram participants
        let apply_sequence_diagram_participants_config = DataConfig {
            lhs_name: Some(
                self.session_tasks_to_sequence_diagram_participants_task_name
                    .to_string(),
            ),
            doc_template: Some(
                AvailableJinja2Templates::MermaidSequenceDiagramParticipantsTemplate,
            ),
            doc_name: Some(
                self.apply_sequence_diagram_participants_task_name
                    .to_string(),
            ),
            doc_input: Some(serde_json::to_string(&json!({})).unwrap()),
            format: Some(DataFormat::None),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let apply_sequence_diagram_participants_config_json =
            serde_json::to_vec(&apply_sequence_diagram_participants_config).unwrap();
        let apply_sequence_diagram_participants_config_state = TableBuilder::new()
            .with_name(self.apply_sequence_diagram_participants_processor_name)
            .with_json(&apply_sequence_diagram_participants_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Traces select sequence diagram participants
        let select_sequence_diagram_participants_config = DataConfig {
            lhs_name: Some(
                self.apply_sequence_diagram_participants_task_name
                    .to_string(),
            ),
            lhs_values: Some(vec![
                "role".to_string(),
                "content".to_string(),
                "timestamp".to_string(),
            ]),
            rhs_values: Some(vec!["".to_string(), "".to_string(), "".to_string()]),
            as_columns: Some(vec!["".to_string(), "".to_string(), "".to_string()]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Int64.to_string(),
            ]),
            cast_templates: Some(vec![
                "0".to_string(), // Needs to come first in the aggregation
                "".to_string(),
                "".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let select_sequence_diagram_participants_config_json =
            serde_json::to_vec(&select_sequence_diagram_participants_config).unwrap();
        let select_sequence_diagram_participants_config_state = TableBuilder::new()
            .with_name(self.select_sequence_diagram_participants_processor_name)
            .with_json(&select_sequence_diagram_participants_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Traces apply sequence diagram
        let apply_sequence_diagram_config = DataConfig {
            lhs_name: Some(
                self.traces_aggregate_sequence_diagram_content_task_name
                    .to_string(),
            ),
            doc_template: Some(AvailableJinja2Templates::MermaidSequenceDiagramTemplate),
            doc_name: Some(DiagnosticsVisualizations::TraceSequenceDiagram.to_string()),
            doc_input: Some(serde_json::to_string(&json!({})).unwrap()),
            format: Some(DataFormat::Txt),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let apply_sequence_diagram_config_json =
            serde_json::to_vec(&apply_sequence_diagram_config).unwrap();
        let apply_sequence_diagram_config_state = TableBuilder::new()
            .with_name(self.apply_sequence_diagram_processor_name)
            .with_json(&apply_sequence_diagram_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Errors select and cast kanban
        let errors_select_and_cast_to_kanban_config = DataConfig {
            lhs_name: Some(AvailableSubjects::AnalyticsErrors.to_string()),
            lhs_values: Some(vec![
                "role".to_string(),
                "role".to_string(),
                "timestamp".to_string(),
                "content".to_string(),
                "role".to_string(),
                "timestamp".to_string(),
                "role".to_string(),
            ]),
            rhs_values: Some(vec![
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            as_columns: Some(vec![
                "column_name".to_string(),
                "column_label".to_string(),
                "task_name".to_string(),
                "task_description".to_string(),
                "task_assigned".to_string(),
                "task_ticket".to_string(),
                "task_priority".to_string(),
            ]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
                DataCastOperator::None,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
            ]),
            cast_templates: Some(vec![
                "Error".to_string(),
                "Error".to_string(),
                "".to_string(),
                "".to_string(),
                "''".to_string(),
                "".to_string(),
                "Low".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let errors_select_and_cast_to_kanban_config_json =
            serde_json::to_vec(&errors_select_and_cast_to_kanban_config).unwrap();
        let errors_select_and_cast_to_kanban_config_state = TableBuilder::new()
            .with_name(self.errors_select_and_cast_to_kanban_processor_name)
            .with_json(&errors_select_and_cast_to_kanban_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Errors apply kanban
        let errors_apply_kanban_config = DataConfig {
            lhs_name: Some(self.errors_select_and_cast_to_kanban_task_name.to_string()),
            doc_template: Some(AvailableJinja2Templates::MermaidKanbanTemplate),
            doc_name: Some(DiagnosticsVisualizations::ErrorKanban.to_string()),
            doc_input: Some(serde_json::to_string(&json!({})).unwrap()),
            format: Some(DataFormat::Txt),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let errors_apply_kanban_config_json =
            serde_json::to_vec(&errors_apply_kanban_config).unwrap();
        let errors_apply_kanban_config_state = TableBuilder::new()
            .with_name(self.errors_apply_kanban_processor_name)
            .with_json(&errors_apply_kanban_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Events select and cast kanban
        let events_select_and_cast_to_kanban_1_config = DataConfig {
            lhs_name: Some(AvailableSubjects::AnalyticsEvents.to_string()),
            lhs_values: Some(vec![
                "event_level".to_string(),
                "span_name".to_string(),
                // "function".to_string(),
                "record_name".to_string(),
                "record_value".to_string(),
                // "span_name".to_string(),
                "span_name".to_string(),
                "span_name".to_string(),
                "event_level".to_string(),
                "id".to_string(),
            ]),
            rhs_values: Some(vec![
                "".to_string(),
                "".to_string(),
                // "".to_string(),
                "".to_string(),
                "".to_string(),
                // "function".to_string(),
                "record_name".to_string(),
                "event_level".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            as_columns: Some(vec![
                "".to_string(),
                "".to_string(),
                // "".to_string(),
                "".to_string(),
                "event_level".to_string(),
                // "span_name".to_string(),
                "span_name".to_string(),
                "record_value".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                // DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                // DataColumnOperator::Concat,
                DataColumnOperator::Concat,
                DataColumnOperator::Concat,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                // DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                // DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                // DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                // DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
            ]),
            cast_templates: Some(vec![
                "".to_string(),
                "**Span** {{ span_name }}<br>".to_string(),
                // "**Function** {{ function }}<br>".to_string(),
                "**Record** {{ record_name }}<br>".to_string(),
                "**Value** {{ record_value }}<br>".to_string(),
                // "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let events_select_and_cast_to_kanban_1_config_json =
            serde_json::to_vec(&events_select_and_cast_to_kanban_1_config).unwrap();
        let events_select_and_cast_to_kanban_1_config_state = TableBuilder::new()
            .with_name(self.events_select_and_cast_to_kanban_processor_1_name)
            .with_json(&events_select_and_cast_to_kanban_1_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        let events_select_and_cast_to_kanban_2_config = DataConfig {
            lhs_name: Some(self.events_select_and_cast_tmp.to_string()),
            lhs_values: Some(vec![
                "event_level".to_string(),
                "event_level".to_string(),
                "id".to_string(),
                "record_value".to_string(),
                "event_level".to_string(),
                "id".to_string(),
                "event_level".to_string(),
            ]),
            rhs_values: Some(vec![
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
            ]),
            as_columns: Some(vec![
                "column_name".to_string(),
                "column_label".to_string(),
                "task_name".to_string(),
                "task_description".to_string(),
                "task_assigned".to_string(),
                "task_ticket".to_string(),
                "task_priority".to_string(),
            ]),
            column_operators: Some(vec![
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ]),
            cast_operators: Some(vec![
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Cast,
                DataCastOperator::None,
            ]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
                DataType::Utf8.to_string(),
            ]),
            cast_templates: Some(vec![
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "''".to_string(),
                "".to_string(),
                "Low".to_string(),
            ]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let events_select_and_cast_to_kanban_2_config_json =
            serde_json::to_vec(&events_select_and_cast_to_kanban_2_config).unwrap();
        let events_select_and_cast_to_kanban_2_config_state = TableBuilder::new()
            .with_name(self.events_select_and_cast_to_kanban_processor_2_name)
            .with_json(&events_select_and_cast_to_kanban_2_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Events apply kanban
        let events_apply_kanban_config = DataConfig {
            lhs_name: Some(self.events_select_and_cast_to_kanban_task_name.to_string()),
            doc_template: Some(AvailableJinja2Templates::MermaidKanbanTemplate),
            doc_name: Some(DiagnosticsVisualizations::EventKanban.to_string()),
            doc_input: Some(serde_json::to_string(&json!({})).unwrap()),
            format: Some(DataFormat::Txt),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let events_apply_kanban_config_json =
            serde_json::to_vec(&events_apply_kanban_config).unwrap();
        let events_apply_kanban_config_state = TableBuilder::new()
            .with_name(self.events_apply_kanban_processor_name)
            .with_json(&events_apply_kanban_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // traces and Outbox aggregate
        let aggregator_1_config = DataConfig {
            lhs_values: Some(vec!["role".to_string()]),
            asc: Some(true),
            operator: AvailableCandleOperators::Sort,
            ..Default::default()
        };
        let aggregator_1_config_json = serde_json::to_vec(&aggregator_1_config).unwrap();
        let aggregator_1_state = TableBuilder::new()
            .with_name(self.traces_aggregate_sequence_diagram_content_processor_name)
            .with_json(&aggregator_1_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        let aggregator_2_config = DataConfig {
            lhs_values: Some(vec!["timestamp".to_string()]),
            asc: Some(true),
            operator: AvailableCandleOperators::Sort,
            ..Default::default()
        };
        let aggregator_2_config_json = serde_json::to_vec(&aggregator_2_config).unwrap();
        let aggregator_2_state = TableBuilder::new()
            .with_name(self.aggregate_visualizations_processor_name)
            .with_json(&aggregator_2_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![
            // Processor configs
            metrics_pivot_config_1_state,
            metrics_normalize_time_config_1_state,
            metrics_processors_traces_select_and_cast_to_gantt_config_1_state,
            metrics_elapsed_compute_select_and_cast_to_gantt_config_1_state,
            metrics_output_rows_select_and_cast_to_gantt_config_1_state,
            metrics_processors_traces_apply_gantt_config_state,
            metrics_elapsed_compute_apply_gantt_config_state,
            metrics_output_rows_apply_gantt_config_state,
            traces_to_sequence_diagram_messages_config_state,
            apply_sequence_diagram_messages_config_state,
            select_sequence_diagram_messages_config_state,
            session_tasks_to_sequence_diagram_participants_config_state,
            apply_sequence_diagram_participants_config_state,
            select_sequence_diagram_participants_config_state,
            aggregator_1_state,
            apply_sequence_diagram_config_state,
            errors_select_and_cast_to_kanban_config_state,
            errors_apply_kanban_config_state,
            events_select_and_cast_to_kanban_1_config_state,
            events_select_and_cast_to_kanban_2_config_state,
            events_apply_kanban_config_state,
            aggregator_2_state,
            // Metrics
            AvailableSubjects::AnalyticsMetrics
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::MetricPivot.to_table(None, None).unwrap(),
            AvailableSubjects::MetricPivotNormTime
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::MermaidGanttTemplate
                .to_table(
                    Some(self.metrics_processors_traces_select_and_cast_to_gantt_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::MermaidGanttTemplate
                .to_table(
                    Some(self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::MermaidGanttTemplate
                .to_table(
                    Some(self.metrics_output_rows_select_and_cast_to_gantt_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Blob
                .to_table(
                    Some(
                        DiagnosticsVisualizations::MetricProcessorTracesGantt
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Blob
                .to_table(
                    Some(
                        DiagnosticsVisualizations::MetricElapsedComputeGantt
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Blob
                .to_table(
                    Some(
                        DiagnosticsVisualizations::MetricOutputRowsGantt
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap(),
            // Traces
            AvailableSubjects::AnalyticsTasks
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::MermaidSequenceDiagramParticipantsTemplate
                .to_table(
                    Some(self.session_tasks_to_sequence_diagram_participants_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::AnalyticsTraces
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::MermaidSequenceDiagramMessagesTemplate
                .to_table(
                    Some(self.traces_to_sequence_diagram_messages_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Messages
                .to_table(
                    Some(self.apply_sequence_diagram_participants_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Messages
                .to_table(
                    Some(self.select_sequence_diagram_participants_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Messages
                .to_table(Some(self.apply_sequence_diagram_messages_task_name), None)
                .unwrap(),
            AvailableSubjects::Messages
                .to_table(Some(self.select_sequence_diagram_messages_task_name), None)
                .unwrap(),
            AvailableSubjects::Messages
                .to_table(
                    Some(self.traces_aggregate_sequence_diagram_content_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Blob
                .to_table(
                    Some(
                        DiagnosticsVisualizations::TraceSequenceDiagram
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap(),
            // Errors
            AvailableSubjects::AnalyticsErrors
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::MermaidKanbanTemplate
                .to_table(Some(self.errors_select_and_cast_to_kanban_task_name), None)
                .unwrap(),
            AvailableSubjects::Blob
                .to_table(
                    Some(DiagnosticsVisualizations::ErrorKanban.to_string().as_str()),
                    None,
                )
                .unwrap(),
            // Events
            AvailableSubjects::AnalyticsEvents
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::AnalyticsEvents
                .to_table(Some(self.events_select_and_cast_tmp), None)
                .unwrap(),
            AvailableSubjects::MermaidKanbanTemplate
                .to_table(Some(self.events_select_and_cast_to_kanban_task_name), None)
                .unwrap(),
            AvailableSubjects::Blob
                .to_table(
                    Some(DiagnosticsVisualizations::EventKanban.to_string().as_str()),
                    None,
                )
                .unwrap(),
            // Outbox
            AvailableInterfaceSubjects::AggregatedAttachments
                .to_table(None, None)
                .unwrap(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{BuildableTrait, IPCMessage, MessageBuilderTrait, MessageTrait, TableTrait};
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilderAgentsTrait, SessionContextBuilderTrait, SessionStream,
        SessionStreamState, create_message_map, plans::user_session_inner,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_diagnostic_session() -> Result<()> {
        // initialize the session
        let diagnostic_session = DiagnosticSession::default();
        let session_ctx = diagnostic_session
            .build()
            .with_name(diagnostic_session.session_context_name)
            .with_diagnostics(true) // Debugging
            .add_session_interface(Some(&[AvailableInterfaceSubjects::AggregatedAttachments
                .to_string()
                .as_str()]))?
            .build_with_tables()?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // Make diagnostic data and session tasks data
        let (user_session_stream_state, user_session_stream) = user_session_inner::user_session()?;
        let _user_response: Vec<HashMap<String, IPCMessage>> =
            user_session_stream.try_collect().await?;

        let message_map = {
            let usss = user_session_stream_state.read();
            let table = usss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionMetrics.to_string().as_str())
                .unwrap()
                .read();
            let metrics_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::AnalyticsMetrics.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::AnalyticsMetrics.to_string(),
                })
                .with_publisher(diagnostic_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionTraces.to_string().as_str())
                .unwrap()
                .read();
            let traces_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::AnalyticsTraces.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::AnalyticsTraces.to_string(),
                })
                .with_publisher(diagnostic_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionEvents.to_string().as_str())
                .unwrap()
                .read();
            let events_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::AnalyticsEvents.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::AnalyticsEvents.to_string(),
                })
                .with_publisher(diagnostic_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionTasks.to_string().as_str())
                .unwrap()
                .read();
            let tasks_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::AnalyticsTasks.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::AnalyticsTasks.to_string(),
                })
                .with_publisher(diagnostic_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionErrors.to_string().as_str())
                .unwrap()
                .read();
            if table.count_rows() > 0 {
                let errors_message = IPCMessage::get_builder()
                    .with_message(table.to_ipc_stream()?)
                    .with_subject(AvailableSubjects::AnalyticsErrors.to_string().as_str())
                    .with_update(&TablePublication::Replace {
                        table_name: AvailableSubjects::AnalyticsErrors.to_string(),
                    })
                    .with_publisher(diagnostic_session.session_context_name)
                    .make_name()?
                    .build()?;

                create_message_map(vec![
                    metrics_message,
                    traces_message,
                    events_message,
                    errors_message,
                    tasks_message,
                ])
            } else {
                create_message_map(vec![
                    metrics_message,
                    traces_message,
                    events_message,
                    tasks_message,
                ])
            }
        };

        // Run
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
        let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // let sss = session_stream_state.read();
        // let table = sss
        //     .get_session_context()
        //     .get_states()
        //     .get(AvailableSubjects::SessionErrors.to_string().as_str())
        //     .unwrap()
        //     .read();
        // println!("__ERRORS__");
        // println!("{}", String::from_utf8(table.to_csv(b',', true)?)?);
        // let table = sss
        //     .get_session_context()
        //     .get_states()
        //     .get(AvailableSubjects::SessionTraces.to_string().as_str())
        //     .unwrap()
        //     .read();
        // println!("__TRACES__");
        // println!("{}", String::from_utf8(table.to_csv(b',', true)?)?);

        let bytes = response
            .iter_mut()
            .filter_map(|map| {
                map.remove(&format!(
                    "from_{}_on_{}",
                    diagnostic_session.session_context_name,
                    AvailableInterfaceSubjects::AggregatedAttachments
                ))
                .map(|v| v.get_message_own())
            })
            .collect::<Vec<_>>();
        let attachment_data = bytes
            .into_iter()
            .flat_map(|b| {
                TableBuilder::new_from_ipc_stream(&b)
                    .unwrap()
                    .with_name("")
                    .build()
                    .unwrap()
                    .to_json_object()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for row in &attachment_data {
            let bytes = row["bytes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect::<Vec<u8>>();
            println!(
                "attachment {}.{}: {}",
                row["filename"],
                row["extension"],
                String::from_utf8_lossy(bytes.as_ref()).into_owned()
            )
        }

        Ok(())
    }
}
