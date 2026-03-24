use std::sync::Arc;

use arrow::datatypes::DataType;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableSubscribeEvents, BuildableTrait, BuilderTrait, DataEncoding, DataFormat, DiagnosticsVisualizations, ProcessorPlan, ProcessorPlanBuilder, Publication, RuntimeEnv, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait, Subscription
};
use phymes_data::{
    AvailableCandleOperators, AvailableJinja2Templates, DataAggregatorOperator, DataCastOperator,
    DataColumnOperator, DataConfig,
};
use serde_json::json;

use crate::{AvailableProcessors, TaskPlan, CustomAgentsBuilderTrait};

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
            session_context_name: "diagnostic_session",

            // Metrics analytics
            metrics_pivot_task_name: "metrics_pivot_t",
            metrics_pivot_processor_name: "metrics_pivot_p",
            metrics_normalize_time_task_name: "metrics_normalize_time_t",
            metrics_normalize_time_processor_name: "metrics_normalize_time_p",
            metrics_processors_traces_select_and_cast_to_gantt_task_name: "metrics_processors_traces_select_and_cast_to_gantt_t",
            metrics_processors_traces_select_and_cast_to_gantt_processor_name: "metrics_processors_traces_select_and_cast_to_gantt_p",
            metrics_elapsed_compute_select_and_cast_to_gantt_task_name: "metrics_elapsed_compute_select_and_cast_to_gantt_t",
            metrics_elapsed_compute_select_and_cast_to_gantt_processor_name: "metrics_elapsed_compute_select_and_cast_to_gantt_p",
            metrics_output_rows_select_and_cast_to_gantt_task_name: "metrics_output_rows_select_and_cast_to_gantt_t",
            metrics_output_rows_select_and_cast_to_gantt_processor_name: "metrics_output_rows_select_and_cast_to_gantt_p",
            metrics_processors_traces_apply_gantt_task_name: "Processor_traces_gantt",
            metrics_processors_traces_apply_gantt_processor_name: "metrics_processors_traces_apply_gantt_p",
            metrics_elapsed_compute_apply_gantt_task_name: "Elapsed_compute_barplot",
            metrics_elapsed_compute_apply_gantt_processor_name: "metrics_elapsed_compute_apply_gantt_p",
            metrics_output_rows_apply_gantt_task_name: "Output_rows_barplot",
            metrics_output_rows_apply_gantt_processor_name: "metrics_output_rows_apply_gantt_p",
            metrics_runtime_env_name: "metrics_runtime_env",
            metrics_processors_traces_runtime_env_name: "metrics_processors_traces_runtime_env",
            metrics_elapsed_compute_runtime_env_name: "metrics_elapsed_compute_runtime_env",
            metrics_output_rows_runtime_env_name: "metrics_output_rows_runtime_env",

            // Traces analytics
            traces_to_sequence_diagram_messages_task_name: "traces_to_sequence_diagram_messages_t",
            traces_to_sequence_diagram_messages_processor_name: "traces_to_sequence_diagram_messages_p",
            apply_sequence_diagram_messages_task_name: "apply_sequence_diagram_messages_t",
            apply_sequence_diagram_messages_processor_name: "apply_sequence_diagram_messages_p",
            select_sequence_diagram_messages_task_name: "select_sequence_diagram_messages_t",
            select_sequence_diagram_messages_processor_name: "select_sequence_diagram_messages_p",
            session_tasks_to_sequence_diagram_participants_task_name: "session_tasks_to_sequence_diagram_participants_t",
            session_tasks_to_sequence_diagram_participants_processor_name: "session_tasks_to_sequence_diagram_participants_p",
            apply_sequence_diagram_participants_task_name: "apply_sequence_diagram_participants_t",
            apply_sequence_diagram_participants_processor_name: "apply_sequence_diagram_participants_p",
            select_sequence_diagram_participants_task_name: "select_sequence_diagram_participants_t",
            select_sequence_diagram_participants_processor_name: "select_sequence_diagram_participants_p",
            traces_aggregate_sequence_diagram_content_task_name: "traces_aggregate_sequence_diagram_content_t",
            traces_aggregate_sequence_diagram_content_processor_name: "traces_aggregate_sequence_diagram_content_p",
            apply_sequence_diagram_task_name: "apply_sequence_diagram_t",
            apply_sequence_diagram_processor_name: "apply_sequence_diagram_p",
            traces_runtime_env_name: "traces_runtime_env",

            // Errors analytics
            errors_select_and_cast_to_kanban_task_name: "errors_select_and_cast_to_kanban_t",
            errors_select_and_cast_to_kanban_processor_name: "errors_select_and_cast_to_kanban_p",
            errors_runtime_env_name: "errors_runtime_env",
            errors_apply_kanban_task_name: "errors_apply_kanban_t",
            errors_apply_kanban_processor_name: "errors_apply_kanban_p",

            // Events analytics
            events_select_and_cast_to_kanban_task_name: "events_select_and_cast_to_kanban_t",
            events_select_and_cast_to_kanban_processor_1_name: "events_select_and_cast_to_kanban_processor_1_name",
            events_select_and_cast_to_kanban_processor_2_name: "events_select_and_cast_to_kanban_processor_2_name",
            events_select_and_cast_tmp: "events_select_and_cast_tmp",
            events_apply_kanban_task_name: "events_apply_kanban_t",
            events_apply_kanban_processor_name: "events_apply_kanban_p",
            events_runtime_env_name: "events_runtime_env",
        }
    }
}

impl CustomAgentsBuilderTrait for DiagnosticSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![
            TaskPlan {
                task_name: self.metrics_pivot_task_name.to_string(),
                processor_names: vec![self.metrics_pivot_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_normalize_time_task_name.to_string(),
                processor_names: vec![self.metrics_normalize_time_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self
                    .metrics_processors_traces_select_and_cast_to_gantt_task_name
                    .to_string(),
                processor_names: vec![
                    self.metrics_processors_traces_select_and_cast_to_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                    .to_string(),
                processor_names: vec![
                    self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_output_rows_select_and_cast_to_gantt_task_name
                    .to_string(),
                processor_names: vec![
                    self.metrics_output_rows_select_and_cast_to_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_processors_traces_apply_gantt_task_name
                    .to_string(),
                processor_names: vec![
                    self.metrics_processors_traces_apply_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .metrics_elapsed_compute_apply_gantt_task_name
                    .to_string(),
                processor_names: vec![
                    self.metrics_elapsed_compute_apply_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.metrics_output_rows_apply_gantt_task_name.to_string(),
                processor_names: vec![
                    self.metrics_output_rows_apply_gantt_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .traces_to_sequence_diagram_messages_task_name
                    .to_string(),
                processor_names: vec![
                    self.traces_to_sequence_diagram_messages_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                processor_names: vec![
                    self.apply_sequence_diagram_messages_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.select_sequence_diagram_messages_task_name.to_string(),
                processor_names: vec![
                    self.select_sequence_diagram_messages_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .session_tasks_to_sequence_diagram_participants_task_name
                    .to_string(),
                processor_names: vec![
                    self.session_tasks_to_sequence_diagram_participants_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .apply_sequence_diagram_participants_task_name
                    .to_string(),
                processor_names: vec![
                    self.apply_sequence_diagram_participants_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .select_sequence_diagram_participants_task_name
                    .to_string(),
                processor_names: vec![
                    self.select_sequence_diagram_participants_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .traces_aggregate_sequence_diagram_content_task_name
                    .to_string(),
                processor_names: vec![
                    self.traces_aggregate_sequence_diagram_content_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_task_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.errors_select_and_cast_to_kanban_task_name.to_string(),
                processor_names: vec![
                    self.errors_select_and_cast_to_kanban_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.errors_apply_kanban_task_name.to_string(),
                processor_names: vec![self.errors_apply_kanban_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                processor_names: vec![
                    self.events_select_and_cast_to_kanban_processor_1_name
                        .to_string(),
                    self.events_select_and_cast_to_kanban_processor_2_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.events_apply_kanban_task_name.to_string(),
                processor_names: vec![self.events_apply_kanban_processor_name.to_string()],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Pivot.build_arc(self.metrics_pivot_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableSubjects::MetricPivot.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::AnalyticsMetrics.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.metrics_pivot_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::NormalizeTime
                        .build_arc(self.metrics_normalize_time_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::MetricPivot.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.metrics_normalize_time_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::Select.build_arc(
                    self.metrics_processors_traces_select_and_cast_to_gantt_processor_name,
                ))
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .metrics_processors_traces_select_and_cast_to_gantt_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .metrics_processors_traces_select_and_cast_to_gantt_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::Select.build_arc(
                    self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name,
                ))
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .metrics_elapsed_compute_select_and_cast_to_gantt_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select.build_arc(
                        self.metrics_output_rows_select_and_cast_to_gantt_processor_name,
                    ),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .metrics_output_rows_select_and_cast_to_gantt_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .metrics_output_rows_select_and_cast_to_gantt_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.metrics_processors_traces_apply_gantt_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: DiagnosticsVisualizations::MetricProcessorTracesGantt.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .metrics_processors_traces_select_and_cast_to_gantt_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .metrics_processors_traces_apply_gantt_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.metrics_elapsed_compute_apply_gantt_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: DiagnosticsVisualizations::MetricElapsedComputeGantt.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .metrics_elapsed_compute_select_and_cast_to_gantt_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .metrics_elapsed_compute_apply_gantt_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.metrics_output_rows_apply_gantt_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: DiagnosticsVisualizations::MetricOutputRowsGantt.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .metrics_output_rows_select_and_cast_to_gantt_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .metrics_output_rows_apply_gantt_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleDataProcessor
                        .build_arc(self.traces_to_sequence_diagram_messages_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .traces_to_sequence_diagram_messages_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::AnalyticsTraces.to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::AnalyticsTasks.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .traces_to_sequence_diagram_messages_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.apply_sequence_diagram_messages_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .traces_to_sequence_diagram_messages_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .apply_sequence_diagram_messages_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select
                        .build_arc(self.select_sequence_diagram_messages_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.select_sequence_diagram_messages_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .select_sequence_diagram_messages_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleDataProcessor.build_arc(
                        self.session_tasks_to_sequence_diagram_participants_processor_name,
                    ),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .session_tasks_to_sequence_diagram_participants_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::AnalyticsTasks.to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .traces_to_sequence_diagram_messages_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .session_tasks_to_sequence_diagram_participants_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.apply_sequence_diagram_participants_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .apply_sequence_diagram_participants_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .session_tasks_to_sequence_diagram_participants_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .apply_sequence_diagram_participants_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select
                        .build_arc(self.select_sequence_diagram_participants_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .select_sequence_diagram_participants_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .apply_sequence_diagram_participants_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .select_sequence_diagram_participants_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::AggregatorProcessor
                        .build_arc(self.traces_aggregate_sequence_diagram_content_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self
                        .traces_aggregate_sequence_diagram_content_task_name
                        .to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .select_sequence_diagram_participants_task_name
                            .to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.select_sequence_diagram_messages_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .traces_aggregate_sequence_diagram_content_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.apply_sequence_diagram_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: DiagnosticsVisualizations::TraceSequenceDiagram.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self
                            .traces_aggregate_sequence_diagram_content_task_name
                            .to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.apply_sequence_diagram_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select
                        .build_arc(self.errors_select_and_cast_to_kanban_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.errors_select_and_cast_to_kanban_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::AnalyticsErrors.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .errors_select_and_cast_to_kanban_processor_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.errors_apply_kanban_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: DiagnosticsVisualizations::ErrorKanban.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.errors_select_and_cast_to_kanban_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.errors_apply_kanban_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select
                        .build_arc(self.events_select_and_cast_to_kanban_processor_1_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.events_select_and_cast_tmp.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::AnalyticsEvents.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .events_select_and_cast_to_kanban_processor_1_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select
                        .build_arc(self.events_select_and_cast_to_kanban_processor_2_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.events_select_and_cast_tmp.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self
                            .events_select_and_cast_to_kanban_processor_2_name
                            .to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.events_apply_kanban_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: DiagnosticsVisualizations::EventKanban.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.events_apply_kanban_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        ];

        Some(processors)
    }

    fn make_runtime_env(&self) -> Option<Arc<RuntimeEnv>> {
        Some(RuntimeEnv::get_builder().with_name(self.session_context_name).build_arc().unwrap())
    }

    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
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
        let metrics_pivot_config_1_state = SubjectBuilder::new()
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
        let metrics_normalize_time_config_1_state = SubjectBuilder::new()
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
        let metrics_processors_traces_select_and_cast_to_gantt_config_1_state = SubjectBuilder::new()
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
        let metrics_elapsed_compute_select_and_cast_to_gantt_config_1_state = SubjectBuilder::new()
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
        let metrics_output_rows_select_and_cast_to_gantt_config_1_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Txt),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let metrics_processors_traces_apply_gantt_config_json =
            serde_json::to_vec(&metrics_processors_traces_apply_gantt_config).unwrap();
        let metrics_processors_traces_apply_gantt_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Txt),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let metrics_elapsed_compute_apply_gantt_config_json =
            serde_json::to_vec(&metrics_elapsed_compute_apply_gantt_config).unwrap();
        let metrics_elapsed_compute_apply_gantt_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Txt),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let metrics_output_rows_apply_gantt_config_json =
            serde_json::to_vec(&metrics_output_rows_apply_gantt_config).unwrap();
        let metrics_output_rows_apply_gantt_config_state = SubjectBuilder::new()
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
        let traces_to_sequence_diagram_messages_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::None),
            schema: Some(AvailableSubjects::Messages),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let apply_sequence_diagram_messages_config_json =
            serde_json::to_vec(&apply_sequence_diagram_messages_config).unwrap();
        let apply_sequence_diagram_messages_config_state = SubjectBuilder::new()
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
        let select_sequence_diagram_messages_config_state = SubjectBuilder::new()
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
        let session_tasks_to_sequence_diagram_participants_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::None),
            schema: Some(AvailableSubjects::Messages),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let apply_sequence_diagram_participants_config_json =
            serde_json::to_vec(&apply_sequence_diagram_participants_config).unwrap();
        let apply_sequence_diagram_participants_config_state = SubjectBuilder::new()
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
        let select_sequence_diagram_participants_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Txt),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let apply_sequence_diagram_config_json =
            serde_json::to_vec(&apply_sequence_diagram_config).unwrap();
        let apply_sequence_diagram_config_state = SubjectBuilder::new()
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
        let errors_select_and_cast_to_kanban_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Txt),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let errors_apply_kanban_config_json =
            serde_json::to_vec(&errors_apply_kanban_config).unwrap();
        let errors_apply_kanban_config_state = SubjectBuilder::new()
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
        let events_select_and_cast_to_kanban_1_config_state = SubjectBuilder::new()
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
        let events_select_and_cast_to_kanban_2_config_state = SubjectBuilder::new()
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
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Txt),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableCandleOperators::ApplyTemplate,
            ..Default::default()
        };
        let events_apply_kanban_config_json =
            serde_json::to_vec(&events_apply_kanban_config).unwrap();
        let events_apply_kanban_config_state = SubjectBuilder::new()
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
        let aggregator_1_state = SubjectBuilder::new()
            .with_name(self.traces_aggregate_sequence_diagram_content_processor_name)
            .with_json(&aggregator_1_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let subjects = vec![
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
            // Metrics
            AvailableSubjects::AnalyticsMetrics
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::MetricPivot.to_subject(None, None).unwrap(),
            AvailableSubjects::MetricPivotNormTime
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::MermaidGanttTemplate
                .to_subject(
                    Some(self.metrics_processors_traces_select_and_cast_to_gantt_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::MermaidGanttTemplate
                .to_subject(
                    Some(self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::MermaidGanttTemplate
                .to_subject(
                    Some(self.metrics_output_rows_select_and_cast_to_gantt_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Attachments
                .to_subject(
                    Some(
                        DiagnosticsVisualizations::MetricProcessorTracesGantt
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Attachments
                .to_subject(
                    Some(
                        DiagnosticsVisualizations::MetricElapsedComputeGantt
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Attachments
                .to_subject(
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
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::MermaidSequenceDiagramParticipantsTemplate
                .to_subject(
                    Some(self.session_tasks_to_sequence_diagram_participants_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::AnalyticsTraces
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::MermaidSequenceDiagramMessagesTemplate
                .to_subject(
                    Some(self.traces_to_sequence_diagram_messages_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(
                    Some(self.apply_sequence_diagram_participants_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(
                    Some(self.select_sequence_diagram_participants_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(Some(self.apply_sequence_diagram_messages_task_name), None)
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(Some(self.select_sequence_diagram_messages_task_name), None)
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(
                    Some(self.traces_aggregate_sequence_diagram_content_task_name),
                    None,
                )
                .unwrap(),
            AvailableSubjects::Attachments
                .to_subject(
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
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::MermaidKanbanTemplate
                .to_subject(Some(self.errors_select_and_cast_to_kanban_task_name), None)
                .unwrap(),
            AvailableSubjects::Attachments
                .to_subject(
                    Some(DiagnosticsVisualizations::ErrorKanban.to_string().as_str()),
                    None,
                )
                .unwrap(),
            // Events
            AvailableSubjects::AnalyticsEvents
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::AnalyticsEvents
                .to_subject(Some(self.events_select_and_cast_tmp), None)
                .unwrap(),
            AvailableSubjects::MermaidKanbanTemplate
                .to_subject(Some(self.events_select_and_cast_to_kanban_task_name), None)
                .unwrap(),
            AvailableSubjects::Attachments
                .to_subject(
                    Some(DiagnosticsVisualizations::EventKanban.to_string().as_str()),
                    None,
                )
                .unwrap(),
        ];
        let subject_plans = subjects.into_iter().map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap()).collect::<Vec<_>>();
        Some(subject_plans)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{
        BuildableTrait, IPCMessage, MessageBuilderTrait, MessageTrait, Subject, SubjectTrait
    };
    use phymes_diagnostics::{HashMap, HashSet};

    use crate::{
        SessionContextBuilderAgentsTrait, SessionContextBuilderTrait, SessionStream, SessionStreamStep, SessionStreamStepTrait, SubscriptionTrait, create_message_map, test_session_context_builder, test_task
    };

    use super::*;

    /// Make the test data for the diagnostic session
    async fn make_test_data(name: &str) -> Result<HashMap<String, IPCMessage>> {
        // Make the test sequential session
        let (session_context, session_messages) =
            test_session_context_builder::make_test_session_context_builder_sequential(
                "session_1",
                2,
            )?
            .with_diagnostics(true)
            .add_session_interface(Some(&["state_1"]))?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;

        // Mimic a session run for 1 steps
        let session_ctx_arc = Arc::new(session_context);
        let _ = session_ctx_arc.update_subjects_from_messages(session_messages.unwrap_or_default(), 0).await;
        let messages = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::Replace {
                subject_name: "state_1".to_string(),
            },
            true,
        )?;
        let session_stream = SessionStream::new(messages, Arc::clone(&session_ctx_arc));
        let _response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // Extract the subjects
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionMetrics.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionMetrics.to_string().as_str())
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();
        let metrics_message = IPCMessage::get_builder()
            .with_message(subject.to_ipc_stream()?)
            .with_subject(AvailableSubjects::AnalyticsMetrics.to_string().as_str())
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::AnalyticsMetrics.to_string(),
            })
            .with_publisher(name)
            .make_name()?
            .build()?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionTraces.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionTraces.to_string().as_str())
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();
        let traces_message = IPCMessage::get_builder()
            .with_message(subject.to_ipc_stream()?)
            .with_subject(AvailableSubjects::AnalyticsTraces.to_string().as_str())
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::AnalyticsTraces.to_string(),
            })
            .with_publisher(name)
            .make_name()?
            .build()?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionEvents.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionEvents.to_string().as_str())
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();
        let events_message = IPCMessage::get_builder()
            .with_message(subject.to_ipc_stream()?)
            .with_subject(AvailableSubjects::AnalyticsEvents.to_string().as_str())
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::AnalyticsEvents.to_string(),
            })
            .with_publisher(name)
            .make_name()?
            .build()?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionTasks.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionTasks.to_string().as_str())
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();
        let tasks_message = IPCMessage::get_builder()
            .with_message(subject.to_ipc_stream()?)
            .with_subject(AvailableSubjects::AnalyticsTasks.to_string().as_str())
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::AnalyticsTasks.to_string(),
            })
            .with_publisher(name)
            .make_name()?
            .build()?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionErrors.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;

        let messages = if batches.len() > 0 {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
                .with_record_batches(batches)
                .unwrap()
                .build()
                .unwrap();
            let errors_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::AnalyticsErrors.to_string().as_str())
                .with_update(&Publication::Replace {
                    subject_name: AvailableSubjects::AnalyticsErrors.to_string(),
                })
                .with_publisher(name)
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
        };
        Ok(messages)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_diagnostic_session() -> Result<()> {
        // initialize the session
        let diagnostic_session = DiagnosticSession::default();
        let (session_ctx, session_messages) = diagnostic_session
            .build()
            .with_name(diagnostic_session.session_context_name)
            .with_diagnostics(true) // Debugging
            .add_session_interface(Some(&[
                DiagnosticsVisualizations::MetricProcessorTracesGantt
                    .to_string()
                    .as_str(),
                DiagnosticsVisualizations::MetricElapsedComputeGantt
                    .to_string()
                    .as_str(),
                DiagnosticsVisualizations::MetricOutputRowsGantt
                    .to_string()
                    .as_str(),
                DiagnosticsVisualizations::TraceSequenceDiagram
                    .to_string()
                    .as_str(),
                DiagnosticsVisualizations::EventKanban.to_string().as_str(),
                DiagnosticsVisualizations::ErrorKanban.to_string().as_str(),
            ]))?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Make diagnostic data and session tasks data
        let message_map = make_test_data(diagnostic_session.session_context_name).await?;
        let _ = session_ctx_arc.update_subjects_from_messages(session_messages.unwrap_or_default(), 0).await;

        // Run
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // Check the response
        let keys = response
            .iter()
            .flat_map(|m| m.keys().map(|k| k.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        dbg!(&keys);
        assert_eq!(keys.len(), 5);
        let keys_set = keys.into_iter().collect::<HashSet<_>>();
        let expected = [
            DiagnosticsVisualizations::MetricProcessorTracesGantt.to_string(),
            DiagnosticsVisualizations::MetricElapsedComputeGantt.to_string(),
            DiagnosticsVisualizations::MetricOutputRowsGantt.to_string(),
            DiagnosticsVisualizations::TraceSequenceDiagram.to_string(),
            DiagnosticsVisualizations::EventKanban.to_string(),
            // DiagnosticsVisualizations::ErrorKanban.to_string(),
        ]
        .into_iter()
        .map(|s| format!("from_{}_on_{s}", diagnostic_session.session_context_name))
        .collect::<HashSet<_>>();
        assert_eq!(keys_set, expected);

        // Extract the response
        let tables = response
            .into_iter()
            .flat_map(|map| {
                map.into_iter()
                    .filter_map(|(k, v)| {
                        if k.contains(diagnostic_session.session_context_name) {
                            let subject_name = v.get_subject().to_string();
                            Some((
                                k,
                                SubjectBuilder::new_from_ipc_stream(&v.get_message_own())
                                    .unwrap()
                                    .with_name(subject_name.as_str())
                                    .build()
                                    .unwrap(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<HashMap<_, _>>()
            })
            .collect::<HashMap<_, _>>();
        let subject = tables
            .get(
                format!(
                    "from_{}_on_{}",
                    diagnostic_session.session_context_name,
                    DiagnosticsVisualizations::MetricProcessorTracesGantt
                )
                .as_str(),
            )
            .unwrap();
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["MetricProcessorTracesGantt"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..15], "\n        gantt\n");
        let subject = tables
            .get(
                format!(
                    "from_{}_on_{}",
                    diagnostic_session.session_context_name,
                    DiagnosticsVisualizations::MetricElapsedComputeGantt
                )
                .as_str(),
            )
            .unwrap();
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["MetricElapsedComputeGantt"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..15], "\n        gantt\n");
        let subject = tables
            .get(
                format!(
                    "from_{}_on_{}",
                    diagnostic_session.session_context_name,
                    DiagnosticsVisualizations::MetricOutputRowsGantt
                )
                .as_str(),
            )
            .unwrap();
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["MetricOutputRowsGantt"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..15], "\n        gantt\n");
        let subject = tables
            .get(
                format!(
                    "from_{}_on_{}",
                    diagnostic_session.session_context_name,
                    DiagnosticsVisualizations::TraceSequenceDiagram
                )
                .as_str(),
            )
            .unwrap();
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["TraceSequenceDiagram"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..25], "\n        sequenceDiagram\n");
        let subject = tables
            .get(
                format!(
                    "from_{}_on_{}",
                    diagnostic_session.session_context_name,
                    DiagnosticsVisualizations::EventKanban
                )
                .as_str(),
            )
            .unwrap();
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["EventKanban"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..16], "\n        kanban\n");

        // Check the session
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: DiagnosticsVisualizations::MetricProcessorTracesGantt.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(DiagnosticsVisualizations::MetricProcessorTracesGantt.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["MetricProcessorTracesGantt"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..15], "\n        gantt\n");
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: DiagnosticsVisualizations::MetricElapsedComputeGantt.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(DiagnosticsVisualizations::MetricElapsedComputeGantt.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["MetricElapsedComputeGantt"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..15], "\n        gantt\n");
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: DiagnosticsVisualizations::MetricOutputRowsGantt.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(DiagnosticsVisualizations::MetricOutputRowsGantt.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["MetricOutputRowsGantt"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..15], "\n        gantt\n");
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: DiagnosticsVisualizations::TraceSequenceDiagram.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(DiagnosticsVisualizations::TraceSequenceDiagram.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["TraceSequenceDiagram"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..25], "\n        sequenceDiagram\n");
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: DiagnosticsVisualizations::EventKanban.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(DiagnosticsVisualizations::EventKanban.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let columns = subject.get_column_as_vec_str("filename");
        assert_eq!(columns, ["EventKanban"]);
        let columns = subject.get_column_as_vec_str("extension");
        assert_eq!(columns, ["txt"]);
        let columns = subject.get_column_as_vec_str("metadata");
        assert_eq!(columns, ["assistant"]);
        let bytes = subject
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let column = String::from_utf8_lossy(bytes.as_ref()).into_owned();
        assert_eq!(&column[..16], "\n        kanban\n");

        Ok(())
    }
}
