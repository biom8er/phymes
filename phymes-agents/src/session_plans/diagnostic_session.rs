use std::sync::Arc;
use anyhow::Result;

use arrow::datatypes::DataType;
use phymes_core::{
    schemas::available_subjects::{AvailableSubjects, AvailableSubjectsTrait}, session::{
        common_traits::BuilderTrait,
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::TaskPlan,
    }, table::{
        data_format::DataFormat, table_trait::{Table, TableBuilder, TableBuilderTrait}, table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, AnyTableNameSubscribe, SubscribeTrait, TableSubscribe}
    }, task::processor::{ProcessorEcho, ProcessorTrait}
};
use phymes_data::{candle_data::{data_config::{DataAggregatorOperator, DataCastOperator, DataConfig}, data_processor::CandleDataProcessor, summary_config::DataSummaryConfig, summary_processor::DataSummaryProcessor}, candle_operators::available_candle_operators::AvailableCandleOperators};
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
    
    /// Traces analytics
    pub traces_to_sequence_diagram_messages_task_name: &'a str,
    pub traces_to_sequence_diagram_messages_processor_name: &'a str,
    pub apply_sequence_diagram_messages_task_name: &'a str,
    pub apply_sequence_diagram_messages_processor_name: &'a str,
    pub session_tasks_to_sequence_diagram_participants_task_name: &'a str,
    pub session_tasks_to_sequence_diagram_participants_processor_name: &'a str,
    pub apply_sequence_diagram_participants_task_name: &'a str,
    pub apply_sequence_diagram_participants_processor_name: &'a str,
    pub traces_aggregate_sequence_diagram_content_task_name: &'a str,
    pub traces_aggregate_sequence_diagram_content_processor_name: &'a str,
    pub traces_select_and_cast_to_sequence_diagram_content_task_name: &'a str,
    pub traces_select_and_cast_to_sequence_diagram_content_processor_name: &'a str,
    pub apply_sequence_diagram_task_name: &'a str,
    pub apply_sequence_diagram_processor_name: &'a str,
    pub traces_runtime_env_name: &'a str,
    
    /// Events analytics
    pub events_select_and_cast_to_kanban_task_name: &'a str,
    pub events_select_and_cast_to_kanban_processor_name: &'a str,
    pub apply_kanban_task_name: &'a str,
    pub apply_kanban_task_processor_name: &'a str,
    pub events_runtime_env_name: &'a str,

    /// Errors analytics
    // todo!()

    /// Outbox
    pub aggregate_visualizations_task_name: &'a str,
    pub aggregate_visualizations_processor_name: &'a str,

    /// Session
    pub session_context_name: &'a str,
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
                task_name: self.metrics_processors_traces_select_and_cast_to_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_processors_traces_select_and_cast_to_gantt_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_output_rows_select_and_cast_to_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_output_rows_select_and_cast_to_gantt_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_processors_traces_apply_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_processors_traces_apply_gantt_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_elapsed_compute_apply_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_elapsed_compute_apply_gantt_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.metrics_output_rows_apply_gantt_task_name.to_string(),
                runtime_env_name: self.metrics_runtime_env_name.to_string(),
                processor_names: vec![self.metrics_output_rows_apply_gantt_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.traces_to_sequence_diagram_messages_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.traces_to_sequence_diagram_messages_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_messages_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.session_tasks_to_sequence_diagram_participants_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.session_tasks_to_sequence_diagram_participants_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_participants_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_participants_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![self.events_select_and_cast_to_kanban_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_kanban_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![self.apply_kanban_task_processor_name.to_string()],
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
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_pivot_processor_name,
                &[TablePublish::Replace {
                    table_name: AvailableSubjects::MetricPivot.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::Metrics.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_pivot_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_normalize_time_processor_name,
                &[TablePublish::Replace {
                    table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivot.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_normalize_time_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_processors_traces_select_and_cast_to_gantt_processor_name,
                &[TablePublish::Replace {
                    table_name: self.metrics_processors_traces_select_and_cast_to_gantt_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_processors_traces_select_and_cast_to_gantt_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name,
                &[TablePublish::Replace {
                    table_name: self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_output_rows_select_and_cast_to_gantt_processor_name,
                &[TablePublish::Replace {
                    table_name: self.metrics_output_rows_select_and_cast_to_gantt_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::MetricPivotNormTime.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_output_rows_select_and_cast_to_gantt_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_processors_traces_apply_gantt_processor_name,
                &[TablePublish::Replace {
                    table_name: self.metrics_processors_traces_apply_gantt_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.metrics_processors_traces_select_and_cast_to_gantt_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_processors_traces_apply_gantt_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_elapsed_compute_apply_gantt_processor_name,
                &[TablePublish::Replace {
                    table_name: self.metrics_elapsed_compute_apply_gantt_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_elapsed_compute_apply_gantt_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.metrics_output_rows_apply_gantt_processor_name,
                &[TablePublish::Replace {
                    table_name: self.metrics_output_rows_apply_gantt_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.metrics_output_rows_select_and_cast_to_gantt_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.metrics_output_rows_apply_gantt_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.traces_to_sequence_diagram_messages_processor_name,
                &[TablePublish::Replace {
                    table_name: self.traces_to_sequence_diagram_messages_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::SessionTasks.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::Traces.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.traces_to_sequence_diagram_messages_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.apply_sequence_diagram_messages_processor_name,
                &[TablePublish::Replace {
                    table_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.traces_to_sequence_diagram_messages_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.apply_sequence_diagram_messages_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.session_tasks_to_sequence_diagram_participants_processor_name,
                &[TablePublish::Replace {
                    table_name: self.session_tasks_to_sequence_diagram_participants_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::SessionTasks.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.traces_to_sequence_diagram_messages_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.session_tasks_to_sequence_diagram_participants_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.apply_sequence_diagram_participants_processor_name,
                &[TablePublish::Replace {
                    table_name: self.apply_sequence_diagram_participants_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.session_tasks_to_sequence_diagram_participants_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.apply_sequence_diagram_participants_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.traces_aggregate_sequence_diagram_content_processor_name,
                &[TablePublish::Replace {
                    table_name: self.traces_aggregate_sequence_diagram_content_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.apply_sequence_diagram_participants_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.traces_aggregate_sequence_diagram_content_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.traces_select_and_cast_to_sequence_diagram_content_processor_name,
                &[TablePublish::Replace {
                    table_name: self.traces_select_and_cast_to_sequence_diagram_content_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.traces_aggregate_sequence_diagram_content_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.traces_select_and_cast_to_sequence_diagram_content_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.apply_sequence_diagram_processor_name,
                &[TablePublish::Replace {
                    table_name: self.apply_sequence_diagram_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.traces_select_and_cast_to_sequence_diagram_content_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.apply_sequence_diagram_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.events_select_and_cast_to_kanban_processor_name,
                &[TablePublish::Replace {
                    table_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::Events.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.events_select_and_cast_to_kanban_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.apply_kanban_task_processor_name,
                &[TablePublish::Replace {
                    table_name: self.apply_kanban_task_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.events_select_and_cast_to_kanban_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.apply_kanban_task_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            CandleDataProcessor::new_arc_with_pub_sub(
                self.aggregate_visualizations_processor_name,
                &[TablePublish::Replace {
                    table_name: AvailableInterfaceSubjects::AggregatedAttachments.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.metrics_processors_traces_apply_gantt_task_name.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.metrics_elapsed_compute_apply_gantt_task_name.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.metrics_output_rows_apply_gantt_task_name.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.apply_sequence_diagram_task_name.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.apply_sequence_diagram_task_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.aggregate_visualizations_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),

        ];

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name(self.metrics_runtime_env_name),
            RuntimeEnv::new().with_name(self.traces_runtime_env_name),
            RuntimeEnv::new().with_name(self.events_runtime_env_name),
        ])
    }
    
    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Metrics pivot
        let metrics_pivot_config = DataConfig {
            lhs_name: AvailableSubjects::Metrics.to_string(),
            lhs_values: vec!["span_name".to_string(), "span_id".to_string(), "parent_name".to_string(), "parent_id".to_string()],
            agg_columns: Some(vec!["metric_value".to_string()]),
            agg_operators: Some(vec![DataAggregatorOperator::Sum]),
            pvt_columns: Some(vec!["metric_name".to_string()]),
            operator: AvailableCandleOperators::Pivot,
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
            lhs_name: AvailableSubjects::MetricPivot.to_string(),
            operator: AvailableCandleOperators::NormalizeTime,
            ..Default::default()
        };
        let metrics_normalize_time_config_json = serde_json::to_vec(&metrics_normalize_time_config).unwrap();
        let metrics_normalize_time_config_1_state = TableBuilder::new()
            .with_name(self.metrics_normalize_time_processor_name)
            .with_json(&metrics_normalize_time_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics processor traces select and cast
        let metrics_processors_traces_select_and_cast_to_gantt_config = DataConfig {
            lhs_name: AvailableSubjects::MetricPivot.to_string(),
            lhs_values: vec!["span_name".to_string(), "span_name".to_string(), "start_time_norm".to_string(), "end_time_norm".to_string()],
            as_columns: Some(vec!["section".to_string(), "task".to_string(), "start".to_string(), "end".to_string()]),
            cast_operators: Some(vec![DataCastOperator::None, DataCastOperator::None, DataCastOperator::None, DataCastOperator::None]),
            cast_datatypes: Some(vec![DataType::Utf8.to_string(), DataType::Utf8.to_string(), DataType::Utf8.to_string(), DataType::Utf8.to_string()]),
            cast_templates: Some(vec!["Traces[ns]".to_string(), "".to_string(), "".to_string(), "".to_string()]),
            operator: AvailableCandleOperators::SelectAndCast,
            ..Default::default()
        };
        let metrics_processors_traces_select_and_cast_to_gantt_config_json = serde_json::to_vec(&metrics_processors_traces_select_and_cast_to_gantt_config).unwrap();
        let metrics_processors_traces_select_and_cast_to_gantt_config_1_state = TableBuilder::new()
            .with_name(self.metrics_processors_traces_select_and_cast_to_gantt_processor_name)
            .with_json(&metrics_processors_traces_select_and_cast_to_gantt_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics processor traces select and cast
        let metrics_elapsed_compute_select_and_cast_to_gantt_config = DataConfig {
            lhs_name: AvailableSubjects::MetricPivot.to_string(),
            lhs_values: vec!["span_name".to_string(), "span_name".to_string(), "start_time_norm".to_string(), "end_time_norm".to_string()],
            as_columns: Some(vec!["section".to_string(), "task".to_string(), "start".to_string(), "end".to_string()]),
            cast_operators: Some(vec![DataCastOperator::None, DataCastOperator::None, DataCastOperator::None, DataCastOperator::None]),
            cast_datatypes: Some(vec![DataType::Utf8.to_string(), DataType::Utf8.to_string(), DataType::Utf8.to_string(), DataType::Utf8.to_string()]),
            cast_templates: Some(vec!["Time[ns]".to_string(), "".to_string(), "0".to_string(), "".to_string()]),
            operator: AvailableCandleOperators::SelectAndCast,
            ..Default::default()
        };
        let metrics_elapsed_compute_select_and_cast_to_gantt_config_json = serde_json::to_vec(&metrics_elapsed_compute_select_and_cast_to_gantt_config).unwrap();
        let metrics_elapsed_compute_select_and_cast_to_gantt_config_1_state = TableBuilder::new()
            .with_name(self.metrics_elapsed_compute_select_and_cast_to_gantt_processor_name)
            .with_json(&metrics_elapsed_compute_select_and_cast_to_gantt_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Metrics output rows select and cast
        let metrics_output_rows_select_and_cast_to_gantt_config = DataConfig {
            lhs_name: AvailableSubjects::MetricPivot.to_string(),
            lhs_values: vec!["span_name".to_string(), "span_name".to_string(), "start_time_norm".to_string(), "end_time_norm".to_string()],
            as_columns: Some(vec!["section".to_string(), "task".to_string(), "start".to_string(), "end".to_string()]),
            cast_operators: Some(vec![DataCastOperator::None, DataCastOperator::None, DataCastOperator::None, DataCastOperator::None]),
            cast_datatypes: Some(vec![DataType::Utf8.to_string(), DataType::Utf8.to_string(), DataType::Utf8.to_string(), DataType::Utf8.to_string()]),
            cast_templates: Some(vec!["Counts".to_string(), "".to_string(), "0".to_string(), "".to_string()]),
            operator: AvailableCandleOperators::SelectAndCast,
            ..Default::default()
        };
        let metrics_output_rows_select_and_cast_to_gantt_config_json = serde_json::to_vec(&metrics_output_rows_select_and_cast_to_gantt_config).unwrap();
        let metrics_output_rows_select_and_cast_to_gantt_config_1_state = TableBuilder::new()
            .with_name(self.metrics_output_rows_select_and_cast_to_gantt_processor_name)
            .with_json(&metrics_output_rows_select_and_cast_to_gantt_config_json.clone(), 1)
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

            // Metrics
            AvailableSubjects::Metrics.to_table(None, None).unwrap(),
            AvailableSubjects::MetricPivot.to_table(None, None).unwrap(),
            AvailableSubjects::MermaidGanttTemplate.to_table(Some(self.metrics_processors_traces_select_and_cast_to_gantt_task_name), None).unwrap(),
            AvailableSubjects::MermaidGanttTemplate.to_table(Some(self.metrics_elapsed_compute_select_and_cast_to_gantt_task_name), None).unwrap(),
            AvailableSubjects::MermaidGanttTemplate.to_table(Some(self.metrics_output_rows_select_and_cast_to_gantt_task_name), None).unwrap(),
            AvailableSubjects::Blob.to_table(Some(self.metrics_processors_traces_apply_gantt_task_name), None).unwrap(),
            AvailableSubjects::Blob.to_table(Some(self.metrics_elapsed_compute_apply_gantt_task_name), None).unwrap(),
            AvailableSubjects::Blob.to_table(Some(self.metrics_output_rows_apply_gantt_task_name), None).unwrap(),

            // Traces
            AvailableSubjects::SessionTasks.to_table(None, None).unwrap(),
            AvailableSubjects::MermaidSequenceDiagramParticipantsTemplate.to_table(Some(self.session_tasks_to_sequence_diagram_participants_task_name), None).unwrap(),
            AvailableSubjects::Traces.to_table(None, None).unwrap(),
            AvailableSubjects::MermaidSequenceDiagramMessagesTemplate.to_table(Some(self.traces_to_sequence_diagram_messages_task_name), None).unwrap(),
            AvailableSubjects::Messages.to_table(Some(self.apply_sequence_diagram_participants_task_name), None).unwrap(),
            AvailableSubjects::Messages.to_table(Some(self.apply_sequence_diagram_messages_task_name), None).unwrap(),
            AvailableSubjects::Messages.to_table(Some(self.traces_aggregate_sequence_diagram_content_task_name), None).unwrap(),
            AvailableSubjects::MermaidContentTemplate.to_table(Some(self.traces_select_and_cast_to_sequence_diagram_content_task_name), None).unwrap(),
            AvailableSubjects::Blob.to_table(Some(self.apply_sequence_diagram_task_name), None).unwrap(),

            // Events
            AvailableSubjects::Events.to_table(None, None).unwrap(),
            AvailableSubjects::Errors.to_table(None, None).unwrap(),
            AvailableSubjects::MermaidKanbanTemplate.to_table(Some(self.events_select_and_cast_to_kanban_task_name), None).unwrap(),
            AvailableSubjects::Blob.to_table(Some(self.apply_kanban_task_name), None).unwrap(),

            // Outbox
            AvailableInterfaceSubjects::AggregatedAttachments.to_table(None, None).unwrap(),
        ])
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_user_agent_session() -> Result<()> {
        Ok(())
    }
}