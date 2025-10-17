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
    pub session_tasks_to_sequence_diagram_participants_task_name: &'a str,
    pub session_tasks_to_sequence_diagram_participants_processor_name: &'a str,
    pub apply_sequence_diagram_participants_task_name: &'a str,
    pub apply_sequence_diagram_participants_processor_name: &'a str,
    pub traces_select_and_cast_to_sequence_diagram_messages_task_name: &'a str,
    pub traces_select_and_cast_to_sequence_diagram_messages_processor_name: &'a str,
    pub apply_sequence_diagram_messages_task_name: &'a str,
    pub apply_sequence_diagram_messages_processor_name: &'a str,
    pub apply_sequence_diagram_task_name: &'a str,
    pub apply_sequence_diagram_processor_name: &'a str,
    pub traces_runtime_env_name: &'a str,
    
    /// Events and errors analytics
    pub events_and_errors_aggregation_task_name: &'a str,
    pub events_and_errors_aggregation_processor_name: &'a str,
    pub events_and_errors_select_and_cast_to_kanban_task_name: &'a str,
    pub events_and_errors_select_and_cast_to_kanban_processor_name: &'a str,
    pub apply_kanban_task_name: &'a str,
    pub apply_kanban_task_processor_name: &'a str,
    pub events_runtime_env_name: &'a str,

    /// Session
    pub session_context_name: &'a str,
}

impl CustomAgentsBuilderTrait for MetricSession<'_> {
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
                task_name: self.traces_select_and_cast_to_sequence_diagram_messages_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.traces_select_and_cast_to_sequence_diagram_messages_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_messages_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_messages_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_sequence_diagram_task_name.to_string(),
                runtime_env_name: self.traces_runtime_env_name.to_string(),
                processor_names: vec![self.apply_sequence_diagram_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.events_and_errors_aggregation_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![self.events_and_errors_aggregation_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.events_and_errors_select_and_cast_to_kanban_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![self.events_and_errors_select_and_cast_to_kanban_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.apply_kanban_task_name.to_string(),
                runtime_env_name: self.events_runtime_env_name.to_string(),
                processor_names: vec![self.apply_kanban_task_processor_name.to_string()],
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
                self.events_and_errors_aggregation_processor_name,
                &[TablePublish::Replace {
                    table_name: AvailableSubjects::MetricPivot.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::Events.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableSubjects::Errors.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.events_and_errors_aggregation_processor_name.to_string(),
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
        // TODO

        Some(vec![
            AvailableSubjects::Metrics.to_table(None, None).unwrap(),
            AvailableSubjects::Traces.to_table(None, None).unwrap(),
            AvailableSubjects::Events.to_table(None, None).unwrap(),
            AvailableSubjects::Errors.to_table(None, None).unwrap(),
            AvailableSubjects::SessionTasks.to_table(None, None).unwrap(),
            AvailableSubjects::MermaidGanttTemplate.to_table(Some(self.metrics_processors_traces_select_and_cast_to_gantt_task_name), None).unwrap(),
            AvailableSubjects::MermaidGanttTemplate.to_table(Some(self.metrics_processors_traces_select_and_cast_to_gantt_task_name), None).unwrap(),
            AvailableSubjects::MermaidGanttTemplate.to_table(Some(self.metrics_processors_traces_select_and_cast_to_gantt_task_name), None).unwrap(),
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