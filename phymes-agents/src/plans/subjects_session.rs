use anyhow::Result;
use std::sync::Arc;

use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableTableSubscribePolicies, BuilderTrait,
    DataFormat, ProcessorTrait, RuntimeEnv, RuntimeEnvTrait, Table, TableBuilder,
    TableBuilderTrait, TablePublication, TableSubscription, create_user_batch,
    create_user_session_contexts_batch,
};
use phymes_data::{AvailableCandleOperators, DataConfig, DataSummaryConfig};
use phymes_diagnostics::create_timestamp_micros;

use crate::{
    AvailableInterfaceSubjects, AvailableProcessors, AvailableSessionPlans,
    CustomAgentsBuilderTrait, TaskPlan, make_example_mermaid_table,
};

/// A session for all subject associated tasks
///
/// # Notes
///
/// * Supported tasks include the following:
///
/// 1. Counting the number of rows per subject (i.e., updating the `SubjectNumRows` table)
///   after updates have been made to the `SubjectsChangeLog`
/// 2. Determining what tasks are ready to run for the next super step
/// 3. Retrieving the publications per task and processor that will run for the next super step
/// 4. Updating the `SubjectsChangeLog` cache with the most recent updates and `TasksRunLog` cache with the most recent task runs
///
/// * Caching is implemented to minimize memory and compute
pub struct SubjectsSession<'a> {
    /// Inbox
    pub extract_tasks_task_name: &'a str,
    pub extract_tasks_processor_name: &'a str,

    /// 1, 2, and 3. Aggregate the latest subjects change log
    // Sum aggreggation of delta, and set of session_name;task_name
    pub group_by_subject_change_log_delta_task_name: &'a str,
    pub group_by_subject_change_log_delta_processor_name: &'a str,
    pub select_subject_change_log_delta_task_name: &'a str,
    pub select_subject_change_log_delta_processor_name: &'a str,

    /// 1. Count the number of rows per subject
    pub group_by_subjects_num_rows_task_name: &'a str,
    pub group_by_subjects_num_rows_processor_name: &'a str,
    pub select_subjects_num_rows_task_name: &'a str,
    pub select_subjects_num_rows_processor_name: &'a str,
    pub join_subjects_num_rows_delta_task_name: &'a str,
    pub join_subjects_num_rows_delta_processor_name: &'a str,
    pub add_subjects_num_rows_delta_task_name: &'a str,
    pub add_subjects_num_rows_delta_processor_name: &'a str,
    pub select_subjects_num_rows_delta_task_name: &'a str,
    pub select_subjects_num_rows_delta_processor_name: &'a str,
    // Extend with the new batch

    /// 2 and 3. Aggregate the latest session tasks change log
    pub group_by_tasks_run_log_timestamp_task_name: &'a str,
    pub group_by_tasks_run_log_timestamp_processor_name: &'a str,
    pub select_tasks_run_log_timestamp_task_name: &'a str,
    pub select_tasks_run_log_timestamp_processor_name: &'a str,

    /// 2. Cache filtered subscriptions
    pub filter_processors_subscriptions_task_name: &'a str,
    pub filter_processors_subscriptions_processor_name: &'a str,
    
    /// 2. Retrieve updated subscriptions
    pub join_tasks_run_log_timestamp_task_name: &'a str,
    pub join_tasks_run_log_timestamp_processor_name: &'a str,
    pub join_tasks_processors_subscriptions_task_name: &'a str,
    pub join_tasks_processors_subscriptions_processor_name: &'a str,
    pub join_tasks_processors_subscriptions_subjects_task_name: &'a str,
    pub join_tasks_processors_subscriptions_subjects_processor_name: &'a str,    
    pub select_tasks_processors_subscriptions_subjects_task_name: &'a str,
    pub select_tasks_processors_subscriptions_subjects_processor_name: &'a str,
    // DM: filter for updates that are past the last task run date
    //  and were not updated by the same task
    pub filter_tasks_processors_subscriptions_subjects_task_name: &'a str,
    pub filter_tasks_processors_subscriptions_subjects_processor_name: &'a str,
    
    /// 3. Cache filtered publications
    pub filter_processors_publications_task_name: &'a str,
    pub filter_processors_publications_processor_name: &'a str,

    /// 3. Retrieve the publications
    pub select_tasks_ready_to_run_task_name: &'a str,
    pub select_tasks_ready_to_run_processor_name: &'a str,
    pub join_tasks_processors_publications_task_name: &'a str,
    pub join_tasks_processors_publications_processor_name: &'a str,
    pub select_tasks_processors_publications_task_name: &'a str,
    pub select_tasks_processors_publications_processor_name: &'a str,

    /// Outbox
    pub aggregate_tasks_processors_publications_task_name: &'a str,
    pub aggregate_tasks_processors_publications_processor_name: &'a str,

    // DM: all supersteps need to wait until the list of ready-to-run tasks is produced

    /// Session
    pub session_context_name: &'a str,

    /// Runtime environment
    pub default_runtime_env_name: &'a str,
}

impl Default for SubjectsSession<'_> {
    fn default() -> Self {
        SubjectsSession {
            extract_tasks_task_name: "extract_tasks_task_name",
            extract_tasks_processor_name: "extract_tasks_processor_name",
            group_by_subject_change_log_delta_task_name: "group_by_subject_change_log_delta_task_name",
            group_by_subject_change_log_delta_processor_name: "group_by_subject_change_log_delta_processor_name",
            select_subject_change_log_delta_task_name: "select_subject_change_log_delta_task_name",
            select_subject_change_log_delta_processor_name: "select_subject_change_log_delta_processor_name",
            group_by_subjects_num_rows_task_name: "group_by_subjects_num_rows_task_name",
            group_by_subjects_num_rows_processor_name: "group_by_subjects_num_rows_processor_name",
            select_subjects_num_rows_task_name: "select_subjects_num_rows_task_name",
            select_subjects_num_rows_processor_name: "select_subjects_num_rows_processor_name",
            join_subjects_num_rows_delta_task_name: "join_subjects_num_rows_delta_task_name",
            join_subjects_num_rows_delta_processor_name: "join_subjects_num_rows_delta_processor_name",
            add_subjects_num_rows_delta_task_name: "add_subjects_num_rows_delta_task_name",
            add_subjects_num_rows_delta_processor_name: "add_subjects_num_rows_delta_processor_name",
            select_subjects_num_rows_delta_task_name: "select_subjects_num_rows_delta_task_name",
            select_subjects_num_rows_delta_processor_name: "select_subjects_num_rows_delta_processor_name",
            group_by_tasks_run_log_timestamp_task_name: "group_by_tasks_run_log_timestamp_task_name",
            group_by_tasks_run_log_timestamp_processor_name: "group_by_tasks_run_log_timestamp_processor_name",
            select_tasks_run_log_timestamp_task_name: "select_tasks_run_log_timestamp_task_name",
            select_tasks_run_log_timestamp_processor_name: "select_tasks_run_log_timestamp_processor_name",
            filter_processors_subscriptions_task_name: "filter_processors_subscriptions_task_name",
            filter_processors_subscriptions_processor_name: "filter_processors_subscriptions_processor_name",
            join_tasks_run_log_timestamp_task_name: "join_tasks_run_log_timestamp_task_name",
            join_tasks_run_log_timestamp_processor_name: "join_tasks_run_log_timestamp_processor_name",
            join_tasks_processors_subscriptions_task_name: "join_tasks_processors_subscriptions_task_name",
            join_tasks_processors_subscriptions_processor_name: "join_tasks_processors_subscriptions_processor_name",
            join_tasks_processors_subscriptions_subjects_task_name: "join_tasks_processors_subscriptions_subjects_task_name",
            join_tasks_processors_subscriptions_subjects_processor_name:     "join_tasks_processors_subscriptions_subjects_processor_name", 
            select_tasks_processors_subscriptions_subjects_task_name: "select_tasks_processors_subscriptions_subjects_task_name",
            select_tasks_processors_subscriptions_subjects_processor_name: "select_tasks_processors_subscriptions_subjects_processor_name",
            filter_tasks_processors_subscriptions_subjects_task_name: "filter_tasks_processors_subscriptions_subjects_task_name",
            filter_tasks_processors_subscriptions_subjects_processor_name: "filter_tasks_processors_subscriptions_subjects_processor_name",
            filter_processors_publications_task_name: "filter_processors_publications_task_name",
            filter_processors_publications_processor_name: "filter_processors_publications_processor_name",
            select_tasks_ready_to_run_task_name: "select_tasks_ready_to_run_task_name",
            select_tasks_ready_to_run_processor_name: "select_tasks_ready_to_run_processor_name",
            join_tasks_processors_publications_task_name: "join_tasks_processors_publications_task_name",
            join_tasks_processors_publications_processor_name: "join_tasks_processors_publications_processor_name",
            select_tasks_processors_publications_task_name: "select_tasks_processors_publications_task_name",
            select_tasks_processors_publications_processor_name: "select_tasks_processors_publications_processor_name",
            aggregate_tasks_processors_publications_task_name: "aggregate_tasks_processors_publications_task_name",
            aggregate_tasks_processors_publications_processor_name: "aggregate_tasks_processors_publications_processor_name",
            session_context_name: "subject_session",
            default_runtime_env_name: "default_runtime_env_name",

        }
    }
}

impl<'a> SubjectsSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        SubjectsSession {
            session_context_name,
            ..Default::default()
        }
    }
}

impl CustomAgentsBuilderTrait for SubjectsSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![
            TaskPlan {
                task_name: self.extract_tasks_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.extract_tasks_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.group_by_subject_change_log_delta_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.group_by_subject_change_log_delta_processor_name.to_string(),
                    self.select_subject_change_log_delta_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.group_by_subjects_num_rows_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.group_by_subjects_num_rows_processor_name.to_string(),
                    self.select_subjects_num_rows_processor_name.to_string(),
                    self.join_subjects_num_rows_delta_processor_name.to_string(),
                    self.add_subjects_num_rows_delta_processor_name.to_string(),
                    self.select_subjects_num_rows_delta_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.group_by_tasks_run_log_timestamp_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.group_by_tasks_run_log_timestamp_processor_name.to_string(),
                    self.select_tasks_run_log_timestamp_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.filter_processors_subscriptions_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.filter_processors_subscriptions_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.join_tasks_run_log_timestamp_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.join_tasks_run_log_timestamp_processor_name.to_string(),
                    self.join_tasks_processors_subscriptions_processor_name.to_string(),
                    self.join_tasks_processors_subscriptions_subjects_processor_name.to_string(),
                    self.select_tasks_processors_subscriptions_subjects_processor_name.to_string(),
                    self.filter_tasks_processors_subscriptions_subjects_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.filter_processors_publications_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.filter_processors_publications_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.select_tasks_ready_to_run_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.select_tasks_ready_to_run_processor_name.to_string(),
                    self.join_tasks_processors_publications_processor_name.to_string(),
                    self.select_tasks_processors_publications_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.aggregate_tasks_processors_publications_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.aggregate_tasks_processors_publications_processor_name.to_string(),
                ],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            AvailableProcessors::ExtractTabular.build_arc(
                self.extract_tasks_processor_name,
                &[TablePublication::Replace {
                    table_name: AvailableSubjects::SessionTasksInbox.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableInterfaceSubjects::UserCsv.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.extract_tasks_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::GroupBy.build_arc(
                self.group_by_subject_change_log_delta_processor_name,
                &[TablePublication::Replace {
                    table_name: self.group_by_subject_change_log_delta_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_subject_change_log_delta_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_subject_change_log_delta_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_subject_change_log_delta_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_subject_change_log_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subject_change_log_delta_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            
            AvailableProcessors::GroupBy.build_arc(
                self.group_by_subjects_num_rows_processor_name,
                &[TablePublication::Replace {
                    table_name: self.group_by_subjects_num_rows_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_subjects_num_rows_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_subjects_num_rows_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_subjects_num_rows_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_subjects_num_rows_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subjects_num_rows_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Join.build_arc(
                self.join_subjects_num_rows_delta_processor_name,
                &[TablePublication::Replace {
                    table_name: self.join_subjects_num_rows_delta_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.select_subject_change_log_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subjects_num_rows_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_subjects_num_rows_delta_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.add_subjects_num_rows_delta_processor_name,
                &[TablePublication::Replace {
                    table_name: self.add_subjects_num_rows_delta_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subject_change_log_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.add_subjects_num_rows_delta_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_subjects_num_rows_delta_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_subjects_num_rows_delta_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.add_subjects_num_rows_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subjects_num_rows_delta_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::GroupBy.build_arc(
                self.group_by_subjects_num_rows_processor_name,
                &[TablePublication::Replace {
                    table_name: self.group_by_subjects_num_rows_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.select_subjects_num_rows_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_subjects_num_rows_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_subjects_num_rows_processor_name,
                &[TablePublication::Extend {
                    table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_subjects_num_rows_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subjects_num_rows_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::GroupBy.build_arc(
                self.group_by_tasks_run_log_timestamp_processor_name,
                &[TablePublication::Replace {
                    table_name: self.group_by_tasks_run_log_timestamp_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: AvailableSubjects::SessionTasksRunLog.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_tasks_run_log_timestamp_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_tasks_run_log_timestamp_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_tasks_run_log_timestamp_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_tasks_run_log_timestamp_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_run_log_timestamp_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Filter.build_arc(
                self.filter_processors_subscriptions_processor_name,
                &[TablePublication::Replace {
                    table_name: self.filter_processors_subscriptions_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::SessionProcessors.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.filter_processors_subscriptions_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Join.build_arc(
                self.join_tasks_run_log_timestamp_processor_name,
                &[TablePublication::Replace {
                    table_name: self.join_tasks_run_log_timestamp_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.select_tasks_run_log_timestamp_task_name.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.extract_tasks_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_run_log_timestamp_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Join.build_arc(
                self.join_tasks_processors_subscriptions_processor_name,
                &[TablePublication::Replace {
                    table_name: self.join_tasks_processors_subscriptions_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_run_log_timestamp_task_name.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.filter_processors_subscriptions_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_processors_subscriptions_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Join.build_arc(
                self.join_tasks_processors_subscriptions_subjects_processor_name,
                &[TablePublication::Replace {
                    table_name: self.join_tasks_processors_subscriptions_subjects_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_processors_subscriptions_task_name.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.select_subject_change_log_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_processors_subscriptions_subjects_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_tasks_processors_subscriptions_subjects_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_tasks_processors_subscriptions_subjects_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_processors_subscriptions_subjects_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_processors_subscriptions_subjects_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Filter.build_arc(
                self.filter_tasks_processors_subscriptions_subjects_processor_name,
                &[TablePublication::Replace {
                    table_name: self.filter_tasks_processors_subscriptions_subjects_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_processors_subscriptions_subjects_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.filter_tasks_processors_subscriptions_subjects_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Filter.build_arc(
                self.filter_processors_publications_processor_name,
                &[TablePublication::Replace {
                    table_name: self.filter_processors_publications_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::SessionProcessors.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.filter_processors_publications_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_tasks_ready_to_run_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_tasks_ready_to_run_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.filter_tasks_processors_subscriptions_subjects_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_ready_to_run_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Join.build_arc(
                self.join_tasks_processors_publications_processor_name,
                &[TablePublication::Replace {
                    table_name: self.join_tasks_processors_publications_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_ready_to_run_task_name.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.filter_processors_publications_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_processors_publications_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.select_tasks_processors_publications_processor_name,
                &[TablePublication::Replace {
                    table_name: self.select_tasks_processors_publications_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_tasks_processors_publications_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_processors_publications_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::AttachmentAggregatorProcessor.build_arc(
                self.aggregate_tasks_processors_publications_processor_name,
                &[TablePublication::Replace {
                    table_name: AvailableInterfaceSubjects::AssistantCsv.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_subjects_num_rows_delta_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.filter_tasks_processors_subscriptions_subjects_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.select_tasks_processors_publications_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.aggregate_tasks_processors_publications_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
        ];

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name(self.default_runtime_env_name),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        let extract_tasks_config = DataConfig {
            lhs_name: Some(AvailableInterfaceSubjects::UserCsv.to_string()),
            lhs_values: Some(vec!["bytes".to_string()]),
            format: Some(DataFormat::CsvDefault),
            operator: AvailableCandleOperators::ExtractTabular,
            ..Default::default()
        };
        let extract_tasks_config_json =
            serde_json::to_vec(&extract_tasks_config).unwrap();
        let extract_tasks_state = TableBuilder::new()
            .with_name(self.extract_tasks_processor_name)
            .with_json(&extract_tasks_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let group_by_subject_change_log_delta_config = DataConfig {
            ..Default::default()
        };
        let group_by_subject_change_log_delta_config_json =
            serde_json::to_vec(&group_by_subject_change_log_delta_config).unwrap();
        let group_by_subject_change_log_delta_state = TableBuilder::new()
            .with_name(self.group_by_subject_change_log_delta_processor_name)
            .with_json(&group_by_subject_change_log_delta_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_subject_change_log_delta_config = DataConfig {
            ..Default::default()
        };
        let select_subject_change_log_delta_config_json =
            serde_json::to_vec(&select_subject_change_log_delta_config).unwrap();
        let select_subject_change_log_delta_state = TableBuilder::new()
            .with_name(self.select_subject_change_log_delta_processor_name)
            .with_json(&select_subject_change_log_delta_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let group_by_subjects_num_rows_config = DataConfig {
            ..Default::default()
        };
        let group_by_subjects_num_rows_config_json =
            serde_json::to_vec(&group_by_subjects_num_rows_config).unwrap();
        let group_by_subjects_num_rows_state = TableBuilder::new()
            .with_name(self.group_by_subjects_num_rows_processor_name)
            .with_json(&group_by_subjects_num_rows_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_subjects_num_rows_config = DataConfig {
            ..Default::default()
        };
        let select_subjects_num_rows_config_json =
            serde_json::to_vec(&select_subjects_num_rows_config).unwrap();
        let select_subjects_num_rows_state = TableBuilder::new()
            .with_name(self.select_subjects_num_rows_processor_name)
            .with_json(&select_subjects_num_rows_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let join_subjects_num_rows_delta_config = DataConfig {
            ..Default::default()
        };
        let join_subjects_num_rows_delta_config_json =
            serde_json::to_vec(&join_subjects_num_rows_delta_config).unwrap();
        let join_subjects_num_rows_delta_state = TableBuilder::new()
            .with_name(self.join_subjects_num_rows_delta_processor_name)
            .with_json(&join_subjects_num_rows_delta_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let add_subjects_num_rows_delta_config = DataConfig {
            ..Default::default()
        };
        let add_subjects_num_rows_delta_config_json =
            serde_json::to_vec(&add_subjects_num_rows_delta_config).unwrap();
        let add_subjects_num_rows_delta_state = TableBuilder::new()
            .with_name(self.add_subjects_num_rows_delta_processor_name)
            .with_json(&add_subjects_num_rows_delta_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_subjects_num_rows_delta_config = DataConfig {
            ..Default::default()
        };
        let select_subjects_num_rows_delta_config_json =
            serde_json::to_vec(&select_subjects_num_rows_delta_config).unwrap();
        let select_subjects_num_rows_delta_state = TableBuilder::new()
            .with_name(self.select_subjects_num_rows_delta_processor_name)
            .with_json(&select_subjects_num_rows_delta_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let group_by_tasks_run_log_timestamp_config = DataConfig {
            ..Default::default()
        };
        let group_by_tasks_run_log_timestamp_config_json =
            serde_json::to_vec(&group_by_tasks_run_log_timestamp_config).unwrap();
        let group_by_tasks_run_log_timestamp_state = TableBuilder::new()
            .with_name(self.group_by_tasks_run_log_timestamp_processor_name)
            .with_json(&group_by_tasks_run_log_timestamp_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_tasks_run_log_timestamp_config = DataConfig {
            ..Default::default()
        };
        let select_tasks_run_log_timestamp_config_json =
            serde_json::to_vec(&select_tasks_run_log_timestamp_config).unwrap();
        let select_tasks_run_log_timestamp_state = TableBuilder::new()
            .with_name(self.select_tasks_run_log_timestamp_processor_name)
            .with_json(&select_tasks_run_log_timestamp_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        
        let filter_processors_subscriptions_config = DataConfig {
            ..Default::default()
        };
        let filter_processors_subscriptions_config_json =
            serde_json::to_vec(&filter_processors_subscriptions_config).unwrap();
        let filter_processors_subscriptions_state = TableBuilder::new()
            .with_name(self.filter_processors_subscriptions_processor_name)
            .with_json(&filter_processors_subscriptions_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let join_tasks_run_log_timestamp_config = DataConfig {
            ..Default::default()
        };
        let join_tasks_run_log_timestamp_config_json =
            serde_json::to_vec(&join_tasks_run_log_timestamp_config).unwrap();
        let join_tasks_run_log_timestamp_state = TableBuilder::new()
            .with_name(self.join_tasks_run_log_timestamp_processor_name)
            .with_json(&join_tasks_run_log_timestamp_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let join_tasks_processors_subscriptions_config = DataConfig {
            ..Default::default()
        };
        let join_tasks_processors_subscriptions_config_json =
            serde_json::to_vec(&join_tasks_processors_subscriptions_config).unwrap();
        let join_tasks_processors_subscriptions_state = TableBuilder::new()
            .with_name(self.join_tasks_processors_subscriptions_processor_name)
            .with_json(&join_tasks_processors_subscriptions_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let join_tasks_processors_subscriptions_subjects_config = DataConfig {
            ..Default::default()
        };
        let join_tasks_processors_subscriptions_subjects_config_json =
            serde_json::to_vec(&join_tasks_processors_subscriptions_subjects_config).unwrap();
        let join_tasks_processors_subscriptions_subjects_state = TableBuilder::new()
            .with_name(self.join_tasks_processors_subscriptions_subjects_processor_name)
            .with_json(&join_tasks_processors_subscriptions_subjects_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_tasks_processors_subscriptions_subjects_config = DataConfig {
            ..Default::default()
        };
        let select_tasks_processors_subscriptions_subjects_config_json =
            serde_json::to_vec(&select_tasks_processors_subscriptions_subjects_config).unwrap();
        let select_tasks_processors_subscriptions_subjects_state = TableBuilder::new()
            .with_name(self.select_tasks_processors_subscriptions_subjects_processor_name)
            .with_json(&select_tasks_processors_subscriptions_subjects_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let filter_tasks_processors_subscriptions_subjects_config = DataConfig {
            ..Default::default()
        };
        let filter_tasks_processors_subscriptions_subjects_config_json =
            serde_json::to_vec(&filter_tasks_processors_subscriptions_subjects_config).unwrap();
        let filter_tasks_processors_subscriptions_subjects_state = TableBuilder::new()
            .with_name(self.filter_tasks_processors_subscriptions_subjects_processor_name)
            .with_json(&filter_tasks_processors_subscriptions_subjects_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        
        let filter_processors_publications_config = DataConfig {
            ..Default::default()
        };
        let filter_processors_publications_config_json =
            serde_json::to_vec(&filter_processors_publications_config).unwrap();
        let filter_processors_publications_state = TableBuilder::new()
            .with_name(self.filter_processors_publications_processor_name)
            .with_json(&filter_processors_publications_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_tasks_ready_to_run_config = DataConfig {
            ..Default::default()
        };
        let select_tasks_ready_to_run_config_json =
            serde_json::to_vec(&select_tasks_ready_to_run_config).unwrap();
        let select_tasks_ready_to_run_state = TableBuilder::new()
            .with_name(self.select_tasks_ready_to_run_processor_name)
            .with_json(&select_tasks_ready_to_run_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        
        let join_tasks_processors_publications_config = DataConfig {
            ..Default::default()
        };
        let join_tasks_processors_publications_config_json =
            serde_json::to_vec(&join_tasks_processors_publications_config).unwrap();
        let join_tasks_processors_publications_state = TableBuilder::new()
            .with_name(self.join_tasks_processors_publications_processor_name)
            .with_json(&join_tasks_processors_publications_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let select_tasks_processors_publications_config = DataConfig {
            ..Default::default()
        };
        let select_tasks_processors_publications_config_json =
            serde_json::to_vec(&select_tasks_processors_publications_config).unwrap();
        let select_tasks_processors_publications_state = TableBuilder::new()
            .with_name(self.select_tasks_processors_publications_processor_name)
            .with_json(&select_tasks_processors_publications_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        
        let aggregate_tasks_processors_publications_config = DataConfig {
            ..Default::default()
        };
        let aggregate_tasks_processors_publications_config_json =
            serde_json::to_vec(&aggregate_tasks_processors_publications_config).unwrap();
        let aggregate_tasks_processors_publications_state = TableBuilder::new()
            .with_name(self.aggregate_tasks_processors_publications_processor_name)
            .with_json(&aggregate_tasks_processors_publications_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![extract_tasks_state,
            group_by_subject_change_log_delta_state,
            select_subject_change_log_delta_state,
            group_by_subjects_num_rows_state,
            select_subjects_num_rows_state,
            join_subjects_num_rows_delta_state,
            add_subjects_num_rows_delta_state,
            select_subjects_num_rows_delta_state,
            group_by_tasks_run_log_timestamp_state,
            select_tasks_run_log_timestamp_state,
            filter_processors_subscriptions_state,
            join_tasks_run_log_timestamp_state,
            join_tasks_processors_subscriptions_state,
            join_tasks_processors_subscriptions_subjects_state,
            select_tasks_processors_subscriptions_subjects_state,
            filter_tasks_processors_subscriptions_subjects_state,
            filter_processors_publications_state,
            select_tasks_ready_to_run_state,
            join_tasks_processors_publications_state,
            select_tasks_processors_publications_state,
            aggregate_tasks_processors_publications_state,
            AvailableInterfaceSubjects::UserCsv
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::SessionTasksInbox.to_table(None, None).unwrap(),
            AvailableSubjects::SubjectsChangeLog.to_table(None, None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.group_by_subject_change_log_delta_task_name), None).unwrap(),
            AvailableSubjects::SubjectsChangeLog.to_table(Some(self.select_subject_change_log_delta_task_name), None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.group_by_subjects_num_rows_task_name), None).unwrap(),
            AvailableSubjects::SubjectsNumRows.to_table(Some(self.select_subjects_num_rows_task_name), None).unwrap(),
            AvailableSubjects::SessionTasksRunLog.to_table(None, None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.group_by_tasks_run_log_timestamp_task_name), None).unwrap(),
            AvailableSubjects::SessionTasksRunLog.to_table(Some(self.select_tasks_run_log_timestamp_task_name), None).unwrap(),
            AvailableSubjects::SessionProcessors.to_table(None, None).unwrap(),
            AvailableSubjects::SessionProcessors.to_table(Some(self.filter_processors_subscriptions_task_name), None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.join_tasks_run_log_timestamp_task_name), None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.join_tasks_processors_subscriptions_task_name), None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.join_tasks_processors_subscriptions_subjects_task_name), None).unwrap(),
			// DM: could define the schema
            AvailableSubjects::Empty.to_table(Some(self.select_tasks_processors_subscriptions_subjects_task_name), None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.filter_tasks_processors_subscriptions_subjects_task_name), None).unwrap(),
            AvailableSubjects::SessionProcessors.to_table(Some(self.filter_processors_publications_task_name), None).unwrap(),
            AvailableSubjects::SessionTasksInbox.to_table(Some(self.select_tasks_ready_to_run_task_name), None).unwrap(),
            AvailableSubjects::Empty.to_table(Some(self.join_tasks_processors_publications_task_name), None).unwrap(),
			// DM: could define the schema
            AvailableSubjects::Empty.to_table(Some(self.select_tasks_processors_publications_task_name), None).unwrap(),
            AvailableInterfaceSubjects::AssistantCsv
                .to_table(None, None)
                .unwrap(),
        ])
    }
}

#[allow(dead_code)]
pub(crate) mod user_session_inner {
    use anyhow::Result;
    use parking_lot::RwLock;
    use phymes_core::{
        BlobBuilderTraitExt, BuildableTrait, IPCMessage, MappableTrait, MessageBuilderTrait,
        TableTrait, create_user_inbox_batch,
    };

    use crate::{
        SessionContextBuilderAgentsTrait, SessionContextBuilderTrait, SessionStream,
        SessionStreamState, create_message_map,
    };

    use super::*;

    pub fn user_session() -> Result<(Arc<RwLock<SessionStreamState>>, SessionStream)> {
        // initialize the session
        let user_agent_session = SubjectsSession::default();
        let session_ctx = user_agent_session
            .build()
            .with_name(user_agent_session.session_context_name)
            .with_diagnostics(true)
            .build_with_tables()?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // Make the tabular data
        let batch = create_user_inbox_batch(vec!["contact@biom8er.com".to_string()])?;
        let bytes = Table::get_builder()
            .with_record_batches(vec![batch])?
            .with_name(AvailableInterfaceSubjects::UserJson.to_string().as_str())
            .build()?
            .to_json()?;

        // Wrap into the message
        let blob = AvailableInterfaceSubjects::UserJson
            .to_table_builder(None)
            .with_blob(None, Some("json"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&TablePublication::Replace {
                table_name: blob.get_name().to_string(),
            })
            .with_publisher(user_agent_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![blob_message]);

        let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));

        Ok((session_stream_state, session_stream))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{IPCMessage, MappableTrait, MessageTrait, TableTrait};
    use phymes_diagnostics::HashMap;

    use crate::{SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionStream, SessionStreamState, create_message_map};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_subjects_session() -> Result<()> {
        // Initialize the session
        let subjects_session = SubjectsSession::default();
        let session_ctx = subjects_session
            .build()
            .with_name(subjects_session.session_context_name)
            .to_mermaid_flowchart(false, false)?;
        dbg!(&session_ctx);
        //     .add_session_interface(None)?
        //     .build_with_tables()?;
        // let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));
        
        // // Create the messages
        // let message_map = create_message_map(vec![chat_message, blob_message]);

        // // Run the session
        // let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
        // let mut response: Vec<HashMap<String, IPCMessage>> =
        //     session_stream.try_collect().await?;


        Ok(())
    }
}
