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
    /// 1, 2, and 3. Publication update tasks
    /// 1, 2, and 3. Subscription tasks
    pub extract_tasks_task_name: &'a str,
    pub extract_tasks_processor_name: &'a str,

    pub extract_subject_change_log_task_name: &'a str,
    pub extract_subject_change_log_processor_name: &'a str,

    pub extract_session_tasks_run_log_task_name: &'a str,
    pub extract_session_task_runs_log_processor_name: &'a str,

    /// 1, 2, and 3. Update subjects change log

    /// 1, 3, and 3. Update session tasks run log

    /// 1, 2, and 3. Aggregate the latest subjects change log
    // DM: 
    // Sum aggreggation of delta, and set of session_name;task_name
    pub group_by_updated_subjects_task_name: &'a str,
    pub group_by_updated_subjects_processor_name: &'a str,
    // index = session_name;task_name
    pub index_updated_subjects_task_name: &'a str,
    pub index_updated_subjects_processor_name: &'a str,

    /// 1, 2, and 3. Aggregate the latest subjects change log
    // DM: index session_name;task_name
    // DM: Last aggregation of all fields

    /// 1. Count the number of rows per subject
    pub join_subjects_num_rows_delta_task_name: &'a str,
    pub join_subjects_num_rows_delta_processor_name: &'a str,
    pub add_subjects_num_rows_delta_task_name: &'a str,
    pub add_subjects_num_rows_delta_processor_name: &'a str,
    pub select_subjects_num_rows_delta_task_name: &'a str,
    pub select_subjects_num_rows_delta_processor_name: &'a str,
    // Extend with the new batch
    pub group_by_subjects_task_name: &'a str,
    pub group_by_subjects_processor_name: &'a str,
    pub select_subjects_num_rows_task_name: &'a str,
    pub select_subjects_num_rows_processor_name: &'a str,
    // Replace with the new batch    

    /// 2 and 3. Cache updated tasks
    pub group_by_ran_tasks_task_name: &'a str,
    pub group_by_ran_tasks_processor_name: &'a str,
    pub index_ran_tasks_task_name: &'a str,
    pub index_ran_tasks_processor_name: &'a str,

    /// 2 and 3. Join between incoming tasks and session tasks
    pub join_tasks_inbox_ran_tasks_task_name: &'a str,
    pub join_tasks_inbox_ran_tasks_processor_name: &'a str,
    pub join_tasks_inbox_session_tasks_task_name: &'a str,
    pub join_tasks_inbox_session_tasks_processor_name: &'a str,

    /// 2. Cache filtered subscriptions
    pub filter_session_processors_subscriptions_task_name: &'a str,
    pub filter_session_processors_subscriptions_processor_name: &'a str,
    
    /// 2. Retrieve updated subscriptions
    pub join_tasks_filtered_processors_subscriptions_task_name: &'a str,
    pub join_tasks_filtered_processors_subscriptions_processor_name: &'a str,
    pub join_tasks_filtered_processors_subscriptions_filtered_updated_subjects_task_name: &'a str,
    pub join_tasks_filtered_processors_subscriptions_filtered_updated_subjects_processor_name: &'a str,    
    pub select_updated_subscriptions_task_name: &'a str,
    pub select_updated_subscriptions_processor_name: &'a str,
    
    /// 3. Cache filtered publications
    pub filter_session_processors_publications_task_name: &'a str,
    pub filter_session_processors_publications_processor_name: &'a str,

    /// 3. Retrieve the publications
    pub join_tasks_filtered_processors_publications_task_name: &'a str,
    pub join_tasks_filtered_processors_publications_processor_name: &'a str,
    pub join_tasks_filtered_processors_publications_filtered_updated_subjects_task_name: &'a str,
    pub join_tasks_filtered_processors_publications_filtered_updated_subjects_processor_name: &'a str,    
    pub select_updated_publications_task_name: &'a str,
    pub select_updated_publications_processor_name: &'a str,

    /// Session
    pub session_context_name: &'a str,

    /// Runtime environment
    pub default_runtime_env_name: &'a str,
}

impl Default for SubjectsSession<'_> {
    fn default() -> Self {
        // SubjectsSession {
        //     session_context_name: "subject_session",
        // }
        todo!()
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
                task_name: self.extract_task_inbox_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.extract_task_inbox_processor_name.to_string(),
                    self.index_task_inbox_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.group_by_updated_subjects_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.group_by_updated_subjects_processor_name.to_string(),
                    self.index_updated_subjects_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.join_subjects_num_rows_delta_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.join_subjects_num_rows_delta_processor_name.to_string(),
                    self.add_subjects_num_rows_delta_processor_name.to_string(),
                    self.select_subjects_num_rows_delta_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.group_by_subjects_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.group_by_subjects_processor_name.to_string(),
                    self.select_subjects_num_rows_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.join_tasks_inbox_session_tasks_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.join_tasks_inbox_session_tasks_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.filter_session_processors_subscriptions_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.filter_session_processors_subscriptions_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.join_tasks_filtered_processors_subscriptions_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.join_tasks_filtered_processors_subscriptions_processor_name.to_string(),
                    self.join_tasks_filtered_processors_subscriptions_filtered_updated_subjects_processor_name.to_string(),
                    self.select_updated_subscriptions_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.filter_session_processors_publications_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.filter_session_processors_publications_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.join_tasks_filtered_processors_publications_task_name.to_string(),
                runtime_env_name: self.default_runtime_env_name.to_string(),
                processor_names: vec![
                    self.join_tasks_filtered_processors_publications_processor_name.to_string(),
                    self.join_tasks_filtered_processors_publications_filtered_updated_subjects_processor_name.to_string(),
                    self.select_updated_publications_processor_name.to_string(),
                ],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            AvailableProcessors::ExtractTabular.build_arc(
                self.extract_task_inbox_processor_name,
                &[TablePublication::Replace {
                    table_name: self.extract_task_inbox_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableInterfaceSubjects::UserCsv.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.extract_task_inbox_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.index_task_inbox_processor_name,
                &[TablePublication::Replace {
                    table_name: self.index_task_inbox_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysLastRecordBatch {
                        table_name: self.extract_task_inbox_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.index_task_inbox_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Filter.build_arc(
                self.group_by_updated_subjects_processor_name,
                &[TablePublication::Replace {
                    table_name: self.group_by_updated_subjects_task_name.to_string(),
                }],
                &[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.group_by_updated_subjects_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::Select.build_arc(
                self.index_updated_subjects_processor_name,
                &[TablePublication::Replace {
                    table_name: self.index_updated_subjects_task_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysFullTable {
                        table_name: self.filter_updated_subjects_task_name.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.index_updated_subjects_processor_name.to_string(),
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
                        table_name: AvailableSubjects::SubjectsNumRows.to_string(),
                    },
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: self.join_subjects_num_rows_delta_processor_name.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AnyTableNameSubscribe.build(),
            ),

            AvailableProcessors::Join.build_arc(
                self.filter_user_info_by_email_processor_name,
                &[TablePublication::Replace {
                    table_name: self.filter_user_info_by_email_table_name.to_string(),
                }],
                &[
                    TableSubscription::AlwaysLastRecordBatch {
                        table_name: self.filter_user_info_by_email_processor_name.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::UserInbox.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: AvailableSubjects::User.to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
            AvailableProcessors::DataSummaryProcessor.build_arc(
                self.filter_and_join_session_contexts_by_email_outbox_processor_name,
                &[TablePublication::Replace {
                    table_name: AvailableInterfaceSubjects::AssistantJson.to_string(),
                }],
                &[
                    TableSubscription::AlwaysLastRecordBatch {
                        table_name: self
                            .filter_and_join_session_contexts_by_email_outbox_processor_name
                            .to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: self.filter_user_info_by_email_table_name.to_string(),
                    },
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid
                            .to_string(),
                    },
                ],
                AvailableTableSubscribePolicies::AnyTableNameSubscribe.build(),
            ),
            AvailableProcessors::ProcessorEcho.build_arc(
                self.session_context_name,
                &[
                    TablePublication::Extend {
                        table_name: AvailableSubjects::BuilderMermaid.to_string(),
                    },
                    TablePublication::Extend {
                        table_name: AvailableSubjects::User.to_string(),
                    },
                    TablePublication::Extend {
                        table_name: AvailableSubjects::UserSessionContexts.to_string(),
                    },
                    TablePublication::Replace {
                        table_name: AvailableInterfaceSubjects::UserJson.to_string(),
                    },
                    TablePublication::Replace {
                        table_name: AvailableInterfaceSubjects::AssistantJson.to_string(),
                    },
                ],
                &[TableSubscription::OnUpdateFullTable {
                    table_name: AvailableInterfaceSubjects::AssistantJson.to_string(),
                }],
                AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
            ),
        ];

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name("rt_default"),
            RuntimeEnv::new().with_name(self.filter_session_contexts_by_email_runtime_env_name),
            RuntimeEnv::new()
                .with_name(self.join_session_contexts_with_mermaid_diagrams_runtime_env_name),
            RuntimeEnv::new().with_name(self.filter_user_info_by_email_runtime_env_name),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Extract tabular data config
        let extract_tabular_data_config = DataConfig {
            lhs_name: Some(AvailableInterfaceSubjects::UserJson.to_string()),
            lhs_values: Some(vec!["bytes".to_string()]),
            format: Some(DataFormat::JsonDefault),
            operator: AvailableCandleOperators::ExtractTabular,
            ..Default::default()
        };
        let extract_tabular_data_config_json =
            serde_json::to_vec(&extract_tabular_data_config).unwrap();
        let extract_tabular_data_state = TableBuilder::new()
            .with_name(self.filter_and_join_session_contexts_by_email_inbox_processor_name)
            .with_json(&extract_tabular_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Attachment config
        let attachment_config = DataSummaryConfig {
            summary_format: DataFormat::JsonDefault,
            ..Default::default()
        };
        let attachment_config_json = serde_json::to_vec(&attachment_config).unwrap();
        let attachment_state = TableBuilder::new()
            .with_name(self.filter_and_join_session_contexts_by_email_outbox_processor_name)
            .with_json(&attachment_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for filter
        let filter_user_info_data_config = DataConfig {
            operator: AvailableCandleOperators::Join,
            lhs_name: Some(AvailableSubjects::UserInbox.to_string()),
            lhs_pk: Some("email".to_string()),
            lhs_fk: Some("email".to_string()),
            rhs_name: Some(AvailableSubjects::User.to_string()),
            rhs_pk: Some("email".to_string()),
            rhs_fk: Some("email".to_string()),
            ..Default::default()
        };
        let filter_user_info_data_config_json =
            serde_json::to_vec(&filter_user_info_data_config).unwrap();
        let filter_user_info_data_state = TableBuilder::new()
            .with_name(self.filter_user_info_by_email_processor_name)
            .with_json(&filter_user_info_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for filter
        let filter_user_session_context_data_config = DataConfig {
            operator: AvailableCandleOperators::Join,
            lhs_name: Some(AvailableSubjects::UserInbox.to_string()),
            lhs_pk: Some("email".to_string()),
            lhs_fk: Some("email".to_string()),
            rhs_name: Some(AvailableSubjects::UserSessionContexts.to_string()),
            rhs_pk: Some("email".to_string()),
            rhs_fk: Some("email".to_string()),
            ..Default::default()
        };
        let filter_user_session_context_data_config_json =
            serde_json::to_vec(&filter_user_session_context_data_config).unwrap();
        let filter_user_session_context_data_state = TableBuilder::new()
            .with_name(self.filter_session_contexts_by_email_processor_name)
            .with_json(&filter_user_session_context_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for join
        let join_user_session_context_data_config = DataConfig {
            operator: AvailableCandleOperators::Join,
            lhs_name: Some(AvailableSubjects::JoinUserInboxSessionContexts.to_string()),
            lhs_pk: Some("email".to_string()),
            lhs_fk: Some("session_context_name".to_string()),
            rhs_name: Some(AvailableSubjects::BuilderMermaid.to_string()),
            rhs_pk: Some("session_context_name".to_string()),
            rhs_fk: Some("session_context_name".to_string()),
            ..Default::default()
        };
        let join_user_session_context_data_config_json =
            serde_json::to_vec(&join_user_session_context_data_config).unwrap();
        let join_user_session_context_data_state = TableBuilder::new()
            .with_name(self.join_session_contexts_with_mermaid_diagrams_processor_name)
            .with_json(&join_user_session_context_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![
            extract_tabular_data_state,
            attachment_state,
            filter_user_info_data_state,
            filter_user_session_context_data_state,
            join_user_session_context_data_state,
            AvailableInterfaceSubjects::UserJson
                .to_table(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AssistantJson
                .to_table(None, None)
                .unwrap(),
            self.make_user_table().unwrap(),
            self.make_user_session_context_table().unwrap(),
            AvailableSubjects::User
                .to_table(Some(self.filter_user_info_by_email_table_name), None)
                .unwrap(),
            AvailableSubjects::UserInbox.to_table(None, None).unwrap(),
            AvailableSubjects::JoinUserInboxSessionContexts
                .to_table(None, None)
                .unwrap(),
            AvailableSubjects::JoinUserInboxSessionContextsMermaid
                .to_table(None, None)
                .unwrap(),
            make_example_mermaid_table(false, true).unwrap(),
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
    use phymes_core::{IPCMessage, MappableTrait, MessageTrait, TableTrait};
    use phymes_diagnostics::HashMap;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_user_session() -> Result<()> {
        let (session_stream_state, session_stream) = user_session_inner::user_session()?;
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        let attachment_data = response
            .into_iter()
            .map(|mut r| {
                r.remove(&format!(
                    "from_{}_on_{}",
                    session_stream_state.read().get_session_context().get_name(),
                    AvailableInterfaceSubjects::AssistantJson
                ))
            })
            .filter_map(|m| {
                m.map(|message| {
                    TableBuilder::new_from_ipc_stream(&message.get_message_own())
                        .unwrap()
                        .with_name("")
                        .build()
                        .unwrap()
                        .to_json_object()
                        .unwrap()
                })
            })
            .flatten()
            .collect::<Vec<_>>();
        for row in &attachment_data {
            let bytes = row["bytes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect::<Vec<u8>>();
            println!(
                "attachment {}{}: {}",
                row["filename"].as_str().unwrap(),
                row["extension"].as_str().unwrap(),
                String::from_utf8_lossy(bytes.as_ref()).into_owned()
            )
        }

        Ok(())
    }
}
