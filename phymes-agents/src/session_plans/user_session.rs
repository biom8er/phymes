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

/// A session for all user management tasks
/// 
/// # Notes
/// 
/// Supported tasks include the following:
/// 
/// 1. Filtering the user information by email
/// 2. Joining the user sessions with their mermaid diagrams
/// 3. Registering new users
/// 
/// An inbox and outbox for each support task are provided
///   that trigger the task
pub struct UserSession<'a> {
    /// Extract data from inbox subtask
    pub filter_and_join_session_contexts_by_email_inbox_task_name: &'a str,
    pub filter_and_join_session_contexts_by_email_inbox_processor_name: &'a str,
    /// Make outbox attachment subtask
    pub filter_and_join_session_contexts_by_email_outbox_task_name: &'a str,
    pub filter_and_join_session_contexts_by_email_outbox_processor_name: &'a str,
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

impl Default for UserSession<'_> {
    fn default() -> Self {
        UserSession {
            session_context_name: "session_context_name",
            filter_and_join_session_contexts_by_email_inbox_task_name: "filter_and_join_session_contexts_by_email_inbox_task_name",
            filter_and_join_session_contexts_by_email_inbox_processor_name: "filter_and_join_session_contexts_by_email_inbox_processor_name",
            filter_and_join_session_contexts_by_email_outbox_task_name: "filter_and_join_session_contexts_by_email_outbox_task_name",
            filter_and_join_session_contexts_by_email_outbox_processor_name: "filter_and_join_session_contexts_by_email_outbox_processor_name",
            filter_session_contexts_by_email_runtime_env_name: "filter_session_contexts_by_email_runtime_env_name",
            filter_session_contexts_by_email_task_name: "filter_session_contexts_by_email_task_name",
            filter_session_contexts_by_email_processor_name: "filter_session_contexts_by_email_processor_name",
            join_session_contexts_with_mermaid_diagrams_runtime_env_name: "join_session_contexts_with_mermaid_diagrams_runtime_env_name",
            join_session_contexts_with_mermaid_diagrams_task_name: "join_session_contexts_with_mermaid_diagrams_task_name",
            join_session_contexts_with_mermaid_diagrams_processor_name: "join_session_contexts_with_mermaid_diagrams_processor_name",
            filter_user_info_by_email_runtime_env_name: "filter_user_info_by_email_runtime_env_name",
            filter_user_info_by_email_task_name: "filter_user_info_by_email_task_name",
            filter_user_info_by_email_processor_name: "filter_user_info_by_email_processor_name",
            filter_user_info_by_email_table_name: "filter_user_info_by_email_table_name",
        }
    }
}

impl<'a> UserSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        UserSession {
            session_context_name,
            ..Default::default()
        }
    }

    pub fn make_user_table(&self) -> Result<Table> {
        let batch = create_user_batch(vec!["contact@biom8er.com".to_string()], vec!["con".to_string()], vec!["tact".to_string()], vec!["$2b$12$qJGwWR2rSZ9oBFZff0o2w.RXViv.Mf.BwfsWZTfVm4DmjjVfsaHzi".to_string()], vec![create_timestamp_micros()])?;
        TableBuilder::new()
            .with_name(AvailableSubjects::User.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    pub fn make_user_session_context_table(&self) -> Result<Table> {
        let mut email = Vec::new();
        let mut session_context_name = Vec::new();
        for name in AvailableSessionPlans::get_all_session_plan_names() {
            email.push("contact@biom8er.com".to_string());
            session_context_name.push(name);
        }
        let batch = create_user_session_contexts_batch(email, session_context_name)?;
        TableBuilder::new()
            .with_name(AvailableSubjects::UserSessionContexts.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }
}

impl CustomAgentsBuilderTrait for UserSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![
            TaskPlan {
                task_name: self.filter_and_join_session_contexts_by_email_inbox_task_name.to_string(),
                runtime_env_name: "rt_default".to_string(),
                processor_names: vec![self.filter_and_join_session_contexts_by_email_inbox_processor_name.to_string()],
            }, TaskPlan {
                task_name: self.filter_session_contexts_by_email_task_name.to_string(),
                runtime_env_name: self.filter_session_contexts_by_email_runtime_env_name.to_string(),
                processor_names: vec![self.filter_session_contexts_by_email_processor_name.to_string()],
            }, TaskPlan {
                task_name: self.join_session_contexts_with_mermaid_diagrams_task_name.to_string(),
                runtime_env_name: self.join_session_contexts_with_mermaid_diagrams_runtime_env_name.to_string(),
                processor_names: vec![self.join_session_contexts_with_mermaid_diagrams_processor_name.to_string()],
            }, TaskPlan {
                task_name: self.filter_and_join_session_contexts_by_email_outbox_task_name.to_string(),
                runtime_env_name: "rt_default".to_string(),
                processor_names: vec![self.filter_and_join_session_contexts_by_email_outbox_processor_name.to_string()],
            }, TaskPlan {
                task_name: self.session_context_name.to_string(),
                runtime_env_name: "rt_default".to_string(),
                processor_names: vec![self.session_context_name.to_string()],
            }, TaskPlan {
                task_name: self.filter_user_info_by_email_task_name.to_string(),
                runtime_env_name: self.filter_user_info_by_email_runtime_env_name.to_string(),
                processor_names: vec![self.filter_user_info_by_email_processor_name.to_string()],
            }
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            CandleDataProcessor::new_arc_with_pub_sub(
                self.filter_and_join_session_contexts_by_email_inbox_processor_name,
                &[TablePublish::Replace {
                    table_name: AvailableSubjects::UserInbox.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: AvailableInterfaceSubjects::UserJson.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.filter_and_join_session_contexts_by_email_inbox_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ), CandleDataProcessor::new_arc_with_pub_sub(
                self.filter_session_contexts_by_email_processor_name,
                &[
                    TablePublish::Replace { table_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string() },
                ],
                &[
                    TableSubscribe::AlwaysLastRecordBatch { table_name: self.filter_session_contexts_by_email_processor_name.to_string() },
                    TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::UserInbox.to_string() },
                    TableSubscribe::AlwaysFullTable { table_name: AvailableSubjects::UserSessionContexts.to_string() },
                ],
                AllTableNamesSubscribe::new_box(),
            ), CandleDataProcessor::new_arc_with_pub_sub(
                self.join_session_contexts_with_mermaid_diagrams_processor_name,
                &[
                    TablePublish::Replace { table_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string() },
                ],
                &[
                    TableSubscribe::AlwaysLastRecordBatch { table_name: self.join_session_contexts_with_mermaid_diagrams_processor_name.to_string() },
                    TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string() },
                    TableSubscribe::AlwaysFullTable { table_name: AvailableSubjects::Mermaid.to_string() },
                ],
                AllTableNamesSubscribe::new_box(),
            ), CandleDataProcessor::new_arc_with_pub_sub(
                self.filter_user_info_by_email_processor_name,
                &[
                    TablePublish::Replace { table_name: self.filter_user_info_by_email_table_name.to_string() },
                ],
                &[
                    TableSubscribe::AlwaysLastRecordBatch { table_name: self.filter_user_info_by_email_processor_name.to_string() },
                    TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::UserInbox.to_string() },
                    TableSubscribe::AlwaysFullTable { table_name: AvailableSubjects::User.to_string() },
                ],
                AllTableNamesSubscribe::new_box(),
            ), DataSummaryProcessor::new_arc_with_pub_sub(
                self.filter_and_join_session_contexts_by_email_outbox_processor_name,
                &[
                    TablePublish::Replace { table_name: AvailableInterfaceSubjects::AssistantJson.to_string() }
                ],
                &[
                    TableSubscribe::AlwaysLastRecordBatch { table_name: self.filter_and_join_session_contexts_by_email_outbox_processor_name.to_string() },
                    TableSubscribe::OnUpdateFullTable { table_name: self.filter_user_info_by_email_table_name.to_string() },
                    TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string() },
                ],
                AnyTableNameSubscribe::new_box(),
            ), ProcessorEcho::new_arc_with_pub_sub(
                self.session_context_name,
                &[
                    TablePublish::Extend { table_name: AvailableSubjects::Mermaid.to_string() },
                    TablePublish::Extend { table_name: AvailableSubjects::User.to_string() },
                    TablePublish::Extend { table_name: AvailableSubjects::UserSessionContexts.to_string() },
                    TablePublish::Replace { table_name: AvailableInterfaceSubjects::UserJson.to_string() },
                    TablePublish::Replace { table_name: AvailableInterfaceSubjects::AssistantJson.to_string() },
                ],
                &[
                    TableSubscribe::OnUpdateFullTable { table_name: AvailableInterfaceSubjects::AssistantJson.to_string() },
                ],
                AllTableNamesSubscribe::new_box(),
            )
        ];

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name("rt_default"),
            RuntimeEnv::new().with_name(self.filter_session_contexts_by_email_runtime_env_name),
            RuntimeEnv::new().with_name(self.join_session_contexts_with_mermaid_diagrams_runtime_env_name),
            RuntimeEnv::new().with_name(self.filter_user_info_by_email_runtime_env_name),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Extract tabular data config
        let extract_tabular_data_config = DataConfig {
            lhs_name: AvailableInterfaceSubjects::UserJson.to_string(),
            lhs_values: vec!["bytes".to_string()],
            format: Some(DataFormat::JsonDefault),
            operator: AvailableCandleOperators::ExtractTabularData,
            ..Default::default()
        };
        let extract_tabular_data_config_json = serde_json::to_vec(&extract_tabular_data_config).unwrap();
        let extract_tabular_data_state = TableBuilder::new()
            .with_name(self.filter_and_join_session_contexts_by_email_inbox_processor_name)
            .with_json(&extract_tabular_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Attachment config
        let attachment_config = DataSummaryConfig {
            format: DataFormat::JsonDefault,
            ..Default::default()
        };
        let attachmen_config_json = serde_json::to_vec(&attachment_config).unwrap();
        let attachmen_state = TableBuilder::new()
            .with_name(self.filter_and_join_session_contexts_by_email_outbox_processor_name)
            .with_json(&attachmen_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for filter
        let filter_user_info_data_config = DataConfig {
            operator: AvailableCandleOperators::JoinInner,
            lhs_name: AvailableSubjects::UserInbox.to_string(),
            lhs_pk: "email".to_string(),
            lhs_fk: "email".to_string(),
            rhs_name: Some(AvailableSubjects::User.to_string()),
            rhs_pk: Some("email".to_string()),
            rhs_fk: Some("email".to_string()),
            ..Default::default()
        };
        let filter_user_info_data_config_json = serde_json::to_vec(&filter_user_info_data_config).unwrap();
        let filter_user_info_data_state = TableBuilder::new()
            .with_name(self.filter_user_info_by_email_processor_name)
            .with_json(&filter_user_info_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for filter
        let filter_user_session_context_data_config = DataConfig {
            operator: AvailableCandleOperators::JoinInner,
            lhs_name: AvailableSubjects::UserInbox.to_string(),
            lhs_pk: "email".to_string(),
            lhs_fk: "email".to_string(),
            rhs_name: Some(AvailableSubjects::UserSessionContexts.to_string()),
            rhs_pk: Some("email".to_string()),
            rhs_fk: Some("email".to_string()),
            ..Default::default()
        };
        let filter_user_session_context_data_config_json = serde_json::to_vec(&filter_user_session_context_data_config).unwrap();
        let filter_user_session_context_data_state = TableBuilder::new()
            .with_name(self.filter_session_contexts_by_email_processor_name)
            .with_json(&filter_user_session_context_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for join
        let join_user_session_context_data_config = DataConfig {
            operator: AvailableCandleOperators::JoinInner,
            lhs_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string(),
            lhs_pk: "email".to_string(),
            lhs_fk: "session_context_name".to_string(),
            rhs_name: Some(AvailableSubjects::Mermaid.to_string()),
            rhs_pk: Some("session_context_name".to_string()),
            rhs_fk: Some("session_context_name".to_string()),
            ..Default::default()
        };
        let join_user_session_context_data_config_json = serde_json::to_vec(&join_user_session_context_data_config).unwrap();
        let join_user_session_context_data_state = TableBuilder::new()
            .with_name(self.join_session_contexts_with_mermaid_diagrams_processor_name)
            .with_json(&join_user_session_context_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![
            extract_tabular_data_state,
            attachmen_state,
            filter_user_info_data_state,
            filter_user_session_context_data_state,
            join_user_session_context_data_state,
            AvailableInterfaceSubjects::UserJson.to_table(None, None).unwrap(),
            AvailableInterfaceSubjects::AssistantJson.to_table(None, None).unwrap(),
            self.make_user_table().unwrap(),
            self.make_user_session_context_table().unwrap(),
            AvailableSubjects::User.to_table(Some(self.filter_user_info_by_email_table_name), None).unwrap(),
            AvailableSubjects::UserInbox.to_table(None, None).unwrap(),
            AvailableSubjects::JoinUserInboxSessionContexts.to_table(None, None).unwrap(),
            AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_table(None, None).unwrap(),
            make_example_mermaid_table(false).unwrap(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{schemas::{blob::BlobBuilderTraitExt, user::create_user_inbox_batch}, session::{common_traits::{BuildableTrait, MappableTrait}, session_stream::SessionStream, session_stream_state::SessionStreamState}, table::table_trait::TableTrait, task::message::{IPCMessage, MessageBuilderTrait, MessageTrait}};
    use phymes_diagnostics::HashMap;

    use crate::{session_plans::available_interface_subjects::create_message_map, session_traits::agents::SessionContextBuilderAgentsTrait};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_user_agent_session() -> Result<()> {

        // initialize the session
        let user_agent_session = UserSession::default();
        let session_ctx = user_agent_session
            .build()
            .with_name(user_agent_session.session_context_name)
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
        let blob = AvailableInterfaceSubjects::UserJson.to_table_builder(None)
            .with_blob(None, Some("json"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&TablePublish::Replace { table_name: blob.get_name().to_string() })
            .with_publisher(user_agent_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![blob_message]);

        let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        let attachment_data = response
            .into_iter()
            .map(|mut r| r.remove(&format!(
                "from_{}_on_{}",
                user_agent_session.session_context_name,
                AvailableInterfaceSubjects::AssistantJson
            )))
            .filter_map(|m| {
                m.map(|message| TableBuilder::new_from_ipc_stream(&message.get_message_own()).unwrap()
                    .with_name("")
                    .build().unwrap()
                    .to_json_object().unwrap())
            })
            .flatten()
            .collect::<Vec<_>>();
        for row in &attachment_data {
            let bytes = row["bytes"].as_array().unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect::<Vec<u8>>();
            println!("attachment {}{}: {}", row["filename"].as_str().unwrap(), row["extension"].as_str().unwrap(), String::from_utf8_lossy(bytes.as_ref()).into_owned())
        }

        Ok(())
    }
}
