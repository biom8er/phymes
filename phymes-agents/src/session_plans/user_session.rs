use std::sync::Arc;
use anyhow::Result;

use phymes_core::{
    schemas::{available_subjects::{create_timestamp_micros, AvailableSubjects, AvailableSubjectsTrait}, user::{create_user_batch, create_user_session_contexts_batch}}, session::{
        common_traits::BuilderTrait,
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::TaskPlan,
    }, table::{
        data_format::DataFormat, table::{Table, TableBuilder, TableBuilderTrait}, table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, AnyTableNameSubscribe, SubscribeTrait, TableSubscribe}
    }, task::processor::{ProcessorEcho, ProcessorTrait}
};
use phymes_data::{candle_data::{data_config::DataConfig, data_processor::CandleDataProcessor, summary_config::DataSummaryConfig, summary_processor::DataSummaryProcessor}, candle_operators::available_candle_operators::AvailableCandleOperators};

use crate::{session_plans::{available_interface_subjects::AvailableInterfaceSubjects, available_session_plans::AvailableSessionPlans, builder_session::make_example_mermaid_table}, session_traits::agents::CustomAgentsBuilderTrait};

pub struct UserSession<'a> {
    /// Extract tabular data from the user attachments
    pub extract_tabular_data_task_name: &'a str,
    pub extract_tabular_data_processor_name: &'a str,
    /// Create the attachment for the user
    pub tool_attachment_task_name: &'a str,
    pub tool_attachment_processor_name: &'a str,
    /// Filter session contexts by email task
    pub filter_session_contexts_by_email_task_name: &'a str,
    pub filter_user_session_contexts_by_email_processor_name: &'a str,
    /// Join session contexts by email task
    pub join_session_contexts_with_mermaid_diagrams_task_name: &'a str,
    pub join_user_session_contexts_with_mermaid_diagrams_processor_name: &'a str,
    pub filter_and_join_session_contexts_by_email_runtime_env_name: &'a str,
    /// Session
    pub session_context_name: &'a str,
}

impl Default for UserSession<'_> {
    fn default() -> Self {
        UserSession {
            session_context_name: "session_context_1",
            extract_tabular_data_task_name: "extract_tabular_data_task_1",
            extract_tabular_data_processor_name: "extract_tabular_data_processor_1",
            tool_attachment_task_name: "tool_attachment_task_1",
            tool_attachment_processor_name: "tool_attachment_processor_1",
            filter_session_contexts_by_email_task_name: "filter_session_contexts_by_email_task_1",
            filter_user_session_contexts_by_email_processor_name: "filter_user_session_contexts_by_email_processor_1",
            join_session_contexts_with_mermaid_diagrams_task_name: "join_session_contexts_with_mermaid_diagrams_task_1",
            join_user_session_contexts_with_mermaid_diagrams_processor_name: "join_user_session_contexts_with_mermaid_diagrams_processor_1",
            filter_and_join_session_contexts_by_email_runtime_env_name: "filter_and_join_session_contexts_by_email_runtime_env_1",
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
        let batch = create_user_batch(vec!["contact@bioma8er.com".to_string()], vec!["con".to_string()], vec!["tact".to_string()], vec!["".to_string()], vec![create_timestamp_micros()])?;
        TableBuilder::new()
            .with_name(AvailableSubjects::User.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    pub fn make_user_session_context_table(&self) -> Result<Table> {
        let mut email = Vec::new();
        let mut session_context_name = Vec::new();
        for name in AvailableSessionPlans::get_deployable_session_plan_names() {
            email.push("contact@bioma8er.com".to_string());
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
        let mut tasks = Vec::new();

        tasks.push(TaskPlan {
            task_name: self.extract_tabular_data_task_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![self.extract_tabular_data_processor_name.to_string()],
        });

        tasks.push(TaskPlan {
            task_name: self.filter_session_contexts_by_email_task_name.to_string(),
            runtime_env_name: self.filter_and_join_session_contexts_by_email_runtime_env_name.to_string(),
            processor_names: vec![self.filter_user_session_contexts_by_email_processor_name.to_string()],
        });

        tasks.push(TaskPlan {
            task_name: self.join_session_contexts_with_mermaid_diagrams_task_name.to_string(),
            runtime_env_name: self.filter_and_join_session_contexts_by_email_runtime_env_name.to_string(),
            processor_names: vec![self.join_user_session_contexts_with_mermaid_diagrams_processor_name.to_string()],
        });

        tasks.push(TaskPlan {
            task_name: self.tool_attachment_task_name.to_string(),
            runtime_env_name: self.filter_and_join_session_contexts_by_email_runtime_env_name.to_string(),
            processor_names: vec![self.tool_attachment_processor_name.to_string()],
        });

        tasks.push(TaskPlan {
            task_name: self.session_context_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![self.session_context_name.to_string()],
        });

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let mut processors = Vec::new();

        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.extract_tabular_data_processor_name,
            &[TablePublish::Replace {
                table_name: AvailableSubjects::UserInbox.to_string(),
            }],
            &[
                TableSubscribe::OnUpdateFullTable {
                    table_name: AvailableInterfaceSubjects::UserCsv.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.extract_tabular_data_processor_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));

        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.filter_user_session_contexts_by_email_processor_name,
            &[
                TablePublish::Replace { table_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string() },
            ],
            &[
                TableSubscribe::AlwaysLastRecordBatch { table_name: self.filter_user_session_contexts_by_email_processor_name.to_string() },
                TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::UserInbox.to_string() },
                TableSubscribe::AlwaysFullTable { table_name: AvailableSubjects::UserSessionContexts.to_string() },
            ],
            AllTableNamesSubscribe::new_box(),
        ));

        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.join_user_session_contexts_with_mermaid_diagrams_processor_name,
            &[
                TablePublish::Replace { table_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string() },
            ],
            &[
                TableSubscribe::AlwaysLastRecordBatch { table_name: self.join_user_session_contexts_with_mermaid_diagrams_processor_name.to_string() },
                TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string() },
                TableSubscribe::AlwaysFullTable { table_name: AvailableSubjects::Mermaid.to_string() },
            ],
            AllTableNamesSubscribe::new_box(),
        ));

        processors.push(DataSummaryProcessor::new_arc_with_pub_sub(
            self.tool_attachment_processor_name,
            &[
                TablePublish::Replace { table_name: AvailableInterfaceSubjects::AssistantCsv.to_string() }
            ],
            &[
                TableSubscribe::AlwaysLastRecordBatch { table_name: self.tool_attachment_processor_name.to_string() },
                TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string() },
                TableSubscribe::OnUpdateFullTable { table_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string() },
            ],
            AnyTableNameSubscribe::new_box(),
        ));

        processors.push(ProcessorEcho::new_arc_with_pub_sub(
            self.session_context_name,
            &[
                TablePublish::Extend { table_name: AvailableSubjects::Mermaid.to_string() },
                TablePublish::Extend { table_name: AvailableSubjects::User.to_string() },
                TablePublish::Extend { table_name: AvailableSubjects::UserSessionContexts.to_string() },
                TablePublish::Replace { table_name: AvailableInterfaceSubjects::UserCsv.to_string() },
                TablePublish::Replace { table_name: AvailableInterfaceSubjects::AssistantCsv.to_string() },
            ],
            &[
                TableSubscribe::OnUpdateFullTable { table_name: AvailableInterfaceSubjects::AssistantCsv.to_string() },
            ],
            AllTableNamesSubscribe::new_box(),
        ));

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name("rt_default"),
            RuntimeEnv::new().with_name(self.filter_and_join_session_contexts_by_email_runtime_env_name),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Extract tabular data config        
        let csv_format_str = serde_json::to_string(&DataFormat::CsvDefault).unwrap();
        let extract_tabular_data_config = DataConfig {
            lhs_name: AvailableInterfaceSubjects::UserCsv.to_string(),
            lhs_values: "bytes".to_string(),
            op_kwargs: Some(csv_format_str),
            operator: AvailableCandleOperators::ExtractTabularData,
            ..Default::default()
        };
        let extract_tabular_data_config_json = serde_json::to_vec(&extract_tabular_data_config).unwrap();
        let extract_tabular_data_state = TableBuilder::new()
            .with_name(self.extract_tabular_data_processor_name)
            .with_json(&extract_tabular_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Attachment config
        let attachment_config = DataSummaryConfig {
            format: DataFormat::CsvDefault,
            ..Default::default()
        };
        let attachmen_config_json = serde_json::to_vec(&attachment_config).unwrap();
        let attachmen_state = TableBuilder::new()
            .with_name(self.tool_attachment_processor_name)
            .with_json(&attachmen_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for filter
        let filter_data_config = DataConfig {
            operator: AvailableCandleOperators::JoinInner,
            lhs_name: AvailableSubjects::UserInbox.to_string(),
            lhs_pk: "email".to_string(),
            lhs_fk: "email".to_string(),
            rhs_name: Some(AvailableSubjects::UserSessionContexts.to_string()),
            rhs_pk: Some("email".to_string()),
            rhs_fk: Some("email".to_string()),
            ..Default::default()
        };
        let filter_data_config_json = serde_json::to_vec(&filter_data_config).unwrap();
        let filter_data_state = TableBuilder::new()
            .with_name(self.filter_user_session_contexts_by_email_processor_name)
            .with_json(&filter_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Configs for join
        let join_data_config = DataConfig {
            operator: AvailableCandleOperators::JoinInner,
            lhs_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string(),
            lhs_pk: "email".to_string(),
            lhs_fk: "session_context_name".to_string(),
            rhs_name: Some(AvailableSubjects::Mermaid.to_string()),
            rhs_pk: Some("session_context_name".to_string()),
            rhs_fk: Some("session_context_name".to_string()),
            ..Default::default()
        };
        let join_data_config_json = serde_json::to_vec(&join_data_config).unwrap();
        let join_data_state = TableBuilder::new()
            .with_name(self.join_user_session_contexts_with_mermaid_diagrams_processor_name)
            .with_json(&join_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![
            extract_tabular_data_state,
            attachmen_state,
            filter_data_state,
            join_data_state,
            AvailableInterfaceSubjects::UserCsv.to_table(None, None).unwrap(),
            AvailableInterfaceSubjects::AssistantCsv.to_table(None, None).unwrap(),
            self.make_user_table().unwrap(),
            self.make_user_session_context_table().unwrap(),
            AvailableSubjects::UserInbox.to_table(None, None).unwrap(),
            AvailableSubjects::JoinUserInboxSessionContexts.to_table(None, None).unwrap(),
            AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_table(None, None).unwrap(),
            make_example_mermaid_table().unwrap(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{metrics::{ArrowTaskMetricsSet, HashMap}, schemas::{blob::BlobBuilderTraitExt, user::create_user_inbox_batch}, session::{common_traits::{BuildableTrait, MappableTrait}, session_context::{SessionStream, SessionStreamState}, session_context_builder::SessionContextBuilderTrait}, table::{data_format::CsvFormat, table::TableTrait}, task::message::{IPCMessage, MessageBuilderTrait, MessageTrait}};

    use crate::{session_plans::available_interface_subjects::create_message_map, session_traits::agents::SessionContextBuilderAgentsTrait};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_user_agent_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let user_agent_session = UserSession::default();
        let session_ctx = user_agent_session
            .build()
            .with_metrics(metrics.clone())
            .with_name(user_agent_session.session_context_name)
            .build_with_tables()?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // Make the tabular data
        let csv_format = CsvFormat::default();
        let batch = create_user_inbox_batch(vec!["contact@biom8er.com".to_string()])?;
        let bytes = Table::get_builder()
            .with_record_batches(vec![batch])?
            .with_name(AvailableInterfaceSubjects::UserCsv.to_string().as_str())
            .build()?
            .to_csv(csv_format.delimiter, csv_format.header)?;

        // Wrap into the message
        let blob = AvailableInterfaceSubjects::UserCsv.to_table_builder(None)
            .with_blob(None, Some("csv"), &bytes, None)?
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
                AvailableInterfaceSubjects::AssistantCsv
            )))
            .filter_map(|m| {
                if m.is_none() {
                    None
                } else {
                    Some(TableBuilder::new_from_ipc_stream(&m.unwrap().get_message_own()).unwrap()
                        .with_name("")
                        .build().unwrap()
                        .to_json_object().unwrap())
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        for row in &attachment_data {
            let bytes = row["bytes"].as_array().unwrap()
                .into_iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect::<Vec<u8>>();
            println!("attachment {}{}: {}", row["filename"].as_str().unwrap(), row["extension"].as_str().unwrap(), String::from_utf8_lossy(bytes.as_ref()).into_owned())
        }

        Ok(())
    }
}
