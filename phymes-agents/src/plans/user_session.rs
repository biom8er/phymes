use anyhow::Result;
use std::sync::Arc;

use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableSubscribeEvents, BuildableTrait, BuilderTrait, ProcessorPlan, ProcessorPlanBuilder, Publication, RuntimeEnv, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait, Subscription, create_user_batch, create_user_session_contexts_batch
};
use phymes_data::{AvailableCandleOperators, DataConfig, DataJoinOperator};
use phymes_diagnostics::create_timestamp_micros;

use crate::{
    AvailableProcessors, AvailableSessionPlans, CustomAgentsBuilderTrait,
    make_example_mermaid_table, TaskPlan,
};

/// A session for all user management tasks
///
/// # Notes
///
/// Supported tasks include the following:
///
/// 1. Filtering the user information by email
/// 2. Joining the user sessions with their mermaid diagrams
/// 3. Registering new users
pub struct UserSession<'a> {
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
            session_context_name: "user_session",
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

    pub fn make_user_table(&self) -> Result<Subject> {
        let batch = create_user_batch(
            vec!["contact@biom8er.com".to_string()],
            vec!["con".to_string()],
            vec!["tact".to_string()],
            vec!["$2b$12$qJGwWR2rSZ9oBFZff0o2w.RXViv.Mf.BwfsWZTfVm4DmjjVfsaHzi".to_string()],
            vec![create_timestamp_micros()],
        )?;
        SubjectBuilder::new()
            .with_name(AvailableSubjects::User.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    pub fn make_user_session_context_table(&self) -> Result<Subject> {
        let mut email = Vec::new();
        let mut session_context_name = Vec::new();
        for name in AvailableSessionPlans::get_all_session_plan_names() {
            email.push("contact@biom8er.com".to_string());
            session_context_name.push(name);
        }
        let batch = create_user_session_contexts_batch(email, session_context_name)?;
        SubjectBuilder::new()
            .with_name(AvailableSubjects::UserSessionContexts.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }
}

impl CustomAgentsBuilderTrait for UserSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![
            TaskPlan {
                task_name: self.filter_session_contexts_by_email_task_name.to_string(),
                processor_names: vec![
                    self.filter_session_contexts_by_email_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self
                    .join_session_contexts_with_mermaid_diagrams_task_name
                    .to_string(),
                processor_names: vec![
                    self.join_session_contexts_with_mermaid_diagrams_processor_name
                        .to_string(),
                ],
            },
            TaskPlan {
                task_name: self.filter_user_info_by_email_task_name.to_string(),
                processor_names: vec![self.filter_user_info_by_email_processor_name.to_string()],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Join
                        .build_arc(self.filter_session_contexts_by_email_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self
                            .filter_session_contexts_by_email_processor_name
                            .to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::UserInbox.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableSubjects::UserSessionContexts.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Join
                        .build_arc(self.join_session_contexts_with_mermaid_diagrams_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self
                            .join_session_contexts_with_mermaid_diagrams_processor_name
                            .to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::JoinUserInboxSessionContexts.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableSubjects::BuilderMermaid.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Join
                        .build_arc(self.filter_user_info_by_email_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.filter_user_info_by_email_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.filter_user_info_by_email_processor_name.to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::UserInbox.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableSubjects::User.to_string(),
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
        // Configs for filter
        let filter_user_info_data_config = DataConfig {
            operator: AvailableCandleOperators::Join,
            lhs_name: Some(AvailableSubjects::UserInbox.to_string()),
            lhs_pk: Some("email".to_string()),
            lhs_fk: Some("email".to_string()),
            rhs_name: Some(AvailableSubjects::User.to_string()),
            rhs_pk: Some("email".to_string()),
            rhs_fk: Some("email".to_string()),
            join_operators: Some(DataJoinOperator::Inner),
            ..Default::default()
        };
        let filter_user_info_data_config_json =
            serde_json::to_vec(&filter_user_info_data_config).unwrap();
        let filter_user_info_data_state = SubjectBuilder::new()
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
            join_operators: Some(DataJoinOperator::Inner),
            ..Default::default()
        };
        let filter_user_session_context_data_config_json =
            serde_json::to_vec(&filter_user_session_context_data_config).unwrap();
        let filter_user_session_context_data_state = SubjectBuilder::new()
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
            join_operators: Some(DataJoinOperator::Inner),
            ..Default::default()
        };
        let join_user_session_context_data_config_json =
            serde_json::to_vec(&join_user_session_context_data_config).unwrap();
        let join_user_session_context_data_state = SubjectBuilder::new()
            .with_name(self.join_session_contexts_with_mermaid_diagrams_processor_name)
            .with_json(&join_user_session_context_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        let subjects = vec![
            filter_user_info_data_state,
            filter_user_session_context_data_state,
            join_user_session_context_data_state,
            self.make_user_table().unwrap(),
            self.make_user_session_context_table().unwrap(),
            AvailableSubjects::User
                .to_subject(Some(self.filter_user_info_by_email_table_name), None)
                .unwrap(),
            AvailableSubjects::UserInbox.to_subject(None, None).unwrap(),
            AvailableSubjects::JoinUserInboxSessionContexts
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::JoinUserInboxSessionContextsMermaid
                .to_subject(None, None)
                .unwrap(),
            make_example_mermaid_table(false, true).unwrap(),
        ];
        let subject_plans = subjects.into_iter().map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap()).collect::<Vec<_>>();
        Some(subject_plans)
    }
}

#[allow(dead_code)]
pub(crate) mod user_session_inner {
    use anyhow::Result;
    use phymes_core::{
        BuildableTrait, IPCMessage, MappableTrait, MessageBuilderTrait, SubjectTrait,
        create_user_inbox_batch,
    };

    use crate::{
        SessionContext, SessionContextBuilderAgentsTrait, SessionContextBuilderTrait,
        SessionStream, create_message_map,
    };

    use super::*;

    pub async fn user_session() -> Result<(Arc<SessionContext>, SessionStream)> {
        // initialize the session
        let user_agent_session = UserSession::default();
        let (session_ctx, session_messages) = user_agent_session
            .build()
            .with_name(user_agent_session.session_context_name)
            .with_diagnostics(true)
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Make the user inbox message
        let batch = create_user_inbox_batch(vec!["contact@biom8er.com".to_string()])?;
        let table = Subject::get_builder()
            .with_record_batches(vec![batch])?
            .with_name(AvailableSubjects::UserInbox.to_string().as_str())
            .build()?;
        let message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&Publication::Replace {
                subject_name: table.get_name().to_string(),
            })
            .with_publisher(user_agent_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![message]);
        let _ = session_ctx_arc.update_subjects_from_messages(session_messages.unwrap_or_default(), 0).await;

        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));

        Ok((session_ctx_arc, session_stream))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{IPCMessage, MappableTrait, SubjectTrait};
    use phymes_diagnostics::HashMap;

    use crate::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_user_session() -> Result<()> {
        let (session_ctx_arc, session_stream) = user_session_inner::user_session().await?;
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;
        assert!(response.is_empty());

        // Check the User subject
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::User.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::User.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("email");
        assert_eq!(column, ["contact@biom8er.com"]);
        let column = subject.get_column_as_vec_str("first_name");
        assert_eq!(column, ["con"]);
        let column = subject.get_column_as_vec_str("last_name");
        assert_eq!(column, ["tact"]);
        let column = subject.get_column_as_vec_str("password_hash");
        assert_eq!(column.len(), 1);
        let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
        for c in column {
            assert!(c > 0);
        }
        
        // Check the Join subject
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("email");
        assert_eq!(
            column,
            [
                "contact@biom8er.com",
                "contact@biom8er.com",
                "contact@biom8er.com",
                "contact@biom8er.com"
            ]
        );
        let column = subject.get_column_as_vec_str("session_context_name");
        assert_eq!(column, ["Builder", "Chat", "DocChat", "ToolChat"]);
        let column = subject.get_column_as_vec_str("er_diagram");
        for c in column {
            assert!(!c.is_empty());
        }
        let column = subject.get_column_as_vec_str("flowchart_diagram");
        for c in column {
            assert!(!c.is_empty());
        }
        let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
        for c in column {
            assert!(c > 0);
        }

        Ok(())
    }
}
