use anyhow::Result;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, IPCMessageMap,
    MessageBuilderTrait, Table, TableBuilderTrait, TablePublication, TableTrait,
    create_session_tasks_subscribe_publish_batch,
};
use phymes_diagnostics::HashMap;

use crate::create_message_map;

/// A session for determining the next superstep task publications and subscriptions
pub struct ViewTaskSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl Default for ViewTaskSession<'_> {
    fn default() -> Self {
        ViewTaskSession {
            session_context_name: "view_task_session",
        }
    }
}

impl<'a> ViewTaskSession<'a> {
    /// Return the pre-compiled task subscriptions and publications as messages
    ///
    /// # Notes
    /// * Messages 1, 2, and 4 trigger SuperSteps
    /// * Message 3 is empty and is meant to trigger `tasks_subscribe` method of [SessionContext]
    ///
    /// [SessionContext]: crate::SessionContext
    pub fn as_task_messages(&self) -> Result<Vec<IPCMessageMap>> {
        // 1. Message to trigger the first superstep
        let task_names = vec![
            "group_by_tasks_run_log_timestamp_t",
            "group_by_tasks_run_log_timestamp_t",
            "filter_processors_subscriptions_t",
            "filter_processors_subscriptions_t",
            "filter_processors_subscriptions_t",
            "filter_processors_publications_t",
            "filter_processors_publications_t",
            "filter_processors_publications_t",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_names = vec![
            "group_by_tasks_run_log_timestamp_p",
            "select_tasks_run_log_timestamp_p",
            "cmp_processors_subscriptions_p",
            "filter_processors_subscriptions_p",
            "select_processors_subscriptions_p",
            "select_processors_p",
            "filter_processors_publications_p",
            "select_tasks_processors_p",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_types = vec![
            "GroupBy", "Select", "Select", "Filter", "Select", "Select", "Filter", "Select",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["OnUpdateFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["OnUpdateFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["OnUpdateFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["SessionTasksRunLog", "group_by_tasks_run_log_timestamp_p"],
            vec![
                "group_by_tasks_run_log_timestamp_t",
                "select_tasks_run_log_timestamp_p",
            ],
            vec!["SessionProcessors", "cmp_processors_subscriptions_p"],
            vec![
                "cmp_processors_subscriptions_t",
                "filter_processors_subscriptions_p",
            ],
            vec![
                "filter_processors_subscriptions_t",
                "select_processors_subscriptions_p",
            ],
            vec!["SessionProcessors", "select_processors_p"],
            vec![
                "select_processors_s",
                "filter_processors_publications_p",
            ],
            vec![
                "filter_processors_publications_t",
                "select_tasks_processors_p",
            ],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_table_names = vec![
            vec!["group_by_tasks_run_log_timestamp_t"],
            vec!["select_tasks_run_log_timestamp_t"],
            vec!["cmp_processors_subscriptions_t"],
            vec!["filter_processors_subscriptions_t"],
            vec!["select_processors_subscriptions_t"],
            vec!["select_processors_s"],
            vec!["filter_processors_publications_t"],
            vec!["select_tasks_processors_s"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| self.session_context_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Table::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.session_context_name)
            .make_name()?
            .build()?;
        let messages_1 = create_message_map(vec![tasks_publish_subscribe_message]);

        // 2. Message to trigger the second superstep
        let task_names = vec![
            "join_tasks_processors_s",
            "join_tasks_processors_s",
            "join_tasks_processors_s",
            "join_tasks_processors_s",
            "join_tasks_processors_s",
            "join_tasks_processors_s",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_names = vec![
            "group_by_processors_p",
            "join_tasks_processors_p",
            "join_tasks_processors_subscriptions_p",
            "join_tasks_processors_subscriptions_subjects_p",
            "select_tasks_processors_subscriptions_subjects_p",
            "group_by_tasks_processors_subscriptions_p",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Join", "Join", "Join", "Select", "GroupBy"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["OnUpdateFullTable", "AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec![
                "SubjectsChangeLog",
                "group_by_processors_p",
            ],
            vec![
                "select_tasks_run_log_timestamp_t",
                "SessionTasks",
                "join_tasks_processors_p",
            ],
            vec![
                "join_tasks_processors_s",
                "select_processors_subscriptions_t",
                "join_tasks_processors_subscriptions_p",
            ],
            vec![
                "join_tasks_processors_subscriptions_t",
                "group_by_processors_s",
                "join_tasks_processors_subscriptions_subjects_p",
            ],
            vec![
                "join_tasks_processors_subscriptions_subjects_t",
                "select_tasks_processors_subscriptions_subjects_p",
            ],
            vec![
                "select_tasks_processors_subscriptions_subjects_t",
                "group_by_tasks_processors_subscriptions_p",
            ],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_table_names = vec![
            vec!["group_by_processors_s"],
            vec!["join_tasks_processors_s"],
            vec!["join_tasks_processors_subscriptions_t"],
            vec!["join_tasks_processors_subscriptions_subjects_t"],
            vec!["select_tasks_processors_subscriptions_subjects_t"],
            vec!["SessionTasksSubscribeAggregate"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| self.session_context_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Table::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.session_context_name)
            .make_name()?
            .build()?;
        let messages_2 = create_message_map(vec![tasks_publish_subscribe_message]);

        // Calculate the tasks subscribe
        let messages_none = HashMap::<String, IPCMessage>::new();

        // 3. Message to trigger the third superstep
        let task_names = vec![
            "select_tasks_processors_publications_t",
            "select_tasks_processors_publications_t",
            "select_tasks_processors_publications_t",
            "select_tasks_processors_publications_t",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_names = vec![
            "group_by_tasks_processors_subscriptions_subjects_p",
            "group_by_tasks_processors_publications_p",
            "join_tasks_processors_publications_p",
            "select_tasks_processors_publications_p",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "GroupBy", "Join", "Select"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["OnUpdateFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable", "AlwaysFullTable"],
            vec!["AlwaysFullTable", "AlwaysFullTable"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec![
                "SessionTasksSubscribe",
                "group_by_tasks_processors_subscriptions_subjects_p",
            ],
            vec![
                "select_tasks_processors_s",
                "group_by_tasks_processors_publications_p",
            ],
            vec![
                "group_by_tasks_processors_subscriptions_subjects_t",
                "group_by_tasks_processors_publications_t",
                "join_tasks_processors_publications_p",
            ],
            vec![
                "join_tasks_processors_publications_t",
                "select_tasks_processors_publications_p",
            ],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
            vec!["Replace"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_table_names = vec![
            vec!["group_by_tasks_processors_subscriptions_subjects_t"],
            vec!["group_by_tasks_processors_publications_t"],
            vec!["join_tasks_processors_publications_t"],
            vec!["SessionTasksSubscribePublish"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| self.session_context_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Table::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.session_context_name)
            .make_name()?
            .build()?;
        let messages_3 = create_message_map(vec![tasks_publish_subscribe_message]);

        Ok(vec![messages_1, messages_2, messages_none, messages_3])
    }

    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    ViewTaskSession_runtime_env-rt@{shape: subproc, label: ViewTaskSession_runtime_env}

	subgraph join_tasks_processors_t
		SessionProcessors-subject-.->|FullTable|group_by_processors_p-subscribe
		group_by_processors_p-subscribe-->group_by_processors_p-processor
		group_by_processors_p-processor-->group_by_processors_p-publish
		group_by_processors_p-publish-->|Replace|group_by_processors_s-subject
		group_by_processors_s-subject-->|FullTable|select_processors_p-subscribe
		select_processors_p-subscribe-->select_processors_p-processor
		select_processors_p-processor-->select_processors_p-publish
		select_processors_p-publish-->|Replace|select_processors_s-subject
		group_by_processors_s-subject-->|FullTable|join_tasks_processors_p-subscribe
		SessionTasks-subject-.->|FullTable|join_tasks_processors_p-subscribe
		join_tasks_processors_p-subscribe-->join_tasks_processors_p-processor
		join_tasks_processors_p-processor-->join_tasks_processors_p-publish
		join_tasks_processors_p-publish-->|Replace|join_tasks_processors_s-subject
		join_tasks_processors_s-subject-->|FullTable|select_tasks_processors_p-subscribe
		select_tasks_processors_p-subscribe-->select_tasks_processors_p-processor
		select_tasks_processors_p-processor-->select_tasks_processors_p-publish
		select_tasks_processors_p-publish-->|Replace|select_tasks_processors_s-subject
	end
	ViewTaskSession_runtime_env-rt-->join_tasks_processors_s
	SessionProcessors-subject@{shape: doc, label: SessionProcessors}
	group_by_processors_p-subscribe@{shape: diamond, label: All}
	group_by_processors_p-processor@{shape: rect, label: GroupBy}
	group_by_processors_p-publish@{shape: fork}
	group_by_processors_s-subject@{shape: doc, label: group_by_processors_s}
	select_processors_p-subscribe@{shape: diamond, label: All}
	select_processors_p-processor@{shape: rect, label: Select}
	select_processors_p-publish@{shape: fork}
	select_processors_s-subject@{shape: doc, label: select_processors_s}
	SessionTasks-subject@{shape: doc, label: SessionTasks}
	join_tasks_processors_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_p-processor@{shape: rect, label: Join}
	join_tasks_processors_p-publish@{shape: fork}
	join_tasks_processors_s-subject@{shape: doc, label: join_tasks_processors_s}
	select_tasks_processors_p-subscribe@{shape: diamond, label: All}
	select_tasks_processors_p-processor@{shape: rect, label: Select}
	select_tasks_processors_p-publish@{shape: fork}
	select_tasks_processors_s-subject@{shape: doc, label: select_tasks_processors_s}"#
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    SessionProcessors["SessionProcessors"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }
    group_by_processors_p["group_by_processors_p"] {
        List-Utf8 agg_columns "['timestamp']"
        List-Utf8 agg_operators "['Max']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsChangeLog"
        List-Utf8 lhs_values "['subject_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_p["select_processors_p"] {
        List-Utf8 as_columns "['','','','','','','','','publication']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Zeros']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription','publication']"
        Utf8 operator "Select"
    }
    SessionTasks["SessionTasks"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 runtime_env_name
    }
    join_tasks_processors_p["join_tasks_processors_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "task_name"
        Utf8 lhs_name "select_tasks_run_log_timestamp_t"
        Utf8 lhs_pk "task_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "task_name"
        Utf8 rhs_name "SessionTasks"
        Utf8 rhs_pk "task_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_p["select_tasks_processors_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_publications_t"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_s["select_tasks_processors_s"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, MessageBuilderTrait,
        TablePublication, TableTrait, test_task,
    };
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, SessionStreamStep, SessionStreamStepTrait,
        create_message_map, test_session_context_builder,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_view_task_session() -> Result<()> {
        // Initialize the session
        let view_task_session = ViewTaskSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            view_task_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(view_task_session.as_mermaid_erdiagram(), false, true)?
        .with_name(view_task_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_supersteps()?
        .with_max_iter(1) // DM: prevent continued execution after the final superstep for testing
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test session data
        let mut message_map = {
            // Make the test sequential session
            let session_context =
                test_session_context_builder::make_test_session_context_builder_sequential(
                    "session_1",
                    4,
                )?
                .with_diagnostics(false)
                .add_session_interface(Some(&["state_1"]))?
                .add_next_tasks()? // DM required for 'SessionTasksSubscribePublish' table
                .add_next_supersteps()?
                .build_with_tables()?;
            let session_context_arc = Arc::new(RwLock::new(session_context));

            // Mimic a superstep update without running the superstep
            let messages = test_task::make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::Replace {
                    table_name: "state_1".to_string(),
                },
                true,
            )?;
            let _step = SessionStreamStep::current_superstep(&session_context_arc).await;
            SessionStreamStep::update_subjects_and_changelog_from_messages(
                &session_context_arc,
                messages,
            )?;

            // Extract out the subjects for the test
            let session_ctx_reading = session_context_arc.read();
            let table = session_ctx_reading
                .get_states()
                .get(AvailableSubjects::SessionProcessors.to_string().as_str())
                .unwrap()
                .read();
            let session_processor_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionProcessors.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SessionProcessors.to_string(),
                })
                .with_publisher(view_task_session.session_context_name)
                .make_name()?
                .build()?;
            let table = session_ctx_reading
                .get_states()
                .get(AvailableSubjects::SessionTasks.to_string().as_str())
                .unwrap()
                .read();
            let session_tasks_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionTasks.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SessionTasks.to_string(),
                })
                .with_publisher(view_task_session.session_context_name)
                .make_name()?
                .build()?;
            let table = session_ctx_reading
                .get_states()
                .get(AvailableSubjects::SessionTasksRunLog.to_string().as_str())
                .unwrap()
                .read();
            let session_tasks_run_log_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionTasksRunLog.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SessionTasksRunLog.to_string(),
                })
                .with_publisher(view_task_session.session_context_name)
                .make_name()?
                .build()?;
            let table = session_ctx_reading
                .get_states()
                .get(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .unwrap()
                .read();
            let subjects_change_log_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                })
                .with_publisher(view_task_session.session_context_name)
                .make_name()?
                .build()?;
            create_message_map(vec![
                session_processor_message,
                session_tasks_message,
                session_tasks_run_log_message,
                subjects_change_log_message,
            ])
        };

        let mut tasks_publish_subscribe_messages = view_task_session
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        // Run the session
        message_map.extend(tasks_publish_subscribe_messages.pop().unwrap());
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 1
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get("select_tasks_run_log_timestamp_t")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("task_name");
            assert_eq!(column, ["session_1", "task_1"]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            for timestamp in column {
                assert_eq!(timestamp, 0);
            }

            let table_reading = session_reading
                .get_states()
                .get("select_processors_subscriptions_t")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1"
                ]
            );
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "processor_1",
                    "processor_1",
                    "processor_2",
                    "processor_2",
                    "processor_3",
                    "processor_3",
                    "session_1"
                ]
            );
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorEcho"
                ]
            );
            let column = table_reading.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(
                column,
                [
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateLastRecordBatch"
                ]
            );
            let column = table_reading.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(
                column,
                [
                    "state_1",
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1"
                ]
            );
            let column = table_reading.get_column_as_vec_str("subscribe_type");
            assert_eq!(column, ["All", "All", "All", "All", "All", "All", "Any"]);
            let column = table_reading.get_column_as_vec_str("update_type");
            assert_eq!(
                column,
                [
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate"
                ]
            );
            let column = table_reading.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [1, 1, 1, 1, 1, 1, 1]);

            let table_reading = session_reading
                .get_states()
                .get("select_tasks_processors_s")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["session_1", "session_1", "session_1", "session_1"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                ["processor_1", "processor_2", "processor_3", "session_1"]
            );
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorEcho"
                ]
            );
            let column = table_reading.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(column, ["Extend", "Extend", "Extend", "Extend"]);
            let column = table_reading.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(column, ["state_1", "state_1", "state_1", "state_1"]);
            let column = table_reading.get_column_as_vec_str("subscribe_type");
            assert_eq!(column, ["All", "All", "All", "Any"]);
            let column = table_reading.get_column_as_vec_str("update_type");
            assert_eq!(
                column,
                [
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate"
                ]
            );
            let column = table_reading.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [0, 0, 0, 0]);
        }

        // Run the session
        let session_stream = SessionStream::new(
            tasks_publish_subscribe_messages.pop().unwrap(),
            Arc::clone(&session_ctx_arc),
        );
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 2
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get("SessionTasksSubscribeAggregate")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["session_1", "session_1", "session_1", "session_1"]);
            let column = table_reading.get_column_as_vec_str("task_name");
            assert_eq!(column, ["task_1", "task_1", "task_1", "session_1"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                ["processor_1", "processor_2", "processor_3", "session_1"]
            );
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorEcho",
                ]
            );
            let column = table_reading
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_name-List")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "OnUpdateLastRecordBatch",
                ]
            );
            let column = table_reading
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_name-List")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1",
                    "state_1",
                ]
            );
            let column = table_reading.get_column_as_vec_str("subscribe_type-Last");
            assert_eq!(column, ["All", "All", "All", "Any"]);
            let column = table_reading.get_column_as_vec_str("update_type-Last");
            assert_eq!(
                column,
                [
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate",
                    "TableChangedSinceLastRunUpdate"
                ]
            );
            let column =
                table_reading.get_column_as_vec_nested_primitive::<i64>("timestamp-List")?;
            for timestamps in column {
                for timestamp in timestamps {
                    assert_eq!(timestamp, 0);
                }
            }
            let column =
                table_reading.get_column_as_vec_nested_primitive::<i64>("timestamp-Max-List")?;
            for timestamps in column {
                for timestamp in timestamps {
                    assert!(timestamp >= 0);
                }
            }
        }

        // 3. Calculate the tasks subscribe
        let _ = tasks_publish_subscribe_messages.pop().unwrap();
        session_ctx_arc.read().tasks_subscribe()?;

        {
            // {
            //     // Debug any errors
            //     let subjects_reading = session_ctx_arc.read();
            //     let table_reading = subjects_reading
            //         .get_states()
            //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
            //         .unwrap()
            //         .read();
            //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            //     let subjects_reading = session_ctx_arc.read();
            //     let table_reading = subjects_reading
            //         .get_states()
            //         .get(AvailableSubjects::SessionSupersteps.to_string().as_str())
            //         .unwrap()
            //         .read();
            //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            //     let table_reading = subjects_reading
            //         .get_states()
            //         .get(AvailableSubjects::SessionSuperstepMax.to_string().as_str())
            //         .unwrap()
            //         .read();
            //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            // }

            // Test the tasks subscribe
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(
                    AvailableSubjects::SessionTasksSubscribe
                        .to_string()
                        .as_str(),
                )
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1",
                    "session_1"
                ]
            );
            let column = table_reading.get_column_as_vec_str("task_name");
            assert_eq!(
                column,
                [
                    "task_1",
                    "task_1",
                    "task_1",
                    "task_1",
                    "task_1",
                    "task_1",
                    "session_1",
                ]
            );
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "processor_1",
                    "processor_1",
                    "processor_2",
                    "processor_2",
                    "processor_3",
                    "processor_3",
                    "session_1",
                ]
            );
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorEcho",
                ]
            );
            let column = table_reading.get_column_as_vec_str("subscription_name");
            assert_eq!(
                column,
                [
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "OnUpdateLastRecordBatch",
                ]
            );
            let column = table_reading.get_column_as_vec_str("subscription_table_name");
            assert_eq!(
                column,
                [
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1",
                    "state_1",
                ]
            );
        }

        // Run the session
        let session_stream = SessionStream::new(
            tasks_publish_subscribe_messages.pop().unwrap(),
            Arc::clone(&session_ctx_arc),
        );
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 3
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(
                    AvailableSubjects::SessionTasksSubscribePublish
                        .to_string()
                        .as_str(),
                )
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["session_1", "session_1", "session_1", "session_1"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                ["processor_1", "processor_2", "processor_3", "session_1"]
            );
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorEcho"
                ]
            );
            let column = table_reading
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "AlwaysFullTable",
                    "OnUpdateFullTable",
                    "OnUpdateLastRecordBatch"
                ]
            );
            let column = table_reading
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1",
                    "state_1"
                ]
            );
            let column = table_reading
                .get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["Extend", "Extend", "Extend", "Extend"]);
            let column = table_reading
                .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["state_1", "state_1", "state_1", "state_1"]);
        }

        Ok(())
    }
}
