use anyhow::Result;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, IPCMessageMap,
    MessageBuilderTrait, Table, TableBuilderTrait, TablePublication, TableTrait,
    create_session_tasks_subscribe_publish_batch,
};

use crate::create_message_map;

/// A session for determining the next superstep task publications and subscriptions
pub struct NextSuperstepSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl Default for NextSuperstepSession<'_> {
    fn default() -> Self {
        NextSuperstepSession {
            session_context_name: "next_superstep_session",
        }
    }
}

impl<'a> NextSuperstepSession<'a> {
    /// Return the pre-compiled task subscriptions and publications as messages
    ///
    /// # Notes
    /// * Messages 1, 2, and 4 trigger SuperSteps
    /// * Message 3 is empty and is meant to trigger `tasks_subscribe` method of [SessionContext]
    ///
    /// [SessionContext]: crate::SessionContext
    pub fn as_task_messages(&self) -> Result<Vec<IPCMessageMap>> {
        // 1. Message to trigger the first superstep
        let task_names = vec!["max_superstep_t"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["group_by_session_superstep_p"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![vec!["OnUpdateFullTable", "AlwaysFullTable"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let subscription_table_names =
            vec![vec!["SessionSupersteps", "group_by_session_superstep_p"]]
                .into_iter()
                .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["SessionSuperstepMax"]]
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

        Ok(vec![messages_1])
    }

    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    NextSuperstepSession_runtime_env-rt@{shape: subproc, label: NextSuperstepSession_runtime_env}

	subgraph max_superstep_t
		SessionSupersteps-subject-.->|FullTable|group_by_session_superstep_p-subscribe
		group_by_session_superstep_p-subscribe-->group_by_session_superstep_p-processor
		group_by_session_superstep_p-processor-->group_by_session_superstep_p-publish
		group_by_session_superstep_p-publish-->|Replace|SessionSuperstepMax-subject
	end
	NextSuperstepSession_runtime_env-rt-->max_superstep_t
	SessionSupersteps-subject@{shape: doc, label: SessionSupersteps}
	group_by_session_superstep_p-subscribe@{shape: diamond, label: All}
	group_by_session_superstep_p-processor@{shape: rect, label: GroupBy}
	group_by_session_superstep_p-publish@{shape: fork}
	SessionSuperstepMax-subject@{shape: doc, label: SessionSuperstepMax}"#
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    SessionSupersteps["SessionSupersteps"] {
        Utf8 session_name
        UInt32 superstep
    }
    group_by_session_superstep_p["group_by_session_superstep_p"] {
        List-Utf8 agg_columns "['superstep']"
        List-Utf8 agg_operators "['Max']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionSupersteps"
        List-Utf8 lhs_values "['session_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionSuperstepMax["SessionSuperstepMax"] {
        Utf8 session_name
        UInt32 superstep-Max
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
        TablePublication, TableTrait, create_session_subjects_batch,
        create_session_supersteps_batch, test_task,
    };
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, SessionStreamStep, SessionStreamStepTrait,
        create_message_map, test_session_context_builder,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_next_superstep_session() -> Result<()> {
        // Initialize the session
        let next_superstep_session = NextSuperstepSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            next_superstep_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            next_superstep_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(next_superstep_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test session data
        let session_names = ["session_1", "session_1", "session_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let supersteps = vec![0, 1, 2, 3];
        let batch = create_session_supersteps_batch(session_names, supersteps)?;
        let table = Table::get_builder()
            .with_name(AvailableSubjects::SessionSupersteps.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let superstep_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableSubjects::SessionSupersteps.to_string().as_str())
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionSupersteps.to_string(),
            })
            .with_publisher(next_superstep_session.session_context_name)
            .make_name()?
            .build()?;
        let mut message_map = create_message_map(vec![superstep_message]);

        // Session Tasks
        let mut next_superstep_messages = next_superstep_session
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        // Run the session
        message_map.extend(next_superstep_messages.pop().unwrap());
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 1
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableSubjects::SessionSuperstepMax.to_string().as_str())
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["session_1"]);
            let column = table_reading.get_column_as_vec_primitive::<u32>("superstep-Max")?;
            assert_eq!(column, [3]);
        }

        Ok(())
    }
}
