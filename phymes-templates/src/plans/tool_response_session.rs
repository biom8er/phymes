use crate::plans::tool_call_session::ToolSessionTrait;

/// A session for dynamic tool response summarization
///
/// # Note
/// - Specifying the schema for each subject is not needed because
///   `extend`ing with this session will skip duplicate subjects
///   that are already defined in the source session
/// - Any limits to the row counts should be taken care of prior
pub struct ToolResponseSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// Subjects to listen for
    pub subject_names: &'a [&'a str],
}

impl Default for ToolResponseSession<'_> {
    fn default() -> Self {
        ToolResponseSession {
            session_context_name: "tool_response_session",
            subject_names: &["Bytes"],
        }
    }
}

impl<'a> ToolSessionTrait<'a> for ToolResponseSession<'a> {
    fn subject_names(&self) -> Vec<String> {
        self.subject_names.iter().map(|s| s.to_string()).collect()
    }
}

impl<'a> ToolResponseSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let subgraphs = self
            .subject_names()
            .into_iter()
            .map(|subject_name| {
                let processor = format!("tool_response_{subject_name}_processor");
                let p = format!("{processor}_p");
                format!(
                    r#"
    subgraph {processor}_t
		{}
		{processor}_p-subscribe-->{processor}_p-processor
		{processor}_p-processor-->{processor}_p-publish
		{processor}_p-publish-->|Extend|ToolMessages-subject
	end
	ToolResponseSession_runtime_env-rt-->{processor}_t
    {}
	{processor}_p-processor@{{shape: rect, label: PackTabular}}
	{processor}_p-publish@{{shape: fork}}
	{processor}_p-subscribe@{{shape: diamond, label: Any}}"#,
                    self.flowchart_subject_subscriptions_1(&[&subject_name], &p, "LastRecordBatch"),
                    self.flowchart_subject_subscriptions_2(&[&subject_name])
                )
            })
            .collect::<Vec<_>>();
        [r#"flowchart TD
    ToolResponseSession_runtime_env-rt@{shape: subproc, label: ToolResponseSession_runtime_env}
	ToolMessages-subject@{shape: doc, label: ToolMessages}"#
            .to_string()]
        .into_iter()
        .chain(subgraphs)
        .collect::<Vec<_>>()
        .join("")
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        let subgraphs = self
            .subject_names()
            .into_iter()
            .map(|subject_name| {
                let processor = format!("tool_response_{subject_name}_processor");
                let p = format!("{processor}_p");
                format!(
                    r#"
    {}
    {p}["{p}"] {{
        Utf8 operator "PackTabular"
        Utf8 encoding "None"
        Utf8 format "None"
        Utf8 schema "Messages"
        Boolean cpu "false"
        Utf8 lhs_name "{subject_name}"
        Utf8 doc_name "{subject_name}"
        Utf8 lhs_stream "Accumulate"
    }}"#,
                    self.erdiagram_subject_subscriptions(&[&subject_name])
                )
            })
            .collect::<Vec<_>>();
        [r#"erDiagram
	ToolMessages["ToolMessages"] {
	    Utf8 role
	    Utf8 content
	    Int64 timestamp
	}"#
        .to_string()]
        .into_iter()
        .chain(subgraphs)
        .collect::<Vec<_>>()
        .join("")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait};
    use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, create_bytes_record_batch};
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_task::SubscriptionTrait;
    use phymes_diagnostics::HashMap;
    use phymes_network::{SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_tool_response_session() -> Result<()> {
        // Initialize the session
        let tool_response_session = ToolResponseSession::default();
        dbg!(&tool_response_session.as_mermaid_flowchart());
        dbg!(&tool_response_session.as_mermaid_erdiagram());
        let (session_ctx, session_messages) = SessionContextBuilder::from_mermaid_flowchart(
            &tool_response_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &tool_response_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(tool_response_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_supersteps()?
        .add_next_tasks()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Replace the Bytes to trigger the session
        let message_map = {
            let batch = create_bytes_record_batch(vec!["{}".into()])?;
            let table = AvailableSubjects::Bytes.to_subject(None, Some(vec![batch]))?;
            let session_tasks_message = IPCMessage::get_builder()
                .with_subject(table.get_name())
                .with_update(&Publication::Replace {
                    subject_name: table.get_name().to_string(),
                })
                .with_publisher(tool_response_session.session_context_name)
                .with_message(table.to_ipc_stream()?)
                .make_name()?
                .build()?;
            create_message_map(vec![session_tasks_message])
        };
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableInterfaceSubjects::ToolMessages
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("role");
            assert_eq!(column, ["tool"]);
            let column = subject.get_column_as_vec_str("content");
            assert_eq!(column, ["[{\"bytes\":[123,125]}]"]);
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
        }

        Ok(())
    }
}
