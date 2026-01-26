/// A session for all subject associated tasks
///
/// # Notes
///
/// * Supported tasks include the following:
///
/// 1. Counting the number of rows per subject (i.e., updating the `SubjectNumRows` table)
///    after updates have been made to the `SubjectsChangeLog`
/// 2. Determining what tasks are ready to run for the next super step
/// 3. Retrieving the publications per task and processor that will run for the next super step
/// 4. Updating the `SubjectsChangeLog` cache with the most recent updates and `TasksRunLog` cache with the most recent task runs
///
/// * Caching is implemented to minimize memory and compute
pub struct SubjectsNumRowsSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl Default for SubjectsNumRowsSession<'_> {
    fn default() -> Self {
        SubjectsNumRowsSession {
            session_context_name: "subjects_num_rows_session",
        }
    }
}

impl<'a> SubjectsNumRowsSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        SubjectsNumRowsSession {
            session_context_name,
        }
    }
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    default_runtime_env_name-rt@{shape: subproc, label: default_runtime_env_name}

	subgraph group_by_subject_change_log_delta_t
		SubjectsChangeLog-subject-.->|FullTable|group_by_subject_change_log_delta_p-subscribe
		group_by_subject_change_log_delta_p-subscribe-->group_by_subject_change_log_delta_p-processor
		group_by_subject_change_log_delta_p-processor-->group_by_subject_change_log_delta_p-publish
		group_by_subject_change_log_delta_p-publish-->|Replace|group_by_subject_change_log_delta_t-subject
		group_by_subject_change_log_delta_t-subject-->|FullTable|select_subject_change_log_delta_p-subscribe
		select_subject_change_log_delta_p-subscribe-->select_subject_change_log_delta_p-processor
		select_subject_change_log_delta_p-processor-->select_subject_change_log_delta_p-publish
		select_subject_change_log_delta_p-publish-->|Replace|SubjectsNumRows-subject
	end
	default_runtime_env_name-rt-->group_by_subject_change_log_delta_t
	SubjectsChangeLog-subject@{shape: doc, label: SubjectsChangeLog}
	group_by_subject_change_log_delta_p-subscribe@{shape: diamond, label: All}
	group_by_subject_change_log_delta_p-processor@{shape: rect, label: GroupBy}
	group_by_subject_change_log_delta_p-publish@{shape: fork}
	group_by_subject_change_log_delta_t-subject@{shape: doc, label: group_by_subject_change_log_delta_t}
	select_subject_change_log_delta_p-subscribe@{shape: diamond, label: All}
	select_subject_change_log_delta_p-processor@{shape: rect, label: Select}
	select_subject_change_log_delta_p-publish@{shape: fork}
	SubjectsNumRows-subject@{shape: doc, label: SubjectsNumRows}"#
    }
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    SubjectsChangeLog["SubjectsChangeLog"] {
        Utf8 subject_name
        Utf8 task_name
        Utf8 session_name
        Int64 num_rows_delta
        Int64 timestamp
    }
    group_by_subject_change_log_delta_p["group_by_subject_change_log_delta_p"] {
        List-Utf8 agg_columns "['num_rows_delta']"
        List-Utf8 agg_operators "['Sum']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsChangeLog"
        List-Utf8 lhs_values "['subject_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_subject_change_log_delta_p["select_subject_change_log_delta_p"] {
        List-Utf8 as_columns "['','num_rows']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_subject_change_log_delta_t"
        List-Utf8 lhs_values "['subject_name','num_rows_delta-Sum']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SubjectsNumRows["SubjectsNumRows"] {
        Utf8 subject_name
        Int64 num_rows
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, MessageBuilderTrait, TablePublication, TableTrait, test_task};
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, create_message_map, test_session_context_builder,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_subjects_session() -> Result<()> {
        // Initialize the session
        let subjects_session = SubjectsNumRowsSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            subjects_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(subjects_session.as_mermaid_erdiagram(), false, true)?
        .with_name(subjects_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test session data
        let message_map = {
            // Make the test sequential session
            let session_context = test_session_context_builder::make_test_session_context_builder_sequential("session_1", 2)?
                .with_diagnostics(false)
                .add_session_interface(Some(&["state_1"]))?
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;

            // Mimic a session run for 1 steps
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
            let session_context_arc = Arc::new(RwLock::new(session_context));
            let session_stream = SessionStream::new(messages, Arc::clone(&session_context_arc));
            let _response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

            // Extract out the subjects for the test
            let session_ctx_reading = session_context_arc.read();
            let table = session_ctx_reading
                .get_states()
                .get(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .unwrap()
                .read();
            dbg!(&table);
            let subjects_change_log_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_update(&TablePublication::Extend {
                    table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                })
                .with_publisher(subjects_session.session_context_name)
                .make_name()?
                .build()?;
            create_message_map(vec![
                subjects_change_log_message,
            ])
        };

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test session stream
            // {
            //     // Debug any errors
            //     let subjects_reading = session_ctx_arc.read();
            //     let table_reading = subjects_reading
            //         .get_states()
            //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
            //         .unwrap()
            //         .read();
            //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            // }

            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableSubjects::SubjectsNumRows.to_string().as_str())
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("subject_name");
            assert_eq!(column, ["SessionTasksRunLog", "SubjectsChangeLog", "SubjectsNumRows", "group_by_subject_change_log_delta_p", "group_by_subject_change_log_delta_t", "processor_1", "processor_2", "processor_3", "select_subject_change_log_delta_p", "state_1"]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("num_rows")?;
            assert_eq!(column, [4, 12, 0, 0, 0, 0, 0, 0, 0, 72]);
        }

        Ok(())
    }
}
