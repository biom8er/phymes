/// Count the number of rows for each subject
pub struct CountSubjectRowsNetwork<'a> {
    /// Network
    pub network_name: &'a str,
}

impl Default for CountSubjectRowsNetwork<'_> {
    fn default() -> Self {
        CountSubjectRowsNetwork {
            network_name: "count_subject_rows_network",
        }
    }
}

impl<'a> CountSubjectRowsNetwork<'a> {
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    CountSubjectRowsNetwork_runtime_env-rt@{shape: subproc, label: CountSubjectRowsNetwork_runtime_env}

	subgraph group_by_subject_change_log_delta_t
		SubjectsChangeLog-subject-.->|AllRecordBatches|group_by_subject_change_log_delta_p-subscribe
		group_by_subject_change_log_delta_p-subscribe-->group_by_subject_change_log_delta_p-processor
		group_by_subject_change_log_delta_p-processor-->group_by_subject_change_log_delta_p-publish
		group_by_subject_change_log_delta_p-publish-->|Replace|group_by_subject_change_log_delta_t-subject
		group_by_subject_change_log_delta_t-subject-->|AllRecordBatches|select_subject_change_log_delta_p-subscribe
		select_subject_change_log_delta_p-subscribe-->select_subject_change_log_delta_p-processor
		select_subject_change_log_delta_p-processor-->select_subject_change_log_delta_p-publish
		select_subject_change_log_delta_p-publish-->|Replace|SubjectsNumRows-subject
	end
	CountSubjectRowsNetwork_runtime_env-rt-->group_by_subject_change_log_delta_t
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
        Utf8 network_name
        Int64 num_rows
        Int64 superstep
    }
    group_by_subject_change_log_delta_p["group_by_subject_change_log_delta_p"] {
        List-Utf8 agg_columns "['num_rows']"
        List-Utf8 agg_operators "['Sum']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsChangeLog"
        List-Utf8 lhs_values "['subject_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    select_subject_change_log_delta_p["select_subject_change_log_delta_p"] {
        List-Utf8 as_columns "['','num_rows']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_subject_change_log_delta_t"
        List-Utf8 lhs_values "['subject_name','num_rows-Sum']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
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
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_schemas::AvailableSubjects;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::{SubscriptionTrait, extended_diagnostic_subjects, test_task, write_diagnostic_subjects_to_csv};

    use crate::{
        NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait,
        NetworkStream, test_network_builder,
    };

    use super::*;

    #[tokio::test]
    async fn test_count_subject_rows_network() -> Result<()> {
        // Initialize the network
        let subjects_network = CountSubjectRowsNetwork::default();
        let (network, network_messages) =
            NetworkBuilder::from_mermaid_flowchart(subjects_network.as_mermaid_flowchart(), false)?
                .with_subjects_from_mermaid_erdiagram(
                    subjects_network.as_mermaid_erdiagram(),
                    false,
                    true,
                )?
                .with_name(subjects_network.network_name)
                .with_diagnostics(true)
                .add_processor_subjects()?
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test network data
        let message_map = {
            // Make the test sequential network
            let (network, network_messages) =
                test_network_builder::make_test_network_builder_sequential("network_1", 2)?
                    .with_diagnostics(false)
                    .add_network_interface(Some(&["state_1"]))?
                    .add_next_tasks()?
                    .add_next_supersteps()?
                    .build_with_tables()?;

            // Mimic a network run for 1 steps
            let network_arc = Arc::new(network);
            let _ = network_arc
                .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
                .await;
            let messages = test_task::make_test_input_message(
                "task_1",
                "network_1",
                "state_1",
                "state_1",
                &Publication::Replace {
                    subject_name: "state_1".to_string(),
                },
                true,
            )?;
            let network_stream = NetworkStream::new(messages, Arc::clone(&network_arc));
            let _response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            // Extract out the subjects for the test
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_record_batches(batches)
                .unwrap()
                .build()
                .unwrap();
            let subjects_change_log_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_update(&Publication::Extend {
                    subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                })
                .with_publisher(subjects_network.network_name)
                .make_name()?
                .build()?;
            create_message_map(vec![subjects_change_log_message])
        };
        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;

        // Run the network
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        let extended_diagnostic_subjects = extended_diagnostic_subjects();
        let subject_names = extended_diagnostic_subjects
            .iter()
            .map(|s| s.as_str())
            .chain([
                "SubjectsNumRows",
                "state_1",
            ])
            .collect::<Vec<_>>();
        write_diagnostic_subjects_to_csv(
            &subject_names,
            network_arc.runtime_env(),
            network_arc.get_name(),
        )
        .await?;

        assert_eq!(response.len(), 0);

        // Test network stream
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();
        let column = subject.get_column_as_vec_str("subject_name");
        assert_eq!(
            column,
            [
                "NetworkTasksRunLog",
                "SubjectsChangeLog",
                "SubjectsNumRows",
                "group_by_subject_change_log_delta_p",
                "group_by_subject_change_log_delta_t",
                "processor_1",
                "processor_2",
                "processor_3",
                "select_subject_change_log_delta_p",
                "state_1"
            ]
        );
        let column = subject.get_column_as_vec_primitive::<i64>("num_rows")?;
        assert_eq!(column, [3, 9, 0, 0, 0, 0, 0, 0, 0, 33]);

        Ok(())
    }
}
