use anyhow::Result;
use phymes_diagnostics::HashMap;
use phymes_event::Publication;
use phymes_message::{IPCMessage, IPCMessageMap, MessageBuilderTrait, create_message_map};
use phymes_schemas::{AvailableSubjects, create_network_tasks_subscribe_publish_batch};
use phymes_subject::{BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait};

/// A network for determining the next superstep task publications and subscriptions
pub struct NextTaskNetwork<'a> {
    /// Network
    pub network_name: &'a str,
}

impl Default for NextTaskNetwork<'_> {
    fn default() -> Self {
        NextTaskNetwork {
            network_name: "next_task_network",
        }
    }
}

impl<'a> NextTaskNetwork<'a> {
    /// Return the pre-compiled task subscriptions and publications as messages
    ///
    /// # Notes
    /// * Messages 1, 2, and 4 trigger SuperSteps
    /// * Message 3 is empty and is meant to trigger `tasks_subscribe` method of [Network]
    ///
    /// [Network]: crate::Network
    pub fn as_task_messages(&self) -> Result<Vec<IPCMessageMap>> {
        // 1. Message to trigger the first superstep
        let task_names = vec![
            "group_by_tasks_run_log_superstep_t",
            "group_by_tasks_run_log_superstep_t",
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
            "group_by_tasks_run_log_superstep_p",
            "select_tasks_run_log_superstep_p",
            "cmp_processors_subscriptions_p",
            "filter_processors_subscriptions_p",
            "select_processors_subscriptions_p",
            "cmp_processors_publications_p",
            "filter_processors_publications_p",
            "select_processors_publications_p",
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
            vec!["OnUpdateAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["OnUpdateAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["OnUpdateAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["NetworkTasksRunLog", "group_by_tasks_run_log_superstep_p"],
            vec![
                "group_by_tasks_run_log_superstep_s",
                "select_tasks_run_log_superstep_p",
            ],
            vec!["NetworkProcessors", "cmp_processors_subscriptions_p"],
            vec![
                "cmp_processors_subscriptions_s",
                "filter_processors_subscriptions_p",
            ],
            vec![
                "filter_processors_subscriptions_s",
                "select_processors_subscriptions_p",
            ],
            vec!["NetworkProcessors", "cmp_processors_publications_p"],
            vec![
                "cmp_processors_publications_s",
                "filter_processors_publications_p",
            ],
            vec![
                "filter_processors_publications_s",
                "select_processors_publications_p",
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
            vec!["group_by_tasks_run_log_superstep_s"],
            vec!["select_tasks_run_log_superstep_s"],
            vec!["cmp_processors_subscriptions_s"],
            vec!["filter_processors_subscriptions_s"],
            vec!["select_processors_subscriptions_s"],
            vec!["cmp_processors_publications_s"],
            vec!["filter_processors_publications_s"],
            vec!["select_processors_publications_s"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let network_names = task_names
            .iter()
            .map(|_| self.network_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_network_tasks_subscribe_publish_batch(
            network_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::NetworkTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.network_name)
            .make_name()?
            .build()?;
        let messages_1 = create_message_map(vec![tasks_publish_subscribe_message]);

        // 2. Message to trigger the second superstep
        let task_names = vec![
            "join_tasks_run_log_superstep_t",
            "join_tasks_run_log_superstep_t",
            "join_tasks_run_log_superstep_t",
            "join_tasks_run_log_superstep_t",
            "join_tasks_run_log_superstep_t",
            "join_tasks_run_log_superstep_t",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let processor_names = vec![
            "group_by_subject_change_log_superstep_p",
            "join_tasks_run_log_superstep_p",
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
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec![
                "OnUpdateAllRecordBatches",
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
            ],
            vec![
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
            ],
            vec![
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
            ],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec![
                "SubjectsChangeLog",
                "group_by_subject_change_log_superstep_p",
            ],
            vec![
                "select_tasks_run_log_superstep_s",
                "NetworkTasks",
                "join_tasks_run_log_superstep_p",
            ],
            vec![
                "join_tasks_run_log_superstep_s",
                "select_processors_subscriptions_s",
                "join_tasks_processors_subscriptions_p",
            ],
            vec![
                "join_tasks_processors_subscriptions_s",
                "group_by_subject_change_log_superstep_s",
                "join_tasks_processors_subscriptions_subjects_p",
            ],
            vec![
                "join_tasks_processors_subscriptions_subjects_s",
                "select_tasks_processors_subscriptions_subjects_p",
            ],
            vec![
                "select_tasks_processors_subscriptions_subjects_s",
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
            vec!["group_by_subject_change_log_superstep_s"],
            vec!["join_tasks_run_log_superstep_s"],
            vec!["join_tasks_processors_subscriptions_s"],
            vec!["join_tasks_processors_subscriptions_subjects_s"],
            vec!["select_tasks_processors_subscriptions_subjects_s"],
            vec!["NetworkTasksSubscribeAggregate"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let network_names = task_names
            .iter()
            .map(|_| self.network_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_network_tasks_subscribe_publish_batch(
            network_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::NetworkTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.network_name)
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
            vec!["OnUpdateAllRecordBatches", "AlwaysAllRecordBatches"],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            vec![
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
                "AlwaysAllRecordBatches",
            ],
            vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec![
                "NetworkTasksSubscribe",
                "group_by_tasks_processors_subscriptions_subjects_p",
            ],
            vec![
                "select_processors_publications_s",
                "group_by_tasks_processors_publications_p",
            ],
            vec![
                "group_by_tasks_processors_subscriptions_subjects_s",
                "group_by_tasks_processors_publications_s",
                "join_tasks_processors_publications_p",
            ],
            vec![
                "join_tasks_processors_publications_s",
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
            vec!["group_by_tasks_processors_subscriptions_subjects_s"],
            vec!["group_by_tasks_processors_publications_s"],
            vec!["join_tasks_processors_publications_s"],
            vec!["NetworkTasksSubscribePublish"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let network_names = task_names
            .iter()
            .map(|_| self.network_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_network_tasks_subscribe_publish_batch(
            network_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::NetworkTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.network_name)
            .make_name()?
            .build()?;
        let messages_3 = create_message_map(vec![tasks_publish_subscribe_message]);

        Ok(vec![messages_1, messages_2, messages_none, messages_3])
    }

    /// Return the Mermaid.js flowchart representation of the network
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    NextTaskNetwork_runtime_env-rt@{shape: subproc, label: NextTaskNetwork_runtime_env}

	subgraph group_by_tasks_run_log_superstep_t
		NetworkTasksRunLog-subject-.->|AllRecordBatches|group_by_tasks_run_log_superstep_p-subscribe
		group_by_tasks_run_log_superstep_p-subscribe-->group_by_tasks_run_log_superstep_p-processor
		group_by_tasks_run_log_superstep_p-processor-->group_by_tasks_run_log_superstep_p-publish
		group_by_tasks_run_log_superstep_p-publish-->|Replace|group_by_tasks_run_log_superstep_s-subject
		group_by_tasks_run_log_superstep_s-subject-->|AllRecordBatches|select_tasks_run_log_superstep_p-subscribe
		select_tasks_run_log_superstep_p-subscribe-->select_tasks_run_log_superstep_p-processor
		select_tasks_run_log_superstep_p-processor-->select_tasks_run_log_superstep_p-publish
		select_tasks_run_log_superstep_p-publish-->|Replace|select_tasks_run_log_superstep_s-subject
	end
	NextTaskNetwork_runtime_env-rt-->group_by_tasks_run_log_superstep_t
	NetworkTasksRunLog-subject@{shape: doc, label: NetworkTasksRunLog}
	group_by_tasks_run_log_superstep_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_run_log_superstep_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_run_log_superstep_p-publish@{shape: fork}
	group_by_tasks_run_log_superstep_s-subject@{shape: doc, label: group_by_tasks_run_log_superstep_s}
	select_tasks_run_log_superstep_p-subscribe@{shape: diamond, label: All}
	select_tasks_run_log_superstep_p-processor@{shape: rect, label: Select}
	select_tasks_run_log_superstep_p-publish@{shape: fork}
	select_tasks_run_log_superstep_s-subject@{shape: doc, label: select_tasks_run_log_superstep_s}

	subgraph filter_processors_subscriptions_t
		NetworkProcessors-subject-.->|AllRecordBatches|cmp_processors_subscriptions_p-subscribe
		cmp_processors_subscriptions_p-subscribe-->cmp_processors_subscriptions_p-processor
		cmp_processors_subscriptions_p-processor-->cmp_processors_subscriptions_p-publish
		cmp_processors_subscriptions_p-publish-->|Replace|cmp_processors_subscriptions_s-subject
		cmp_processors_subscriptions_s-subject-->|AllRecordBatches|filter_processors_subscriptions_p-subscribe
		filter_processors_subscriptions_p-subscribe-->filter_processors_subscriptions_p-processor
		filter_processors_subscriptions_p-processor-->filter_processors_subscriptions_p-publish
		filter_processors_subscriptions_p-publish-->|Replace|filter_processors_subscriptions_s-subject
		filter_processors_subscriptions_s-subject-->|AllRecordBatches|select_processors_subscriptions_p-subscribe
		select_processors_subscriptions_p-subscribe-->select_processors_subscriptions_p-processor
		select_processors_subscriptions_p-processor-->select_processors_subscriptions_p-publish
		select_processors_subscriptions_p-publish-->|Replace|select_processors_subscriptions_s-subject
	end
	NextTaskNetwork_runtime_env-rt-->filter_processors_subscriptions_t
	NetworkProcessors-subject@{shape: doc, label: NetworkProcessors}
	cmp_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	cmp_processors_subscriptions_p-processor@{shape: rect, label: Select}
	cmp_processors_subscriptions_p-publish@{shape: fork}
	cmp_processors_subscriptions_s-subject@{shape: doc, label: cmp_processors_subscriptions_s}
	filter_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	filter_processors_subscriptions_p-processor@{shape: rect, label: Filter}
	filter_processors_subscriptions_p-publish@{shape: fork}
	filter_processors_subscriptions_s-subject@{shape: doc, label: filter_processors_subscriptions_s}
	select_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	select_processors_subscriptions_p-processor@{shape: rect, label: Select}
	select_processors_subscriptions_p-publish@{shape: fork}
	select_processors_subscriptions_s-subject@{shape: doc, label: select_processors_subscriptions_s}

	subgraph join_tasks_run_log_superstep_t
		SubjectsChangeLog-subject-->|AllRecordBatches|group_by_subject_change_log_superstep_p-subscribe
		group_by_subject_change_log_superstep_p-subscribe-->group_by_subject_change_log_superstep_p-processor
		group_by_subject_change_log_superstep_p-processor-->group_by_subject_change_log_superstep_p-publish
		group_by_subject_change_log_superstep_p-publish-->|Replace|group_by_subject_change_log_superstep_s-subject
		select_tasks_run_log_superstep_s-subject-.->|AllRecordBatches|join_tasks_run_log_superstep_p-subscribe
		NetworkTasks-subject-->|AllRecordBatches|join_tasks_run_log_superstep_p-subscribe
		join_tasks_run_log_superstep_p-subscribe-->join_tasks_run_log_superstep_p-processor
		join_tasks_run_log_superstep_p-processor-->join_tasks_run_log_superstep_p-publish
		join_tasks_run_log_superstep_p-publish-->|Replace|join_tasks_run_log_superstep_s-subject
		join_tasks_run_log_superstep_s-subject-->|AllRecordBatches|join_tasks_processors_subscriptions_p-subscribe
		select_processors_subscriptions_s-subject-->|AllRecordBatches|join_tasks_processors_subscriptions_p-subscribe
		join_tasks_processors_subscriptions_p-subscribe-->join_tasks_processors_subscriptions_p-processor
		join_tasks_processors_subscriptions_p-processor-->join_tasks_processors_subscriptions_p-publish
		join_tasks_processors_subscriptions_p-publish-->|Replace|join_tasks_processors_subscriptions_s-subject
		join_tasks_processors_subscriptions_s-subject-->|AllRecordBatches|join_tasks_processors_subscriptions_subjects_p-subscribe
		group_by_subject_change_log_superstep_s-subject-->|AllRecordBatches|join_tasks_processors_subscriptions_subjects_p-subscribe
		join_tasks_processors_subscriptions_subjects_p-subscribe-->join_tasks_processors_subscriptions_subjects_p-processor
		join_tasks_processors_subscriptions_subjects_p-processor-->join_tasks_processors_subscriptions_subjects_p-publish
		join_tasks_processors_subscriptions_subjects_p-publish-->|Replace|join_tasks_processors_subscriptions_subjects_s-subject
		join_tasks_processors_subscriptions_subjects_s-subject-->|AllRecordBatches|select_tasks_processors_subscriptions_subjects_p-subscribe
		select_tasks_processors_subscriptions_subjects_p-subscribe-->select_tasks_processors_subscriptions_subjects_p-processor
		select_tasks_processors_subscriptions_subjects_p-processor-->select_tasks_processors_subscriptions_subjects_p-publish
		select_tasks_processors_subscriptions_subjects_p-publish-->|Replace|select_tasks_processors_subscriptions_subjects_s-subject
		select_tasks_processors_subscriptions_subjects_s-subject-->|AllRecordBatches|group_by_tasks_processors_subscriptions_p-subscribe
		group_by_tasks_processors_subscriptions_p-subscribe-->group_by_tasks_processors_subscriptions_p-processor
		group_by_tasks_processors_subscriptions_p-processor-->group_by_tasks_processors_subscriptions_p-publish
		group_by_tasks_processors_subscriptions_p-publish-->|Replace|NetworkTasksSubscribeAggregate-subject
	end
	NextTaskNetwork_runtime_env-rt-->join_tasks_run_log_superstep_t
	SubjectsChangeLog-subject@{shape: doc, label: SubjectsChangeLog}
	group_by_subject_change_log_superstep_p-subscribe@{shape: diamond, label: All}
	group_by_subject_change_log_superstep_p-processor@{shape: rect, label: GroupBy}
	group_by_subject_change_log_superstep_p-publish@{shape: fork}
	group_by_subject_change_log_superstep_s-subject@{shape: doc, label: group_by_subject_change_log_superstep_s}
	NetworkTasks-subject@{shape: doc, label: NetworkTasks}
	join_tasks_run_log_superstep_p-subscribe@{shape: diamond, label: All}
	join_tasks_run_log_superstep_p-processor@{shape: rect, label: Join}
	join_tasks_run_log_superstep_p-publish@{shape: fork}
	join_tasks_run_log_superstep_s-subject@{shape: doc, label: join_tasks_run_log_superstep_s}
	join_tasks_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_subscriptions_p-processor@{shape: rect, label: Join}
	join_tasks_processors_subscriptions_p-publish@{shape: fork}
	join_tasks_processors_subscriptions_s-subject@{shape: doc, label: join_tasks_processors_subscriptions_s}
	join_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: Join}
	join_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	join_tasks_processors_subscriptions_subjects_s-subject@{shape: doc, label: join_tasks_processors_subscriptions_subjects_s}
	select_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	select_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: Select}
	select_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	select_tasks_processors_subscriptions_subjects_s-subject@{shape: doc, label: select_tasks_processors_subscriptions_subjects_s}
	group_by_tasks_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_processors_subscriptions_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_processors_subscriptions_p-publish@{shape: fork}
	NetworkTasksSubscribeAggregate-subject@{shape: doc, label: NetworkTasksSubscribeAggregate}

	subgraph filter_processors_publications_t
		NetworkProcessors-subject-.->|AllRecordBatches|cmp_processors_publications_p-subscribe
		cmp_processors_publications_p-subscribe-->cmp_processors_publications_p-processor
		cmp_processors_publications_p-processor-->cmp_processors_publications_p-publish
		cmp_processors_publications_p-publish-->|Replace|cmp_processors_publications_s-subject
		cmp_processors_publications_s-subject-->|AllRecordBatches|filter_processors_publications_p-subscribe
		filter_processors_publications_p-subscribe-->filter_processors_publications_p-processor
		filter_processors_publications_p-processor-->filter_processors_publications_p-publish
		filter_processors_publications_p-publish-->|Replace|filter_processors_publications_s-subject
		filter_processors_publications_s-subject-->|AllRecordBatches|select_processors_publications_p-subscribe
		select_processors_publications_p-subscribe-->select_processors_publications_p-processor
		select_processors_publications_p-processor-->select_processors_publications_p-publish
		select_processors_publications_p-publish-->|Replace|select_processors_publications_s-subject
	end
	NextTaskNetwork_runtime_env-rt-->filter_processors_publications_t
	cmp_processors_publications_p-subscribe@{shape: diamond, label: All}
	cmp_processors_publications_p-processor@{shape: rect, label: Select}
	cmp_processors_publications_p-publish@{shape: fork}
	cmp_processors_publications_s-subject@{shape: doc, label: cmp_processors_publications_s}
	filter_processors_publications_p-subscribe@{shape: diamond, label: All}
	filter_processors_publications_p-processor@{shape: rect, label: Filter}
	filter_processors_publications_p-publish@{shape: fork}
	filter_processors_publications_s-subject@{shape: doc, label: filter_processors_publications_s}
	select_processors_publications_p-subscribe@{shape: diamond, label: All}
	select_processors_publications_p-processor@{shape: rect, label: Select}
	select_processors_publications_p-publish@{shape: fork}
	select_processors_publications_s-subject@{shape: doc, label: select_processors_publications_s}

	subgraph select_tasks_processors_publications_t
		NetworkTasksSubscribe-subject-.->|AllRecordBatches|group_by_tasks_processors_subscriptions_subjects_p-subscribe
		group_by_tasks_processors_subscriptions_subjects_p-subscribe-->group_by_tasks_processors_subscriptions_subjects_p-processor
		group_by_tasks_processors_subscriptions_subjects_p-processor-->group_by_tasks_processors_subscriptions_subjects_p-publish
		group_by_tasks_processors_subscriptions_subjects_p-publish-->|Replace|group_by_tasks_processors_subscriptions_subjects_s-subject
		select_processors_publications_s-subject-->|AllRecordBatches|group_by_tasks_processors_publications_p-subscribe
		group_by_tasks_processors_publications_p-subscribe-->group_by_tasks_processors_publications_p-processor
		group_by_tasks_processors_publications_p-processor-->group_by_tasks_processors_publications_p-publish
		group_by_tasks_processors_publications_p-publish-->|Replace|group_by_tasks_processors_publications_s-subject
		group_by_tasks_processors_subscriptions_subjects_s-subject-->|AllRecordBatches|join_tasks_processors_publications_p-subscribe
		group_by_tasks_processors_publications_s-subject-->|AllRecordBatches|join_tasks_processors_publications_p-subscribe
		join_tasks_processors_publications_p-subscribe-->join_tasks_processors_publications_p-processor
		join_tasks_processors_publications_p-processor-->join_tasks_processors_publications_p-publish
		join_tasks_processors_publications_p-publish-->|Replace|join_tasks_processors_publications_s-subject
		join_tasks_processors_publications_s-subject-->|AllRecordBatches|select_tasks_processors_publications_p-subscribe
		select_tasks_processors_publications_p-subscribe-->select_tasks_processors_publications_p-processor
		select_tasks_processors_publications_p-processor-->select_tasks_processors_publications_p-publish
		select_tasks_processors_publications_p-publish-->|Replace|NetworkTasksSubscribePublish-subject
	end
	NextTaskNetwork_runtime_env-rt-->select_tasks_processors_publications_t
	NetworkTasksSubscribe-subject@{shape: doc, label: NetworkTasksSubscribe}
	group_by_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	group_by_tasks_processors_subscriptions_subjects_s-subject@{shape: doc, label: group_by_tasks_processors_subscriptions_subjects_s}
	group_by_tasks_processors_publications_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_processors_publications_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_processors_publications_p-publish@{shape: fork}
	group_by_tasks_processors_publications_s-subject@{shape: doc, label: group_by_tasks_processors_publications_s}
	join_tasks_processors_publications_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_publications_p-processor@{shape: rect, label: Join}
	join_tasks_processors_publications_p-publish@{shape: fork}
	join_tasks_processors_publications_s-subject@{shape: doc, label: join_tasks_processors_publications_s}
	select_tasks_processors_publications_p-subscribe@{shape: diamond, label: All}
	select_tasks_processors_publications_p-processor@{shape: rect, label: Select}
	select_tasks_processors_publications_p-publish@{shape: fork}
	NetworkTasksSubscribePublish-subject@{shape: doc, label: NetworkTasksSubscribePublish}"#
    }

    /// Return the Mermaid.js ER Diagram representation of the network
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    NetworkTasksRunLog["NetworkTasksRunLog"] {
        Utf8 network_name
        Utf8 task_name
        Int64 superstep
        Int64 timestamp
    }
    group_by_tasks_run_log_superstep_p["group_by_tasks_run_log_superstep_p"] {
        List-Utf8 agg_columns "['superstep']"
        List-Utf8 agg_operators "['Max']"
        Boolean cpu "false"
        Utf8 lhs_name "NetworkTasksRunLog"
        List-Utf8 lhs_values "['task_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    select_tasks_run_log_superstep_p["select_tasks_run_log_superstep_p"] {
        List-Utf8 as_columns "['','superstep']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_tasks_run_log_superstep_s"
        List-Utf8 lhs_values "['task_name','superstep-Max']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_tasks_run_log_superstep_s["select_tasks_run_log_superstep_s"] {
        Utf8 task_name
        Int64 superstep
    }
    NetworkProcessors["NetworkProcessors"] {
        Utf8 network_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }
    cmp_processors_subscriptions_p["cmp_processors_subscriptions_p"] {
        List-Utf8 as_columns "['','','','','','','','','subscription']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Ones']"
        Boolean cpu "false"
        Utf8 lhs_name "NetworkProcessors"
        List-Utf8 lhs_values "['network_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription','subscription']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_processors_subscriptions_p["filter_processors_subscriptions_p"] {
        List-Utf8 cmp_columns "['subscription']"
        List-Utf8 cmp_operators "['Equals']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_processors_subscriptions_s"
        List-Utf8 lhs_values "['is_subscription']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_processors_subscriptions_p["select_processors_subscriptions_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_subscriptions_s"
        List-Utf8 lhs_values "['network_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_processors_subscriptions_s["select_processors_subscriptions_s"] {
        Utf8 network_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }
    cmp_processors_publications_p["cmp_processors_publications_p"] {
        List-Utf8 as_columns "['','','','','','','','','publication']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Zeros']"
        Boolean cpu "false"
        Utf8 lhs_name "NetworkProcessors"
        List-Utf8 lhs_values "['network_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription','publication']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    filter_processors_publications_p["filter_processors_publications_p"] {
        List-Utf8 cmp_columns "['publication']"
        List-Utf8 cmp_operators "['Equals']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_processors_publications_s"
        List-Utf8 lhs_values "['is_subscription']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }
    select_processors_publications_p["select_processors_publications_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_publications_s"
        List-Utf8 lhs_values "['network_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_processors_publications_s["select_processors_publications_s"] {
        Utf8 network_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }
    SubjectsChangeLog["SubjectsChangeLog"] {
        Utf8 subject_name
        Utf8 task_name
        Utf8 network_name
        Int64 num_rows
        Int64 superstep
    }
    group_by_subject_change_log_superstep_p["group_by_subject_change_log_superstep_p"] {
        List-Utf8 agg_columns "['superstep']"
        List-Utf8 agg_operators "['Max']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsChangeLog"
        List-Utf8 lhs_values "['subject_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    join_tasks_run_log_superstep_p["join_tasks_run_log_superstep_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "task_name"
        Utf8 lhs_name "select_tasks_run_log_superstep_s"
        Utf8 lhs_pk "task_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "task_name"
        Utf8 rhs_name "NetworkTasks"
        Utf8 rhs_pk "task_name"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }
    NetworkTasks["NetworkTasks"] {
        Utf8 network_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 runtime_env_name
    }
    join_tasks_processors_subscriptions_p["join_tasks_processors_subscriptions_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "join_tasks_run_log_superstep_s"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "select_processors_subscriptions_s"
        Utf8 rhs_pk "processor_name"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }
    join_tasks_processors_subscriptions_subjects_p["join_tasks_processors_subscriptions_subjects_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "publication_subscription_table_name"
        Utf8 lhs_name "join_tasks_processors_subscriptions_s"
        Utf8 lhs_pk "publication_subscription_table_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "subject_name"
        Utf8 rhs_name "group_by_subject_change_log_superstep_s"
        Utf8 rhs_pk "subject_name"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }
    select_tasks_processors_subscriptions_subjects_p["select_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 as_columns "['','','','','subscription_name','subscription_table_name','','','','']"
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_subscriptions_subjects_s"
        List-Utf8 lhs_values "['network_name','task_name','processor_name','processor_type','publication_subscription_name','subject_name','subscribe_type','update_type','superstep','superstep-Max']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    select_tasks_processors_subscriptions_subjects_s["select_tasks_processors_subscriptions_subjects_s"] {
        Utf8 network_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 subscription_name
        Utf8 subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        Int64 superstep
        Int64 superstep-Max
    }
    group_by_tasks_processors_subscriptions_p["group_by_tasks_processors_subscriptions_p"] {
        List-Utf8 agg_columns "['subscription_name','subscription_table_name','subscribe_type','update_type','superstep','superstep-Max']"
        List-Utf8 agg_operators "['List','List','Last','Last','List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_tasks_processors_subscriptions_subjects_s"
        List-Utf8 lhs_values "['network_name','task_name','processor_type','processor_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    NetworkTasksSubscribeAggregate["NetworkTasksSubscribeAggregate"] {
        Utf8 network_name
        Utf8 task_name
        Utf8 processor_type
        Utf8 processor_name
        List-Utf8 subscription_name-List
        List-Utf8 subscription_table_name-List
        Utf8 subscribe_type-Last
        Utf8 update_type-Last
        List-Int64 superstep-List
        List-Int64 superstep-Max-List
    }
    NetworkTasksSubscribe["NetworkTasksSubscribe"] {
        Utf8 network_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 subscription_name
        Utf8 subscription_table_name
    }
    group_by_tasks_processors_subscriptions_subjects_p["group_by_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 agg_columns "['subscription_name','subscription_table_name']"
        List-Utf8 agg_operators "['List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "NetworkTasksSubscribe"
        List-Utf8 lhs_values "['network_name','task_name','processor_type','processor_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    group_by_tasks_processors_publications_p["group_by_tasks_processors_publications_p"] {
        List-Utf8 agg_columns "['publication_subscription_name','publication_subscription_table_name']"
        List-Utf8 agg_operators "['List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_processors_publications_s"
        List-Utf8 lhs_values "['network_name','processor_type','processor_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    join_tasks_processors_publications_p["join_tasks_processors_publications_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "group_by_tasks_processors_subscriptions_subjects_s"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "group_by_tasks_processors_publications_s"
        Utf8 rhs_pk "processor_name"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }
    select_tasks_processors_publications_p["select_tasks_processors_publications_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_publications_s"
        List-Utf8 as_columns "['','','','','subscription_names','subscription_table_names','publication_names','publication_table_names']"
        List-Utf8 lhs_values "['network_name','task_name','processor_name','processor_type','subscription_name-List','subscription_table_name-List','publication_subscription_name-List','publication_subscription_table_name-List']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }
    NetworkTasksSubscribePublish["NetworkTasksSubscribePublish"] {
        Utf8 network_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
        List-Utf8 publication_names
        List-Utf8 publication_table_names
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
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait,
        SubjectTrait,
    };
    use phymes_task::{SubscriptionTrait, test_task};

    use crate::{
        NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait,
        NetworkStream, NetworkStreamStep, NetworkStreamStepTrait, test_network_builder,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_next_task_network() -> Result<()> {
        // Initialize the testing runtime
        let runtime_env = RuntimeEnv::get_builder()
            .with_name("rt")
            .with_max_steps(1) // DM: prevent continued execution after the final superstep for testing
            .build_arc()?;

        // Initialize the network
        let next_task_network = NextTaskNetwork::default();
        let (network, network_messages) = NetworkBuilder::from_mermaid_flowchart(
            next_task_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            next_task_network.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(next_task_network.network_name)
        .with_diagnostics(true)
        .with_runtime_env(runtime_env)
        .add_processor_subjects()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test network data
        let mut message_map = {
            // Make the test sequential network
            let (network, network_messages) =
                test_network_builder::make_test_network_builder_sequential("network_1", 4)?
                    .with_diagnostics(false)
                    .add_network_interface(Some(&["state_1"]))?
                    .add_next_tasks()? // DM required for 'NetworkTasksSubscribePublish' table
                    .add_next_supersteps()?
                    .build_with_tables()?;

            // Mimic a superstep update without running the superstep
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
            let step = NetworkStreamStep::current_superstep(&network_arc).await;
            NetworkStreamStep::update_subjects_and_changelog_from_messages(
                &network_arc,
                messages,
                step,
            )
            .await?;

            // Extract out the subjects for the test
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::NetworkProcessors.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::NetworkProcessors.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let network_processor_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::NetworkProcessors.to_string().as_str())
                .with_update(&Publication::Replace {
                    subject_name: AvailableSubjects::NetworkProcessors.to_string(),
                })
                .with_publisher(next_task_network.network_name)
                .make_name()?
                .build()?;
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::NetworkTasks.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::NetworkTasks.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let network_tasks_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::NetworkTasks.to_string().as_str())
                .with_update(&Publication::Replace {
                    subject_name: AvailableSubjects::NetworkTasks.to_string(),
                })
                .with_publisher(next_task_network.network_name)
                .make_name()?
                .build()?;
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::NetworkTasksRunLog.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::NetworkTasksRunLog.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let network_tasks_run_log_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::NetworkTasksRunLog.to_string().as_str())
                .with_update(&Publication::Replace {
                    subject_name: AvailableSubjects::NetworkTasksRunLog.to_string(),
                })
                .with_publisher(next_task_network.network_name)
                .make_name()?
                .build()?;
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let subjects_change_log_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_update(&Publication::Replace {
                    subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                })
                .with_publisher(next_task_network.network_name)
                .make_name()?
                .build()?;
            create_message_map(vec![
                network_processor_message,
                network_tasks_message,
                network_tasks_run_log_message,
                subjects_change_log_message,
            ])
        };

        let mut tasks_publish_subscribe_messages = next_task_network
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        // Run the network
        message_map.extend(tasks_publish_subscribe_messages.pop().unwrap());
        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 1
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_tasks_run_log_superstep_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_tasks_run_log_superstep_s")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("task_name");
            assert_eq!(
                column,
                [
                    "filter_processors_publications_t",
                    "filter_processors_subscriptions_t",
                    "group_by_tasks_run_log_superstep_t",
                    "network_1",
                    "task_1"
                ]
            );
            let column = subject.get_column_as_vec_primitive::<i64>("superstep")?;
            assert_eq!(column, [1, 1, 1, 0, 0]);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_processors_subscriptions_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_processors_subscriptions_s")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("network_name");
            assert_eq!(
                column,
                [
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "processor_1",
                    "processor_1",
                    "processor_2",
                    "processor_2",
                    "processor_3",
                    "processor_3",
                    "network_1"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
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
            let column = subject.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(
                column,
                [
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateLastRecordBatch"
                ]
            );
            let column = subject.get_column_as_vec_str("publication_subscription_table_name");
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
            let column = subject.get_column_as_vec_str("subscribe_type");
            assert_eq!(column, ["All", "All", "All", "All", "All", "All", "Any"]);
            let column = subject.get_column_as_vec_str("update_type");
            assert_eq!(
                column,
                [
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate"
                ]
            );
            let column = subject.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [1, 1, 1, 1, 1, 1, 1]);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_processors_publications_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_processors_publications_s")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("network_name");
            assert_eq!(column, ["network_1", "network_1", "network_1", "network_1"]);
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                ["processor_1", "processor_2", "processor_3", "network_1"]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorEcho"
                ]
            );
            let column = subject.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(column, ["Extend", "Extend", "Extend", "Extend"]);
            let column = subject.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(column, ["state_1", "state_1", "state_1", "state_1"]);
            let column = subject.get_column_as_vec_str("subscribe_type");
            assert_eq!(column, ["All", "All", "All", "Any"]);
            let column = subject.get_column_as_vec_str("update_type");
            assert_eq!(
                column,
                [
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate"
                ]
            );
            let column = subject.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [0, 0, 0, 0]);
        }

        // Run the network
        let network_stream = NetworkStream::new(
            tasks_publish_subscribe_messages.pop().unwrap(),
            Arc::clone(&network_arc),
        );
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 2
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "NetworkTasksSubscribeAggregate".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("NetworkTasksSubscribeAggregate")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("network_name");
            assert_eq!(column, ["network_1", "network_1", "network_1", "network_1"]);
            let column = subject.get_column_as_vec_str("task_name");
            assert_eq!(column, ["network_1", "task_1", "task_1", "task_1"]);
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                ["network_1", "processor_1", "processor_2", "processor_3"]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorEcho",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_name-List")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "OnUpdateLastRecordBatch", "AlwaysAllRecordBatches", "OnUpdateAllRecordBatches", "AlwaysAllRecordBatches", "OnUpdateAllRecordBatches", "AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_name-List")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "state_1",
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1",
                ]
            );
            let column = subject.get_column_as_vec_str("subscribe_type-Last");
            assert_eq!(column, ["Any", "All", "All", "All"]);
            let column = subject.get_column_as_vec_str("update_type-Last");
            assert_eq!(
                column,
                [
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate"
                ]
            );
            let column = subject.get_column_as_vec_nested_primitive::<i64>("superstep-List")?;
            for supersteps in column {
                for superstep in supersteps {
                    assert_eq!(superstep, 0);
                }
            }
            let column = subject.get_column_as_vec_nested_primitive::<i64>("superstep-Max-List")?;
            for supersteps in column {
                for superstep in supersteps {
                    assert!(superstep >= 0);
                }
            }
        }

        // 3. Calculate the tasks subscribe
        let _ = tasks_publish_subscribe_messages.pop().unwrap();
        network_arc.tasks_subscribe().await?;

        {
            // Test the tasks subscribe
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::NetworkTasksSubscribe.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableSubjects::NetworkTasksSubscribe
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("network_name");
            assert_eq!(
                column,
                [
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1",
                    "network_1"
                ]
            );
            let column = subject.get_column_as_vec_str("task_name");
            assert_eq!(
                column,
                [
                    "network_1",
                    "task_1",
                    "task_1",
                    "task_1",
                    "task_1",
                    "task_1",
                    "task_1",
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "network_1",
                    "processor_1",
                    "processor_1",
                    "processor_2",
                    "processor_2",
                    "processor_3",
                    "processor_3",
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorEcho",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                ]
            );
            let column = subject.get_column_as_vec_str("subscription_name");
            assert_eq!(
                column,
                [
                    "OnUpdateLastRecordBatch",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                ]
            );
            let column = subject.get_column_as_vec_str("subscription_table_name");
            assert_eq!(
                column,
                [
                    "state_1",
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1",
                ]
            );
        }

        // Run the network
        let network_stream = NetworkStream::new(
            tasks_publish_subscribe_messages.pop().unwrap(),
            Arc::clone(&network_arc),
        );
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 3
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::NetworkTasksSubscribePublish.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableSubjects::NetworkTasksSubscribePublish
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("network_name");
            assert_eq!(column, ["network_1", "network_1", "network_1", "network_1"]);
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                ["network_1", "processor_1", "processor_2", "processor_3"]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ProcessorEcho",
                    "ProcessorMock",
                    "ProcessorMock",
                    "ProcessorMock",
                ]
            );
            let column =
                subject.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "OnUpdateLastRecordBatch",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "state_1",
                    "processor_1",
                    "state_1",
                    "processor_2",
                    "state_1",
                    "processor_3",
                    "state_1",
                ]
            );
            let column =
                subject.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["Extend", "Extend", "Extend", "Extend"]);
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["state_1", "state_1", "state_1", "state_1"]);
        }

        Ok(())
    }
}
