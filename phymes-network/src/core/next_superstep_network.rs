use anyhow::Result;
use phymes_event::Publication;
use phymes_message::{IPCMessage, IPCMessageMap, MessageBuilderTrait, create_message_map};
use phymes_schemas::{AvailableSubjects, create_network_tasks_subscribe_publish_batch};
use phymes_subject::{BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait};

/// A network for determining the next superstep task publications and subscriptions
pub struct NextSuperstepNetwork<'a> {
    /// Network
    pub network_name: &'a str,
}

impl Default for NextSuperstepNetwork<'_> {
    fn default() -> Self {
        NextSuperstepNetwork {
            network_name: "next_superstep_network",
        }
    }
}

impl<'a> NextSuperstepNetwork<'a> {
    /// Return the pre-compiled task subscriptions and publications as messages
    pub fn as_task_messages(&self) -> Result<Vec<IPCMessageMap>> {
        // 1. Message to trigger the first superstep
        let task_names = vec!["max_superstep_t"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["group_by_network_superstep_p"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![vec!["OnUpdateAllRecordBatches", "AlwaysAllRecordBatches"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let subscription_table_names =
            vec![vec!["NetworkSupersteps", "group_by_network_superstep_p"]]
                .into_iter()
                .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["NetworkSuperstepMax"]]
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

        Ok(vec![messages_1])
    }

    /// Return the Mermaid.js flowchart representation of the network
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    NextSuperstepNetwork_runtime_env-rt@{shape: subproc, label: NextSuperstepNetwork_runtime_env}

	subgraph max_superstep_t
		NetworkSupersteps-subject-.->|AllRecordBatches|group_by_network_superstep_p-subscribe
		group_by_network_superstep_p-subscribe-->group_by_network_superstep_p-processor
		group_by_network_superstep_p-processor-->group_by_network_superstep_p-publish
		group_by_network_superstep_p-publish-->|Replace|NetworkSuperstepMax-subject
	end
	NextSuperstepNetwork_runtime_env-rt-->max_superstep_t
	NetworkSupersteps-subject@{shape: doc, label: NetworkSupersteps}
	group_by_network_superstep_p-subscribe@{shape: diamond, label: All}
	group_by_network_superstep_p-processor@{shape: rect, label: GroupBy}
	group_by_network_superstep_p-publish@{shape: fork}
	NetworkSuperstepMax-subject@{shape: doc, label: NetworkSuperstepMax}"#
    }

    /// Return the Mermaid.js ER Diagram representation of the network
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    NetworkSupersteps["NetworkSupersteps"] {
        Utf8 network_name
        UInt32 superstep
    }
    group_by_network_superstep_p["group_by_network_superstep_p"] {
        List-Utf8 agg_columns "['superstep']"
        List-Utf8 agg_operators "['Max']"
        Boolean cpu "false"
        Utf8 lhs_name "NetworkSupersteps"
        List-Utf8 lhs_values "['network_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }
    NetworkSuperstepMax["NetworkSuperstepMax"] {
        Utf8 network_name
        UInt32 superstep-Max
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
    use phymes_schemas::create_network_supersteps_batch;
    use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, SubjectTrait};
    use phymes_task::SubscriptionTrait;

    use crate::{
        NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait,
        NetworkStream, NetworkStreamStepMinimal, NetworkStreamStepTrait,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_next_superstep_network() -> Result<()> {
        // Initialize the network
        let next_superstep_network = NextSuperstepNetwork::default();
        let (network, network_messages) = NetworkBuilder::from_mermaid_flowchart(
            next_superstep_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            next_superstep_network.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(next_superstep_network.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Network Tasks
        let mut next_superstep_messages = next_superstep_network
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        // Run the network
        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(
            next_superstep_messages.pop().unwrap(),
            Arc::clone(&network_arc),
        );
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supserstep 1
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::NetworkSuperstepMax.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkSuperstepMax.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("network_name");
        assert_eq!(column, ["next_superstep_network"]);
        let column = subject.get_column_as_vec_primitive::<u32>("superstep-Max")?;
        assert_eq!(column, [2]); // Should be one without the forced execution of network step network tasks

        // Make the test network data
        let network_names = ["network_1", "network_1", "network_1", "network_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let supersteps = vec![3, 4, 5, 6];
        let batch = create_network_supersteps_batch(network_names, supersteps)?;
        let table = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkSupersteps.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let superstep_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableSubjects::NetworkSupersteps.to_string().as_str())
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::NetworkSupersteps.to_string(),
            })
            .with_publisher(next_superstep_network.network_name)
            .make_name()?
            .build()?;
        let mut message_map = create_message_map(vec![superstep_message]);

        // Network Tasks
        let mut next_superstep_messages = next_superstep_network
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        message_map.extend(next_superstep_messages.pop().unwrap());
        let _response =
            NetworkStreamStepMinimal::run_superstep(Arc::clone(&network_arc), message_map).await?;

        // Test supserstep 1
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::NetworkSuperstepMax.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkSuperstepMax.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("network_name");
        assert_eq!(column, ["network_1"]);
        let column = subject.get_column_as_vec_primitive::<u32>("superstep-Max")?;
        assert_eq!(column, [6]);

        Ok(())
    }
}
