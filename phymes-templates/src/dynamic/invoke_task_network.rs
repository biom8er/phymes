use anyhow::Result;
use phymes_data::items_to_list;

pub trait DynamicNetworkTrait<'a> {
    /// Subjects to listen for
    fn subject_names(&self) -> Vec<String>;

    /// ER Diagram subjects as `Bytes`
    fn erdiagram_subject_subscriptions(&self, subject_names: &[&str]) -> String {
        let mut subscriptions_vec = Vec::new();
        for subject_name in subject_names {
            let line = format!(
                r#"{subject_name}["{subject_name}"] {{
        List-UInt8 bytes
    }}"#
            );
            subscriptions_vec.push(line);
        }
        subscriptions_vec.join("\n\t")
    }

    /// List of subjects compatible with List-Utf8
    fn subject_columns(&self) -> Result<String> {
        items_to_list(
            &self
                .subject_names()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    }

    /// Flowchart subjects subscriptions part 1
    fn flowchart_subject_subscriptions_1(
        &self,
        subject_names: &[&str],
        processor: &str,
        subscription: &str,
    ) -> String {
        let mut subscriptions_vec = Vec::new();
        for subject_name in subject_names {
            let line = format!("{subject_name}-subject-.->|{subscription}|{processor}-subscribe");
            subscriptions_vec.push(line);
        }
        subscriptions_vec.join("\n\t\t")
    }

    /// Flowchart subjects subscriptions part 2
    fn flowchart_subject_subscriptions_2(&self, subject_names: &[&str]) -> String {
        let mut subscriptions_vec = Vec::new();
        for subject_name in subject_names {
            let line = format!("{subject_name}-subject@{{shape: doc, label: {subject_name}}}");
            subscriptions_vec.push(line);
        }
        subscriptions_vec.join("\n\t")
    }
}

/// A session for dynamic tool calling
pub struct InvokeTaskNetwork<'a> {
    /// Session
    pub network_name: &'a str,
    /// Subjects to listen for
    pub subject_names: &'a [&'a str],
}

impl Default for InvokeTaskNetwork<'_> {
    fn default() -> Self {
        InvokeTaskNetwork {
            network_name: "invoke_task_network",
            subject_names: &["Bytes"],
        }
    }
}

impl<'a> DynamicNetworkTrait<'a> for InvokeTaskNetwork<'a> {
    fn subject_names(&self) -> Vec<String> {
        self.subject_names.iter().map(|s| s.to_string()).collect()
    }
}

impl<'a> InvokeTaskNetwork<'a> {
    pub fn new(network_name: &'a str, subject_names: &'a [&'a str]) -> Self {
        InvokeTaskNetwork {
            network_name,
            subject_names,
        }
    }
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        format!(
            r#"flowchart TD
    ToolCallNetwork_runtime_env-rt@{{shape: subproc, label: ToolCallNetwork_runtime_env}}
    
	subgraph group_by_processors_subscriptions_t
        {}
		select_processors_subscriptions_s-subject-->|AllRecordBatches|group_by_processors_subscriptions_p-subscribe
		group_by_processors_subscriptions_p-subscribe-->group_by_processors_subscriptions_p-processor
		group_by_processors_subscriptions_p-processor-->group_by_processors_subscriptions_p-publish
		group_by_processors_subscriptions_p-publish-->|Replace|group_by_processors_subscriptions_s-subject
		group_by_processors_subscriptions_s-subject-->|AllRecordBatches|select_processors_subscriptions_aggregated_p-subscribe
		select_processors_subscriptions_aggregated_p-subscribe-->select_processors_subscriptions_aggregated_p-processor
		select_processors_subscriptions_aggregated_p-processor-->select_processors_subscriptions_aggregated_p-publish
		select_processors_subscriptions_aggregated_p-publish-->|Replace|select_processors_subscriptions_aggregated_s-subject
    end
	ToolCallNetwork_runtime_env-rt-->group_by_processors_subscriptions_t
    {}
	select_processors_subscriptions_s-subject@{{shape: doc, label: select_processors_subscriptions_s}}
	group_by_processors_subscriptions_p-subscribe@{{shape: diamond, label: Any}}
	group_by_processors_subscriptions_p-processor@{{shape: rect, label: GroupBy}}
	group_by_processors_subscriptions_p-publish@{{shape: fork}}
	group_by_processors_subscriptions_s-subject@{{shape: doc, label: group_by_processors_subscriptions_s}}
	select_processors_subscriptions_aggregated_p-subscribe@{{shape: diamond, label: All}}
	select_processors_subscriptions_aggregated_p-processor@{{shape: rect, label: Select}}
	select_processors_subscriptions_aggregated_p-publish@{{shape: fork}}
	select_processors_subscriptions_aggregated_s-subject@{{shape: doc, label: select_processors_subscriptions_aggregated_s}}
    
	subgraph group_by_processors_publications_t
        {}
		select_processors_publications_s-subject-->|AllRecordBatches|group_by_processors_publications_p-subscribe
		group_by_processors_publications_p-subscribe-->group_by_processors_publications_p-processor
		group_by_processors_publications_p-processor-->group_by_processors_publications_p-publish
		group_by_processors_publications_p-publish-->|Replace|group_by_processors_publications_s-subject
		group_by_processors_publications_s-subject-->|AllRecordBatches|select_processors_publications_aggregated_p-subscribe
		select_processors_publications_aggregated_p-subscribe-->select_processors_publications_aggregated_p-processor
		select_processors_publications_aggregated_p-processor-->select_processors_publications_aggregated_p-publish
		select_processors_publications_aggregated_p-publish-->|Replace|select_processors_publications_aggregated_s-subject
    end
	ToolCallNetwork_runtime_env-rt-->group_by_processors_publications_t
	select_processors_publications_s-subject@{{shape: doc, label: select_processors_publications_s}}
	group_by_processors_publications_p-subscribe@{{shape: diamond, label: Any}}
	group_by_processors_publications_p-processor@{{shape: rect, label: GroupBy}}
	group_by_processors_publications_p-publish@{{shape: fork}}
	group_by_processors_publications_s-subject@{{shape: doc, label: group_by_processors_publications_s}}
	select_processors_publications_aggregated_p-subscribe@{{shape: diamond, label: All}}
	select_processors_publications_aggregated_p-processor@{{shape: rect, label: Select}}
	select_processors_publications_aggregated_p-publish@{{shape: fork}}
	select_processors_publications_aggregated_s-subject@{{shape: doc, label: select_processors_publications_aggregated_s}}

	subgraph join_tasks_processors_subscriptions_publications_aggregated_t
		select_processors_subscriptions_aggregated_s-subject-.->|AllRecordBatches|join_processors_subscriptions_publications_aggregated_p-subscribe
		select_processors_publications_aggregated_s-subject-.->|AllRecordBatches|join_processors_subscriptions_publications_aggregated_p-subscribe
		join_processors_subscriptions_publications_aggregated_p-subscribe-->join_processors_subscriptions_publications_aggregated_p-processor
		join_processors_subscriptions_publications_aggregated_p-processor-->join_processors_subscriptions_publications_aggregated_p-publish
		join_processors_subscriptions_publications_aggregated_p-publish-->|Replace|join_processors_subscriptions_publications_aggregated_s-subject
		join_processors_subscriptions_publications_aggregated_s-subject-->|AllRecordBatches|join_tasks_processors_subscriptions_publications_aggregated_p-subscribe
		SessionTasks-subject-->|AllRecordBatches|join_tasks_processors_subscriptions_publications_aggregated_p-subscribe
		join_tasks_processors_subscriptions_publications_aggregated_p-subscribe-->join_tasks_processors_subscriptions_publications_aggregated_p-processor
		join_tasks_processors_subscriptions_publications_aggregated_p-processor-->join_tasks_processors_subscriptions_publications_aggregated_p-publish
		join_tasks_processors_subscriptions_publications_aggregated_p-publish-->|Replace|join_tasks_processors_subscriptions_publications_aggregated_s-subject
		join_tasks_processors_subscriptions_publications_aggregated_s-subject-->|AllRecordBatches|select_tasks_processors_subscriptions_publications_aggregated_p-subscribe
		select_tasks_processors_subscriptions_publications_aggregated_p-subscribe-->select_tasks_processors_subscriptions_publications_aggregated_p-processor
		select_tasks_processors_subscriptions_publications_aggregated_p-processor-->select_tasks_processors_subscriptions_publications_aggregated_p-publish
		select_tasks_processors_subscriptions_publications_aggregated_p-publish-->|Replace|select_tasks_processors_subscriptions_publications_aggregated_s-subject
	end
	ToolCallNetwork_runtime_env-rt-->join_tasks_processors_subscriptions_publications_aggregated_t
	join_processors_subscriptions_publications_aggregated_p-subscribe@{{shape: diamond, label: All}}
	join_processors_subscriptions_publications_aggregated_p-processor@{{shape: rect, label: Join}}
	join_processors_subscriptions_publications_aggregated_p-publish@{{shape: fork}}
	join_processors_subscriptions_publications_aggregated_s-subject@{{shape: doc, label: join_processors_subscriptions_publications_aggregated_s}}
	SessionTasks-subject@{{shape: doc, label: SessionTasks}}
	join_tasks_processors_subscriptions_publications_aggregated_p-subscribe@{{shape: diamond, label: All}}
	join_tasks_processors_subscriptions_publications_aggregated_p-processor@{{shape: rect, label: Join}}
	join_tasks_processors_subscriptions_publications_aggregated_p-publish@{{shape: fork}}
	join_tasks_processors_subscriptions_publications_aggregated_s-subject@{{shape: doc, label: join_tasks_processors_subscriptions_publications_aggregated_s}}
	select_tasks_processors_subscriptions_publications_aggregated_p-subscribe@{{shape: diamond, label: All}}
	select_tasks_processors_subscriptions_publications_aggregated_p-processor@{{shape: rect, label: Select}}
	select_tasks_processors_subscriptions_publications_aggregated_p-publish@{{shape: fork}}
	select_tasks_processors_subscriptions_publications_aggregated_s-subject@{{shape: doc, label: select_tasks_processors_subscriptions_publications_aggregated_s}}  
    
	%% ------------------------------------------------------------------------------
	%% Tool call processor that enables calling processors from their config
	%% ------------------------------------------------------------------------------
	subgraph call_processor_t
        select_tasks_processors_subscriptions_publications_aggregated_s-subject-.->|AllRecordBatches|echo_processor_p-subscribe
		echo_processor_p-subscribe-->echo_processor_p-processor
		echo_processor_p-processor-->echo_processor_p-publish
		echo_processor_p-publish-->|Extend|select_tasks_processors_subscriptions_publications_aggregated_s-subject
        select_tasks_processors_subscriptions_publications_aggregated_s-subject-->|AllRecordBatches|call_processor_p-subscribe
		{}
		call_processor_p-subscribe-->call_processor_p-processor
		call_processor_p-processor-->call_processor_p-publish
		call_processor_p-publish-->|Extend|SessionTasksSubscribePublish-subject
	end
	ToolCallNetwork_runtime_env-rt-->call_processor_t
	echo_processor_p-processor@{{shape: rect, label: ProcessorEcho}}
	echo_processor_p-publish@{{shape: fork}}
	echo_processor_p-subscribe@{{shape: diamond, label: All}}
	call_processor_p-processor@{{shape: rect, label: ToolCallProcessor}}
	call_processor_p-publish@{{shape: fork}}
	call_processor_p-subscribe@{{shape: diamond, label: Any}}
	SessionTasksSubscribePublish-subject@{{shape: doc, label: SessionTasksSubscribePublish}}"#,
            self.flowchart_subject_subscriptions_1(
                &self
                    .subject_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                "group_by_processors_subscriptions_p",
                "OnUpdateEmpty"
            ),
            self.flowchart_subject_subscriptions_2(
                &self
                    .subject_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
            ),
            self.flowchart_subject_subscriptions_1(
                &self
                    .subject_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                "group_by_processors_publications_p",
                "OnUpdateEmpty"
            ),
            self.flowchart_subject_subscriptions_1(
                &self
                    .subject_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                "call_processor_p",
                "LastRecordBatch"
            )
        )
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> Result<String> {
        let er_diagram = format!(
            r#"erDiagram
    {}
    select_processors_subscriptions_s["select_processors_subscriptions_s"] {{
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }}
    select_processors_publications_s["select_processors_publications_s"] {{
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }}
    group_by_processors_subscriptions_p["group_by_processors_subscriptions_p"] {{
        List-Utf8 agg_columns "['publication_subscription_name','publication_subscription_table_name']"
        List-Utf8 agg_operators "['List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_processors_subscriptions_s"
        List-Utf8 lhs_values "['session_name','processor_type','processor_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }}
    select_processors_subscriptions_aggregated_p["select_processors_subscriptions_aggregated_p"] {{
        List-Utf8 as_columns "['','','','subscription_names','subscription_table_names']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_processors_subscriptions_s"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name-List','publication_subscription_table_name-List']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    select_processors_subscriptions_aggregated_s["select_processors_subscriptions_aggregated_s"] {{
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
    }}
    group_by_processors_publications_p["group_by_processors_publications_p"] {{
        List-Utf8 agg_columns "['publication_subscription_name','publication_subscription_table_name']"
        List-Utf8 agg_operators "['List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_processors_publications_s"
        List-Utf8 lhs_values "['session_name','processor_type','processor_name']"
        Utf8 operator "GroupBy"
        Utf8 lhs_stream "Accumulate"
    }}
    select_processors_publications_aggregated_p["select_processors_publications_aggregated_p"] {{
        List-Utf8 as_columns "['','','','publication_names','publication_table_names']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_processors_publications_s"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name-List','publication_subscription_table_name-List']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    select_processors_publications_aggregated_s["select_processors_publications_aggregated_s"] {{
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 publication_names
        List-Utf8 publication_table_names
    }}
    join_processors_subscriptions_publications_aggregated_p["join_processors_subscriptions_publications_aggregated_p"] {{
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "select_processors_subscriptions_aggregated_s"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "select_processors_publications_aggregated_s"
        Utf8 rhs_pk "processor_name"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }}
    SessionTasks["SessionTasks"] {{
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 runtime_env_name
    }}
    join_tasks_processors_subscriptions_publications_aggregated_p["join_tasks_processors_subscriptions_publications_aggregated_p"] {{
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "join_processors_subscriptions_publications_aggregated_s"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "SessionTasks"
        Utf8 rhs_pk "processor_name"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
        Utf8 join_operators "Inner"
    }}
    select_tasks_processors_subscriptions_publications_aggregated_p["select_tasks_processors_subscriptions_publications_aggregated_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_subscriptions_publications_aggregated_s"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','subscription_names','subscription_table_names','publication_names','publication_table_names']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    select_tasks_processors_subscriptions_publications_aggregated_s["select_tasks_processors_subscriptions_publications_aggregated_s"] {{
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
        List-Utf8 publication_names
        List-Utf8 publication_table_names
    }}
    call_processor_p["call_processor_p"] {{
        Utf8 subject_name "select_tasks_processors_subscriptions_publications_aggregated_s"
        List-Utf8 subject_names "[{}]"
        List-Utf8 subscription_table_names "['lhs_name', 'rhs_name', 'subject_name']"
    }}
    SessionTasksSubscribePublish["SessionTasksSubscribePublish"] {{
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
        List-Utf8 publication_names
        List-Utf8 publication_table_names
    }}"#,
            self.erdiagram_subject_subscriptions(
                &self
                    .subject_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
            ),
            self.subject_columns()?
        );
        Ok(er_diagram)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{AvailableSubjects, AvailableSubjectsTrait, create_bytes_record_batch};
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_invoke_task_network() -> Result<()> {
        // Initialize the session
        let invoke_task_network = InvokeTaskNetwork::default();
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &invoke_task_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &invoke_task_network.as_mermaid_erdiagram()?,
            false,
            true,
        )?
        .with_name(invoke_task_network.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_supersteps()?
        .add_next_tasks()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Replace the Bytes to trigger the session
        let message_map = {
            let batch = create_bytes_record_batch(vec!["{}".into()])?;
            let table = AvailableSubjects::Bytes.to_subject(None, Some(vec![batch]))?;
            let session_tasks_message = IPCMessage::get_builder()
                .with_subject(table.get_name())
                .with_update(&Publication::Replace {
                    subject_name: table.get_name().to_string(),
                })
                .with_publisher(invoke_task_network.network_name)
                .with_message(table.to_ipc_stream()?)
                .make_name()?
                .build()?;
            create_message_map(vec![session_tasks_message])
        };
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Run the session
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 1
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
            let column = subject.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "group_by_processors_subscriptions_p",
                    "group_by_processors_subscriptions_p",
                    "group_by_processors_subscriptions_p",
                    "select_processors_subscriptions_aggregated_p",
                    "select_processors_subscriptions_aggregated_p",
                    "group_by_processors_publications_p",
                    "group_by_processors_publications_p",
                    "group_by_processors_publications_p",
                    "select_processors_publications_aggregated_p",
                    "select_processors_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_p",
                    "echo_processor_p",
                    "echo_processor_p",
                    "call_processor_p",
                    "call_processor_p",
                    "call_processor_p"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "GroupBy",
                    "GroupBy",
                    "GroupBy",
                    "Select",
                    "Select",
                    "GroupBy",
                    "GroupBy",
                    "GroupBy",
                    "Select",
                    "Select",
                    "Join",
                    "Join",
                    "Join",
                    "Join",
                    "Join",
                    "Join",
                    "Select",
                    "Select",
                    "ProcessorEcho",
                    "ProcessorEcho",
                    "ToolCallProcessor",
                    "ToolCallProcessor",
                    "ToolCallProcessor"
                ]
            );
            let column = subject.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(
                column,
                [
                    "OnUpdateEmpty",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateEmpty",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateLastRecordBatch",
                    "AlwaysAllRecordBatches"
                ]
            );
            let column = subject.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(
                column,
                [
                    "Bytes",
                    "select_processors_subscriptions_s",
                    "group_by_processors_subscriptions_p",
                    "group_by_processors_subscriptions_s",
                    "select_processors_subscriptions_aggregated_p",
                    "Bytes",
                    "select_processors_publications_s",
                    "group_by_processors_publications_p",
                    "group_by_processors_publications_s",
                    "select_processors_publications_aggregated_p",
                    "select_processors_subscriptions_aggregated_s",
                    "select_processors_publications_aggregated_s",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_s",
                    "SessionTasks",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "echo_processor_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "Bytes",
                    "call_processor_p"
                ]
            );
            let column = subject.get_column_as_vec_str("subscribe_type");
            assert_eq!(
                column,
                [
                    "Any", "Any", "Any", "All", "All", "Any", "Any", "Any", "All", "All", "All",
                    "All", "All", "All", "All", "All", "All", "All", "All", "All", "Any", "Any",
                    "Any"
                ]
            );
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
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
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
            assert_eq!(
                column,
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]
            );

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
            let column = subject.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "group_by_processors_subscriptions_p",
                    "select_processors_subscriptions_aggregated_p",
                    "group_by_processors_publications_p",
                    "select_processors_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_p",
                    "echo_processor_p",
                    "call_processor_p"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "GroupBy",
                    "Select",
                    "GroupBy",
                    "Select",
                    "Join",
                    "Join",
                    "Select",
                    "ProcessorEcho",
                    "ToolCallProcessor"
                ]
            );
            let column = subject.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(
                column,
                [
                    "Replace", "Replace", "Replace", "Replace", "Replace", "Replace", "Replace",
                    "Extend", "Extend"
                ]
            );
            let column = subject.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(
                column,
                [
                    "group_by_processors_subscriptions_s",
                    "select_processors_subscriptions_aggregated_s",
                    "group_by_processors_publications_s",
                    "select_processors_publications_aggregated_s",
                    "join_processors_subscriptions_publications_aggregated_s",
                    "join_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "SessionTasksSubscribePublish"
                ]
            );
            let column = subject.get_column_as_vec_str("subscribe_type");
            assert_eq!(
                column,
                [
                    "Any", "All", "Any", "All", "All", "All", "All", "All", "Any"
                ]
            );
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
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate",
                    "SubjectChangedSinceLastRunUpdate"
                ]
            );
            let column = subject.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [0, 0, 0, 0, 0, 0, 0, 0, 0]);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_processors_subscriptions_aggregated_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_processors_subscriptions_aggregated_s")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "call_processor_p",
                    "echo_processor_p",
                    "group_by_processors_publications_p",
                    "group_by_processors_subscriptions_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_processors_publications_aggregated_p",
                    "select_processors_subscriptions_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_p"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ToolCallProcessor",
                    "ProcessorEcho",
                    "GroupBy",
                    "GroupBy",
                    "Join",
                    "Join",
                    "Select",
                    "Select",
                    "Select"
                ]
            );
            let column =
                subject.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "AlwaysAllRecordBatches",
                    "OnUpdateLastRecordBatch",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateEmpty",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateEmpty",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches"
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "Bytes",
                    "call_processor_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "echo_processor_p",
                    "Bytes",
                    "select_processors_publications_s",
                    "group_by_processors_publications_p",
                    "Bytes",
                    "select_processors_subscriptions_s",
                    "group_by_processors_subscriptions_p",
                    "select_processors_subscriptions_aggregated_s",
                    "select_processors_publications_aggregated_s",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_s",
                    "SessionTasks",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "group_by_processors_publications_s",
                    "select_processors_publications_aggregated_p",
                    "group_by_processors_subscriptions_s",
                    "select_processors_subscriptions_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_p"
                ]
            );

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_processors_publications_aggregated_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_processors_publications_aggregated_s")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "call_processor_p",
                    "echo_processor_p",
                    "group_by_processors_publications_p",
                    "group_by_processors_subscriptions_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_processors_publications_aggregated_p",
                    "select_processors_subscriptions_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_p"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ToolCallProcessor",
                    "ProcessorEcho",
                    "GroupBy",
                    "GroupBy",
                    "Join",
                    "Join",
                    "Select",
                    "Select",
                    "Select"
                ]
            );
            let column =
                subject.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "Extend", "Extend", "Replace", "Replace", "Replace", "Replace", "Replace",
                    "Replace", "Replace"
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "SessionTasksSubscribePublish",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "group_by_processors_publications_s",
                    "group_by_processors_subscriptions_s",
                    "join_processors_subscriptions_publications_aggregated_s",
                    "join_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_processors_publications_aggregated_s",
                    "select_processors_subscriptions_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_s"
                ]
            );

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "select_tasks_processors_subscriptions_publications_aggregated_s"
                    .to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("select_tasks_processors_subscriptions_publications_aggregated_s")
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("session_name");
            assert_eq!(
                column,
                [
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network",
                    "invoke_task_network"
                ]
            );
            let column = subject.get_column_as_vec_str("task_name");
            assert_eq!(
                column,
                [
                    "call_processor_t",
                    "call_processor_t",
                    "group_by_processors_publications_t",
                    "group_by_processors_subscriptions_t",
                    "join_tasks_processors_subscriptions_publications_aggregated_t",
                    "join_tasks_processors_subscriptions_publications_aggregated_t",
                    "group_by_processors_publications_t",
                    "group_by_processors_subscriptions_t",
                    "join_tasks_processors_subscriptions_publications_aggregated_t"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_name");
            assert_eq!(
                column,
                [
                    "call_processor_p",
                    "echo_processor_p",
                    "group_by_processors_publications_p",
                    "group_by_processors_subscriptions_p",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "select_processors_publications_aggregated_p",
                    "select_processors_subscriptions_aggregated_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_p"
                ]
            );
            let column = subject.get_column_as_vec_str("processor_type");
            assert_eq!(
                column,
                [
                    "ToolCallProcessor",
                    "ProcessorEcho",
                    "GroupBy",
                    "GroupBy",
                    "Join",
                    "Join",
                    "Select",
                    "Select",
                    "Select"
                ]
            );
            let column =
                subject.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "AlwaysAllRecordBatches",
                    "OnUpdateLastRecordBatch",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateEmpty",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateEmpty",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "OnUpdateAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches",
                    "AlwaysAllRecordBatches"
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "Bytes",
                    "call_processor_p",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "echo_processor_p",
                    "Bytes",
                    "select_processors_publications_s",
                    "group_by_processors_publications_p",
                    "Bytes",
                    "select_processors_subscriptions_s",
                    "group_by_processors_subscriptions_p",
                    "select_processors_subscriptions_aggregated_s",
                    "select_processors_publications_aggregated_s",
                    "join_processors_subscriptions_publications_aggregated_p",
                    "join_processors_subscriptions_publications_aggregated_s",
                    "SessionTasks",
                    "join_tasks_processors_subscriptions_publications_aggregated_p",
                    "group_by_processors_publications_s",
                    "select_processors_publications_aggregated_p",
                    "group_by_processors_subscriptions_s",
                    "select_processors_subscriptions_aggregated_p",
                    "join_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_p"
                ]
            );
            let column =
                subject.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "Extend", "Extend", "Replace", "Replace", "Replace", "Replace", "Replace",
                    "Replace", "Replace"
                ]
            );
            let column = subject
                .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(
                flattened,
                [
                    "SessionTasksSubscribePublish",
                    "select_tasks_processors_subscriptions_publications_aggregated_s",
                    "group_by_processors_publications_s",
                    "group_by_processors_subscriptions_s",
                    "join_processors_subscriptions_publications_aggregated_s",
                    "join_tasks_processors_subscriptions_publications_aggregated_s",
                    "select_processors_publications_aggregated_s",
                    "select_processors_subscriptions_aggregated_s",
                    "select_tasks_processors_subscriptions_publications_aggregated_s"
                ]
            );

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert_eq!(batches.len(), 0);
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::SessionErrors.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);
            let column = subject.get_column_as_vec_str("content");
            assert!(column.first().unwrap().contains("Tool call subscription subject `Bytes` was not found in the All task/publisher subscriptions and publications."));
        }

        Ok(())
    }
}
