use phymes_data::ToolTrait;
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_network::{DynamicTaskNetworkBuilder, DynamicTaskNetworkTypes, NetworkBuilder, NetworkBuilderCustomTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, create_tools_record_batch};
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, SubjectPlan, SubjectPlanBuilderTrait};

use crate::{AttachmentsNetworkBuilder, GenerateTextNetworkBuilder};

/// Tabular (columnar data) operator network
/// 
/// # Notes
/// * Pre-specified operators than can be called sequentially to build a full SQL-like SELECT command
/// * Results can be exported to CSV or HTML
/// * Function-calling LLM "assistant" for operator calling hints
/// * View of all operators calls sorted by timestamp
pub struct TabularDataOperatorNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
}

impl TabularDataOperatorNetworkBuilder {
    /// Helper to create a Dynamic Unary Operator Network
    fn unary_network_builder(processor: AvailableProcessors, subject_name_lhs: &str, subject_name_out: &str) -> NetworkBuilder {
        let task_name = processor.to_string();
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&task_name),
                // Some(&task_name),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let builder = DynamicTaskNetworkBuilder {
            network_name: task_name,
            dynamic_type: DynamicTaskNetworkTypes::Function,
            processor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name_lhs.to_string(),
            },
            publication: Publication::Replace {
                subject_name: subject_name_out.to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_processor,
            ..Default::default()
        };
        builder.build_dynamic()
    }

    /// Helper to create a Dynamic Binary Operator Network
    fn binary_network_builder(processor: AvailableProcessors, subject_name_lhs: &str, subject_name_rhs: &str, subject_name_out: &str) -> NetworkBuilder {
        let task_name = processor.to_string();
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&task_name),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let builder = DynamicTaskNetworkBuilder {
            network_name: task_name,
            dynamic_type: DynamicTaskNetworkTypes::Function,
            processor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name_lhs.to_string(),
            },
            subscription_rhs: Some(Subscription::AlwaysAllRecordBatches {
                subject_name: subject_name_rhs.to_string(),
            }),
            publication: Publication::Replace {
                subject_name: subject_name_out.to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_processor,
            ..Default::default()
        };
        builder.build_dynamic()
    }
}

impl Default for TabularDataOperatorNetworkBuilder {
    fn default() -> Self {
        // Constant names used throughout the network
        let subject_name_lhs = "left_hand_side_s";
        let subject_name_rhs = "right_hand_side_s";
        let subject_name_out = "out_s";

        // Tabular data operators
        let unary_operators = [AvailableProcessors::Select,
            AvailableProcessors::Sort,
            AvailableProcessors::GroupBy,
            AvailableProcessors::Filter,
            AvailableProcessors::Pivot,
            AvailableProcessors::Melt,
            // AvailableProcessors::LimitProcessor,
            // AvailableProcessors::CoalesceProcessor,
        ];
        let binary_operators = [AvailableProcessors::Patch,
            AvailableProcessors::Diff,
            AvailableProcessors::Join,
            // AvailableProcessors::AggregatorProcessor,
        ];
        let tabular_data_operator_network_builder = unary_operators
            .into_iter()
            .map(|op| Self::unary_network_builder(op, subject_name_lhs, subject_name_out))
            .chain(binary_operators
                .into_iter()
                .map(|op| Self::binary_network_builder(op, subject_name_lhs, subject_name_rhs, subject_name_out))
            )
            .chain([Self::unary_network_builder(AvailableProcessors::ExtractTabular, &AvailableInterfaceSubjects::UserCsv.to_string(), subject_name_lhs)])
            .chain([Self::unary_network_builder(AvailableProcessors::PackTabular, subject_name_out, &AvailableInterfaceSubjects::AssistantCsv.to_string())])
            .chain([Self::unary_network_builder(AvailableProcessors::ApplyTemplate, subject_name_out, &AvailableInterfaceSubjects::AssistantScript.to_string())])
            .reduce(|tabular_data_operator_network_builder, e| tabular_data_operator_network_builder.extend(e).unwrap())
            .unwrap();

        // Generate text network
        let generate_text_network = GenerateTextNetworkBuilder::default();
        let network_builder = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_network.as_mermaid_flowchart(),
            false,
        ).unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_network.as_mermaid_erdiagram(),
            false,
            true,
        ).unwrap()
        .with_name(generate_text_network.network_name);
        let tabular_data_operator_network_builder = tabular_data_operator_network_builder.extend(network_builder).unwrap();

        // DM: Not needed so long as the `UserMessage` are `Replace`d after each execution
        // DM, todo!(): Need to update the UI to not assume `UserMessage` are `Extend`ed
        // // Task response network
        // let subject_names = &[subject_name_out];
        // let task_response_network = TaskResponseNetworkBuilder::new("task_response_network", subject_names);
        // let network_builder = NetworkBuilder::from_mermaid_flowchart(
        //     &task_response_network.as_mermaid_flowchart(),
        //     false,
        // ).unwrap()
        // .with_subjects_from_mermaid_erdiagram(
        //     &task_response_network.as_mermaid_erdiagram(),
        //     false,
        //     true,
        // ).unwrap()
        // .with_name(task_response_network.network_name);
        // let tabular_data_operator_network_builder = tabular_data_operator_network_builder.extend(network_builder).unwrap();

        // Attachment aggregation for the UI
        let subject_names = [
            AvailableInterfaceSubjects::UserCsv.to_string(),
            AvailableInterfaceSubjects::AssistantCsv.to_string(),
            AvailableInterfaceSubjects::AssistantScript.to_string(),
        ];
        let binding = subject_names.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        let attachments_network = AttachmentsNetworkBuilder::new("attachments_network", &binding);
        let network_builder = attachments_network.build().with_name(attachments_network.network_name);
        let mut tabular_data_operator_network_builder = tabular_data_operator_network_builder.extend(network_builder).unwrap();

        // Add the available tool subjects
        let tool_ids = unary_operators.into_iter()
            .chain(binary_operators)
            .chain([AvailableProcessors::ExtractTabular, AvailableProcessors::PackTabular, AvailableProcessors::ApplyTemplate])
            .map(|p| p.to_string())
            .collect::<Vec<_>>();
        let tools = unary_operators.into_iter()
            .chain(binary_operators)
            .chain([AvailableProcessors::ExtractTabular, AvailableProcessors::PackTabular, AvailableProcessors::ApplyTemplate])
            .map(|p| p.to_json_tool_schema())
            .collect::<Vec<_>>();
        let batch = create_tools_record_batch(tool_ids, tools).unwrap();
        let subject_plan = AvailableSubjects::Tools.to_subject_plan(None, Some(vec![batch])).unwrap();

        // Add the LHS, RHS, OUT subjects
        let subject_plans = [subject_name_lhs, subject_name_rhs, subject_name_out].into_iter()
            .map(|s| AvailableSubjects::None
                .to_subject_plan(Some(s), None)
                .unwrap())
            .chain([
                AvailableInterfaceSubjects::UserCsv.to_subject_plan(None, None).unwrap(),
                AvailableInterfaceSubjects::AssistantCsv.to_subject_plan(None, None).unwrap(),
                AvailableInterfaceSubjects::AssistantScript.to_subject_plan(None, None).unwrap(),
            ])
            .collect::<Vec<_>>();

        let subjects = tabular_data_operator_network_builder.subjects
            .take()
            .unwrap()
            .into_iter()
            .filter(|s| s.get_name() != AvailableSubjects::Tools.to_string().as_str())
            .chain(subject_plans)
            .chain([subject_plan])
            .collect::<Vec<_>>();
        let tabular_data_operator_network_builder = tabular_data_operator_network_builder.with_subjects(subjects);

        TabularDataOperatorNetworkBuilder {
            inner: Some(tabular_data_operator_network_builder.with_name("tabular_data_operator_network")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_diagnostics::{HashMap, create_timestamp_micros};
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{DynamicTaskNetworkNames, NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream};
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjectsTrait, create_attachments_batch, create_chat_record_batch,
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait, test_subject,
    };
    use phymes_task::SubscriptionTrait;

use super::*;

    #[tokio::test]
    async fn test_tabular_data_operator_network() -> Result<()> {
        // Initialize the session
        let tabular_data_operator_network_builder = TabularDataOperatorNetworkBuilder::default().inner.take().unwrap();
        let network_name = tabular_data_operator_network_builder.name.clone().unwrap();
        let (network, session_messages) = tabular_data_operator_network_builder
            .with_runtime_env(
                RuntimeEnv::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_max_steps(20)
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test session data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Make the CSV attachments
        let bytes = test_subject::make_test_subject("test", 128, 0, 1)?.to_csv(b',', true)?;
        let filename = vec!["table".to_string()];
        let extension = vec!["csv".to_string()];
        let bytes = vec![bytes];
        let metadata = vec!["user".to_string()];
        let timestamp = vec![0_i64];
        let batch = create_attachments_batch(filename, extension, bytes, metadata, timestamp)?;
        let blob = AvailableInterfaceSubjects::UserCsv
            .to_subject_builder(None)
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            blob.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(blob.get_name())
                .with_subject(blob.get_name())
                .with_update(&Publication::Replace {
                    subject_name: blob.get_name().to_string(),
                })
                .with_message(blob.to_ipc_stream()?)
                .with_publisher(network_arc.get_name())
                .make_name()?
                .build()?,
        );

        // 1. Make the extraction query
        let role = vec!["user".to_string()];
        let content =vec!["ExtractTabular with lhs_name UserCsv, lhs_values bytes, format CsvDefault, encoding None, and schema None.".to_string()];
        let timestamp = vec![create_timestamp_micros()];
        let batch = create_chat_record_batch(role, content, timestamp)?;
        let queries = AvailableInterfaceSubjects::UserMessages
            .to_subject_builder(None)
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            queries.get_name().to_string(),
            IPCMessage::get_builder()
                .with_message(queries.to_ipc_stream()?)
                .with_subject(queries.get_name())
                .with_update(&Publication::Replace { // was Extend
                    subject_name: queries.get_name().to_string(),
                })
                .with_publisher(network_arc.get_name())
                .make_name()?
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // 1. Extract the CSV files
            let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            assert_eq!(response.len(), 0);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableProcessors::ExtractTabular.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(&AvailableProcessors::ExtractTabular.to_string())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "left_hand_side_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("left_hand_side_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 128);
            let column = subject.get_column_as_vec_primitive::<i64>("id")?;
            assert_eq!(column.first().unwrap(), &0);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "right_hand_side_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert!(batches.is_empty());

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "out_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert!(batches.is_empty());

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert!(batches.is_empty());
            // let subject = Subject::get_builder()
            //     .with_name(
            //         AvailableInterfaceSubjects::ToolMessages
            //             .to_string()
            //             .as_str(),
            //     )
            //     .with_record_batches(batches)?
            //     .build()?;
            // assert_eq!(subject.count_rows(), 1);
            // let column = subject.get_column_as_vec_str("role");
            // assert_eq!(column.first().unwrap(), &"tool");
            // assert_eq!(column.last().unwrap(), &"tool");
            // let column = subject.get_column_as_vec_str("content");
            // dbg!(column.first().unwrap());
            // dbg!(column.last().unwrap());
            // let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            // for t in column {
            //     assert!(t > 0);
            // }

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            assert!(batches.is_empty());
            // let subject = Subject::get_builder()
            //     .with_name(
            //         AvailableInterfaceSubjects::AssistantMessages
            //             .to_string()
            //             .as_str(),
            //     )
            //     .with_record_batches(batches)?
            //     .build()?;
            // assert_eq!(subject.count_rows(), 1);
            // let column = subject.get_column_as_vec_str("role");
            // assert_eq!(column.first().unwrap(), &"assistant");
            // assert_eq!(column.last().unwrap(), &"assistant");
            // let column = subject.get_column_as_vec_str("content");
            // dbg!(column.first().unwrap());
            // dbg!(column.last().unwrap());
            // let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            // for t in column {
            //     assert!(t > 0);
            // }

            // 2. Make the sort query
            let mut message_map = HashMap::<String, IPCMessage>::new();
            let role = vec!["user".to_string()];
            let content =vec!["Sort with lhs_name left_hand_side_s, lhs_values id, and asc false".to_string()];
            let timestamp = vec![create_timestamp_micros()];
            let batch = create_chat_record_batch(role, content, timestamp)?;
            let queries = AvailableInterfaceSubjects::UserMessages
                .to_subject_builder(None)
                .with_record_batches(vec![batch])?
                .build()?;
            let _ = message_map.insert(
                queries.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_message(queries.to_ipc_stream()?)
                    .with_subject(queries.get_name())
                    .with_update(&Publication::Replace { // was Extend
                        subject_name: queries.get_name().to_string(),
                    })
                    .with_publisher(network_arc.get_name())
                    .make_name()?
                    .build()?,
            );

            // 2. Sort
            let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            assert_eq!(response.len(), 0);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableProcessors::Sort.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(&AvailableProcessors::Sort.to_string())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: "out_s".to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name("out_s")
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 128);
            let column = subject.get_column_as_vec_primitive::<i64>("id")?;
            assert_eq!(column.first().unwrap(), &127);

            // 3. Make the sort query
            let mut message_map = HashMap::<String, IPCMessage>::new();
            let role = vec!["user".to_string()];
            let content =vec!["PackTabular with lhs_name out_s, encoding None, format CsvDefault, schema Attachments, and doc_name Out".to_string()];
            let timestamp = vec![create_timestamp_micros()];
            let batch = create_chat_record_batch(role, content, timestamp)?;
            let queries = AvailableInterfaceSubjects::UserMessages
                .to_subject_builder(None)
                .with_record_batches(vec![batch])?
                .build()?;
            let _ = message_map.insert(
                queries.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_message(queries.to_ipc_stream()?)
                    .with_subject(queries.get_name())
                    .with_update(&Publication::Replace { // was Extend
                        subject_name: queries.get_name().to_string(),
                    })
                    .with_publisher(network_arc.get_name())
                    .make_name()?
                    .build()?,
            );

            // 3. PackTabular
            let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            // let extended_diagnostic_subjects = extended_diagnostic_subjects();
            // let subject_names = extended_diagnostic_subjects
            //     .iter()
            //     .map(|s| s.as_str())
            //     .chain(["left_hand_side_s", "right_hand_side_s", "out_s", "generate_text_inference_s"])
            //     .collect::<Vec<_>>();
            // write_diagnostic_subjects_to_csv(
            //     &subject_names,
            //     network_arc.runtime_env(),
            //     network_arc.get_name(),
            // )
            // .await?;

            assert_eq!(response.len(), 0);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableProcessors::PackTabular.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(&AvailableProcessors::PackTabular.to_string())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserCsv.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserCsv.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);

            // 4. Extract a new table

            // 5. Join

            // 6. ...
        }

        Ok(())
    }
}
