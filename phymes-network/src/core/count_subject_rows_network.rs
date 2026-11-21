use std::collections::VecDeque;

use phymes_data::{AvailableOperators, DataAggregatorOperator, DataConfig, DataStreamManager};
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{AvailableSubjects, AvailableSubjectsTrait, create_network_tasks_subscribe_publish_batch};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait, SubjectTrait,
};
use crate::{DynamicTaskNetworkBuilder, DynamicTaskNetworkNames, DynamicTaskNetworkTypes, NetworkBuilder, core::PreCompiledNetworkTrait};

/// Network builder to count the number of rows per subject
/// 
/// # Notes
/// * Currently, only tracks the most recent number of rows written to the object store
pub struct CountSubjectRowsNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
    pub network_name: Option<String>,
}

impl Default for CountSubjectRowsNetworkBuilder {
    fn default() -> Self {
        // Count subject rows task
        let network_builder = {
            let task_name = "count_subject_rows";
            let mut tasks = VecDeque::new();
            {
                let network_name = "group_by_subjects_object_store_meta";
                let config = DataConfig {
                    lhs_name: Some(AvailableSubjects::SubjectsObjectStoreMeta.to_string()),
                    lhs_values: Some(vec![
                        "subject_name".to_string(),
                        "task_name".to_string(),
                        "network_name".to_string(),
                        "superstep".to_string()
                    ]),
                    agg_columns: Some(vec!["num_rows".to_string()]),
                    agg_operators: Some(vec![DataAggregatorOperator::Sum]),
                    cpu: false,
                    operator: AvailableOperators::GroupBy,
                    lhs_stream: DataStreamManager::Accumulate,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::GroupBy,
                    subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableSubjects::SubjectsObjectStoreMeta.to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "group_by_num_rows_subjects_object_store_meta";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),),
                    lhs_values: Some(vec![
                        "subject_name".to_string(),
                    ]),
                    agg_columns: Some(vec!["num_rows-Sum".to_string()]),
                    agg_operators: Some(vec![DataAggregatorOperator::Last]),
                    cpu: false,
                    operator: AvailableOperators::GroupBy,
                    lhs_stream: DataStreamManager::Accumulate,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::GroupBy,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "select_num_rows_subjects_object_store_meta";
                let cols = [
                    "subject_name",
                    "num_rows-Sum-Last",
                ];
                let schema_cols = [
                    "subject_name",
                    "num_rows",
                ];
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        cols
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    as_columns: Some(
                        schema_cols
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cpu: false,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let subject_out = AvailableSubjects::SubjectsNumRows
                    .to_subject_plan(None, None)
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    subscription_rhs: None,
                    publication: Publication::Replace {
                        subject_name: subject_out.get_name().to_string(),
                    },
                    subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
                    subject_out: Some(subject_out),
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            let mut network_builder = tasks.pop_front().unwrap().build_dynamic();
            while let Some(task) = tasks.pop_front() {
                network_builder = network_builder.extend(task.build_dynamic()).unwrap();
            }
            network_builder
        };

        let network_name = "count_subject_rows_network";
        Self {
            inner: Some(network_builder.with_name(network_name)),
            network_name: Some(network_name.to_string())
        }
    }
}

impl PreCompiledNetworkTrait for CountSubjectRowsNetworkBuilder {
    fn as_task_messages(&self) -> anyhow::Result<Vec<phymes_message::IPCMessageMap>> {
        // 1. Message to trigger the first superstep
        let task_names = vec![
                "count_subject_rows_t",
                "count_subject_rows_t",
                "count_subject_rows_t",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec![
                "group_by_subjects_object_store_meta_p",
                "group_by_num_rows_subjects_object_store_meta_p",
                "select_num_rows_subjects_object_store_meta_p"
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec![
                "GroupBy",
                "GroupBy",
                "Select",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
                vec!["OnUpdateAllRecordBatches", "AlwaysAllRecordBatches"],
                vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
                vec!["AlwaysAllRecordBatches", "AlwaysAllRecordBatches"],
            ]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let subscription_table_names = vec![
                vec!["SubjectsObjectStoreMeta", "group_by_subjects_object_store_meta_p"],
                vec!["group_by_subjects_object_store_meta_s", "group_by_num_rows_subjects_object_store_meta_p"],
                vec!["group_by_num_rows_subjects_object_store_meta_s", "select_num_rows_subjects_object_store_meta_p"],
            ]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_names = vec![
                vec!["Replace"],
                vec!["Replace"],
                vec!["Replace"],
            ]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![
                vec!["group_by_subjects_object_store_meta_s"],
                vec!["group_by_num_rows_subjects_object_store_meta_s"],
                vec!["SubjectsNumRows"],
            ]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let network_names = task_names
            .iter()
            .map(|_| self.network_name.clone().unwrap_or("count_subject_rows_network".to_string()))
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
        let subject = Subject::get_builder()
            .with_name(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(subject.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::NetworkTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&Publication::Replace {
                subject_name: AvailableSubjects::NetworkTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.network_name.clone().unwrap_or("count_subject_rows_network".to_string()).as_str())
            .make_name()?
            .build()?;
        let messages_1 = create_message_map(vec![tasks_publish_subscribe_message]);

        Ok(vec![messages_1])
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
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnvBuilder, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::{SubscriptionTrait, test_task};

    use crate::{NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream, test_network_builder};

    use super::*;

    #[tokio::test]
    async fn test_count_subject_rows_network() -> Result<()> {
        // Initialize the network
        let subjects_network_builder = CountSubjectRowsNetworkBuilder::default().inner.take().unwrap();
        let network_name = subjects_network_builder.name.clone().unwrap();
        let (network, network_messages) = subjects_network_builder
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test network data
        let mut message_map = {
            // Make the test sequential network
            let (network, network_messages) =
                test_network_builder::make_test_network_builder_sequential("network_1", 2)?
                    .with_diagnostics(false)
                    .add_network_interface(Some(&["state_1"]))?
                    .add_next_tasks()?
                    .add_next_supersteps()?
                    .add_subjects_num_rows()?
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
                subject_name: AvailableSubjects::SubjectsObjectStoreMeta.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SubjectsObjectStoreMeta.to_string().as_str())
                .with_record_batches(batches)
                .unwrap()
                .build()
                .unwrap();
            let subjects_change_log_message = IPCMessage::get_builder()
                .with_message(subject.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SubjectsObjectStoreMeta.to_string().as_str())
                .with_update(&Publication::Extend {
                    subject_name: AvailableSubjects::SubjectsObjectStoreMeta.to_string(),
                })
                .with_publisher(&network_name)
                .make_name()?
                .build()?;
            create_message_map(vec![subjects_change_log_message])
        };
        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;

        let mut tasks_publish_subscribe_messages = CountSubjectRowsNetworkBuilder::default()
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        // Run the network
        message_map.extend(tasks_publish_subscribe_messages.pop().unwrap());
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

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
                "NetworkTasksRunLog", "NetworkTasksSubscribePublish", "SubjectsNumRows", "SubjectsObjectStoreMeta", "group_by_num_rows_subjects_object_store_meta_p", "group_by_num_rows_subjects_object_store_meta_s", "group_by_subjects_object_store_meta_p", "group_by_subjects_object_store_meta_s", "processor_1", "processor_2", "processor_3", "select_num_rows_subjects_object_store_meta_p", "state_1"           
            ]
        );
        let column = subject.get_column_as_vec_primitive::<i64>("num_rows")?;
        assert_eq!(column, [3, 3, 12, 25, 0, 0, 0, 0, 0, 0, 0, 0, 21]);

        Ok(())
    }
}
