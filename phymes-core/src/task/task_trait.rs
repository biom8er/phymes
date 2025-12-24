use std::sync::Arc;

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use parking_lot::{Mutex, RwLock};
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, TraceBuilderTrait};
use tracing::{Level, event};

use crate::{BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, PublishAndSubscribeTrait, RunnableTrait, RuntimeEnv, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, TablePublication, TableSubscription, TaskBuilder};

/// Trait to implement the actual task which could involve one or
///   more operators over [`RecordBatch`]s often originating from
///   structs implementing the [`TableTrait`].
///
/// [`TableTrait`]: crate::table::TableTrait
///
/// The trait allows for the schema of the data to change (e.g. after joins),
///   but the logic must be implemented by the user
/// The trait allows for tasks to have access to local data
///
/// # Example: Chaining
///
/// Result = Task (t1) -> Task (t2) -> Task (t3)
/// where t3 represents the leaf node in the computation and t1 and t2 represent
///   intermediate nodes in the computation tree. The `run` method of [`RunnableTrait`] of
///   t3 will most likely just produce a stream of its underlying [`RecordBatch`]s
///   while t1 and t2 will operate over incoming streams of [`RecordBatch`]s
///
/// Chaining use cases would include RAG, database query, etc.
///
/// # Example: Directed Cyclic Graph
///
/// Result = Task (t1) -> Or(Task (t2) -> Task (t1), Task(t3) -> Task (t1), End)
/// where t1 can call one or more tasks which that run a task and return the results to t1
///    or stop the loop when a criteria is reached
///
/// DCG use cases would include an agentic AI application, etc.
///
/// # Example: Parallel execution
///
/// Result = Apply Task (t1) over (Taable (d1), Taable (d2), Taable (d3), ...)
/// where the same task is run over different ArrowTables in parallel. The results can then
///    be collected is a single stream per table using [`collect_partitions_runs`] or
///    as a single stream using [`collect_task_runs`]
///
/// [`collect_partitions_runs`]: super::test_exec::collect_partitions_runs
/// [`collect_task_runs`]: super::test_exec::collect_task_runs
///
/// Parallel execution could be integrated into any uses case to improve execution speed
pub trait TaskTrait:
    MappableTrait + BuildableTrait + RunnableTrait + PublishAndSubscribeTrait + Sync + Send
{
    /// Make the outbox
    ///
    /// # Note
    ///
    /// A unique name to protect against collisions when building
    ///   the final message map
    fn make_outbox(
        &self,
        outbox: SendableRecordBatchStreamMessageMap,
    ) -> SendableRecordBatchStreamMessageMap {
        let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for (name, message) in outbox.into_iter() {
            let publications = self.get_publications();
            let update = publications
                .iter()
                .filter(|p| p.get_table_name() == message.get_subject())
                .collect::<Vec<_>>();

            // Skip messages that are not in the publications
            if update.is_empty() {
                event!(
                    Level::ERROR,
                    "No publications found for message {} on {} from {} during {}",
                    &name,
                    message.get_subject(),
                    message.get_publisher(),
                    self.get_name()
                );
                continue;
            }

            // Build the output message
            let out = SendableRecordBatchStreamMessage::get_builder()
                // .with_name(name.as_str())
                .with_publisher(self.get_name())
                .with_subject(message.get_subject())
                .with_update(update.first().unwrap())
                .with_message(message.get_message_own())
                .make_name()
                .unwrap()
                .build()
                .unwrap();
            let _ = map.insert(out.get_name().to_string(), out);
        }
        map
    }

    /// Get an immutable reference to the processors
    fn get_processors(&self) -> &Vec<Arc<dyn ProcessorTrait>>;

    /// Get an immutable reference to the runtime env
    fn get_runtime_env(&self) -> &Arc<Mutex<RuntimeEnv>>;
}

/// The actual task to execute
#[derive(Default, Debug)]
pub struct Task {
    /// Name of the task
    pub(crate) name: String,
    /// Runtime environment for the task and processors
    pub(crate) runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Entry processor
    pub(crate) processor: Vec<Arc<dyn ProcessorTrait>>,
}

impl MappableTrait for Task {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for Task {
    type T = TaskBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl RunnableTrait for Task {
    fn run(
        &self,
        mut messages: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Running task {}", self.get_name());
        // Trace the inbox
        let trace = if let Some(diagnostic_builder) = diagnostic_builder {
            let trace_builder = diagnostic_builder.clone().to_child(self.get_name())?;
            let trace = trace_builder
                .clone()
                .messages(line!(), file!(), self.get_name());
            trace.enter(&messages.values().collect::<Vec<_>>());
            Some((trace, trace_builder))
        } else {
            None
        };

        // Process the incoming message resulting in a `SendableRecordBatchStream`
        for processor in self.processor.iter() {
            let processor_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
            messages = processor.process(
                messages,
                processor_diagnostic_builder.as_ref(),
                self.runtime_env.clone(),
            )?;
        }

        // make the output message
        let outbox = self.make_outbox(messages);

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&outbox.values().collect::<Vec<_>>());
        }
        Ok(outbox)
    }
}

impl TaskTrait for Task {
    fn get_runtime_env(&self) -> &Arc<Mutex<RuntimeEnv>> {
        &self.runtime_env
    }
    fn get_processors(&self) -> &Vec<Arc<dyn ProcessorTrait>> {
        &self.processor
    }
}

impl PublishAndSubscribeTrait for Task {
    fn get_subscriptions(&self) -> Vec<&TableSubscription> {
        self.get_processors()
            .iter()
            .flat_map(|p| p.get_subscriptions())
            .collect::<Vec<&TableSubscription>>()
    }
    fn get_publications(&self) -> Vec<&TablePublication> {
        self.get_processors()
            .iter()
            .flat_map(|p| p.get_publications())
            .collect::<Vec<&TablePublication>>()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        for processor in self.get_processors() {
            if !processor.check_subscriptions(updates, state) {
                return false;
            }
        }
        true
    }
}

/// Mock objects and functions for task testing
pub mod test_task {
    use super::*;
    use crate::{AvailableTableSubscribePolicies, BuildableTrait, BuilderTrait, IPCMessage, IPCMessageBuilder, IPCMessageMap, MappableTrait, MessageBuilderTrait, ProcessorBuilder, RuntimeEnv, RuntimeEnvTrait, Table, TableBuilder, TableBuilderTrait, TablePublication, TableTrait, TaskBuilderTrait, test_processor::ProcessorMock, test_table::{make_test_table, make_test_table_chat}};

    use arrow::array::{ArrayRef, StringArray, UInt16Array, UInt32Array};
    use phymes_diagnostics::HashMap;
    use std::sync::Arc;

    pub fn make_state_tables(table_name: &str, config_name: &str) -> Result<Vec<Table>> {
        // mock config for the task
        let a: ArrayRef = Arc::new(StringArray::from(vec!["a".to_string()]));
        let b: ArrayRef = Arc::new(UInt32Array::from(vec![1]));
        let c: ArrayRef = Arc::new(UInt16Array::from(vec![1]));
        let batch = RecordBatch::try_from_iter(vec![("a", a), ("b", b), ("c", c)])?;
        let config = TableBuilder::new()
            .with_name(config_name)
            .with_record_batches(vec![batch])?
            .build()?;

        // mock table for the task
        let table = make_test_table(table_name, 4, 8, 3)?;
        Ok(vec![config, table])
    }

    pub fn make_state_tables_empty(table_name: &str, config_name: &str) -> Result<Vec<Table>> {
        // mock config for the task
        let a: ArrayRef = Arc::new(StringArray::from(vec!["".to_string()]));
        let b: ArrayRef = Arc::new(UInt32Array::from(vec![0]));
        let c: ArrayRef = Arc::new(UInt16Array::from(vec![0]));
        let batch = RecordBatch::try_from_iter(vec![("a", a), ("b", b), ("c", c)])?;
        let config = TableBuilder::new()
            .with_name(config_name)
            .with_record_batches(vec![batch])?
            .build()?;

        // mock table for the task
        let table = make_test_table(table_name, 1, 8, 1)?;
        Ok(vec![config, table])
    }

    pub fn make_state(table_name: &str, config_name: &str) -> Result<StateMap> {
        let tables = make_state_tables(table_name, config_name)?;

        // add mock config and table to the state
        let mut state = HashMap::<String, Arc<RwLock<Table>>>::new();
        for table in tables.into_iter() {
            state.insert(table.get_name().to_string(), Arc::new(RwLock::new(table)));
        }
        Ok(state)
    }

    pub fn make_state_updates(table_names: &[&str], updates: &[bool]) -> HashMap<String, bool> {
        let mut updated = HashMap::<String, bool>::new();
        for (i, table_name) in table_names.iter().enumerate() {
            let _ = updated.insert(table_name.to_string(), *updates.get(i).unwrap());
        }
        updated
    }

    pub fn make_runtime_env(name: &str) -> Result<RuntimeEnv> {
        let rt = RuntimeEnv::new().with_name(name);
        Ok(rt)
    }

    pub fn make_test_task_single_processor(
        name: &str,
        runtime_env_name: &str,
        table_name: &str,
        config_name: &str,
    ) -> Result<Task> {
        let processor_name = format!("{name}_processor");
        Task::get_builder()
            .with_name(name)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_processor(vec![
                ProcessorBuilder::default()
                    .with_name(processor_name.as_str())
                    .with_type(ProcessorMock::get_static_name())
                    .with_publications(&[TablePublication::Extend {
                        table_name: table_name.to_string(),
                    }])
                    .with_subscriptions(&[
                        TableSubscription::OnUpdateFullTable {
                            table_name: table_name.to_string(),
                        },
                        TableSubscription::AlwaysFullTable {
                            table_name: config_name.to_string(),
                        },
                    ])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build_arc::<ProcessorMock>()?,
            ])
            .build()
    }

    pub fn make_test_task_multiple_subscriptions(
        name: &str,
        runtime_env_name: &str,
        table_name_1: &str,
        table_name_2: &str,
        config_name: &str,
    ) -> Result<Task> {
        let processor_name = format!("{name}_processor");
        Task::get_builder()
            .with_name(name)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_processor(vec![
                ProcessorBuilder::default()
                    .with_name(processor_name.as_str())
                    .with_type(ProcessorMock::get_static_name())
                    .with_publications(&[TablePublication::Extend {
                        table_name: table_name_1.to_string(),
                    }])
                    .with_subscriptions(&[
                        TableSubscription::OnUpdateFullTable {
                            table_name: table_name_1.to_string(),
                        },
                        TableSubscription::OnUpdateFullTable {
                            table_name: table_name_2.to_string(),
                        },
                        TableSubscription::AlwaysFullTable {
                            table_name: config_name.to_string(),
                        },
                    ])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build_arc::<ProcessorMock>()?,
            ])
            .build()
    }

    pub fn make_test_task_chained_processor(
        name: &str,
        runtime_env_name: &str,
        table_name: &str,
        config_name: &str,
    ) -> Result<Task> {
        let processor_name_1 = format!("{name}_processor_1");
        let processor_name_2 = format!("{name}_processor_2");
        let processor_name_3 = format!("{name}_processor_3");
        Task::get_builder()
            .with_name(name)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_processor(vec![
                ProcessorBuilder::default()
                    .with_name(processor_name_1.as_str())
                    .with_type(ProcessorMock::get_static_name())
                    .with_publications(&[TablePublication::Extend {
                        table_name: table_name.to_string(),
                    }])
                    .with_subscriptions(&[
                        TableSubscription::OnUpdateFullTable {
                            table_name: table_name.to_string(),
                        },
                        TableSubscription::AlwaysFullTable {
                            table_name: config_name.to_string(),
                        },
                    ])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build_arc::<ProcessorMock>()?,
                ProcessorBuilder::default()
                    .with_name(processor_name_2.as_str())
                    .with_type(ProcessorMock::get_static_name())
                    .with_publications(&[])
                    .with_subscriptions(&[])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build_arc::<ProcessorMock>()?,
                ProcessorBuilder::default()
                    .with_name(processor_name_3.as_str())
                    .with_type(ProcessorMock::get_static_name())
                    .with_publications(&[])
                    .with_subscriptions(&[])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build_arc::<ProcessorMock>()?,
            ])
            .build()
    }

    pub fn make_test_input_message(
        name: &str,
        publisher: &str,
        subject: &str,
        table_name: &str,
        update: &TablePublication,
        test_table: bool,
    ) -> Result<IPCMessageMap> {
        // mock table as input
        let table = if test_table {
            make_test_table(table_name, 4, 8, 3)?
        } else {
            make_test_table_chat(table_name)?
        };

        // build the message
        let message = IPCMessageBuilder::new()
            .with_name(name)
            .with_subject(subject)
            .with_publisher(publisher)
            .with_message(table.to_ipc_stream()?)
            .with_update(update)
            .build()?;

        // finish the message map
        let mut map = HashMap::<String, IPCMessage>::new();
        map.insert(message.get_name().to_string(), message);
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{remove_message_by_subject, TableBuilder, TableBuilderTrait, TablePublication, TableTrait, test_table::make_test_table};
    use crate::MessageTrait;
    use phymes_diagnostics::{Diagnostics, HashMap, SpanBuilder};

    /// A compilation test to ensure that the `Task::get_name()` method can
    /// be called from a trait object.
    #[allow(dead_code)]
    fn use_task_name_as_trait_object(plan: &dyn TaskTrait<T = TaskBuilder>) {
        let _ = plan.get_name();
    }

    #[test]
    fn test_get_subscriptions_from_state() -> Result<()> {
        // Single processor with All logic
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
        )?;
        let mut messages = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(
            remove_message_by_subject("test_table", &mut messages)
                .unwrap()
                .get_subject(),
            "test_table"
        );
        assert_eq!(
            remove_message_by_subject("test_config", &mut messages)
                .unwrap()
                .get_subject(),
            "test_config"
        );

        // Multiple processors with no OnUpdates
        let test_task = test_task::make_test_task_multiple_subscriptions(
            "test_task",
            "test_rt",
            "test_table",
            "test_table_2",
            "test_config",
        )?;
        let mut messages = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[false]),
            &test_task::make_state("test_table", "test_config")?,
        );
        assert_eq!(messages.len(), 1);
        assert_eq!(
            remove_message_by_subject("test_config", &mut messages)
                .unwrap()
                .get_subject(),
            "test_config"
        );

        // Multiple processors with one OnUpdates
        let test_task = test_task::make_test_task_multiple_subscriptions(
            "test_task",
            "test_rt",
            "test_table",
            "test_table_2",
            "test_config",
        )?;
        let mut messages = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(
            remove_message_by_subject("test_table", &mut messages)
                .unwrap()
                .get_subject(),
            "test_table"
        );
        assert_eq!(
            remove_message_by_subject("test_config", &mut messages)
                .unwrap()
                .get_subject(),
            "test_config"
        );
        Ok(())
    }

    #[test]
    fn test_run_task_make_outbox() -> Result<()> {
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
        )?;

        // Case 1: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_publisher("s1")
            .with_subject("d1")
            .with_update(&TablePublication::Extend {
                table_name: "d1".to_string(),
            })
            .with_message(make_test_table("d1", 1, 8, 2)?.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let inbox = test_task.make_outbox(messages);
        assert_eq!(inbox.len(), 0);

        // Case 2: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_publisher("s1")
            .with_subject("test_table")
            .with_update(&TablePublication::Extend {
                table_name: "test_table".to_string(),
            })
            .with_message(make_test_table("test_table", 1, 8, 2)?.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let inbox = test_task.make_outbox(messages);
        assert_eq!(inbox.len(), 1);
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            "test_table"
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            TablePublication::Extend {
                table_name: "test_table".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_single_processor() -> Result<()> {
        let span = SpanBuilder::default()
            .with_span("test_run_task_single_processor")
            .build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
        )?;
        let input = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        let mut response = test_task.run(input, Some(&diagnostic_builder))?;
        assert_eq!(response.len(), 1);
        assert!(response.get("from_test_task_on_test_table").is_some());
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            "test_table"
        );
        let stream = response.remove("from_test_task_on_test_table").unwrap();
        let partitions =
            TableBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
                .await?
                .with_name("")
                .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 15); // 3 * (4 + 1) from input + 1 added to each batch
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_chained_processor() -> Result<()> {
        let span = SpanBuilder::default()
            .with_span("test_run_task_chained_processor")
            .build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let test_task = test_task::make_test_task_chained_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
        )?;
        let input = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        let mut response = test_task.run(input, Some(&diagnostic_builder))?;
        assert_eq!(response.len(), 1);
        assert!(response.get("from_test_task_on_test_table").is_some());
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            "test_table"
        );
        let stream = response.remove("from_test_task_on_test_table").unwrap();
        let partitions =
            TableBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
                .await?
                .with_name("")
                .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 21);
        Ok(())
    }
}
