use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::record_batch::RecordBatch;
use parking_lot::{Mutex, RwLock};
use tracing::{Level, event};

use super::{
    message::{
        MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    },
    processor::ProcessorTrait,
    publish_subscribe::PubSubTrait,
};

// Required for documentation
#[allow(unused_imports)]
use super::test_exec::{collect_partitions_runs, collect_task_runs};

// Required for documentation
#[allow(unused_imports)]
use crate::metrics::Metric;

use crate::{
    metrics::{create_random_id, HashMap, MetricBuilder, MetricsSet, SpanMetricsSet},
    session::{
        common_traits::{
            BuildableTrait, BuilderTrait, MappableTrait, RunnableTrait, SendableRecordBatchStreamMessageMap, StateMap
        },
        runtime_env::RuntimeEnv,
    },
    table::{table_publish::TablePublish, table_subscribe::TableSubscribe},
};

/// Trait to implement the actual task which could involve one or
///   more operators over [`RecordBatch`]s often originating from
///   structs implementing the [`TableTrait`].
///
/// [`TableTrait`]: crate::table::table_trait::TableTrait
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
/// Parallel execution could be integrated into any uses case to improve execution speed
pub trait TaskTrait:
    MappableTrait + BuildableTrait + RunnableTrait + PubSubTrait + Sync + Send
{
    /// Make the outbox
    ///
    /// # Note
    ///
    /// A unique name to protect against collisions when building
    ///   the final message map
    fn make_outbox(&self, outbox: SendableRecordBatchStreamMessageMap) -> SendableRecordBatchStreamMessageMap {
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

    /// Return a snapshot of the set of [`Metric`]s for this
    /// [`Task`]. If no [`Metric`]s are available, return None.
    ///
    /// While the values of the metrics in the returned
    /// [`MetricsSet`]s may change as execution progresses, the
    /// specific metrics will not.
    ///
    /// Once `self.run_task()` has returned (technically the future is
    /// resolved) for all available partitions, the set of metrics
    /// should be complete. If this processortion is called prior to
    /// `run_task()` new metrics may appear in subsequent calls.
    ///
    /// self.metrics.clone_inner()
    ///
    fn get_metrics(&self) -> MetricsSet; //{ self.metrics.clone_inner() }
}

/// The actual task to execute
#[derive(Default, Debug)]
pub struct Task {
    /// Name of the task
    name: String,
    /// Metrics for the task and processors
    metrics: SpanMetricsSet,
    /// Runtime environment for the task and processors
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Entry processor
    processor: Vec<Arc<dyn ProcessorTrait>>,
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
    fn run(&self, mut messages: SendableRecordBatchStreamMessageMap, metrics_builder: &MetricBuilder) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Running task {}", self.get_name());
        let span_id = create_random_id()?;

        // Process the incoming message resulting in a `SendableRecordBatchStream`
        for processor in self.processor.iter() {            
            messages = processor.process(
                messages, 
                &metrics_builder.clone().to_child().with_span(self.get_name(), span_id), 
                self.runtime_env.clone())?;
        }

        // make the output message
        let outbox = self.make_outbox(messages);
        Ok(outbox)
    }
}

impl TaskTrait for Task {
    fn get_runtime_env(&self) -> &Arc<Mutex<RuntimeEnv>> {
        &self.runtime_env
    }
    fn get_metrics(&self) -> MetricsSet {
        self.metrics.clone_inner()
    }
    fn get_processors(&self) -> &Vec<Arc<dyn ProcessorTrait>> {
        &self.processor
    }
}

impl PubSubTrait for Task {
    fn get_subscriptions(&self) -> Vec<&TableSubscribe> {
        self.get_processors()
            .iter()
            .flat_map(|p| p.get_subscriptions())
            .collect::<Vec<&TableSubscribe>>()
    }
    fn get_publications(&self) -> Vec<&TablePublish> {
        self.get_processors()
            .iter()
            .flat_map(|p| p.get_publications())
            .collect::<Vec<&TablePublish>>()
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

pub trait TaskBuilderTrait: BuilderTrait {
    fn with_metrics(self, metrics: SpanMetricsSet) -> Self;
    fn with_runtime_env(self, runtime_env: Arc<Mutex<RuntimeEnv>>) -> Self;
    fn with_processor(self, processor: Vec<Arc<dyn ProcessorTrait>>) -> Self;
}

#[derive(Default)]
pub struct TaskBuilder {
    /// Task name
    pub name: Option<String>,
    /// Metrics for the task
    pub metrics: Option<SpanMetricsSet>,
    /// Runtime environment for the task
    pub runtime_env: Option<Arc<Mutex<RuntimeEnv>>>,
    /// Function that implements the logic
    pub processor: Option<Vec<Arc<dyn ProcessorTrait>>>,
}

impl BuilderTrait for TaskBuilder {
    type T = Task;
    fn new() -> Self {
        Self {
            name: None,
            metrics: None,
            runtime_env: None,
            processor: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            metrics: self.metrics.unwrap_or_default(),
            runtime_env: self.runtime_env.unwrap(),
            processor: self.processor.unwrap(),
        })
    }
}

impl TaskBuilderTrait for TaskBuilder {
    fn with_metrics(mut self, metrics: SpanMetricsSet) -> Self {
        self.metrics = Some(metrics);
        self
    }
    fn with_runtime_env(mut self, runtime_env: Arc<Mutex<RuntimeEnv>>) -> Self {
        self.runtime_env = Some(runtime_env);
        self
    }
    fn with_processor(mut self, processor: Vec<Arc<dyn ProcessorTrait>>) -> Self {
        self.processor = Some(processor);
        self
    }
}

/// Checks a `RecordBatch` for `not null` constraints on specified columns.
///
/// # Arguments
///
/// * `batch` - The `RecordBatch` to be checked
/// * `column_indices` - A vector of column indices that should be checked for
///   `not null` constraints.
///
/// # Returns
///
/// * `Result<RecordBatch>` - The original `RecordBatch` if all constraints are met
///
/// This processortion iterates over the specified column indices and ensures that none
/// of the columns contain null values. If any column contains null values, an error
/// is returned.
pub fn check_not_null_constraints(
    batch: RecordBatch,
    column_indices: &Vec<usize>,
) -> Result<RecordBatch> {
    for &index in column_indices {
        if batch.num_columns() <= index {
            return Err(anyhow!(
                "Invalid batch column count {} expected > {}",
                batch.num_columns(),
                index
            ));
        }

        if batch
            .column(index)
            .logical_nulls()
            .map(|nulls| nulls.null_count())
            .unwrap_or_default()
            > 0
        {
            return Err(anyhow!(
                "Invalid batch column at '{}' has null but schema specifies non-nullable",
                index
            ));
        }
    }

    Ok(batch)
}

/// Mock objects and functions for task testing
pub mod test_task {
    use super::*;
    use crate::{
        session::{
            common_traits::{BuildableTrait, BuilderTrait, IPCMessageMap, MappableTrait},
            runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        },
        table::{
            table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, SubscribeTrait}, table_trait::{
                test_table::{make_test_table, make_test_table_chat}, Table, TableBuilder, TableBuilderTrait, TableTrait
            }
        },
        task::{
            message::{
                IPCMessage, IPCMessageBuilder, MessageBuilderTrait,
            },
            processor::test_processor::ProcessorMock,
        },
    };

    use arrow::array::{ArrayRef, StringArray, UInt16Array, UInt32Array};
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
        metrics: SpanMetricsSet,
    ) -> Result<Task> {
        let processor_name = format!("{name}_processor");
        Task::get_builder()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_processor(vec![ProcessorMock::new_arc_with_pub_sub(
                processor_name.as_str(),
                &[TablePublish::Extend {
                    table_name: table_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: table_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: config_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            )])
            .build()
    }

    pub fn make_test_task_multiple_subscriptions(
        name: &str,
        runtime_env_name: &str,
        table_name_1: &str,
        table_name_2: &str,
        config_name: &str,
        metrics: SpanMetricsSet,
    ) -> Result<Task> {
        let processor_name = format!("{name}_processor");
        Task::get_builder()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_processor(vec![ProcessorMock::new_arc_with_pub_sub(
                processor_name.as_str(),
                &[TablePublish::Extend {
                    table_name: table_name_1.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: table_name_1.to_string(),
                    },
                    TableSubscribe::OnUpdateFullTable {
                        table_name: table_name_2.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: config_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            )])
            .build()
    }

    pub fn make_test_task_chained_processor(
        name: &str,
        runtime_env_name: &str,
        table_name: &str,
        config_name: &str,
        metrics: SpanMetricsSet,
    ) -> Result<Task> {
        let processor_name_1 = format!("{name}_processor_1");
        let processor_name_2 = format!("{name}_processor_2");
        let processor_name_3 = format!("{name}_processor_3");
        Task::get_builder()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_processor(vec![
                ProcessorMock::new_arc_with_pub_sub(
                    processor_name_1.as_str(),
                    &[TablePublish::Extend {
                        table_name: table_name.to_string(),
                    }],
                    &[
                        TableSubscribe::OnUpdateFullTable {
                            table_name: table_name.to_string(),
                        },
                        TableSubscribe::AlwaysFullTable {
                            table_name: config_name.to_string(),
                        },
                    ],
                    AllTableNamesSubscribe::new_box(),
                ),
                ProcessorMock::new_arc(processor_name_2.as_str()),
                ProcessorMock::new_arc(processor_name_3.as_str()),
            ])
            .build()
    }

    pub fn make_test_input_message(
        name: &str,
        publisher: &str,
        subject: &str,
        table_name: &str,
        update: &TablePublish,
        test_table: bool
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
    use crate::table::table_trait::TableTrait;
    use crate::table::table_trait::test_table::make_test_table;
    use crate::table::{
        table_trait::{TableBuilder, TableBuilderTrait},
        table_publish::TablePublish,
    };
    use crate::task::message::MessageTrait;
    use arrow::array::{Array, DictionaryArray, Int32Array, NullArray, RunArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use hashbrown::HashMap;

    /// A compilation test to ensure that the `Task::get_name()` method can
    /// be called from a trait object.
    #[allow(dead_code)]
    fn use_task_name_as_trait_object(plan: &dyn TaskTrait<T = TaskBuilder>) {
        let _ = plan.get_name();
    }

    #[test]
    fn test_check_not_null_constraints_accept_non_null() -> Result<()> {
        check_not_null_constraints(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
                vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)]))],
            )?,
            &vec![0],
        )?;
        Ok(())
    }

    #[test]
    fn test_check_not_null_constraints_reject_null() -> Result<()> {
        let result = check_not_null_constraints(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
                vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
            )?,
            &vec![0],
        );
        assert!(result.is_err());
        // assert_eq!(
        //     result.err().unwrap().strip_backtrace(),
        //     "Execution error: Invalid batch column at '0' has null but schema specifies non-nullable",
        // );
        Ok(())
    }

    #[test]
    fn test_check_not_null_constraints_with_run_end_array() -> Result<()> {
        // some null value inside REE array
        let run_ends = Int32Array::from(vec![1, 2, 3, 4]);
        let values = Int32Array::from(vec![Some(0), None, Some(1), None]);
        let run_end_array = RunArray::try_new(&run_ends, &values)?;
        let result = check_not_null_constraints(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "a",
                    run_end_array.data_type().to_owned(),
                    true,
                )])),
                vec![Arc::new(run_end_array)],
            )?,
            &vec![0],
        );
        assert!(result.is_err());
        // assert_eq!(
        //     result.err().unwrap().strip_backtrace(),
        //     "Execution error: Invalid batch column at '0' has null but schema specifies non-nullable",
        // );
        Ok(())
    }

    #[test]
    fn test_check_not_null_constraints_with_dictionary_array_with_null() -> Result<()> {
        let values = Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)]));
        let keys = Int32Array::from(vec![0, 1, 2, 3]);
        let dictionary = DictionaryArray::new(keys, values);
        let result = check_not_null_constraints(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "a",
                    dictionary.data_type().to_owned(),
                    true,
                )])),
                vec![Arc::new(dictionary)],
            )?,
            &vec![0],
        );
        assert!(result.is_err());
        // assert_eq!(
        //     result.err().unwrap().strip_backtrace(),
        //     "Execution error: Invalid batch column at '0' has null but schema specifies non-nullable",
        // );
        Ok(())
    }

    #[test]
    fn test_check_not_null_constraints_with_dictionary_masking_null() -> Result<()> {
        // some null value marked out by dictionary array
        let values = Arc::new(Int32Array::from(vec![
            Some(1),
            None, // this null value is masked by dictionary keys
            Some(3),
            Some(4),
        ]));
        let keys = Int32Array::from(vec![0, /*1,*/ 2, 3]);
        let dictionary = DictionaryArray::new(keys, values);
        check_not_null_constraints(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "a",
                    dictionary.data_type().to_owned(),
                    true,
                )])),
                vec![Arc::new(dictionary)],
            )?,
            &vec![0],
        )?;
        Ok(())
    }

    #[test]
    fn test_check_not_null_constraints_on_null_type() -> Result<()> {
        // null value of Null type
        let result = check_not_null_constraints(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("a", DataType::Null, true)])),
                vec![Arc::new(NullArray::new(3))],
            )?,
            &vec![0],
        );
        assert!(result.is_err());
        // assert_eq!(
        //     result.err().unwrap().strip_backtrace(),
        //     "Execution error: Invalid batch column at '0' has null but schema specifies non-nullable",
        // );
        Ok(())
    }

    #[test]
    fn test_get_subscriptions_from_state() -> Result<()> {
        let metrics = SpanMetricsSet::new();

        // Single processor with All logic
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;
        let messages = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        assert_eq!(messages.len(), 2);
        assert!(messages.get("test_table").is_some());
        assert_eq!(
            messages.get("test_table").unwrap().get_subject(),
            "test_table"
        );
        assert!(messages.get("test_config").is_some());
        assert_eq!(
            messages.get("test_config").unwrap().get_subject(),
            "test_config"
        );

        // Multiple processors with no OnUpdates
        let test_task = test_task::make_test_task_multiple_subscriptions(
            "test_task",
            "test_rt",
            "test_table",
            "test_table_2",
            "test_config",
            metrics.clone(),
        )?;
        let messages = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[false]),
            &test_task::make_state("test_table", "test_config")?,
        );
        assert_eq!(messages.len(), 1);
        assert!(messages.get("test_config").is_some());
        assert_eq!(
            messages.get("test_config").unwrap().get_subject(),
            "test_config"
        );

        // Multiple processors with one OnUpdates
        let test_task = test_task::make_test_task_multiple_subscriptions(
            "test_task",
            "test_rt",
            "test_table",
            "test_table_2",
            "test_config",
            metrics.clone(),
        )?;
        let messages = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        assert_eq!(messages.len(), 2);
        assert!(messages.get("test_table").is_some());
        assert_eq!(
            messages.get("test_table").unwrap().get_subject(),
            "test_table"
        );
        assert!(messages.get("test_config").is_some());
        assert_eq!(
            messages.get("test_config").unwrap().get_subject(),
            "test_config"
        );
        Ok(())
    }

    #[test]
    fn test_run_task_make_outbox() -> Result<()> {
        let metrics = SpanMetricsSet::new();
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;

        // Case 1: Message has subject that the task does not publish on
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            "test_message".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("test_message")
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublish::Extend {
                    table_name: "test_table".to_string(),
                })
                .with_message(make_test_table("test_table", 1, 8, 2)?.to_record_batch_stream())
                .build()?,
        );
        let inbox = test_task.make_outbox(message);
        assert_eq!(inbox.len(), 0);

        // Case 2: Message has subject that the task does not publish on
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            "test_message".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("test_message")
                .with_publisher("s1")
                .with_subject("test_table")
                .with_update(&TablePublish::Extend {
                    table_name: "test_table".to_string(),
                })
                .with_message(make_test_table("test_table", 1, 8, 2)?.to_record_batch_stream())
                .build()?,
        );
        let inbox = test_task.make_outbox(message);
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
            TablePublish::Extend {
                table_name: "test_table".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_single_processor() -> Result<()> {
        let metrics = SpanMetricsSet::new();
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;
        let input = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        let mut response = test_task.run(input)?;
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
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 15);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_chained_processor() -> Result<()> {
        let metrics = SpanMetricsSet::new();
        let test_task = test_task::make_test_task_chained_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;
        let input = test_task.get_subscriptions_from_state(
            &test_task::make_state_updates(&["test_table"], &[true]),
            &test_task::make_state("test_table", "test_config")?,
        );
        let mut response = test_task.run(input)?;
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
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 54);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);
        Ok(())
    }
}
