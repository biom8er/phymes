use std::sync::Arc;

use super::{
    arrow_message::{
        ArrowMessageBuilderTrait, ArrowMessageTrait, ArrowOutgoingMessage,
        ArrowOutgoingMessageBuilderTrait, ArrowOutgoingMessageTrait,
    },
    arrow_processor::ArrowProcessorTrait,
};

// Required for documentation
#[allow(unused_imports)]
use super::test_exec::{collect_partitions_runs, collect_task_runs};

// Required for documentation
#[allow(unused_imports)]
use crate::metrics::Metric;

use crate::metrics::{ArrowTaskMetricsSet, HashMap, MetricsSet};
use crate::session::{
    common_traits::{
        BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap, PubSubTrait,
        RunnableTrait, StateMap,
    },
    runtime_env::RuntimeEnv,
};
use crate::table::{
    arrow_table_publish::ArrowTablePublish,
    arrow_table_subscribe::{ArrowTableSubscribe, ArrowTableSubscribeTrait},
};

use anyhow::{Result, anyhow};
use arrow::record_batch::RecordBatch;
use parking_lot::{Mutex, RwLock};
use tracing::{Level, event};

/// Trait to implement the actual task which could involve one or
///   more operators over [`RecordBatch`]s often originating from
///   structs implementing the [`ArrowTableTrait`].
///
/// [`ArrowTableTrait`]: crate::table::arrow_table::ArrowTableTrait
///
/// The trait allows for the schema of the data to change (e.g. after joins),
///   but the logic must be implemented by the user
/// The trait allows for tasks to have access to local data
///
/// # Example: Chaining
///
/// Result = ArrowTask (t1) -> ArrowTask (t2) -> ArrowTask (t3)
/// where t3 represents the leaf node in the computation and t1 and t2 represent
///   intermediate nodes in the computation tree. The `run` method of [`RunnableTrait`] of
///   t3 will most likely just produce a stream of its underlying [`RecordBatch`]s
///   while t1 and t2 will operate over incoming streams of [`RecordBatch`]s
///
/// Chaining use cases would include RAG, database query, etc.
///
/// # Example: Directed Cyclic Graph
///
/// Result = ArrowTask (t1) -> Or(ArrowTask (t2) -> ArrowTask (t1), ArrowTask(t3) -> ArrowTask (t1), End)
/// where t1 can call one or more tasks which that run a task and return the results to t1
///    or stop the loop when a criteria is reached
///
/// DCG use cases would include an agentic AI application, etc.
///
/// # Example: Parallel execution
///
/// Result = Apply ArrowTask (t1) over (ArrowTable (d1), ArrowTable (d2), ArrowTable (d3), ...)
/// where the same task is run over different ArrowTables in parallel. The results can then
///    be collected is a single stream per table using [`collect_partitions_runs`] or
///    as a single stream using [`collect_task_runs`]
///
/// Parallel execution could be integrated into any uses case to improve execution speed
pub trait ArrowTaskTrait:
    MappableTrait + BuildableTrait + RunnableTrait + PubSubTrait + Sync + Send
{
    /// Short name for the ArrowTask, such as 'AddRows'.
    /// Like [`get_name`](ArrowTask::get_name) but can be called without an instance.
    fn get_static_name() -> &'static str
    where
        Self: Sized,
    {
        let full_name = std::any::type_name::<Self>();
        let maybe_start_idx = full_name.rfind(':');
        match maybe_start_idx {
            Some(start_idx) => &full_name[start_idx + 1..],
            None => "UNKNOWN",
        }
    }

    /// Get subscriptions from the state
    fn get_subscriptions_from_state(&self, state: &StateMap) -> OutgoingMessageMap {
        let mut map = HashMap::<String, ArrowOutgoingMessage>::new();
        for subscription in self.get_subscriptions().iter() {
            let table = state.get(subscription.get_table_name()).unwrap();
            if let Some(message) = table.try_read().unwrap().subscribe_table(subscription) {
                let update = self
                    .get_publications()
                    .iter()
                    .filter(|p| p.get_table_name() == subscription.get_table_name())
                    .collect::<Vec<_>>();
                let update = match update.first() {
                    Some(u) => u,
                    None => &ArrowTablePublish::None,
                };
                let out = ArrowOutgoingMessage::get_builder()
                    .with_name(subscription.get_table_name())
                    .with_publisher("State")
                    .with_subject(subscription.get_table_name())
                    .with_update(update)
                    .with_message(message)
                    .build()
                    .unwrap();
                let _ = map.insert(subscription.get_table_name().to_string(), out);
            }
        }
        map
    }

    /// Make the outbox
    ///
    /// # Note
    ///
    /// A unique name to protect against collisions when building
    ///   the final message map
    fn make_outbox(&self, outbox: OutgoingMessageMap) -> OutgoingMessageMap {
        let mut map = HashMap::<String, ArrowOutgoingMessage>::new();
        for (_name, message) in outbox.into_iter() {
            let update = self
                .get_publications()
                .iter()
                .filter(|p| p.get_table_name() == message.get_subject())
                .collect::<Vec<_>>();

            // Skip messages that are not in the publications
            if update.is_empty() {
                continue;
            }

            // Build the output message
            let out = ArrowOutgoingMessage::get_builder()
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
    fn get_processors(&self) -> &Vec<Arc<dyn ArrowProcessorTrait>>;

    /// Get an immutable reference to the runtime env
    fn get_runtime_env(&self) -> &Arc<Mutex<RuntimeEnv>>;

    /// Return a snapshot of the set of [`Metric`]s for this
    /// [`ArrowTask`]. If no [`Metric`]s are available, return None.
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
pub struct ArrowTask {
    /// Name of the task
    name: String,
    /// Metrics for the task and processors
    metrics: ArrowTaskMetricsSet,
    /// Runtime environment for the task and processors
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Entry processor
    processor: Vec<Arc<dyn ArrowProcessorTrait>>,
    /// Cached subscriptions to listen to
    subscriptions: Vec<ArrowTableSubscribe>,
    /// Cached subjects to publish on
    publications: Vec<ArrowTablePublish>,
}

impl MappableTrait for ArrowTask {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for ArrowTask {
    type T = ArrowTaskBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl RunnableTrait for ArrowTask {
    fn run(&self, mut messages: OutgoingMessageMap) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Running task {}", self.get_name());

        // Process the incoming message resulting in a `SendableRecordBatchStream`
        for processor in self.processor.iter() {
            messages =
                processor.process(messages, self.metrics.clone(), self.runtime_env.clone())?;
        }

        // make the output message
        let outbox = self.make_outbox(messages);
        Ok(outbox)
    }
}

impl ArrowTaskTrait for ArrowTask {
    fn get_runtime_env(&self) -> &Arc<Mutex<RuntimeEnv>> {
        &self.runtime_env
    }
    fn get_metrics(&self) -> MetricsSet {
        self.metrics.clone_inner()
    }
    fn get_processors(&self) -> &Vec<Arc<dyn ArrowProcessorTrait>> {
        &self.processor
    }
}

impl PubSubTrait for ArrowTask {
    fn get_subscriptions(&self) -> &Vec<ArrowTableSubscribe> {
        &self.subscriptions
    }
    fn get_publications(&self) -> &Vec<ArrowTablePublish> {
        &self.publications
    }
}

pub trait ArrowTaskBuilderTrait: BuilderTrait {
    fn with_metrics(self, metrics: ArrowTaskMetricsSet) -> Self;
    fn with_runtime_env(self, runtime_env: Arc<Mutex<RuntimeEnv>>) -> Self;
    fn with_subscriptions(self, subscriptions: Vec<ArrowTableSubscribe>) -> Self;
    fn with_publications(self, publications: Vec<ArrowTablePublish>) -> Self;
    fn with_processor(self, processor: Vec<Arc<dyn ArrowProcessorTrait>>) -> Self;
}

#[derive(Default)]
pub struct ArrowTaskBuilder {
    /// Task name
    pub name: Option<String>,
    /// Metrics for the task
    pub metrics: Option<ArrowTaskMetricsSet>,
    /// Runtime environment for the task
    pub runtime_env: Option<Arc<Mutex<RuntimeEnv>>>,
    /// Subscriptions to listen to
    pub subscriptions: Option<Vec<ArrowTableSubscribe>>,
    /// Subjects to publish on
    pub publications: Option<Vec<ArrowTablePublish>>,
    /// Function that implements the logic
    pub processor: Option<Vec<Arc<dyn ArrowProcessorTrait>>>,
}

impl BuilderTrait for ArrowTaskBuilder {
    type T = ArrowTask;
    fn new() -> Self {
        Self {
            name: None,
            metrics: None,
            runtime_env: None,
            subscriptions: None,
            publications: None,
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
            subscriptions: self.subscriptions.unwrap_or_default(),
            publications: self.publications.unwrap_or_default(),
            processor: self.processor.unwrap(),
        })
    }
}

impl ArrowTaskBuilderTrait for ArrowTaskBuilder {
    fn with_metrics(mut self, metrics: ArrowTaskMetricsSet) -> Self {
        self.metrics = Some(metrics);
        self
    }
    fn with_runtime_env(mut self, runtime_env: Arc<Mutex<RuntimeEnv>>) -> Self {
        self.runtime_env = Some(runtime_env);
        self
    }
    fn with_subscriptions(mut self, subscriptions: Vec<ArrowTableSubscribe>) -> Self {
        self.subscriptions = Some(subscriptions);
        self
    }
    fn with_publications(mut self, publications: Vec<ArrowTablePublish>) -> Self {
        self.publications = Some(publications);
        self
    }
    fn with_processor(mut self, processor: Vec<Arc<dyn ArrowProcessorTrait>>) -> Self {
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
            common_traits::{BuildableTrait, BuilderTrait, IncomingMessageMap, MappableTrait},
            runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        },
        table::{
            arrow_table::{
                ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, test_table::make_test_table,
            },
            arrow_table_publish::ArrowTablePublish,
        },
        task::{
            arrow_message::{
                ArrowIncomingMessage, ArrowIncomingMessageBuilder,
                ArrowIncomingMessageBuilderTrait, ArrowMessageBuilderTrait,
            },
            arrow_processor::test_processor::ArrowProcessorMock,
        },
    };

    use arrow::array::{ArrayRef, StringArray, UInt16Array, UInt32Array};
    use std::sync::Arc;

    pub fn make_state_tables(table_name: &str, config_name: &str) -> Result<Vec<ArrowTable>> {
        // mock config for the task
        let a: ArrayRef = Arc::new(StringArray::from(vec!["a".to_string()]));
        let b: ArrayRef = Arc::new(UInt32Array::from(vec![1]));
        let c: ArrayRef = Arc::new(UInt16Array::from(vec![1]));
        let batch = RecordBatch::try_from_iter(vec![("a", a), ("b", b), ("c", c)])?;
        let config = ArrowTableBuilder::new()
            .with_name(config_name)
            .with_record_batches(vec![batch])?
            .build()?;

        // mock table for the task
        let table = make_test_table(table_name, 4, 8, 3)?;
        Ok(vec![config, table])
    }

    pub fn make_state_tables_empty(table_name: &str, config_name: &str) -> Result<Vec<ArrowTable>> {
        // mock config for the task
        let a: ArrayRef = Arc::new(StringArray::from(vec!["".to_string()]));
        let b: ArrayRef = Arc::new(UInt32Array::from(vec![0]));
        let c: ArrayRef = Arc::new(UInt16Array::from(vec![0]));
        let batch = RecordBatch::try_from_iter(vec![("a", a), ("b", b), ("c", c)])?;
        let config = ArrowTableBuilder::new()
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
        let mut state = HashMap::<String, Arc<RwLock<ArrowTable>>>::new();
        for table in tables.into_iter() {
            state.insert(table.get_name().to_string(), Arc::new(RwLock::new(table)));
        }
        Ok(state)
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
        metrics: ArrowTaskMetricsSet,
    ) -> Result<ArrowTask> {
        let processor_name = format!("{name}_processor");
        let subscriptions = vec![
            ArrowTableSubscribe::OnUpdateFullTable {
                table_name: table_name.to_string(),
            },
            ArrowTableSubscribe::AlwaysFullTable {
                table_name: config_name.to_string(),
            },
        ];
        let publications = vec![ArrowTablePublish::Extend {
            table_name: table_name.to_string(),
        }];
        ArrowTask::get_builder()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_subscriptions(subscriptions)
            .with_publications(publications)
            .with_processor(vec![ArrowProcessorMock::new_arc(processor_name.as_str())])
            .build()
    }

    pub fn make_test_task_multiple_subscriptions(
        name: &str,
        runtime_env_name: &str,
        table_name_1: &str,
        table_name_2: &str,
        config_name: &str,
        metrics: ArrowTaskMetricsSet,
    ) -> Result<ArrowTask> {
        let processor_name = format!("{name}_processor");
        let subscriptions = vec![
            ArrowTableSubscribe::OnUpdateFullTable {
                table_name: table_name_1.to_string(),
            },
            ArrowTableSubscribe::OnUpdateFullTable {
                table_name: table_name_2.to_string(),
            },
            ArrowTableSubscribe::AlwaysFullTable {
                table_name: config_name.to_string(),
            },
        ];
        let publications = vec![ArrowTablePublish::Extend {
            table_name: table_name_1.to_string(),
        }];
        ArrowTask::get_builder()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_subscriptions(subscriptions)
            .with_publications(publications)
            .with_processor(vec![ArrowProcessorMock::new_arc(processor_name.as_str())])
            .build()
    }

    pub fn make_test_task_chained_processor(
        name: &str,
        runtime_env_name: &str,
        table_name: &str,
        config_name: &str,
        metrics: ArrowTaskMetricsSet,
    ) -> Result<ArrowTask> {
        let processor_name_1 = format!("{name}_processor_1");
        let processor_name_2 = format!("{name}_processor_2");
        let processor_name_3 = format!("{name}_processor_3");
        let subscriptions = vec![
            ArrowTableSubscribe::OnUpdateFullTable {
                table_name: table_name.to_string(),
            },
            ArrowTableSubscribe::AlwaysFullTable {
                table_name: config_name.to_string(),
            },
        ];
        let publications = vec![ArrowTablePublish::Extend {
            table_name: table_name.to_string(),
        }];
        ArrowTask::get_builder()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_env(Arc::new(Mutex::new(make_runtime_env(runtime_env_name)?)))
            .with_subscriptions(subscriptions)
            .with_publications(publications)
            .with_processor(vec![
                ArrowProcessorMock::new_arc(processor_name_1.as_str()),
                ArrowProcessorMock::new_arc(processor_name_2.as_str()),
                ArrowProcessorMock::new_arc(processor_name_3.as_str()),
            ])
            .build()
    }

    pub fn make_test_input_message(
        name: &str,
        publisher: &str,
        subject: &str,
        table_name: &str,
        update: &ArrowTablePublish,
    ) -> Result<IncomingMessageMap> {
        // mock table as input
        let table = make_test_table(table_name, 4, 8, 3)?;

        // build the message
        let message = ArrowIncomingMessageBuilder::new()
            .with_name(name)
            .with_subject(subject)
            .with_publisher(publisher)
            .with_message(table)
            .with_update(update)
            .build()?;

        // finish the message map
        let mut map = HashMap::<String, ArrowIncomingMessage>::new();
        map.insert(message.get_name().to_string(), message);
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::arrow_table::ArrowTableTrait;
    use crate::table::arrow_table::test_table::make_test_table;
    use crate::table::{
        arrow_table::{ArrowTableBuilder, ArrowTableBuilderTrait},
        arrow_table_publish::ArrowTablePublish,
    };
    use crate::task::arrow_message::{ArrowMessageTrait, ArrowOutgoingMessageTrait};
    use arrow::array::{Array, DictionaryArray, Int32Array, NullArray, RunArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use hashbrown::HashMap;

    /// A compilation test to ensure that the `ArrowTask::name()` method can
    /// be called from a trait object.
    #[allow(dead_code)]
    fn use_task_name_as_trait_object(plan: &dyn ArrowTaskTrait<T = ArrowTaskBuilder>) {
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
        let metrics = ArrowTaskMetricsSet::new();
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;
        let test_state = test_task::make_state("test_table", "test_config")?;
        let messages = test_task.get_subscriptions_from_state(&test_state);
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
        let metrics = ArrowTaskMetricsSet::new();
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;

        // Case 1: Message has subject that the task does not publish on
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            "test_message".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("test_message")
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&ArrowTablePublish::Extend {
                    table_name: "test_table".to_string(),
                })
                .with_message(make_test_table("test_table", 1, 8, 2)?.to_record_batch_stream())
                .build()?,
        );
        let inbox = test_task.make_outbox(message);
        assert_eq!(inbox.len(), 0);

        // Case 2: Message has subject that the task does not publish on
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            "test_message".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("test_message")
                .with_publisher("s1")
                .with_subject("test_table")
                .with_update(&ArrowTablePublish::Extend {
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
            ArrowTablePublish::Extend {
                table_name: "test_table".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_single_processor() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let test_task = test_task::make_test_task_single_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;
        let test_state = test_task::make_state("test_table", "test_config")?;
        let input = test_task.get_subscriptions_from_state(&test_state);
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
            ArrowTableBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
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
        let metrics = ArrowTaskMetricsSet::new();
        let test_task = test_task::make_test_task_chained_processor(
            "test_task",
            "test_rt",
            "test_table",
            "test_config",
            metrics.clone(),
        )?;
        let test_state = test_task::make_state("test_table", "test_config")?;
        let input = test_task.get_subscriptions_from_state(&test_state);
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
            ArrowTableBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
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
