use anyhow::{Result, anyhow};
use arrow::datatypes::SchemaRef;
use clap::ValueEnum;
use parking_lot::RwLock;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableTableSubscribePolicies, AvailableTableUpdatePolicies, BuildableTrait, BuilderTrait, IPCMessageBuilder, IPCMessageMap, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorSubjects, ProcessorSubjectsBuilder, ProcessorSubjectsMap, RuntimeEnv, StateMap, Table, TableBuilder, TableBuilderTrait, TablePublication, TablePublicationTrait, TableSubscription, TableTrait, TaskMap, create_session_tasks_subscribe_batch, create_subjects_change_log_batch, create_subjects_num_rows_batch, from_diagnostics_to_tables
};
use phymes_diagnostics::{Diagnostics, HashMap, create_timestamp_micros};
use std::sync::Arc;
use tracing::{Level, event};

use crate::{SessionContextBuilder, create_message_map};

/// The [SessionContext] creates a (dynamic) execution graph based on a [TaskPlan]
///   and manages the running of individual [Task]s and the [Message]s passed between them.
///
/// [TaskPlan]: phymes_core::TaskPlan
/// [Task]: phymes_core::TaskTrait
/// [Message]: phymes_core::MessageTrait
#[derive(Default, Debug, Clone)]
pub struct SessionContext {
    /// A unique UUID that identifies the session
    pub(crate) name: String,
    /// The list of available tasks that can be run during the session
    pub(crate) tasks: TaskMap,
    /// Session data (state) that should be persisted between queries that is composed of local and shared state
    ///
    /// Local state: the session diagnostics (i.e., traces, events, metrics, and errors),
    ///   and task plan (i.e., subjects, tasks, processors, and runtime_envs)
    ///  
    /// Shared state: Message subjects data along with metadata such as
    ///   the row counts for all subjects, and subject changelog (todo)
    pub(crate) state: StateMap,
    /// Runtime environment configuration to use during task runs
    #[allow(dead_code)]
    pub(crate) runtime_envs: HashMap<String, Arc<RuntimeEnv>>,
    /// The maximum number of iterations before stopping
    pub(crate) max_iter: usize,
    /// Whether to gather diagnostic information or not
    pub(crate) diagnostics: bool,
}

impl SessionContext {
    pub fn new(
        name: String,
        tasks: TaskMap,
        state: StateMap,
        runtime_envs: HashMap<String, Arc<RuntimeEnv>>,
        max_iter: usize,
        diagnostics: bool,
    ) -> SessionContext {
        Self {
            name,
            tasks,
            state,
            runtime_envs,
            max_iter,
            diagnostics,
        }
    }

    /// Get a task
    pub fn get_tasks(&self) -> &TaskMap {
        &self.tasks
    }

    /// Get state
    pub fn get_states(&self) -> &StateMap {
        &self.state
    }

    /// Get state
    pub fn get_states_own(self) -> StateMap {
        self.state
    }

    /// Compute the next tasks to subscribe
    pub fn tasks_subscribe(&self) -> Result<()> {
        // Extract the columns
        let batches = self
            .get_states()
            .get(
                AvailableSubjects::SessionTasksSubscribeAggregate
                    .to_string()
                    .as_str(),
            )
            .unwrap_or_else(|| {
                panic!(
                    "Missing table for `{}` in session `{}`.",
                    AvailableSubjects::SessionTasksSubscribeAggregate,
                    self.get_name()
                )
            })
            .write()
            .get_record_batches_mut()
            .drain(0..)
            .collect::<Vec<_>>();
        let table = TableBuilder::default()
            .with_name(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;

        // Extract out the columns
        let session_names = table.get_column_as_vec_str("session_name");
        let task_names = table.get_column_as_vec_str("task_name");
        let processor_names = table.get_column_as_vec_str("processor_name");
        let processor_types = table.get_column_as_vec_str("processor_type");
        let subscription_names = table.get_column_as_vec_nested_nonprimitive::<String>("subscription_name-List")?;
        let subscription_table_names = table.get_column_as_vec_nested_nonprimitive::<String>("subscription_table_name-List")?;
        let subscribe_types = table.get_column_as_vec_str("subscribe_type-Last");
        let update_types = table.get_column_as_vec_str("update_type-Last");
        let timestamps = table.get_column_as_vec_nested_primitive::<i64>("timestamp-List")?;
        let timestamp_lasts = table.get_column_as_vec_nested_primitive::<i64>("timestamp-Last-List")?;

        // Determine the processor subscriptions
        let processors_subscribe = session_names.into_iter()
            .zip(task_names.into_iter())
            .zip(processor_names.into_iter())
            .zip(processor_types.into_iter())
            .zip(subscription_names.into_iter())
            .zip(subscription_table_names.into_iter())
            .zip(subscribe_types.into_iter())
            .zip(update_types.into_iter())
            .zip(timestamps.into_iter())
            .zip(timestamp_lasts.into_iter())
            .map(|(((((((((session_name, task_name), processor_name), processor_type), subscription_names), subscription_table_names), subscribe_type), update_type), timestamps), timestamps_lasts)| {
                let subscriptions = subscription_names.iter()
                    .zip(subscription_table_names.iter())
                    .map(|(name, subject)| TableSubscription::from_str_fuzzy(name, subject).unwrap())
                    .collect::<Vec<_>>();
                let update_policy = AvailableTableUpdatePolicies::from_str(update_type, false).unwrap().build();
                let subjects_change_log = subscription_table_names.iter()
                    .zip(timestamps_lasts.iter())
                    .map(|(subject_name, timestamp)| (subject_name.to_string(), timestamp.to_owned()))
                    .collect::<HashMap<_, _>>();
                let updates = update_policy.determine_updates(&subscriptions, timestamps.last().unwrap(), &subjects_change_log, self.get_states());
                let subscribe_policy = AvailableTableSubscribePolicies::from_str_fuzzy(subscribe_type).unwrap().build();
                let subscribe = subscribe_policy.check_subscriptions(&subscriptions, &updates, self.get_states());
                (session_name, task_name, processor_name, processor_type, subscription_names, subscription_table_names, subscribe_type, update_type, timestamps, timestamps_lasts, subscribe)
            })
            .collect::<Vec<_>>();

        // Determine the task subscriptions
        let mut tasks_subscribe = HashMap::<(String, String), bool>::new();
        for (session_name, task_name, _processor_name, _processor_type, _subscription_names, _subscription_table_names, _subscribe_type, _update_type, _timestamps, _timestamps_lasts, subscribe) in processors_subscribe.iter() {
            if let Some(subscribe_t) = tasks_subscribe.get_mut(&(session_name.to_string(), task_name.to_string())) {
                *subscribe_t = *subscribe_t & subscribe;
            } else {
                let _ = tasks_subscribe.insert((session_name.to_string(), task_name.to_string()), *subscribe);
            }
        }
        let (session_names_subscribe, task_names_subscribe): (String, String) = tasks_subscribe.into_iter()
            .filter_map(|(k, v)| if v {
                Some(k)
            } else {
                None
            })
            .unzip();

        // Determine the subjects to subscribe to
        let (((((session_names, task_names), processor_names), processor_types), subscription_names), subscription_table_names) = processors_subscribe.into_iter()
            .filter_map(|(session_name, task_name, processor_name, processor_type, subscription_names, subscription_table_names, subscribe_type, update_type, timestamps, timestamps_lasts, _subscribe)| if session_names_subscribe.contains(session_name) && task_names_subscribe.contains(task_name) {
                let subscribe = subscription_names.into_iter()
                    .zip(subscription_table_names.into_iter())
                    .zip(timestamps.into_iter())
                    .zip(timestamps_lasts.into_iter())
                    .filter_map(|(((name, subject), timestamp), timestamp_last)| {
                        let subscriptions = vec![TableSubscription::from_str_fuzzy(&name, &subject).unwrap()];
                        let update_policy = AvailableTableUpdatePolicies::from_str(update_type, false).unwrap().build();
                        let mut subjects_change_log = HashMap::<String, i64>::new();
                        let _ = subjects_change_log.insert(subject.to_string(), timestamp_last);
                        let updates = update_policy.determine_updates(&subscriptions, &timestamp, &subjects_change_log, self.get_states());
                        let subscribe_policy = AvailableTableSubscribePolicies::from_str_fuzzy(subscribe_type).unwrap().build();
                        let subscribe = subscribe_policy.check_subscriptions(&subscriptions, &updates, self.get_states());
                        if subscribe {
                            Some((((((session_name.to_string(), task_name.to_string()), processor_name.to_string()), processor_type.to_string()), name), subject))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                Some(subscribe)
            } else {
                None
            })
            .flatten()
            .unzip();

        // Create the table
        let batch = create_session_tasks_subscribe_batch(session_names, task_names, processor_names, processor_types, subscription_names, subscription_table_names)?;
        let table = TableBuilder::default()
            .with_name(AvailableSubjects::SessionTasksSubscribe.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the table
        let message = IPCMessageBuilder::default()
            .with_subject(table.get_name())
            .with_publisher(self.get_name())
            .with_update(&TablePublication::Replace {
                table_name: table.get_name().to_string(),
            })
            .with_message(table.to_ipc_stream()?)
            .make_name()?
            .build()?;
        let messages = create_message_map(vec![message]);
        let _ = self.update_subjects_from_messages(messages)?;

        Ok(())
    }

    /// Update the session tasks subscribe

    /// Update the session subscribe and publish

    /// Take the task subscriptions and publications that are ready to subscribe and publish
    /// 
    /// # Notes
    /// * See schema at [AvailableSubjects::SessionTasksSubscribePublish]
    /// * The columns are taken to prevent infinite loops of the same tasks
    pub fn tasks_subscribe_publish(&self) -> Result<HashMap<(String, String), ProcessorSubjectsMap>> {
        // Extract out the columns
        let batches = self
            .get_states()
            .get(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .unwrap_or_else(|| {
                panic!(
                    "Missing table for `{}` in session `{}`.",
                    AvailableSubjects::SessionTasksSubscribePublish,
                    self.get_name()
                )
            })
            .write()
            .get_record_batches_mut()
            .drain(0..)
            .collect::<Vec<_>>();
        let table = TableBuilder::default()
            .with_name(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;

        let task_names = table.get_column_as_vec_nonprimitive::<String>("task_name")?;
        let processor_names =
            table.get_column_as_vec_nonprimitive::<String>("processor_name")?;
        let processor_types =
            table.get_column_as_vec_nonprimitive::<String>("processor_type")?;
        let subscription_names =
            table.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let subscription_table_names = table
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let publication_names =
            table.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let publication_table_names = table
            .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let session_names =
            table.get_column_as_vec_nonprimitive::<String>("session_name")?;

        // Map to objects
        let combined = task_names
            .into_iter()
            .zip(subscription_names)
            .zip(subscription_table_names)
            .zip(publication_names)
            .zip(publication_table_names)
            .zip(processor_names)
            .zip(processor_types)
            .zip(session_names)
            .map(
                |(
                    (
                        (
                            (
                                (
                                    ((task_name, subscription_names), subscription_table_names),
                                    publication_names,
                                ),
                                publication_table_names,
                            ),
                            processor_name,
                        ),
                        processor_type,
                    ),
                    session_name,
                )| {
                    let subscriptions = subscription_names
                        .iter()
                        .zip(subscription_table_names.iter())
                        .map(|(subscription_name, subscription_table_name)| {
                            TableSubscription::from_str_fuzzy(
                                subscription_name,
                                subscription_table_name,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let publications = publication_names
                        .iter()
                        .zip(publication_table_names.iter())
                        .map(|(publication_name, publication_table_name)| {
                            TablePublication::from_str_fuzzy(
                                publication_name,
                                publication_table_name,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let processor_subjects = ProcessorSubjectsBuilder::default()
                        .with_name(&processor_name)
                        .with_subscriptions(&subscriptions)
                        .with_publications(&publications)
                        .build()
                        .unwrap();
                    (
                        task_name,
                        processor_subjects,
                        processor_name,
                        processor_type,
                        session_name,
                    )
                },
            )
            .collect::<Vec<_>>();

        // Aggregate processors
        // DM: not possible to have two-levels of nesting with Arrow RecordBatches
        let mut tasks = HashMap::<(String, String), ProcessorSubjectsMap>::new();
        for (task_name, processor_subjects, processor_name, _processor_type, session_name) in
            combined
        {
            if let Some(task) = tasks.get_mut(&(task_name.to_string(), session_name.to_string())) {
                let _ = task.insert(processor_name, processor_subjects);
            } else {
                let mut processor = HashMap::<String, ProcessorSubjects>::new();
                let _ = processor.insert(processor_name, processor_subjects);
                let _ = tasks.insert((task_name.to_string(), session_name.to_string()), processor);
            }
        }

        Ok(tasks)
    }

    /// Create the metrics table if it does not exist or update with the new metrics
    pub fn update_metrics_table(
        &mut self,
        diagnostics_vec: &[Diagnostics],
    ) -> Result<(bool, bool, bool)> {
        // create the pivot table and clear the metrics
        let (metrics_table, traces_table, events_table) =
            from_diagnostics_to_tables(diagnostics_vec)?;

        // update the state with the metrics
        let updated_metrics = if let Some(metrics_table) = metrics_table {
            // Add the metrics pivot table to the state or update
            if self
                .state
                .contains_key(AvailableSubjects::SessionMetrics.to_string().as_str())
            {
                self.state
                    .get_mut(AvailableSubjects::SessionMetrics.to_string().as_str())
                    .unwrap()
                    .try_write()
                    .unwrap()
                    .publish_to_table(
                        metrics_table.get_record_batches_own(),
                        TablePublication::Extend {
                            table_name: AvailableSubjects::SessionMetrics
                                .to_string()
                                .as_str()
                                .to_string(),
                        },
                    )?;
            } else {
                self.state.insert(
                    AvailableSubjects::SessionMetrics.to_string(),
                    Arc::new(RwLock::new(metrics_table)),
                );
            }

            true
        } else {
            false
        };

        // update the state with the traces
        let updated_traces = if let Some(traces_table) = traces_table {
            // Add the metrics pivot table to the state or update
            if self
                .state
                .contains_key(AvailableSubjects::SessionTraces.to_string().as_str())
            {
                self.state
                    .get_mut(AvailableSubjects::SessionTraces.to_string().as_str())
                    .unwrap()
                    .try_write()
                    .unwrap()
                    .publish_to_table(
                        traces_table.get_record_batches_own(),
                        TablePublication::Extend {
                            table_name: AvailableSubjects::SessionTraces
                                .to_string()
                                .as_str()
                                .to_string(),
                        },
                    )?;
            } else {
                self.state.insert(
                    AvailableSubjects::SessionTraces.to_string(),
                    Arc::new(RwLock::new(traces_table)),
                );
            }

            true
        } else {
            false
        };

        // update the state with the events
        let updated_events = if let Some(events_table) = events_table {
            // Add the metrics pivot table to the state or update
            if self
                .state
                .contains_key(AvailableSubjects::SessionEvents.to_string().as_str())
            {
                self.state
                    .get_mut(AvailableSubjects::SessionEvents.to_string().as_str())
                    .unwrap()
                    .try_write()
                    .unwrap()
                    .publish_to_table(
                        events_table.get_record_batches_own(),
                        TablePublication::Extend {
                            table_name: AvailableSubjects::SessionEvents
                                .to_string()
                                .as_str()
                                .to_string(),
                        },
                    )?;
            } else {
                self.state.insert(
                    AvailableSubjects::SessionEvents.to_string(),
                    Arc::new(RwLock::new(events_table)),
                );
            }

            true
        } else {
            false
        };

        Ok((updated_metrics, updated_traces, updated_events))
    }

    /// Get the max iterations
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Get the diagnostics
    pub fn get_diagnostics(&self) -> bool {
        self.diagnostics
    }

    /// Update the row counts for the subjects
    pub fn update_subject_num_rows_table(&mut self) {
        let mut subject_names = Vec::new();
        let mut num_rows = Vec::new();

        // Sort the hashmap
        let mut sorted_map = self.state.iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        for (_name, state) in sorted_map.iter() {
            let name = state.read().get_name().to_string();
            let num_row = state.read().count_rows() as i64;
            subject_names.push(name.clone());
            num_rows.push(num_row);
        }

        // create the record batch
        let batch = create_subjects_num_rows_batch(subject_names, num_rows).unwrap();

        // create the table
        let subject_num_rows_table = Table::get_builder()
            .with_name(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .with_record_batches(vec![batch])
            .unwrap()
            .build()
            .unwrap();

        // Add the subjects num rows table to the state or update
        if self
            .state
            .contains_key(AvailableSubjects::SubjectsNumRows.to_string().as_str())
        {
            self.state
                .get_mut(AvailableSubjects::SubjectsNumRows.to_string().as_str())
                .unwrap()
                .write()
                .publish_to_table(
                    subject_num_rows_table.get_record_batches_own(),
                    TablePublication::Replace {
                        table_name: AvailableSubjects::SubjectsNumRows.to_string(),
                    },
                )
                .unwrap();
        } else {
            self.state.insert(
                AvailableSubjects::SubjectsNumRows.to_string(),
                Arc::new(RwLock::new(subject_num_rows_table)),
            );
        }
    }

    /// Find the table by matching schemas
    pub fn get_table_name_by_schema(&self, schema: &SchemaRef) -> Option<&str> {
        let mut sorted_map = self.state.iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        for (name, table) in sorted_map.iter() {
            if schema.eq(&table.read().get_schema()) {
                return Some(name);
            }
        }
        None
    }

    /// Get the subject as a csv string
    pub fn get_subject_as_csv_str(
        &self,
        name: &str,
        delimiter: u8,
        header: bool,
    ) -> Result<String> {
        let csv = self
            .state
            .get(name)
            .unwrap()
            .read()
            .to_csv(delimiter, header)?;
        let csv_str = String::from_utf8_lossy(csv.as_ref()).into_owned();
        Ok(csv_str)
    }

    /// Save the current state to disk
    pub fn write_state(&self, path: &str, tag: &str) -> Result<()> {
        for (name, subject) in self.state.iter() {
            let pathname = format!("{path}/{tag}-{}-{name}", self.get_name());
            let mut file = std::fs::File::create(pathname)?;
            match subject.read().to_ipc_file(&mut file) {
                Ok(()) => (),
                Err(e) => event!(Level::ERROR, "Error writing state: {e:?}"),
            };
        }
        Ok(())
    }

    /// Read state
    pub fn read_state(&mut self, path: &str, tag: &str) -> Result<()> {
        for (name, subject) in self.state.iter() {
            let pathname = format!("{path}/{tag}-{}-{name}", self.get_name());
            let file = std::fs::File::open(pathname)?;
            match TableBuilder::new_from_ipc_file(&file) {
                Ok(table_builder) => {
                    let table = table_builder.with_name(name).build()?;
                    let update = TablePublication::Replace {
                        table_name: name.to_string(),
                    };
                    subject
                        .write()
                        .publish_to_table(table.get_record_batches_own(), update)?;
                }
                Err(e) => event!(Level::ERROR, "Error reading state: {e:?}"),
            };
        }
        Ok(())
    }

    /// Update the state from the published messages
    /// and return a map of changed subscriptions along with their publishers
    pub fn update_subjects_from_messages(&self, messages: IPCMessageMap) -> Result<Table> {
        let mut subject_names = Vec::new();
        let mut task_names = Vec::new();
        let mut session_names = Vec::new();
        let mut num_rows_deltas = Vec::new();
        let mut timestamps = Vec::new();
        for (_name, message) in messages.into_iter() {
            // Should the subject be updated?
            let update = message.get_update().clone();
            if update == TablePublication::None {
                continue;
            }

            // Try to update the state with the new record batches
            let table_name = message.get_update().get_table_name().to_string();
            if let Some(state) = self.get_states().get(table_name.as_str()) {
                let publisher = message.get_publisher().to_string();

                // Check for any inconsistencies in the message and intercept any errors
                let table = TableBuilder::new_from_ipc_stream(&message.get_message_own())?
                    .with_name(table_name.as_str())
                    .build()?;
                let _num_rows = table.count_rows(); // DM: not used currently...
                let batches = table.get_record_batches_own();

                // Update the state
                // Check for a mismatch in the schema and intercept any errors
                let num_rows_old = state.read().count_rows();
                state.write().publish_to_table(batches, update)?;
                let num_rows_new = state.read().count_rows();

                // Record the table name that was updated and the pubisher who updated it
                subject_names.push(state.read().get_name().to_string());
                task_names.push(publisher);
                session_names.push(self.get_name().to_string());
                num_rows_deltas.push(num_rows_old as i64 - num_rows_new as i64);
                timestamps.push(create_timestamp_micros());
            } else {
                // Mismatch in table names of the update and state
                return Err(anyhow!(
                    "Subject '{table_name}' with update '{update:?}' is not in the session state tables! Available tables are {:?}",
                    self.get_states().keys()
                ));
            }
        }
        let batches = create_subjects_change_log_batch(
            subject_names,
            task_names,
            session_names,
            num_rows_deltas,
            timestamps,
        )?;
        AvailableSubjects::SubjectsChangeLog.to_table(None, Some(vec![batches]))
    }
}

impl MappableTrait for SessionContext {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for SessionContext {
    type T = SessionContextBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_session_context_builder::{
        make_test_session_context_parallel_task, make_test_session_context_parallel_task_empty,
    };
    use arrow::array::Int64Array;
    use phymes_core::{
        IPCMessage,
        test_table::{self, make_test_table_schema},
        test_task,
    };
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempdir;

    #[test]
    fn test_session_get_table_name_by_schema() -> Result<()> {
        let session_context = make_test_session_context_parallel_task("session_1", 25)?;

        // table should be found
        let schema = make_test_table_schema(8)?;
        let name = session_context.get_table_name_by_schema(&schema).unwrap();
        assert_eq!(name, "state_1");

        // table should not be found
        let schema = make_test_table_schema(2)?;
        let name = session_context.get_table_name_by_schema(&schema);
        assert!(name.is_none());
        Ok(())
    }

    #[test]
    fn test_session_update_subject_num_rows_table() -> Result<()> {
        let mut session_context = make_test_session_context_parallel_task("session_1", 25)?;
        session_context.update_subject_num_rows_table();
        let info = session_context
            .get_states()
            .get(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .unwrap()
            .read();

        assert_eq!(
            info.get_column_as_vec_str("subject_name"),
            [
                "config_1", "config_2", "config_3", "state_1", "state_2", "state_3",
            ]
        );
        let num_rows = info
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("num_rows")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(num_rows, [1, 1, 1, 12, 12, 12]);

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_session_read_write_state() -> Result<()> {
        // Create the session
        let session_context = make_test_session_context_parallel_task("session_1", 25)?;

        // Write the session to disk
        let tmp_dir = tempdir()?;
        session_context.write_state(tmp_dir.path().to_str().unwrap(), "tag")?;

        // Read the state
        let mut session_context_empty =
            make_test_session_context_parallel_task_empty("session_1", 25)?;
        session_context_empty.read_state(tmp_dir.path().to_str().unwrap(), "tag")?;

        for subject in session_context.get_states().keys() {
            assert_eq!(
                session_context
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_record_batches(),
                session_context_empty
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_record_batches()
            );
            assert_eq!(
                session_context
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_schema(),
                session_context_empty
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_schema()
            );
            assert_eq!(
                session_context
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_name(),
                session_context_empty
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_name()
            );
        }
        tmp_dir.close()?;
        Ok(())
    }
    
    #[test]
    fn test_session_update_subjects_from_messages() -> Result<()> {
        // Case 1: no state update
        let session_context = make_test_session_context_parallel_task("session_1", 25)?;
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::None,
            true,
        )?;
        let updates = session_context.update_subjects_from_messages(input)?;

        // check the response
        assert_eq!(updates.count_rows(), 0);
        assert_eq!(
            session_context
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .get_states()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .get_states()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            4
        );

        // Case 2: update state
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Extend {
                table_name: "state_1".to_string(),
            },
            true,
        )?;
        let updates = session_context.update_subjects_from_messages(input)?;

        // check the response
        assert_eq!(updates.count_rows(), 1);
        let col = updates.get_column_as_vec_str("subject_name");
        assert_eq!(col, [""]);
        let col = updates.get_column_as_vec_str("task_name");
        assert_eq!(col, [""]);
        let col = updates.get_column_as_vec_str("session_name");
        assert_eq!(col, [""]);
        let col = updates.get_column_as_vec_primitive::<i64>("num_rows_delta")?;
        assert_eq!(col, [0]);
        assert_eq!(
            session_context
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        ); // Originally 3
        assert_eq!(
            session_context
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            4
        );
        assert_eq!(
            session_context
                .get_states()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .get_states()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );

        // Case 3: Error due to mismatching schemas
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Extend {
                table_name: "state_1".to_string(),
            },
            false,
        )?;
        let updates = session_context.update_subjects_from_messages(input);
        assert!(updates.is_err());

        // Case 4: Error due to mismatching table names
        let message = IPCMessage::new(
            "task_1",
            "state_1",
            "session_1",
            Some(test_table::make_test_table("state_1", 4, 8, 3)?.to_ipc_stream()?),
            Some(TablePublication::Extend {
                table_name: "NotFound".to_string(),
            }),
        );
        let mut input = HashMap::<String, IPCMessage>::new();
        input.insert(message.get_name().to_string(), message);
        let updates = session_context.update_subjects_from_messages(input);
        assert!(updates.is_err());

        Ok(())
    }
    
    #[test]
    fn test_session_tasks_subscribe() -> Result<()> {
        // Make the test data
        Ok(())
    }
}
