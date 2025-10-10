use anyhow::Result;
use arrow::array::{ArrayRef, UInt64Array};
use arrow::array::StringArray;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parking_lot::{Mutex, RwLock};
use std::fmt::Display;
use std::sync::Arc;
use tracing::{Level, event};

use super::{
    common_traits::{
        BuildableTrait, BuilderTrait, MappableTrait, StateMap, TaskMap,
    },
    runtime_env::RuntimeEnv,
    session_context_builder::SessionContextBuilder,
};
use crate::metrics::{
    SpanMetricsSet, HashMap, get_metrics_as_gantt_table, get_metrics_as_mermaid_gantt,
    get_metrics_as_pivot_table,
};
use crate::table::table_publish::TablePublish;
use crate::table::{
    table_trait::{Table, TableBuilder, TableBuilderTrait, TableTrait},
    table_publish::TableUpdateTrait,
};
use crate::task::publish_subscribe::PubSubTrait;

/// Reserved table names for the [SessionContext]
#[derive(Debug)]
pub enum SessionContextTableNames {
    Metrics,
    Tasks,
    Processors,
    Subjects,
    RuntimeEnvironments,
    MermaidJS,
    SubjectsNumRows,
    MetricMermaidGantt,
    Errors,
    Logs,
}

impl Display for SessionContextTableNames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionContextTableNames::Metrics => write!(f, "Metrics"),
            SessionContextTableNames::Tasks => write!(f, "Tasks"),
            SessionContextTableNames::Processors => write!(f, "Processors"),
            SessionContextTableNames::Subjects => write!(f, "Subjects"),
            SessionContextTableNames::RuntimeEnvironments => write!(f, "RuntimeEnvironments"),
            SessionContextTableNames::MermaidJS => write!(f, "MermaidJS"),
            SessionContextTableNames::SubjectsNumRows => write!(f, "SubjectsNumRows"),
            SessionContextTableNames::MetricMermaidGantt => write!(f, "MetricMermaidGantt"),
            SessionContextTableNames::Errors => write!(f, "Errors"),
            SessionContextTableNames::Logs => write!(f, "Logs"),
        }
    }
}

/// The `SessionContext` creates an execution graph based on a
/// `SessionPlan` and manages the running of individual tasks
/// and the messages passed between tasks.
#[derive(Default, Debug, Clone)]
#[allow(dead_code)]
pub struct SessionContext {
    /// A unique UUID that identifies the session
    pub(crate) name: String,
    /// The list of available tasks that can be run during the session
    pub(crate) tasks: TaskMap,
    /// Data that should be persisted between queries
    pub(crate) state: StateMap,
    /// Runtime environment configuration to use during task runs
    pub(crate) runtime_envs: HashMap<String, Arc<Mutex<RuntimeEnv>>>,
    /// The maximum number of iterations before stopping
    pub(crate) max_iter: usize,
}

impl SessionContext {
    pub fn new(
        name: String,
        tasks: TaskMap,
        state: StateMap,
        runtime_envs: HashMap<String, Arc<Mutex<RuntimeEnv>>>,
        max_iter: usize,
    ) -> SessionContext {
        Self {
            name,
            tasks,
            state,
            runtime_envs,
            max_iter,
        }
    }

    /// Get a task
    pub(crate) fn get_tasks(&self) -> &TaskMap {
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

    /// Create the metrics table if it does not exist or update with the new metrics
    pub fn update_metrics_table(&mut self, metrics_vec: &[SpanMetricsSet]) -> Result<bool> {
        // create the pivot table and clear the metrics
        let pivot_table = get_metrics_as_pivot_table(
            metrics_vec,
            SessionContextTableNames::Metrics.to_string().as_str(),
        )?;

        // update the state with the metrics
        if pivot_table.count_rows() > 0 {

            // Add the metrics pivot table to the state or update
            if self
                .state
                .contains_key(SessionContextTableNames::Metrics.to_string().as_str())
            {
                self.state
                    .get_mut(SessionContextTableNames::Metrics.to_string().as_str())
                    .unwrap()
                    .try_write()
                    .unwrap()
                    .update_table(
                        pivot_table.get_record_batches_own(),
                        TablePublish::Extend {
                            table_name: SessionContextTableNames::Metrics.to_string().as_str().to_string(),
                        },
                    )?;
            } else {
                self.state.insert(
                    SessionContextTableNames::Metrics.to_string(),
                    Arc::new(RwLock::new(pivot_table)),
                );
            }

            // Create the gantt view
            let gantt_table = get_metrics_as_gantt_table(self.state.get(SessionContextTableNames::Metrics.to_string().as_str()).unwrap().read().clone(), SessionContextTableNames::MetricMermaidGantt.to_string().as_str())?;
            let mermaid_gantt_table = get_metrics_as_mermaid_gantt(gantt_table)?;

            // Add the metrics gantt table to the state or update
            if self
                .state
                .contains_key(SessionContextTableNames::MetricMermaidGantt.to_string().as_str())
            {
                self.state
                    .get_mut(SessionContextTableNames::MetricMermaidGantt.to_string().as_str())
                    .unwrap()
                    .write()
                    .update_table(
                        mermaid_gantt_table.get_record_batches_own(),
                        TablePublish::Extend {
                            table_name: SessionContextTableNames::MetricMermaidGantt.to_string(),
                        },
                    )?;
            } else {
                self.state.insert(
                    SessionContextTableNames::MetricMermaidGantt.to_string(),
                    Arc::new(RwLock::new(mermaid_gantt_table)),
                );
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the max iterations
    pub(crate) fn get_max_iter(&self) -> usize {
        self.max_iter
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
            let num_row = state.read().count_rows() as u64;
            subject_names.push(name.clone());
            num_rows.push(num_row);
        }

        // create the record batch
        let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
        let num_rows: ArrayRef = Arc::new(UInt64Array::from(num_rows));
        let batch = RecordBatch::try_from_iter(vec![
            ("subject_name", subject_names),
            ("num_rows", num_rows),
        ]).unwrap();

        // create the table
        let subject_num_rows_table = Table::get_builder()
            .with_name(SessionContextTableNames::SubjectsNumRows.to_string().as_str())
            .with_record_batches(vec![batch]).unwrap()
            .build()
            .unwrap();

        // Add the metrics pivot table to the state or update
        if self
            .state
            .contains_key(SessionContextTableNames::SubjectsNumRows.to_string().as_str())
        {
            self.state
                .get_mut(SessionContextTableNames::SubjectsNumRows.to_string().as_str())
                .unwrap()
                .write()
                .update_table(
                    subject_num_rows_table.get_record_batches_own(),
                    TablePublish::Replace {
                        table_name: SessionContextTableNames::SubjectsNumRows.to_string(),
                    },
                ).unwrap();
        } else {
            self.state.insert(
                SessionContextTableNames::SubjectsNumRows.to_string(),
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

    /// Initialize the superstep_update with all tasks and their update subscriptions
    pub fn init_superstep_updates(&self) -> HashMap<String, HashMap<String, bool>> {
        let mut init = HashMap::<String, HashMap<String, bool>>::new();
        for (task_name, task) in self.tasks.iter() {
            let mut subscriptions = HashMap::<String, bool>::new();
            for subscription in task.get_subscriptions() {
                subscriptions.insert(subscription.get_table_name().to_string(), false);
            }
            init.insert(task_name.to_string(), subscriptions);
        }
        init
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
                    let update = TablePublish::Replace {
                        table_name: name.to_string(),
                    };
                    subject
                        .write()
                        .update_table(table.get_record_batches_own(), update)?;
                }
                Err(e) => event!(Level::ERROR, "Error reading state: {e:?}"),
            };
        }
        Ok(())
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
    use crate::metrics::HashSet;
    use crate::table::table_trait::test_table::make_test_table_schema;
    use crate::session::session_context_builder::test_session_context_builder::{make_test_session_context_parallel_task, make_test_session_context_parallel_task_empty};
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempdir;

    #[test]
    fn test_session_get_table_name_by_schema() -> Result<()> {
        let session_context =
            make_test_session_context_parallel_task("session_1", 25)?;

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
        let mut session_context =
            make_test_session_context_parallel_task("session_1", 25)?;
        session_context.update_subject_num_rows_table();
        let info = session_context.get_states().get(SessionContextTableNames::SubjectsNumRows.to_string().as_str()).unwrap().read();

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
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(num_rows, [1, 1, 1, 12, 12, 12]);

        Ok(())
    }

    #[test]
    fn test_session_init_superstep_updates() -> Result<()> {
        let session_context =
            make_test_session_context_parallel_task("session_1", 25)?;
        let init = session_context.init_superstep_updates();
        assert_eq!(init.len(), 4);
        assert_eq!(
            init.keys().map(|k| k.as_str()).collect::<HashSet<_>>(),
            ["task_1", "task_2", "task_3", "session_1"]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        let mut subscriptions = init
            .get("task_1")
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect::<Vec<_>>();
        subscriptions.sort();
        assert_eq!(subscriptions, &["config_1", "state_1"]);
        for (_k, v) in init.get("task_1").unwrap() {
            assert!(!v);
        }
        let mut subscriptions = init
            .get("task_2")
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect::<Vec<_>>();
        subscriptions.sort();
        assert_eq!(subscriptions, &["config_2", "state_2"]);
        for (_k, v) in init.get("task_2").unwrap() {
            assert!(!v);
        }
        let mut subscriptions = init
            .get("task_3")
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect::<Vec<_>>();
        subscriptions.sort();
        assert_eq!(subscriptions, &["config_3", "state_3"]);
        for (_k, v) in init.get("task_3").unwrap() {
            assert!(!v);
        }
        assert_eq!(
            init.get("session_1")
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect::<HashSet<_>>(),
            ["state_1", "state_2", "state_3"]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        for (_k, v) in init.get("session_1").unwrap() {
            assert!(!v);
        }

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_session_read_write_state() -> Result<()> {
        // Create the session
        let session_context =
            make_test_session_context_parallel_task("session_1", 25)?;

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
}
