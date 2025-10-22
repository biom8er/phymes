use anyhow::{anyhow, Result};
use arrow::array::ArrayRef;
use arrow::array::{BooleanArray, StringArray};
use arrow::record_batch::RecordBatch;
use phymes_diagnostics::HashMap;
use std::fs::File;
use std::sync::Arc;
use tracing::{Level, event, instrument};

use super::common_traits::{BuildableTrait, BuilderTrait, IPCMessageMap, MappableTrait};
use crate::session::session_context::SessionContext;
use crate::table::{Table, TableBuilder, TableBuilderTrait, TableTrait, TableUpdateTrait, TablePublish};
use crate::task::MessageTrait;

/// State tracked during the course of running a [`SessionStream`]
#[derive(Default, Debug, Clone)]
pub struct SessionStreamState {
    /// The session context
    session_context: SessionContext,
    /// The current iteration
    iter: usize,
    /// The changes from the last superstep
    /// where keys are tasks and values are subjects
    superstep_updates: HashMap<String, HashMap<String, bool>>,
}

impl SessionStreamState {
    pub fn new(session_context: SessionContext) -> Self {
        let init = session_context.init_superstep_updates();
        Self {
            session_context,
            iter: 0,
            superstep_updates: init,
        }
    }

    /// Get the session context
    pub fn get_session_context(&self) -> &SessionContext {
        &self.session_context
    }

    /// Get the session context
    pub fn get_session_context_own(self) -> SessionContext {
        self.session_context
    }

    /// Get the session context
    pub fn get_session_context_mut(&mut self) -> &mut SessionContext {
        &mut self.session_context
    }

    /// Get the current iteration
    pub fn get_iter(&self) -> usize {
        self.iter
    }

    /// Update the current iteration
    pub fn set_iter(&mut self, iter: usize) {
        self.iter = iter;
    }

    /// Get the superstep update
    pub fn get_superstep_updates(&self) -> &HashMap<String, HashMap<String, bool>> {
        &self.superstep_updates
    }

    /// Extend the superstep update
    ///
    /// # Notes
    ///
    /// * We assume that all tasks available have already been added
    ///   upon initialization of `superstep_updates`
    /// * Subject updates where the executing task and publisher are the same are ignored
    ///
    /// # Arguments
    ///
    /// * `updates` - Map where keys are subjects and values are publishers
    pub fn extend_superstep_updates(&mut self, updates: HashMap<String, Vec<String>>) {
        for (task_name, subjects) in self.superstep_updates.iter_mut() {
            for (subject, publishers) in updates.iter() {
                for publisher in publishers.iter() {
                    if publisher != task_name && subjects.contains_key(subject) {
                        *subjects.get_mut(subject).unwrap() = true;
                    }
                }
            }
        }
    }

    /// Set the last superstep update
    pub fn set_superstep_updates(&mut self, updates: HashMap<String, HashMap<String, bool>>) {
        self.superstep_updates = updates;
    }

    /// Clear task from superstep update
    pub fn clear_subjects_from_task_for_superstep_updates(&mut self, task_name: &str) {
        let task_update = self.superstep_updates.get(task_name).unwrap();
        let task_update = task_update
            .iter()
            .map(|(s, _)| (s.to_string(), false))
            .collect::<HashMap<_, _>>();
        self.superstep_updates
            .insert(task_name.to_string(), task_update);
    }

    /// Update the state from the published messages
    /// and return a map of changed subscriptions along with their publishers
    #[instrument(skip(self, messages))]
    pub fn update_state_from_messages(
        &self,
        messages: IPCMessageMap,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut subjects_updated = HashMap::<String, Vec<String>>::new();
        for (_name, message) in messages.into_iter() {

            // Should the subject be updated?            
            let update = message.get_update().clone();
            if  update == TablePublish::None {
                continue;
            }

            // Try to update the state with the new record batches
            let table_name = message.get_update().get_table_name().to_string();
            if let Some(state) = self
                .session_context
                .get_states()
                .get(table_name.as_str())
            {
                let publisher = message.get_publisher().to_string();

                // Check for any inconsistencies in the message and intercept any errors
                let batches =  TableBuilder::new_from_ipc_stream(&message.get_message_own())?
                    .with_name(table_name.as_str())
                    .build()?
                    .get_record_batches_own();

                // Update the state
                // Check for a mismatch in the schema and intercept any errors
                state.write().update_table(batches, update)?;

                // Record the table name that was updated and the pubisher who updated it
                if let Some(v) = subjects_updated.get_mut(state.read().get_name()) {
                    v.push(publisher);
                } else {
                    subjects_updated.insert(
                        state.read().get_name().to_string(),
                        vec![publisher],
                    );
                }
            } else {
                // Mismatch in table names of the update and state
                return Err(anyhow!("Subject '{table_name}' with update '{update:?}' is not in the session state tables! Available tables are {:?}", self.session_context.get_states().keys()));
            }
        }
        Ok(subjects_updated)
    }

    /// Write superstep updates to file
    pub fn write_superstep_updates(&self, file: &mut File) -> Result<()> {
        // Convert the superstep updates to a record batch
        let mut task_vec = Vec::new();
        let mut subject_vec = Vec::new();
        let mut status_vec = Vec::new();
        for (task_name, subjects) in self.superstep_updates.iter() {
            for (subject_name, status) in subjects.iter() {
                task_vec.push(task_name.to_string());
                subject_vec.push(subject_name.to_string());
                status_vec.push(status.to_owned());
            }
        }
        let task_names: ArrayRef = Arc::new(StringArray::from(task_vec));
        let subject_names: ArrayRef = Arc::new(StringArray::from(subject_vec));
        let status_vec: ArrayRef = Arc::new(BooleanArray::from(status_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("task_name", task_names),
            ("subject_name", subject_names),
            ("status_value", status_vec),
        ])?;

        // Write to IPC file
        let table = Table::get_builder()
            .with_name("superstep_updates")
            .with_record_batches(vec![batch])?
            .build()?;
        table.to_ipc_file(file)
    }

    /// Read superstep updates to file
    pub fn read_superstep_updates(&mut self, file: &File) -> Result<()> {
        // Read in the IPC file
        let table = TableBuilder::new_from_ipc_file(file)?
            .with_name("superstep_updates")
            .build()?;

        // Extract out the data
        let task_vec = table.get_column_as_vec_str("task_name");
        let subject_vec = table.get_column_as_vec_str("subject_name");
        let status_vec = table
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("status_value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.superstep_updates = HashMap::<String, HashMap<String, bool>>::new();
        for iter in 0..task_vec.len() {
            let mut superstep_update = HashMap::<String, bool>::new();
            superstep_update.insert(
                subject_vec.get(iter).unwrap().to_string(),
                status_vec.get(iter).unwrap().to_owned(),
            );
            if self
                .superstep_updates
                .contains_key(task_vec.get(iter).unwrap().to_owned())
            {
                self.superstep_updates
                    .get_mut(task_vec.get(iter).unwrap().to_owned())
                    .unwrap()
                    .insert(
                        subject_vec.get(iter).unwrap().to_string(),
                        status_vec.get(iter).unwrap().to_owned(),
                    );
            } else {
                self.superstep_updates
                    .insert(task_vec.get(iter).unwrap().to_string(), superstep_update);
            }
        }
        Ok(())
    }

    /// Write the session stream state to disk
    pub fn write_state(&self, path: &str, tag: &str) -> Result<()> {
        // write the session context state
        match self.session_context.write_state(path, tag) {
            Ok(()) => (),
            Err(_e) => (),
        }

        // Prepare the file
        let pathname = format!(
            "{path}/{tag}-{}-superstep_updates",
            self.get_session_context().get_name()
        );
        let mut file = std::fs::File::create(pathname)?;

        // write the session context state
        match self.write_superstep_updates(&mut file) {
            Ok(()) => (),
            Err(e) => event!(Level::ERROR, "Error writing superstep updates: {e:?}"),
        }
        Ok(())
    }

    /// Read the session state from disk
    pub fn read_state(&mut self, path: &str, tag: &str) -> Result<()> {
        // write the session context state
        match self.session_context.read_state(path, tag) {
            Ok(()) => (),
            Err(_e) => (),
        }

        // Prepare the file
        let pathname = format!(
            "{path}/{tag}-{}-superstep_updates",
            self.get_session_context().get_name()
        );
        let file = std::fs::File::open(pathname)?;

        // write the session context state
        match self.read_superstep_updates(&file) {
            Ok(()) => (),
            Err(e) => event!(Level::ERROR, "Error reading superstep updates: {e:?}"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        session::session_context_builder::test_session_context_builder::{
            make_test_session_context_parallel_task,
            make_test_session_context_sequential_task,
        }, table::TablePublish, task::test_task::make_test_input_message
    };
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempfile;

    #[test]
    fn test_session_update_state() -> Result<()> {
        // Case 1: no state update
        let session_context =
            make_test_session_context_parallel_task("session_1", 25)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublish::None,
            true
        )?;
        let session_stream_step = SessionStreamState::new(session_context);
        let updates = session_stream_step.update_state_from_messages(input)?;

        // check the response
        assert!(updates.is_empty());
        assert_eq!(
            session_stream_step
                .get_session_context()
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
            session_stream_step
                .get_session_context()
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
            session_stream_step
                .get_session_context()
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
            session_stream_step
                .get_session_context()
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
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublish::Extend {
                table_name: "state_1".to_string(),
            },
            true
        )?;
        let updates = session_stream_step.update_state_from_messages(input)?;

        // check the response
        assert_eq!(updates.len(), 1);
        assert_eq!(updates.get("state_1").unwrap().len(), 1);
        assert_eq!(
            updates.get("state_1").unwrap().first().unwrap(),
            "session_1"
        );
        assert_eq!(
            session_stream_step
                .get_session_context()
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
            session_stream_step
                .get_session_context()
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
            session_stream_step
                .get_session_context()
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
            session_stream_step
                .get_session_context()
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
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",            
            &TablePublish::Extend { table_name: "state_1".to_string() },
            false
        )?;
        let updates = session_stream_step.update_state_from_messages(input);
        assert!(updates.is_err());

        // Case 4: Error due to mismatching table names
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",            
            &TablePublish::Extend { table_name: "NotFound".to_string() },
            true
        )?;
        let updates = session_stream_step.update_state_from_messages(input);
        assert!(updates.is_err());

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_session_read_write_superstep_update() -> Result<()> {
        // initialize the session stream state

        use parking_lot::RwLock;

        let session_context =
            make_test_session_context_parallel_task("session_1", 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));

        // write the session stream state to file
        let mut file = tempfile()?;
        session_stream_state
            .try_read()
            .unwrap()
            .write_superstep_updates(&mut file)?;

        // read the session stream state back to file
        let session_context =
            make_test_session_context_sequential_task("session_1", 4)?;
        let session_stream_state_test =
            Arc::new(RwLock::new(SessionStreamState::new(session_context)));

        assert_ne!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates(),
            session_stream_state_test
                .try_read()
                .unwrap()
                .get_superstep_updates()
        );
        session_stream_state_test
            .try_write()
            .unwrap()
            .read_superstep_updates(&file)?;
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates(),
            session_stream_state_test
                .try_read()
                .unwrap()
                .get_superstep_updates()
        );

        Ok(())
    }
}
