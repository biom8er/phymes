use anyhow::{Result, anyhow};
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuilderTrait, IPCMessageMap, MappableTrait,
    MessageTrait, Table, TableBuilder, TableBuilderTrait, TablePublication, TablePublicationTrait,
    TableTrait, create_subjects_change_log_batch,
};
use phymes_diagnostics::create_timestamp_micros;

use crate::SessionContext;

/// State tracked during the course of running a [SessionStream]
///
/// [SessionStream]: super::session_stream::SessionStream
#[derive(Default, Debug, Clone)]
pub struct SessionStreamState {
    /// The session context
    session_context: SessionContext,
    /// The current iteration
    iter: usize,
}

impl SessionStreamState {
    pub fn new(session_context: SessionContext) -> Self {
        Self {
            session_context,
            iter: 0,
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

    /// Update the state from the published messages
    /// and return a map of changed subscriptions along with their publishers
    pub fn update_state_from_messages(&self, messages: IPCMessageMap) -> Result<Table> {
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
            if let Some(state) = self.session_context.get_states().get(table_name.as_str()) {
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
                session_names.push(self.get_session_context().get_name().to_string());
                num_rows_deltas.push(num_rows_old as i64 - num_rows_new as i64);
                timestamps.push(create_timestamp_micros());
            } else {
                // Mismatch in table names of the update and state
                return Err(anyhow!(
                    "Subject '{table_name}' with update '{update:?}' is not in the session state tables! Available tables are {:?}",
                    self.session_context.get_states().keys()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_session_context_builder::{
        make_test_session_context_parallel_task, make_test_session_context_sequential_task,
    };
    use parking_lot::RwLock;
    use phymes_core::{
        IPCMessage, TablePublication, test_table::make_test_table,
        test_task::make_test_input_message,
    };
    use phymes_diagnostics::HashMap;
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempfile;

    #[test]
    fn test_session_update_state() -> Result<()> {
        // Case 1: no state update
        let session_context = make_test_session_context_parallel_task("session_1", 25)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::None,
            true,
        )?;
        let session_stream_step = SessionStreamState::new(session_context);
        let updates = session_stream_step.update_state_from_messages(input)?;

        // check the response
        assert_eq!(updates.count_rows(), 0);
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
            &TablePublication::Extend {
                table_name: "state_1".to_string(),
            },
            true,
        )?;
        let updates = session_stream_step.update_state_from_messages(input)?;

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
            &TablePublication::Extend {
                table_name: "state_1".to_string(),
            },
            false,
        )?;
        let updates = session_stream_step.update_state_from_messages(input);
        assert!(updates.is_err());

        // Case 4: Error due to mismatching table names
        let message = IPCMessage::new(
            "task_1",
            "state_1",
            "session_1",
            Some(make_test_table("state_1", 4, 8, 3)?.to_ipc_stream()?),
            Some(TablePublication::Extend {
                table_name: "NotFound".to_string(),
            }),
        );
        let mut input = HashMap::<String, IPCMessage>::new();
        input.insert(message.get_name().to_string(), message);
        let updates = session_stream_step.update_state_from_messages(input);
        assert!(updates.is_err());

        Ok(())
    }
}
