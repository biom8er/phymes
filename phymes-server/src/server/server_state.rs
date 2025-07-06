// General imports
use anyhow::{Result, anyhow};
use parking_lot::RwLock;
use phymes_agents::session_plans::available_session_plans::AvailableSessionPlans;
use std::sync::Arc;

// From crates
use phymes_core::{metrics::HashMap, session::session_context::SessionStreamState};

use crate::handlers::sign_in::{create_session_name, test_sign_in_handler};

#[derive(Clone)]
pub struct ServerState {
    /// Session context
    /// HashMap of sessions indexed by session name
    ///   where the session name = session_name + user_name
    pub session_contexts: Arc<RwLock<HashMap<String, Arc<RwLock<SessionStreamState>>>>>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            session_contexts: Arc::new(RwLock::new(HashMap::<
                String,
                Arc<RwLock<SessionStreamState>>,
            >::new())),
        }
    }

    /// Check that state for the email exists
    ///
    /// # Arguments
    ///
    /// `email` - &str, the user email
    pub fn check_email_in_state(&self, email: &str) -> bool {
        if let Some((_session_plans, session_names)) = self.get_session_names_by_email(email) {
            let missing = self.find_session_names_not_in_state(&session_names);
            missing.is_empty()
        } else {
            false
        }
    }

    /// Find missing session_names in the state
    ///
    /// # Arguments
    ///
    /// `session_names` - &[String], vector of session_names
    ///
    /// # Returns
    ///
    /// Vec<String> of missing session_names
    pub fn find_session_names_not_in_state(&self, session_names: &[String]) -> Vec<String> {
        let mut missing = Vec::new();
        for session_name in session_names.iter() {
            if !self
                .session_contexts
                .try_read()
                .unwrap()
                .contains_key(session_name)
            {
                missing.push(session_name.to_owned());
            }
        }
        missing
    }

    /// Get the sessions by email
    ///
    /// # Arguments
    ///
    /// `email` - &str, the user email
    ///
    /// # Returns
    ///
    /// Option<(Vec<String>, Vec<String>)> of created (session_plans, session_names)
    pub fn get_session_names_by_email(&self, email: &str) -> Option<(Vec<String>, Vec<String>)> {
        match test_sign_in_handler::retrieve_session_plans_by_email(email) {
            Some(session_plans) => {
                let mut session_names = Vec::new();
                for session_plan in session_plans.iter() {
                    let session_name = create_session_name(email, session_plan.as_str());
                    session_names.push(session_name);
                }
                Some((session_plans, session_names))
            }
            None => None,
        }
    }

    /// Create the sessions by email
    ///
    /// # Arguments
    ///
    /// `email` - &str, the user email
    ///
    /// # Returns
    ///
    /// Option<(Vec<String>, Vec<String>)> of created (session_plans, session_names)
    pub fn create_session_names_by_email(
        &mut self,
        email: &str,
    ) -> Option<(Vec<String>, Vec<String>)> {
        match test_sign_in_handler::retrieve_session_plans_by_email(email) {
            Some(session_plans) => {
                // Add user sessions to the state if it does not exist
                let mut session_names = Vec::new();
                for session_plan in session_plans.iter() {
                    let session_name = create_session_name(email, session_plan.as_str());
                    session_names.push(session_name.clone());
                    let _ = self.session_contexts.try_write().unwrap().insert(
                        session_name.clone(),
                        AvailableSessionPlans::get_session_stream_state_by_name(
                            session_plan.as_str(),
                            session_name.as_str(),
                        )
                        .unwrap(),
                    );
                    tracing::debug!(
                        "Creating session_plan {} for session_name {}",
                        session_plan.as_str(),
                        session_name.as_str()
                    );
                }
                Some((session_plans, session_names))
            }
            None => None,
        }
    }

    /// Read the session state by email
    ///
    /// # Arguments
    ///
    /// `path` - &str, the path to the files
    /// `email` - &str, the user email
    pub fn read_state_by_email(&mut self, path: &str, email: &str) -> Result<()> {
        if let Some((_session_plans, session_names)) = self.create_session_names_by_email(email) {
            for session_name in session_names.iter() {
                self.session_contexts
                    .try_write()
                    .unwrap()
                    .get_mut(session_name)
                    .unwrap()
                    .try_write()
                    .unwrap()
                    .read_state(path, email)?;
            }
            Ok(())
        } else {
            Err(anyhow!("Could not read state for email {email}"))
        }
    }

    /// Write the session state by email
    ///
    /// # Arguments
    ///
    /// `path` - &str, the path to the files
    /// `email` - &str, the user email
    pub fn write_state_by_email(&self, path: &str, email: &str) -> Result<()> {
        if let Some((_session_plans, session_names)) = self.get_session_names_by_email(email) {
            for session_name in session_names.iter() {
                self.session_contexts
                    .try_read()
                    .unwrap()
                    .get(session_name)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .write_state(path, email)?;
            }
            Ok(())
        } else {
            Err(anyhow!("Could not write state for email {email}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phymes_core::metrics::HashSet;

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    use phymes_core::{session::common_traits::MappableTrait, table::arrow_table::ArrowTableTrait};

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    use tempfile::tempdir;

    #[test]
    fn test_server_state_get_session_names_by_email() {
        let state = ServerState::new();
        let (session_plans, session_names) = state
            .get_session_names_by_email("myemail@gmail.com")
            .unwrap();
        assert_eq!(session_plans, &["Chat", "DocChat", "ToolChat"]);
        assert_eq!(
            session_names,
            &[
                "myemail@gmail.comChat",
                "myemail@gmail.comDocChat",
                "myemail@gmail.comToolChat"
            ]
        );
    }

    #[test]
    fn test_server_state_create_session_names_by_email() {
        let mut state = ServerState::new();

        // Test creation of state
        let (session_plans, session_names) = state
            .create_session_names_by_email("myemail@gmail.com")
            .unwrap();
        assert_eq!(session_plans, &["Chat", "DocChat", "ToolChat"]);
        assert_eq!(
            session_names
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>(),
            [
                "myemail@gmail.comChat",
                "myemail@gmail.comToolChat",
                "myemail@gmail.comDocChat"
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            state
                .session_contexts
                .try_read()
                .unwrap()
                .keys()
                .map(|s| s.to_owned())
                .collect::<HashSet<_>>(),
            session_names
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>()
        );

        // Test that we can find all session_names
        let missing = state.find_session_names_not_in_state(&session_names);
        assert!(missing.is_empty());

        // Test that we can find the missing session_names
        let _ = state
            .session_contexts
            .try_write()
            .unwrap()
            .remove("myemail@gmail.comChat")
            .unwrap();
        let missing = state.find_session_names_not_in_state(&session_names);
        assert_eq!(missing, &["myemail@gmail.comChat"]);
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    #[test]
    fn test_server_state_read_write_state() -> Result<()> {
        // Create the state
        let mut state = ServerState::new();
        let (_session_plans, _session_names) = state
            .create_session_names_by_email("myemail@gmail.com")
            .unwrap();

        // Write the state to disk
        let tmp_dir = tempdir()?;
        state.write_state_by_email(tmp_dir.path().to_str().unwrap(), "myemail@gmail.com")?;

        // Read in the state
        let mut state_empty = ServerState::new();
        state_empty.read_state_by_email(tmp_dir.path().to_str().unwrap(), "myemail@gmail.com")?;

        let state_keys = state
            .session_contexts
            .try_read()
            .unwrap()
            .keys()
            .map(|s| s.to_owned())
            .collect::<Vec<_>>();
        for key in state_keys.iter() {
            let subjects = state
                .session_contexts
                .try_read()
                .unwrap()
                .get(key)
                .unwrap()
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .keys()
                .map(|s| s.to_owned())
                .collect::<Vec<_>>();
            for subject in subjects.iter() {
                assert_eq!(
                    state
                        .session_contexts
                        .try_read()
                        .unwrap()
                        .get(key)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(subject)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_record_batches(),
                    state_empty
                        .session_contexts
                        .try_read()
                        .unwrap()
                        .get(key)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(subject)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_record_batches()
                );
                assert_eq!(
                    state
                        .session_contexts
                        .try_read()
                        .unwrap()
                        .get(key)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(subject)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_schema(),
                    state_empty
                        .session_contexts
                        .try_read()
                        .unwrap()
                        .get(key)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(subject)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_schema()
                );
                assert_eq!(
                    state
                        .session_contexts
                        .try_read()
                        .unwrap()
                        .get(key)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(subject)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_name(),
                    state_empty
                        .session_contexts
                        .try_read()
                        .unwrap()
                        .get(key)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(subject)
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_name()
                );
            }
        }
        tmp_dir.close()?;
        Ok(())
    }
}
