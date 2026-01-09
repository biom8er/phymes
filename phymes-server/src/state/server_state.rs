use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_agents::{
    AvailableInterfaceSubjects, AvailableSessionPlans, SessionContextBuilder,
    SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
    SessionContextBuilderTrait, SessionStream, SessionStreamState, create_message_map,
};
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BlobBuilderTraitExt, BuildableTrait, BuilderTrait,
    IPCMessage, IPCMessageBuilder, JoinUserInboxSessionContextsMermaidDiagrams, JsonFormat,
    MappableTrait, MessageBuilderTrait, MessageTrait, Table, TableBuilder, TableBuilderTrait,
    TablePublication, TableTrait, UserSubject, create_session_mermaid_batch,
    create_user_inbox_batch, create_user_session_contexts_batch,
};
use phymes_diagnostics::HashMap;

use crate::handlers::create_session_name;

/// The user state
///
/// # Notes
///
/// The user state is independent of the [ServerState] so that it can be safetly
///   copied to middleware without creating locks
///
/// A default user "contact at biom8er dot com" is created upon initialization
#[derive(Clone)]
pub struct UserState {
    /// Users information
    pub users: Arc<RwLock<SessionStreamState>>,
}

impl Default for UserState {
    fn default() -> Self {
        Self::new(None)
    }
}

impl UserState {
    /// Make a new [UserState] with an optional name for the user state
    ///   and initialize with the default user
    pub fn new(user_session_context_name: Option<&str>) -> Self {
        let session_name = user_session_context_name.unwrap_or("Users");
        let users =
            AvailableSessionPlans::get_session_stream_state_by_name("Users", session_name).unwrap();
        Self { users }
    }

    /// Get the user information by their email
    pub async fn get_user_by_email(
        &self,
        email: &str,
    ) -> Result<(
        Vec<UserSubject>,
        Vec<JoinUserInboxSessionContextsMermaidDiagrams>,
    )> {
        // To prevent locks and other performance issues
        let session_context_name = self
            .users
            .read()
            .get_session_context()
            .get_name()
            .to_string();

        // Prepare the input message
        let batch = create_user_inbox_batch(vec![email.to_string()])?;
        let bytes = Table::get_builder()
            .with_name(AvailableInterfaceSubjects::UserJson.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?
            .to_json()?;
        let blob = AvailableInterfaceSubjects::UserJson
            .to_table_builder(None)
            .with_blob(None, Some("json"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&TablePublication::Replace {
                table_name: blob.get_name().to_string(),
            })
            .with_publisher(session_context_name.as_str())
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![blob_message]);

        // Run the tasks for the user session
        let session_stream = SessionStream::new(message_map, self.users.clone());
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // Parse the response
        let mut attachment_data = response
            .into_iter()
            .map(|mut r| {
                r.remove(&format!(
                    "from_{}_on_{}",
                    session_context_name.as_str(),
                    AvailableInterfaceSubjects::AssistantJson
                ))
            })
            .filter_map(|m| {
                if let Some(message) = m {
                    let bytes = TableBuilder::new_from_ipc_stream(&message.get_message_own())
                        .unwrap()
                        .with_name("")
                        .build()
                        .unwrap()
                        .get_column_as_vec_nested_primitive::<u8>("bytes")
                        .unwrap();
                    let json_format = JsonFormat::default();
                    let table = Table::get_builder()
                        .with_name("attachment_data")
                        .with_json(bytes.first().unwrap(), json_format.batch_size)
                        .unwrap()
                        .build()
                        .unwrap();
                    Some(table)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let user = attachment_data.swap_remove(0).to_struct::<UserSubject>()?;
        let join = attachment_data
            .swap_remove(0)
            .to_struct::<JoinUserInboxSessionContextsMermaidDiagrams>()?;

        // Reset the iter
        self.users.write().set_iter(0);

        Ok((user, join))
    }

    /// Get the user information by their email
    pub fn update_user_session_contexts(
        &self,
        email: &str,
        session_context_name: &[String],
        flowchart_diagram: &[String],
        er_diagram: &[String],
        timestamp: &[i64],
    ) -> Result<()> {
        // To prevent locks and other performance issues
        let session_plan = self
            .users
            .try_read()
            .unwrap()
            .get_session_context()
            .get_name()
            .to_string();

        // Prepare the update messages
        let email_vec = session_context_name
            .iter()
            .map(|_| email.to_string())
            .collect::<Vec<_>>();
        let user_session_contexts =
            create_user_session_contexts_batch(email_vec, session_context_name.to_owned())?;
        let user_session_contexts_bytes = Table::get_builder()
            .with_record_batches(vec![user_session_contexts])?
            .with_name(AvailableSubjects::UserSessionContexts.to_string().as_str())
            .build()?
            .to_ipc_stream()?;
        let mermaid = create_session_mermaid_batch(
            session_context_name.to_owned(),
            flowchart_diagram.to_owned(),
            er_diagram.to_owned(),
            timestamp.to_owned(),
        )?;
        let mermaid_bytes = Table::get_builder()
            .with_record_batches(vec![mermaid])?
            .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
            .build()?
            .to_ipc_stream()?;

        // Create the update message
        let user_session_contexts_message = IPCMessageBuilder::new()
            .with_subject(AvailableSubjects::UserSessionContexts.to_string().as_str())
            .with_publisher(&create_session_name(email, session_plan.as_str()))
            .with_message(user_session_contexts_bytes)
            .with_update(&TablePublication::Extend {
                table_name: AvailableSubjects::UserSessionContexts.to_string(),
            })
            .make_name()?
            .build()?;
        let mermaid_message = IPCMessageBuilder::new()
            .with_subject(AvailableSubjects::BuilderMermaid.to_string().as_str())
            .with_publisher(&create_session_name(email, session_plan.as_str()))
            .with_message(mermaid_bytes)
            .with_update(&TablePublication::Extend {
                table_name: AvailableSubjects::BuilderMermaid.to_string(),
            })
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![user_session_contexts_message, mermaid_message]);

        // Update the session state with the new message
        let update = self
            .users
            .try_write()
            .unwrap()
            .update_state_from_messages(message_map)
            .unwrap();


        // Update the subjects change log
        let messages = create_message_map(vec![
            IPCMessageBuilder::new()
                .with_name(update.get_name())
                .with_subject(update.get_name())
                .with_publisher("")
                .with_update(&phymes_core::TablePublication::Extend {
                    table_name: update.get_name().to_string(),
                })
                .with_message(update.to_ipc_stream().unwrap())
                .build().unwrap(),
        ]);
        let _ = self
            .users
            .try_write()
            .unwrap()
            .update_state_from_messages(messages)
            .unwrap();

        Ok(())
    }
}

/// The server state
///
/// # Notes
///
/// The server state is composed of two parts:
/// 1. the session contexts which store the available sessions for each user
/// 2. the user session names cache which store session context names per user
///
/// A default user "contact at biom8er dot com" is created upon initialization
#[derive(Clone)]
pub struct ServerState {
    /// Session context
    /// HashMap of sessions indexed by session name
    ///   where the session name = session_name + user_name
    pub session_contexts: Arc<RwLock<HashMap<String, Arc<RwLock<SessionStreamState>>>>>,
    /// Cache of user session_names indexed by user_name
    pub user_session_names: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerState {
    /// Make a new server state
    pub fn new() -> Self {
        Self {
            session_contexts: Arc::new(RwLock::new(HashMap::<
                String,
                Arc<RwLock<SessionStreamState>>,
            >::new())),
            user_session_names: Arc::new(RwLock::new(HashMap::<String, Vec<String>>::new())),
        }
    }

    /// Create the sessions
    ///
    /// # Arguments
    ///
    /// `user_session_contexts` - &[JoinUserInboxSessionContextsMermaidDiagrams], session plans to create for the user
    /// `make_session_contexts` - makes the session contexts if true or just returns the session names if false
    ///
    /// # Returns
    ///
    /// `Vec<String>` of created session_names
    pub fn make_session_contexts(
        &mut self,
        user_session_contexts: &[JoinUserInboxSessionContextsMermaidDiagrams],
        make_session_contexts: bool,
    ) -> Result<Vec<String>> {
        let mut session_names = Vec::new();
        for user_session_context in user_session_contexts {
            // Create the session name
            let session_name = create_session_name(
                &user_session_context.email,
                &user_session_context.session_context_name,
            );

            if make_session_contexts {
                // Create the session stream state if it does not yet exist
                if self
                    .session_contexts
                    .try_read()
                    .unwrap()
                    .contains_key(&session_name)
                {
                    tracing::debug!(
                        "Session_context {} already exists for session_name {}",
                        &user_session_context.session_context_name,
                        &session_name
                    );
                } else if AvailableSessionPlans::get_all_session_plan_names()
                    .contains(&user_session_context.session_context_name)
                {
                    // Prioritize the available session plans with initialized configs and other state
                    let session_stream_state =
                        AvailableSessionPlans::get_session_stream_state_by_name(
                            &user_session_context.session_context_name,
                            &session_name,
                        )?;

                    // Add the session stream state to the state
                    let _ = self
                        .session_contexts
                        .try_write()
                        .unwrap()
                        .insert(session_name.to_string(), session_stream_state);
                    tracing::debug!(
                        "Creating session_context {} for session_name {} from AvailableSessionPlans",
                        &user_session_context.session_context_name,
                        &session_name
                    );
                } else {
                    // Build the session stream state with tables from Mermaid
                    // and leave the upload of configs and other initial session state to another step
                    // DM: turn agent subject tests back on after refactoring BuilderSession
                    let session_context = SessionContextBuilder::from_mermaid_flowchart(
                        &user_session_context.flowchart_diagram,
                        false,
                    )?
                    .with_name(&session_name)
                    .with_state_from_mermaid_erdiagram(
                        &user_session_context.er_diagram,
                        false,
                        true,
                    )?
                    .add_processor_subjects()?
                    .add_session_interface(None)?
                    .with_diagnostics(true)
                    .build_with_tables()?;
                    let session_stream_state =
                        Arc::new(RwLock::new(SessionStreamState::new(session_context)));

                    // Add the session stream state to the state
                    let _ = self
                        .session_contexts
                        .try_write()
                        .unwrap()
                        .insert(session_name.to_string(), session_stream_state);
                    tracing::debug!(
                        "Creating session_context {} for session_name {} from mermaid diagrams.",
                        &user_session_context.session_context_name,
                        &session_name
                    );
                }

                // Update the cache if it exists
                if self
                    .user_session_names
                    .try_read()
                    .unwrap()
                    .contains_key(&user_session_context.email)
                {
                    self.user_session_names
                        .try_write()
                        .unwrap()
                        .get_mut(&user_session_context.email)
                        .unwrap()
                        .push(session_name.to_string());
                } else {
                    let _ = self.user_session_names.try_write().unwrap().insert(
                        user_session_context.email.to_string(),
                        vec![session_name.to_string()],
                    );
                }
            }
            session_names.push(session_name);
        }
        Ok(session_names)
    }

    /// Read the session state by email
    ///
    /// # Arguments
    ///
    /// `path` - &str, the path to the files
    /// `email` - &str, the user email
    pub fn read_session_contexts(&mut self, path: &str, email: &str) -> Result<()> {
        if let Some(session_names) = self.user_session_names.try_read().unwrap().get(email) {
            for session_name in session_names.iter() {
                self.session_contexts
                    .try_write()
                    .unwrap()
                    .get_mut(session_name)
                    .unwrap()
                    .try_write()
                    .unwrap()
                    .get_session_context_mut()
                    .read_state(path, email)?;
            }
        } else {
            return Err(anyhow!(
                "{email} was not found in the cache so no state was read from disk."
            ));
        }
        Ok(())
    }

    /// Write the session state by email
    ///
    /// # Arguments
    ///
    /// `path` - &str, the path to the files
    /// `email` - &str, the user email
    pub fn write_session_contexts(&self, path: &str, email: &str) -> Result<()> {
        if let Some(session_names) = self.user_session_names.try_read().unwrap().get(email) {
            for session_name in session_names.iter() {
                self.session_contexts
                    .try_read()
                    .unwrap()
                    .get(session_name)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_session_context()
                    .write_state(path, email)?;
            }
        } else {
            return Err(anyhow!(
                "{email} was not found in the cache so no state was written to disk."
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phymes_agents::make_example_mermaid_table;
    use phymes_diagnostics::HashSet;

    #[cfg(not(target_family = "wasm"))]
    use phymes_core::{MappableTrait, TableTrait};

    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempdir;

    #[test]
    fn test_server_state_update_user_session_contexts() -> Result<()> {
        let user = UserState::new(None);
        let table = make_example_mermaid_table(true, false)?;
        user.update_user_session_contexts(
            "user@biom8er.com",
            &table.get_column_as_vec_nonprimitive::<String>("session_context_name")?,
            &table.get_column_as_vec_nonprimitive::<String>("flowchart_diagram")?,
            &table.get_column_as_vec_nonprimitive::<String>("er_diagram")?,
            &table.get_column_as_vec_primitive::<i64>("timestamp")?,
        )?;

        assert_eq!(
            user.users
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("UserSessionContexts")
                .unwrap()
                .try_read()
                .unwrap()
                .get_column_as_vec_str("email"),
            [
                "contact@biom8er.com",
                "contact@biom8er.com",
                "contact@biom8er.com",
                "contact@biom8er.com",
                "user@biom8er.com",
                "user@biom8er.com",
                "user@biom8er.com"
            ]
        );
        assert_eq!(
            user.users
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("UserSessionContexts")
                .unwrap()
                .try_read()
                .unwrap()
                .get_column_as_vec_str("session_context_name"),
            [
                "Chat", "DocChat", "ToolChat", "Builder", "Chat", "DocChat", "ToolChat"
            ]
        );
        assert_eq!(
            user.users
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("BuilderMermaid")
                .unwrap()
                .try_read()
                .unwrap()
                .get_column_as_vec_str("session_context_name"),
            [
                "Chat", "DocChat", "ToolChat", "Builder", "Chat", "DocChat", "ToolChat"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_server_state_get_user_by_email() -> Result<()> {
        let user = UserState::new(None);
        let (user_info, user_session_contexts) =
            user.get_user_by_email("contact@biom8er.com").await?;
        assert_eq!(user_info.len(), 1);
        assert_eq!(user_session_contexts.len(), 4);
        assert_eq!(user_info.first().unwrap().email, "contact@biom8er.com");
        assert_eq!(user_info.first().unwrap().first_name, "con");
        assert_eq!(user_info.first().unwrap().last_name, "tact");
        assert_eq!(
            user_session_contexts.first().unwrap().email,
            "contact@biom8er.com"
        );
        assert_eq!(
            user_session_contexts.first().unwrap().session_context_name,
            "Builder"
        );
        assert_eq!(
            user_session_contexts.get(1).unwrap().email,
            "contact@biom8er.com"
        );
        assert_eq!(
            user_session_contexts.get(1).unwrap().session_context_name,
            "Chat"
        );
        assert_eq!(
            user_session_contexts.get(2).unwrap().email,
            "contact@biom8er.com"
        );
        assert_eq!(
            user_session_contexts.get(2).unwrap().session_context_name,
            "DocChat"
        );
        assert_eq!(
            user_session_contexts.get(3).unwrap().email,
            "contact@biom8er.com"
        );
        assert_eq!(
            user_session_contexts.get(3).unwrap().session_context_name,
            "ToolChat"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_server_state_make_session_contexts_from_mermaid_diagrams() -> Result<()> {
        let user = UserState::new(None);
        let (_user_info, user_session_contexts) =
            user.get_user_by_email("contact@biom8er.com").await?;
        let mut state = ServerState::new();
        let session_names = state.make_session_contexts(&user_session_contexts, true)?;
        assert_eq!(
            session_names
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>(),
            [
                "contactbiom8ercomDocChat",
                "contactbiom8ercomToolChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomBuilder"
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

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_server_state_read_write_state() -> Result<()> {
        // Create the state
        let user = UserState::new(None);
        let (_user_info, user_session_contexts) =
            user.get_user_by_email("contact@biom8er.com").await?;
        let mut state = ServerState::new();
        let _session_names = state.make_session_contexts(&user_session_contexts, true)?;

        // Write the state to disk
        let tmp_dir = tempdir()?;
        state.write_session_contexts(tmp_dir.path().to_str().unwrap(), "contact@biom8er.com")?;

        // Read in the state
        let mut state_empty = ServerState::new();
        assert!(
            state_empty
                .read_session_contexts(tmp_dir.path().to_str().unwrap(), "contact@biom8er.com")
                .is_err()
        );

        // Read in the state after initializing the cache
        let _session_names = state_empty.make_session_contexts(&user_session_contexts, true)?;
        state_empty
            .read_session_contexts(tmp_dir.path().to_str().unwrap(), "contact@biom8er.com")?;

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
