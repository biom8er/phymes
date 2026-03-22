use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_agents::{
    AvailableSessionPlans, SessionContext, SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream, SessionStreamStep, SessionStreamStepTrait, SubscriptionTrait, create_message_map
};
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, IPCMessageBuilder, JoinUserInboxSessionContextsMermaidDiagrams, MappableTrait, MessageBuilderTrait, Publication, Subject, SubjectBuilderTrait, SubjectTrait, Subscription, UserSubject, create_session_mermaid_batch, create_user_inbox_batch, create_user_session_contexts_batch
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
    pub users: Arc<SessionContext>,
}

impl UserState {
    /// Make a new [UserState] with an optional name for the user state
    ///   and initialize with the default user
    pub fn new(user_session_context_name: Option<&str>) -> impl std::future::Future<Output = Result<Self>> + Send { async move {
        let session_name = user_session_context_name.unwrap_or("Users");
        let (session_ctx_arc, session_messages) =
            AvailableSessionPlans::get_session_stream_state_by_name("Users", session_name)?;

        // Write the session messages to the store
        let _ = SessionStreamStep::update_subjects_and_changelog_from_messages(&session_ctx_arc, session_messages.unwrap_or_default()).await?;
        Ok(Self { users: session_ctx_arc })
    }}

    /// Get the user information by their email
    pub async fn get_user_by_email(
        &self,
        email: &str,
    ) -> Result<(
        Vec<UserSubject>,
        Vec<JoinUserInboxSessionContextsMermaidDiagrams>,
    )> {
        // Prepare the input message
        let batch = create_user_inbox_batch(vec![email.to_string()])?;
        let table = Subject::get_builder()
            .with_name(AvailableSubjects::UserInbox.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&Publication::Replace {
                subject_name: table.get_name().to_string(),
            })
            .with_publisher(self.users.get_name())
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![message]);

        // Run the tasks for the user session
        let session_stream = SessionStream::new(message_map, self.users.clone());
        let _response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // Parse out the results
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::User.to_string() }
			.subscribe_to_subject(self.users.runtime_env())?
			.ok_or(anyhow!("Unable to get the subject `{}` from object storage for session `{}` while getting the user by email.", 
				AvailableSubjects::User,
				self.users.get_name()
			))?
			.try_collect()
			.await?;
        let user = Subject::get_builder()
            .with_name(&AvailableSubjects::User.to_string())
            .with_record_batches(batches)?
            .build()?
            .to_struct::<UserSubject>()?;
        
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string() }
			.subscribe_to_subject(self.users.runtime_env())?
			.ok_or(anyhow!("Unable to get the subject `{}` from object storage for session `{}` while getting the user by email.", 
				AvailableSubjects::JoinUserInboxSessionContextsMermaid,
				self.users.get_name()
			))?
			.try_collect()
			.await?;
        let join = Subject::get_builder()
            .with_name(&AvailableSubjects::JoinUserInboxSessionContextsMermaid.to_string())
            .with_record_batches(batches)?
            .build()?
            .to_struct::<JoinUserInboxSessionContextsMermaidDiagrams>()?;

        Ok((user, join))
    }

    /// Get the user information by their email
    pub async fn update_user_session_contexts(
        &self,
        email: &str,
        session_context_name: &[String],
        flowchart_diagram: &[String],
        er_diagram: &[String],
        timestamp: &[i64],
    ) -> Result<()> {
        // Prepare the update messages
        let email_vec = session_context_name
            .iter()
            .map(|_| email.to_string())
            .collect::<Vec<_>>();
        let user_session_contexts =
            create_user_session_contexts_batch(email_vec, session_context_name.to_owned())?;
        let user_session_contexts_bytes = Subject::get_builder()
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
        let mermaid_bytes = Subject::get_builder()
            .with_record_batches(vec![mermaid])?
            .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
            .build()?
            .to_ipc_stream()?;

        // Create the update message
        let user_session_contexts_message = IPCMessageBuilder::new()
            .with_subject(AvailableSubjects::UserSessionContexts.to_string().as_str())
            .with_publisher(&create_session_name(email, self.users.get_name()))
            .with_message(user_session_contexts_bytes)
            .with_update(&Publication::Extend {
                subject_name: AvailableSubjects::UserSessionContexts.to_string(),
            })
            .make_name()?
            .build()?;
        let mermaid_message = IPCMessageBuilder::new()
            .with_subject(AvailableSubjects::BuilderMermaid.to_string().as_str())
            .with_publisher(&create_session_name(email, self.users.get_name()))
            .with_message(mermaid_bytes)
            .with_update(&Publication::Extend {
                subject_name: AvailableSubjects::BuilderMermaid.to_string(),
            })
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![user_session_contexts_message, mermaid_message]);

        // Update the session state with the new message
        let (changelog, meta, _errors) = self.users.update_subjects_from_messages(message_map).await;

        let mut messages = Vec::new();
        if let Some(subject) = changelog {
            let message = IPCMessageBuilder::new()
                .with_subject(subject.get_name())
                .with_publisher(&self.users.get_name())
                .with_update(&Publication::Extend {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
                .make_random_name()?
                .build()?;
            messages.push(message);
        }
        if let Some(subject) = meta {
            let message = IPCMessageBuilder::new()
                .with_subject(subject.get_name())
                .with_publisher(&self.users.get_name())
                .with_update(&Publication::Extend {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
                .make_random_name()?
                .build()?;
            messages.push(message);
        }

        let messages = create_message_map(messages);
        let _ = self.users.update_subjects_from_messages(messages).await;

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
    pub session_contexts: Arc<RwLock<HashMap<String, Arc<SessionContext>>>>,
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
            session_contexts: Arc::new(RwLock::new(
                HashMap::<String, Arc<SessionContext>>::new(),
            )),
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
    ) -> impl std::future::Future<Output = Result<Vec<String>>> + Send { async move {
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
                    let (session_ctx_arc, session_messages) = AvailableSessionPlans::get_session_stream_state_by_name(
                        &user_session_context.session_context_name,
                        &session_name,
                    )?;

                    // Write the session messages to the store
                    let _ = SessionStreamStep::update_subjects_and_changelog_from_messages(&session_ctx_arc, session_messages.unwrap_or_default()).await?;

                    // Add the session stream state to the state
                    let _ = self
                        .session_contexts
                        .try_write()
                        .unwrap()
                        .insert(session_name.to_string(), session_ctx_arc);
                    tracing::debug!(
                        "Creating session_context {} for session_name {} from AvailableSessionPlans",
                        &user_session_context.session_context_name,
                        &session_name
                    );
                } else {
                    // Build the session stream state with tables from Mermaid
                    // and leave the upload of configs and other initial session state to another step
                    // DM: turn agent subject tests back on after refactoring BuilderSession
                    let (session_context, session_messages) = SessionContextBuilder::from_mermaid_flowchart(
                        &user_session_context.flowchart_diagram,
                        false,
                    )?
                    .with_name(&session_name)
                    .with_subjects_from_mermaid_erdiagram(
                        &user_session_context.er_diagram,
                        false,
                        true,
                    )?
                    .add_processor_subjects()?
                    .add_session_interface(None)?
                    .with_diagnostics(true)
                    .build_with_tables()?;
                    let session_ctx_arc = Arc::new(session_context);

                    // Write the session messages to the store
                    let _ = SessionStreamStep::update_subjects_and_changelog_from_messages(&session_ctx_arc, session_messages.unwrap_or_default()).await?;

                    // Add the session stream state to the state
                    let _ = self
                        .session_contexts
                        .try_write()
                        .unwrap()
                        .insert(session_name.to_string(), session_ctx_arc);
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
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    use phymes_agents::make_example_mermaid_table;
    use phymes_diagnostics::HashSet;

    #[cfg(not(target_family = "wasm"))]
    use phymes_core::SubjectTrait;

    #[tokio::test]
    async fn test_server_state_update_user_session_contexts() -> Result<()> {
        let user = UserState::new(None).await?;
        let table = make_example_mermaid_table(true, false)?;
        user.update_user_session_contexts(
            "user@biom8er.com",
            &table.get_column_as_vec_nonprimitive::<String>("session_context_name")?,
            &table.get_column_as_vec_nonprimitive::<String>("flowchart_diagram")?,
            &table.get_column_as_vec_nonprimitive::<String>("er_diagram")?,
            &table.get_column_as_vec_primitive::<i64>("timestamp")?,
        ).await?;

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: "UserSessionContexts".to_string() }
            .subscribe_to_subject(user.users.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name("UserSessionContexts")
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.get_column_as_vec_str("email"),
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
        assert_eq!(subject.get_column_as_vec_str("session_context_name"),
            [
                "Chat", "DocChat", "ToolChat", "Builder", "Chat", "DocChat", "ToolChat"
            ]
        );
        assert_eq!(subject.get_column_as_vec_str("session_context_name"),
            [
                "Chat", "DocChat", "ToolChat", "Builder", "Chat", "DocChat", "ToolChat"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_server_state_get_user_by_email() -> Result<()> {
        let user = UserState::new(None).await?;
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
        let user = UserState::new(None).await?;
        let (_user_info, user_session_contexts) =
            user.get_user_by_email("contact@biom8er.com").await?;
        let mut state = ServerState::new();
        let session_names = state.make_session_contexts(&user_session_contexts, true).await?;
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
}
