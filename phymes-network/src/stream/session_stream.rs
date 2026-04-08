use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::Result;
use futures::{FutureExt, Stream};
use phymes_message::{IPCMessage, IPCMessageMap};
use phymes_diagnostics::HashMap;
use tracing::{Level, event};

use crate::{SessionContext, SessionStreamStep, SessionStreamStepTrait};

pub struct SessionStream {
    /// The session context
    session_context: Arc<SessionContext>,
    /// The next superstep
    #[allow(clippy::type_complexity)]
    next_step: Option<Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>>,
    /// The maximum number of supersteps
    max_steps: usize,
    /// The current dynamic step
    step: usize,
}

impl SessionStream {
    /// New [SessionStream]
    pub fn new(messages: IPCMessageMap, session_context: Arc<SessionContext>) -> Self {
        let max_steps = session_context.get_max_steps();
        let step = 0;
        #[allow(clippy::type_complexity)]
        let next_step: Option<
            Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>,
        > = Some(Box::pin(SessionStreamStep::run_superstep(
            Arc::clone(&session_context),
            messages,
        )));

        Self {
            session_context,
            next_step,
            max_steps,
            step,
        }
    }
}

impl Stream for SessionStream {
    type Item = Result<IPCMessageMap>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // An internal while loop is needed to skip empty messages without causing an infinite poll
        while self.step < self.max_steps {
            // Poll the next superstep
            let messages = if let Some(fut) = self.next_step.as_mut() {
                match ready!(fut.poll_unpin(cx)) {
                    Ok(Some(messages)) => messages,
                    Ok(None) => return Poll::Ready(None),
                    Err(err) => {
                        event!(Level::ERROR, "{err:?}");
                        HashMap::<String, IPCMessage>::new()
                    }
                }
            } else {
                return Poll::Ready(None);
            };

            // Prepare the next superstep
            self.next_step = Some(Box::pin(SessionStreamStep::run_superstep(
                Arc::clone(&self.session_context),
                HashMap::<String, IPCMessage>::new(),
            )));
            self.step += 1;

            // Return the superstep result
            if messages.is_empty() {
                // Skip empty results
                continue;
            } else {
                return Poll::Ready(Some(Ok(messages)));
            }
        }
        event!(
            Level::DEBUG,
            "Maximum iterations {} exeeded.",
            self.max_steps
        );
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.max_steps))
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use phymes_core::{BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
    use phymes_schemas::AvailableSubjects;
    use phymes_message::MessageTrait;
    use phymes_event::{Publication, Subscription};
    use phymes_task::{SubscriptionTrait, test_task};

    use super::*;
    use crate::{SessionContextBuilderAgentsTrait, SessionContextBuilderTrait, test_session_context_builder};

    #[tokio::test]
    async fn test_session_stream_replace_state_update_sequential_tasks() -> Result<()> {
        // Build the session
        let (session_context, session_messages) =
            test_session_context_builder::make_test_session_context_builder_sequential(
                "session_1",
                2,
            )?
            .with_diagnostics(true)
            .add_session_interface(Some(&["state_1"]))?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_context);
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let messages = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::Replace {
                subject_name: "state_1".to_string(),
            },
            true,
        )?;
        let session_stream = SessionStream::new(messages, Arc::clone(&session_ctx_arc));
        let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // Check the response
        assert_eq!(response.len(), 1);
        assert_eq!(response.last().unwrap().len(), 1);
        assert_eq!(
            response
                .last()
                .unwrap()
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_name(),
            "from_session_1_on_state_1"
        );
        assert_eq!(
            response
                .last()
                .unwrap()
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_publisher(),
            "session_1"
        );
        assert_eq!(
            response
                .last()
                .unwrap()
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_subject(),
            "state_1"
        );
        assert_eq!(
            *response
                .last()
                .unwrap()
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_update(),
            Publication::Extend {
                subject_name: "state_1".to_string()
            }
        );
        let bytes = response
            .pop()
            .unwrap()
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = SubjectBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 7); // DM, Check!(): changed from 4

        // check the session and session_context
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6);
        assert_eq!(subscriptions.last().unwrap().num_rows(), 7);

        // Check the traces, events, and metrics tables
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3); // DM, Check!(): changed from 5
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 5); // DM, Check!(): changed from 8
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);

        Ok(())
    }
}
