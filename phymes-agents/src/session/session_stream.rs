use std::{future::Future, pin::Pin, sync::Arc, task::{Context, Poll, ready}};

use anyhow::Result;
use futures::{FutureExt, Stream};
use parking_lot::RwLock;
use phymes_core::{IPCMessage, IPCMessageMap};
use phymes_diagnostics::HashMap;
use tracing::{Level, event};

use crate::{SessionContext, SessionStreamStep, session::session_stream_step::SessionStreamStepTrait};

pub struct SessionStream {
    /// The session context
    session_context: Arc<RwLock<SessionContext>>,
    /// The next superstep
    next_step: Option<Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>>,
    /// The maximum number of supersteps
    max_steps: usize,
    /// The current dynamic step
    step: usize,
}

impl SessionStream {
    /// New [SessionStream]
    pub fn new(messages: IPCMessageMap, session_context: Arc<RwLock<SessionContext>>) -> Self {
        let max_steps = session_context.read().get_max_iter();
        let step = 0;
        #[allow(clippy::type_complexity)]
        let next_step: Option<Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>> = Some(Box::pin(SessionStreamStep::run_superstep(Arc::clone(&session_context), messages, step)));

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
                self.step
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
        event!(Level::DEBUG, "Maximum iterations {} exeeded.", self.max_steps);
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.max_steps))
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use phymes_core::{
        AvailableSubjects, BuilderTrait, MappableTrait, MessageTrait, TableBuilder,
        TableBuilderTrait, TablePublication, TableTrait, test_task::make_test_input_message,
    };

    use super::*;
    use crate::{SessionContextBuilderAgentsTrait, SessionContextBuilderTrait, test_session_context_builder::make_test_session_context_builder_sequential};

    #[tokio::test]
    async fn test_session_stream_replace_state_update_sequential_tasks() -> Result<()> {
        // Build the session
        let session_context = make_test_session_context_builder_sequential("session_1", 2)?
            .with_diagnostics(true)
            .add_session_interface(Some(&["state_1"]))?
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Replace {
                table_name: "state_1".to_string(),
            },
            true,
        )?;
        let session_context_arc = Arc::new(RwLock::new(session_context));
        let session_stream = SessionStream::new(input, Arc::clone(&session_context_arc));
        let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // Check the response
        assert_eq!(response.len(), 2);
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
            TablePublication::Extend {
                table_name: "state_1".to_string()
            }
        );
        let bytes = response
            .pop()
            .unwrap()
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 7);
        let bytes = response
            .pop()
            .unwrap()
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 4);

        // check the session and session_context
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            12
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            10
        );

        // Check the traces, events, and metrics tables
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SessionTasksRunLog.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            5
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            8
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SubjectsNumRows.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            1
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SessionMetrics.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            2
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SessionErrors.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            0
        );

        Ok(())
    }
}
