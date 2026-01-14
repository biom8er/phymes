use std::{future::Future, pin::Pin, sync::Arc, task::{Context, Poll, ready}};

use anyhow::Result;
use futures::{FutureExt, Stream};
use parking_lot::RwLock;
use phymes_core::{IPCMessage, IPCMessageMap};
use phymes_diagnostics::HashMap;
use tracing::{Level, event};

use crate::{SessionContext, SessionStreamStep, create_message_map, session::session_stream_step::SessionStreamStepTrait};

/// The state of the [SessionStream]
pub enum SessionStreamState {
    Step(Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>),
    Message(HashMap<String, IPCMessage>),
    Done,
}

pub struct SessionStream {
    /// The current state of the stream
    stream_state: SessionStreamState,
    /// The session context
    session_context: Arc<RwLock<SessionContext>>,
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
        let next_step = Box::pin(SessionStreamStep::run_superstep(Arc::clone(&session_context), messages, step));
        let stream_state = SessionStreamState::Step(next_step);

        Self {
            stream_state,
            session_context,
            max_steps,
            step,
        }
    }
}

impl Stream for SessionStream {
    type Item = Result<IPCMessageMap>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match &mut self.stream_state {
            SessionStreamState::Step(step) => {
                // Poll the step
                let stream_state = match ready!(step.poll_unpin(cx)) {
                    Ok(Some(message)) => SessionStreamState::Message(message),
                    Ok(None) => SessionStreamState::Done,
                    Err(err) => {
                        event!(Level::ERROR, "{err:?}");
                        SessionStreamState::Message(HashMap::<String, IPCMessage>::new())
                    }
                };

                // Update the stream state
                self.stream_state = stream_state;
                self.poll_next(cx)
            },
            SessionStreamState::Message(message) => {
                // Prepare the poll
                let poll = Poll::Ready(Some(Ok(message.drain().collect::<HashMap<_, _>>())));

                // Prepare the next step
                dbg!(&self.step);
                if self.step < self.max_steps {
                    self.step += 1;
                    let stream_state = SessionStreamState::Step(Box::pin(SessionStreamStep::run_superstep(
                        Arc::clone(&self.session_context),
                        HashMap::<String, IPCMessage>::new(),
                        self.step,
                    )));
                    self.stream_state = stream_state;
                } else {
                    self.stream_state = SessionStreamState::Done;
                }

                // Return the poll
                poll
                
            }
            SessionStreamState::Done => {
                println!("Poll is Done");
                Poll::Ready(None)
            },
        }
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
    use crate::test_session_context_builder::make_test_session_context_sequential_task;

    #[tokio::test]
    async fn test_session_stream_replace_state_update_sequential_tasks() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        let session_context = make_test_session_context_sequential_task("session_1", 4)?;
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

        // check the response
        assert_eq!(response.len(), 3); //was 2...
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
            .get_mut(0)
            .unwrap()
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 6);

        // Check the traces, events, and metrics tables
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
            4
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SessionTraces.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            4
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(AvailableSubjects::SessionEvents.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            4
        );

        Ok(())
    }
}
