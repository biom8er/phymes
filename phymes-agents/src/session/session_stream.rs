use anyhow::Result;
use futures::{FutureExt, Stream};
use parking_lot::RwLock;
use phymes_core::{IPCMessage, IPCMessageMap};
use phymes_diagnostics::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use tracing::{Level, event};

use crate::{SessionContext, SessionStreamStep};

pub struct SessionStream {
    /// The session context
    session_context: Arc<RwLock<SessionContext>>,
    /// The next result
    #[allow(clippy::type_complexity)]
    next: Option<Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>>,
    /// The current iteration
    iter: usize,
}

impl SessionStream {
    pub fn new(input: IPCMessageMap, state: Arc<RwLock<SessionContext>>) -> Self {
        #[allow(clippy::type_complexity)]
        let next: Option<
            Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>,
        > = Some(Box::pin(SessionStreamStep::run_superstep(
            Arc::clone(&state),
            input,
            0,
        )));
        Self {
            session_context: state,
            next,
            iter: 0,
        }
    }
}

impl Stream for SessionStream {
    type Item = Result<IPCMessageMap>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Get the current iter
        let max_iter = self.session_context.read().get_max_iter();
        while self.iter < max_iter {
            // Poll the next item
            let res = if let Some(fut) = self.next.as_mut() {
                match ready!(fut.poll_unpin(cx)) {
                    Ok(Some(res)) => res,
                    Ok(None) => return Poll::Ready(None),
                    Err(err) => {
                        event!(Level::ERROR, "{err:?}");
                        println!("unhandled session step error: {err:?}");
                        HashMap::<String, IPCMessage>::new()
                    }
                }
            } else {
                return Poll::Ready(None);
            };

            // Prepare the next item
            self.iter += 1;
            self.next = Some(Box::pin(SessionStreamStep::run_superstep(
                Arc::clone(&self.session_context),
                HashMap::<String, IPCMessage>::new(),
                self.iter,
            )));

            // Return the poll
            if res.is_empty() {
                // Skip empty results
                continue;
            } else {
                return Poll::Ready(Some(Ok(res)));
            }
        }
        event!(Level::DEBUG, "Maximum iterations {} exeeded.", max_iter);
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.session_context.read().get_max_iter()))
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
        let session_stream = SessionStream::new(input, session_context_arc.clone());
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
