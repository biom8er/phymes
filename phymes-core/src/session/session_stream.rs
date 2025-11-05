use anyhow::Result;
use futures::{FutureExt, Stream};
use parking_lot::RwLock;
use phymes_diagnostics::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use tracing::{Level, event};

use super::common_traits::IPCMessageMap;
use crate::session::session_stream_state::SessionStreamState;
use crate::session::session_stream_step::SessionStreamStep;
use crate::task::IPCMessage;

pub struct SessionStream {
    /// The state
    state: Arc<RwLock<SessionStreamState>>,
    /// The next result
    #[allow(clippy::type_complexity)]
    next: Option<Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>>,
}

impl SessionStream {
    pub fn new(input: IPCMessageMap, state: Arc<RwLock<SessionStreamState>>) -> Self {
        #[allow(clippy::type_complexity)]
        let next: Option<
            Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>,
        > = Some(Box::pin(SessionStreamStep::run_superstep(
            Arc::clone(&state),
            input,
        )));
        Self { state, next }
    }
}

impl Stream for SessionStream {
    type Item = Result<IPCMessageMap>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Get the current iter
        let mut iter = self.state.read().get_iter();
        let max_iter = self.state.read().get_session_context().get_max_iter();
        while iter < max_iter {
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

            // Prepare the next itme
            self.next = Some(Box::pin(SessionStreamStep::run_superstep(
                Arc::clone(&self.state),
                HashMap::<String, IPCMessage>::new(),
            )));
            iter = self.state.read().get_iter();

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
        (
            1,
            Some(self.state.read().get_session_context().get_max_iter()),
        )
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;

    use super::*;
    use crate::{
        schemas::AvailableSubjects,
        session::{
            common_traits::{BuilderTrait, MappableTrait},
            session_context_builder::test_session_context_builder::make_test_session_context_sequential_task,
        },
        table::{TableBuilder, TableBuilderTrait, TablePublication, TableTrait},
        task::{MessageTrait, test_task::make_test_input_message},
    };

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
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let session_stream = SessionStream::new(input, session_stream_state.clone());
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
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
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
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
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
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionEvents.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            4
        );

        // Check gantt
        {
            let update = session_stream_state
                .write()
                .get_session_context_mut()
                .update_metrics_mermaid_gantt_table()?;
            assert!(update);
            let sss = session_stream_state.read();
            let metrics_table = sss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::MetricPivot.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap();
            let output_rows = metrics_table.get_column_as_vec_primitive::<i64>("output_rows")?;
            assert_eq!(output_rows.iter().sum::<i64>(), 5385);
            let gantt = sss
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::MetricMermaidGantt.to_string().as_str())
                .unwrap()
                .read();
            assert!(gantt.get_column_as_vec_str("processor_traces").join("").contains("gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tProcessor Traces\n\n\tsection Traces[ns]\n\t"));
            assert!(gantt.get_column_as_vec_str("elapsed_compute").join("").contains("gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tElapsed compute\n\n\tsection Time[ns]\n\t"));
            assert!(gantt.get_column_as_vec_str("output_rows").join("").contains("gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tRow count\n\n\tsection Counts\n\t"));
        }

        Ok(())
    }
}
