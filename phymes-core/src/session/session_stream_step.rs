use anyhow::Result;
use arrow::record_batch::RecordBatch;
use futures::TryStreamExt;
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{Level, event, instrument};

use super::common_traits::{BuilderTrait, IPCMessageMap, MappableTrait, SendableRecordBatchStreamMessageMap, RunnableTrait};
use crate::metrics::HashMap;
use crate::schemas::error::{create_error_message_map, create_error_message_map_stream};
use crate::session::session_stream_state::SessionStreamState;
use crate::table::table_trait::{TableBuilder, TableBuilderTrait, TableTrait};
use crate::task::{
    message::{
        IPCMessage, IPCMessageBuilder, MessageBuilderTrait, MessageTrait,
        SendableRecordBatchStreamMessage,
    },
    publish_subscribe::PubSubTrait,
};

/// A single step of a [`SessionStream`]
pub struct SessionStreamStep {}

impl SessionStreamStep {
    /// Join the message streams using JointSet
    async fn join_message_streams(messages: SendableRecordBatchStreamMessageMap) -> Result<IPCMessageMap> {
        event!(Level::DEBUG, "Messages to join: {:?}.", &messages.keys());
        // Inspect each of the response futures
        let mut response_builder = HashMap::<String, IPCMessageBuilder>::new();
        let mut join_set = JoinSet::new();
        messages.into_iter().for_each(|(resp_name, resp)| {
            // Copy over name, source, destination for later building of the complete response
            let message = IPCMessageBuilder::new()
                .with_name(resp_name.as_str())
                .with_subject(resp.get_subject())
                .with_publisher(resp.get_publisher())
                .with_update(resp.get_update());
            let _ = response_builder.insert(resp_name.clone(), message);

            // Spawn the future
            join_set.spawn(async move {
                let result: Result<Vec<RecordBatch>> = resp.get_message_own().try_collect().await;
                (resp_name, result)
            });
        });

        // Collect each of the response RecordBatches
        let mut response_batches = HashMap::<String, IPCMessage>::new();
        // Note that currently this doesn't identify the thread that panicked
        //
        // TODO: Replace with [join_next_with_id](https://docs.rs/tokio/latest/tokio/task/struct.JoinSet.html#method.join_next_with_id
        // once it is stable
        while let Some(response) = join_set.join_next().await {
            match response {
                Ok((resp_name, resp)) => {
                    // Complete the input message with the processed stream
                    let table = TableBuilder::new()
                        .with_name(resp_name.as_str())
                        .with_record_batches(resp?)?
                        .build()?;
                    let message = response_builder
                        .remove(resp_name.as_str())
                        .unwrap()
                        .with_message(table.to_ipc_stream()?)
                        .build()?;
                    let message_map = message.to_map()?;
                    response_batches.extend(message_map);
                }
                Err(err) => {
                    // Intercept the error and forward to the error subject
                    event!(Level::ERROR, "{err}"); 
                    let message_map = create_error_message_map(&err.into(), "SessionStreamStep")?;
                    response_batches.extend(message_map);
                }
            }
        }

        Ok(response_batches)
    }

    /// Run a super-step
    ///
    /// Inspired by the Pregel model for large-scale graph processing, introduced
    /// by Google in a paper titled "Pregel: A System for Large-Scale Graph
    /// Processing" in 2010.
    ///
    /// The Pregel model is a distributed computing model for processing graph data
    /// in a distributed and parallel manner. It is designed for efficiently processing
    /// large-scale graphs with billions or trillions of vertices and edges.
    ///
    /// For agentic AI, and more generally, simulation of dynamic networks, greater
    /// complexity is required than the original Pregel models provides for.
    /// The additional complexity that is added by the `SessionContext`
    /// includes dynamical computational graph where edges are conditionally executed
    /// based on the outputs of nodes, state that can be shared between computational
    /// nodes besides the messages that are passed between nodes, and more granular
    /// control over the runtime environment for each node so that computations can
    /// be optimized based on the available hardware
    ///
    /// To account for the added complexity, the Pregal model is modified to align with a
    /// subject-based messaging paradigm which allows for publish-subscribe, request-reply,
    /// and queue group networking patterns found in production systems such as Kafka and NATS.io.
    ///
    /// # Components
    ///
    /// - Tasks: Represent the entities in the graph that subscribe to subjects,
    ///   perform computations on the subjects messages, and publish the resulting messages
    ///   to the state.
    ///
    /// - Subjects: The tables (data) that compose the state of the application.
    ///
    /// - Computation: Each task performs a user-defined computation during each
    ///   super-step as defined by the processor network and based on its subscriptions
    ///   that have changed in the previous super-step.
    ///
    /// - Messages: Subset of the state tables that are passed to tasks at each super-step.
    ///   Messages are used for communication and coordination between tasks.
    ///
    /// # Usage
    ///
    /// The algorithm follows a sequence of super-step, where each super-step consists
    /// of subscription, computation, and publishing. Tasks perform their computations
    /// in parallel according to which subscriptions were updated.
    /// The computation continues in a series of super-steps until a termination condition is met.
    ///
    /// # Returns
    ///
    /// [IPCMessageMap] if the the `Session` subsject was updated and None otherwise.
    #[instrument(skip(state, messages))]
    pub async fn run_superstep(
        state: Arc<RwLock<SessionStreamState>>,
        messages: IPCMessageMap,
    ) -> Result<Option<IPCMessageMap>> {
        // Update the state and handle any errors
        let update = state.write().update_state_from_messages(messages)?;
        state.write().extend_superstep_updates(update);

        // DM, TODO: initialize channels for metrics and logs

        // Iterate through each task and collect the resulting stream responses
        let mut session_streams = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let mut response_streams = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let mut tasks = Vec::new();
        for (task_name, task) in state.read().get_session_context().get_tasks().iter() {
            // Continue to the next task if all subscribed subjects are not updated
            let state_rwlock = state.read();
            let updates = state_rwlock.get_superstep_updates().get(task_name).unwrap();
            let states = state_rwlock.get_session_context().get_states();
            if !task.check_subscriptions(updates, states) {
                continue;
            } else {
                tasks.push(task_name.to_owned());
            }
            event!(Level::INFO, "Superstep for task {}", &task_name);           

            // Run the task and collect the stream responses
            let messages = task.get_subscriptions_from_state(updates, states);
            match task.run(messages) {
                Ok(result) => {
                    for (resp_name, resp) in result.into_iter() {
                        if task_name == state.read().get_session_context().get_name() {
                            session_streams.insert(resp_name, resp);
                        } else {
                            response_streams.insert(resp_name, resp);
                        }
                    }
                }
                Err(err) => {
                    // Intercept the error and wrap into a `SendableRecordBatch` for consumption
                    event!(Level::ERROR, "{} for task {}", err.to_string(), &task_name);
                    let message_map = create_error_message_map_stream(&err.into(), &task_name)?;
                    response_streams.extend(message_map);
                },
            }
        }        

        // DM, TODO: collect channel responses for metrics and logs
        //  and update the metric and log subjects

        // Break if there is nothing to update
        if session_streams.is_empty() && response_streams.is_empty() {
            return Ok(None);
        }

        // Remove the ran tasks from the update
        for task_name in tasks.iter() {
            state
                .write()
                .clear_subjects_from_task_for_superstep_updates(task_name.as_str());
        }

        // Join each of the response futures
        let response_batches = SessionStreamStep::join_message_streams(response_streams).await?;

        // Update the state and handle any errors
        let update = state.write().update_state_from_messages(response_batches)?;
        state.write().extend_superstep_updates(update);

        // Increment the step
        let iter = state.read().get_iter() + 1;
        state.write().set_iter(iter);

        // Return the session stream if any
        if session_streams.is_empty() {
            return Ok(Some(HashMap::<String, IPCMessage>::new()));
        } else {
            // Join each of the session futures
            let session_batches = SessionStreamStep::join_message_streams(session_streams).await?;
            return Ok(Some(session_batches));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ArrowTaskMetricsSet;
    use crate::schemas::available_subjects::{AvailableSubjects, AvailableSubjectsTrait};
    use crate::session::session_context::SessionContextTableNames;
    use crate::session::session_context_builder::{SessionContextBuilder, SessionContextBuilderTrait, TaskPlan};
    use crate::table::table_subscribe::{AllTableNamesSubscribe, SubscribeTrait, TableSubscribe};
    use crate::task::processor::test_processor::{ProcessorError, ProcessorMock};
    use crate::task::processor::ProcessorTrait;
    use crate::task::task_trait::test_task::{make_runtime_env, make_state_tables};
    use crate::{
        session::session_context_builder::test_session_context_builder::{
            make_test_session_context_parallel_task,
            make_test_session_context_sequential_task,
        },
        table::table_publish::TablePublish,
        task::task_trait::test_task::make_test_input_message,
    };

    #[tokio::test]
    async fn test_session_run_superstep_no_state_update() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublish::None,
                true
            )?,
        )
        .await?;
        assert!(response.is_none());

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 0);

        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
        assert!(metrics.clone_inner().output_rows().is_none());
        assert!(metrics.clone_inner().elapsed_compute().is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_extend_state_update_single_task() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublish::Extend {
                    table_name: "state_1".to_string(),
                },
                true
            )?,
        )
        .await?
        .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 1);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            12
        ); // Originally 3
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            5
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 30);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_single_task() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublish::Replace {
                    table_name: "state_1".to_string(),
                },
                true
            )?,
        )
        .await?
        .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 1);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
            5
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 15);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_parallel_tasks() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        // Superstep 1
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 4)?;
        let mut input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublish::Replace {
                table_name: "state_1".to_string(),
            },
            true
        )?;
        input.extend(make_test_input_message(
            "task_2",
            "session_1",
            "state_2",
            "state_2",
            &TablePublish::Replace {
                table_name: "state_2".to_string(),
            },
            true
        )?);
        input.extend(make_test_input_message(
            "task_3",
            "session_1",
            "state_3",
            "state_3",
            &TablePublish::Replace {
                table_name: "state_3".to_string(),
            },
            true
        )?);
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await?
            .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 1);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            session_stream_state
                .try_read()
                .unwrap()
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
            5
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            5
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            5
        );
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 45);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        // Superstep 2
        let mut response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        // check the response
        assert_eq!(response.len(), 3);
        assert_eq!(
            response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_name(),
            "from_session_1_on_state_1"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_publisher(),
            "session_1"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_subject(),
            "state_1"
        );
        assert_eq!(
            *response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_update(),
            TablePublish::Extend {
                table_name: "state_1".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 6);

        assert_eq!(
            response
                .get("from_session_1_on_state_2")
                .unwrap()
                .get_name(),
            "from_session_1_on_state_2"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_2")
                .unwrap()
                .get_publisher(),
            "session_1"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_2")
                .unwrap()
                .get_subject(),
            "state_2"
        );
        assert_eq!(
            *response
                .get("from_session_1_on_state_2")
                .unwrap()
                .get_update(),
            TablePublish::Extend {
                table_name: "state_2".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_2")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 6);

        assert_eq!(
            response
                .get("from_session_1_on_state_3")
                .unwrap()
                .get_name(),
            "from_session_1_on_state_3"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_3")
                .unwrap()
                .get_publisher(),
            "session_1"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_3")
                .unwrap()
                .get_subject(),
            "state_3"
        );
        assert_eq!(
            *response
                .get("from_session_1_on_state_3")
                .unwrap()
                .get_update(),
            TablePublish::Extend {
                table_name: "state_3".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_3")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 6);

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 2);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        ); // The same as superstep 1
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            5
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            5
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            5
        );
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 63);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_sequential_tasks() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        // Superstep 1
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_sequential_task("session_1", metrics.clone(), 4)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublish::Replace {
                table_name: "state_1".to_string(),
            },
            true
        )?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await?
            .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 1);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            12
        ); // Originally 3
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            5
        );
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 45);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        // Supersteps 2, 3, and 4
        let _ = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?;
        assert_eq!(session_stream_state.read().get_iter(), 2);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        let _ = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?;
        assert_eq!(session_stream_state.read().get_iter(), 3);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        let mut response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?
        .unwrap();

        // check the response
        assert_eq!(response.len(), 1);
        assert_eq!(
            response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_name(),
            "from_session_1_on_state_1"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_publisher(),
            "session_1"
        );
        assert_eq!(
            response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_subject(),
            "state_1"
        );
        assert_eq!(
            *response
                .get("from_session_1_on_state_1")
                .unwrap()
                .get_update(),
            TablePublish::Extend {
                table_name: "state_1".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 8);

        // check the session and state
        assert_eq!(session_stream_state.read().get_iter(), 4);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            768
        ); // Originally 3
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
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
            8
        );
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 5385);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        Ok(())
    }    

    #[tokio::test]
    async fn test_session_run_superstep_schema_mismatch_error() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_sequential_task("session_1", metrics.clone(), 4)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublish::Replace {
                table_name: "state_1".to_string(),
            },
            false
        )?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await;
        assert!(response.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_processor_error() -> Result<()> {
        // Create an error emitting session plan
        let task_plans = vec![
            TaskPlan {
                task_name: "task_1".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["processor_1".to_string()],
            },
            TaskPlan {
                task_name: "task_2".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["error_1".to_string()],
            },
        ];
        let processors = vec![
            ProcessorMock::new_arc_with_pub_sub(
                "processor_1",
                &[TablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "config_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorError::new_arc_with_pub_sub("error_1",
                &[TablePublish::Extend { table_name: "state_1".to_string() }],
                &[TableSubscribe::OnUpdateFullTable { table_name: "state_1".to_string() }],
            AllTableNamesSubscribe::new_box()
        )];
        let state = make_state_tables("state_1", "config_1")?;
        let metrics = ArrowTaskMetricsSet::new();
        let mut session_context = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .with_metrics(metrics)
            .with_max_iter(1)
            .build()?;

        // Run the session context without adding the Error table
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublish::Replace {
                table_name: "state_1".to_string(),
            },
            true
        )?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context.clone())));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input.clone())
            .await;
        assert!(response.is_err());

        // Add the Error table and retry
        session_context.state.insert(SessionContextTableNames::Errors.to_string(), Arc::new(RwLock::new(AvailableSubjects::Errors.to_table(None, None)?)));
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await?.unwrap();
        
        assert!(response.is_empty());
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get(SessionContextTableNames::Errors.to_string().as_str())
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            1
        );

        Ok(())
    }
}
