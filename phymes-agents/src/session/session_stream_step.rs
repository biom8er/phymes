use anyhow::{Result, anyhow};
use arrow::record_batch::RecordBatch;
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuilderTrait, IPCMessage, IPCMessageBuilder, IPCMessageMap, MappableTrait, MessageBuilderTrait, MessageTrait, RunnableTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, TableBuilder, TableBuilderTrait, TableTrait, create_error_message_map, create_error_message_map_stream, create_session_tasks_run_log_batch, create_subjects_change_log_batch
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, EventBuilderTrait, HashMap,
    SpanBuilder, TraceBuilderTrait, create_timestamp_micros,
};
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{Level, event, instrument};

use crate::{SessionStreamState, create_message_map};

/// A single step of a [`SessionStream`]
///
/// [`SessionStream`]: crate::session::session_stream::SessionStream
pub struct SessionStreamStep {}

impl SessionStreamStep {
    /// Join the message streams using JointSet
    async fn join_message_streams(
        messages: SendableRecordBatchStreamMessageMap,
    ) -> Result<IPCMessageMap> {
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
                    // Check the response
                    let message_map = match resp {
                        Ok(batches) => match TableBuilder::new()
                            .with_name(resp_name.as_str())
                            .with_record_batches(batches)
                        {
                            Ok(builder) => {
                                let table = builder.build()?;

                                // Complete the input message with the processed stream
                                let message = response_builder
                                    .remove(resp_name.as_str())
                                    .unwrap()
                                    .with_message(table.to_ipc_stream()?)
                                    .build()?;
                                message.to_map()?
                            }
                            Err(err) => create_error_message_map(&err, "SessionStreamStep", true)?,
                        },
                        Err(err) => create_error_message_map(&err, "SessionStreamStep", true)?,
                    };

                    // Add the message to the joined responses
                    response_batches.extend(message_map);
                }
                Err(err) => {
                    // Intercept the error and forward to the error subject
                    event!(Level::ERROR, "{err}");
                    let message_map =
                        create_error_message_map(&anyhow!("{err}"), "SessionStreamStep", true)?;
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
        // Initialize the channels for collecting the metrics, events, and traces)
        let mut diagnostics_vec = Vec::new();
        let span = SpanBuilder::default()
            .with_span(state.read().get_session_context().get_name())
            .build()?;

        // Create the diagnostics for the session step
        let collect_diagnostics = state.read().get_session_context().get_diagnostics();
        let trace = if collect_diagnostics {
            let diagnostics = Diagnostics::new();
            let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
            diagnostics_vec.push(diagnostics);

            // Trace the session step
            let trace = diagnostic_builder.clone().messages(
                line!(),
                file!(),
                state.read().get_session_context().get_name(),
            );
            trace.enter(&messages.values().collect::<Vec<_>>());
            let event = diagnostic_builder.clone().info(
                line!(),
                file!(),
                state.read().get_session_context().get_name(),
            );
            event.insert(
                "superstep",
                &serde_json::Value::Number(state.read().get_iter().into()),
            );
            Some(trace)
        } else {
            None
        };

        let mut response_streams = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        {  // Update the state and handle any errors (without locking the state)
            let update = match state.write().update_state_from_messages(messages) {
                Ok(update) => update,
                Err(err) => {
                    let message_map = create_error_message_map_stream(&err, span.span().0, true)?;
                    response_streams.extend(message_map);
                    AvailableSubjects::SubjectsChangeLog.to_table(None, None)?
                }
            };

            // Update the subjects change log
            let messages = create_message_map(vec![IPCMessageBuilder::new()
                .with_subject(update.get_name())
                .with_publisher("")
                .with_update(&phymes_core::TablePublication::Extend { table_name: update.get_name().to_string() })
                .with_message(update.to_ipc_stream()?)
                .make_random_name()?
                .build()?]);
            let _ = state.write().update_state_from_messages(messages)?;
        }

        // Retrieve the task ready to subscribe and their corresponding publications
        // DM: run the task session...
        let tasks = state.read().tasks_to_run()?;

        // Iterate through each task and collect the resulting stream responses
        let mut session_streams = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for ((task_name, session_name), processor_subjects_map) in tasks.iter() {
            event!(Level::INFO, "Superstep for task {}", &task_name);

            // Subscribe to the task subjects
            let task = state.read()
                .get_session_context()
                .get_tasks()
                .get(task_name)
                .expect(format!("Missing task `{task_name}` in session `{}` state.", state.read().get_session_context().get_name()).as_str())
                .clone();

            // Create the diagnostics for the task
            let diagnostic_builder = if collect_diagnostics {
                let diagnostics = Diagnostics::new();
                let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
                diagnostics_vec.push(diagnostics);
                Some(diagnostic_builder)
            } else {
                None
            };

            // Run the task and collect the stream responses
            // DM: need to add the publications as an additional parameter
            // DM: alternatively, let the tasks/processors handle the publications from the state
            match task.run(messages, diagnostic_builder.as_ref()) {
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
                    let message_map = create_error_message_map_stream(&err, task_name, true)?;
                    response_streams.extend(message_map);
                }
            }
        }

        {  // Update the tasks run log
            let (session_names, (task_names, timestamps)): (Vec<_>, (Vec<_>, Vec<_>)) = tasks.into_iter()
                .map(|((task_name, session_name), _ )| (session_name, (task_name, create_timestamp_micros())))
                .unzip();
            let tasks_run_log_batch = create_session_tasks_run_log_batch(session_names, task_names, timestamps)?;
            let tasks_run_log_table = AvailableSubjects::SessionTasksRunLog.to_table(None, Some(vec![tasks_run_log_batch]))?;
            let messages = create_message_map(vec![
                IPCMessageBuilder::new()
                    .with_subject(tasks_run_log_table.get_name())
                    .with_publisher("")
                    .with_update(&phymes_core::TablePublication::Extend { table_name: tasks_run_log_table.get_name().to_string() })
                    .with_message(tasks_run_log_table.to_ipc_stream()?)
                    .make_random_name()?
                    .build()?]);
            let state_update = state.write().update_state_from_messages(messages)?;

            // Update the subjects change log
            let messages = create_message_map(vec![
                IPCMessageBuilder::new()
                    .with_subject(state_update.get_name())
                    .with_publisher("")
                    .with_update(&phymes_core::TablePublication::Extend { table_name: state_update.get_name().to_string() })
                    .with_message(state_update.to_ipc_stream()?)
                    .make_random_name()?
                    .build()?]);
            let _ = state.write().update_state_from_messages(messages)?;
        }

        // Break if there is nothing to update
        if session_streams.is_empty() && response_streams.is_empty() {
            // Collect metrics, logs, and traces and update their corresponding subjects
            let (_metrics_updated, _traces_updated, _events_updated) = state
                .write()
                .get_session_context_mut()
                .update_metrics_table(&diagnostics_vec)?;

            Ok(None)
        } else {
            // Join each of the response futures
            let response_batches =
                match SessionStreamStep::join_message_streams(response_streams).await {
                    Ok(response_batches) => response_batches,
                    Err(err) => create_error_message_map(&err, span.span().0, true)?,
                };

            {  // Update the state and handle any errors (without locking the state)
                let mut error_messages = HashMap::<String, IPCMessage>::new();
                let state_update = match state.write().update_state_from_messages(response_batches) {
                    Ok(update) => update,
                    Err(err) => {
                        let message_map = create_error_message_map(&err, span.span().0, true)?;
                        error_messages.extend(message_map);
                        AvailableSubjects::SubjectsChangeLog.to_table(None, None)?
                    }
                };
                let errors_update = state.write().update_state_from_messages(error_messages)?;

                // Update the subjects change log
                let messages = create_message_map(vec![
                    IPCMessageBuilder::new()
                        .with_subject(state_update.get_name())
                        .with_publisher("")
                        .with_update(&phymes_core::TablePublication::Extend { table_name: state_update.get_name().to_string() })
                        .with_message(state_update.to_ipc_stream()?)
                        .make_random_name()?
                        .build()?,
                    IPCMessageBuilder::new()
                        .with_subject(errors_update.get_name())
                        .with_publisher("")
                        .with_update(&phymes_core::TablePublication::Extend { table_name: errors_update.get_name().to_string() })
                        .with_message(errors_update.to_ipc_stream()?)
                        .make_random_name()?
                        .build()?,
                    
                    ]);
                let _ = state.write().update_state_from_messages(messages)?;
            }

            // Join each of the response futures
            let session_batches = SessionStreamStep::join_message_streams(session_streams).await?;
            if let Some(trace) = trace {
                trace.exit(&session_batches.values().collect::<Vec<_>>());
            }

            // Collect metrics, logs, and traces and update their corresponding subjects
            let (_metrics_updated, _traces_updated, _events_updated) = state
                .write()
                .get_session_context_mut()
                .update_metrics_table(&diagnostics_vec)?;

            // Increment the step
            let iter = state.read().get_iter() + 1;
            state.write().set_iter(iter);

            Ok(Some(session_batches))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SessionContextBuilder, SessionContextBuilderTrait, TaskPlan,
        test_session_context_builder::{
            make_test_session_context_parallel_task, make_test_session_context_sequential_task,
        },
    };
    use phymes_core::{
        AvailableSubjects, AvailableSubjectsTrait, AvailableTableSubscribePolicies,
        ProcessorBuilder, TablePublication, TableSubscription,
        test_processor::{ProcessorError, ProcessorMock},
        test_task::{make_runtime_env, make_state_tables, make_test_input_message},
    };

    #[tokio::test]
    async fn test_session_run_superstep_no_state_update() -> Result<()> {
        let session_context = make_test_session_context_parallel_task("session_1", 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::None,
                true,
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
        assert!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionMetrics.to_string().as_str())
                .is_none()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_extend_state_update_single_task() -> Result<()> {
        let session_context = make_test_session_context_parallel_task("session_1", 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::Extend {
                    table_name: "state_1".to_string(),
                },
                true,
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
            1
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_single_task() -> Result<()> {
        let session_context = make_test_session_context_parallel_task("session_1", 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::Replace {
                    table_name: "state_1".to_string(),
                },
                true,
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
            1
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_parallel_tasks() -> Result<()> {
        // Superstep 1
        let session_context = make_test_session_context_parallel_task("session_1", 4)?;
        let mut input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Replace {
                table_name: "state_1".to_string(),
            },
            true,
        )?;
        input.extend(make_test_input_message(
            "task_2",
            "session_1",
            "state_2",
            "state_2",
            &TablePublication::Replace {
                table_name: "state_2".to_string(),
            },
            true,
        )?);
        input.extend(make_test_input_message(
            "task_3",
            "session_1",
            "state_3",
            "state_3",
            &TablePublication::Replace {
                table_name: "state_3".to_string(),
            },
            true,
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
            1
        );

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
            TablePublication::Extend {
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
            TablePublication::Extend {
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
            TablePublication::Extend {
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
            2
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_sequential_tasks() -> Result<()> {
        // Superstep 1
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
            1
        );

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
            TablePublication::Extend {
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

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_schema_mismatch_error() -> Result<()> {
        let session_context = make_test_session_context_sequential_task("session_1", 4)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Replace {
                table_name: "state_1".to_string(),
            },
            false,
        )?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response =
            SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input).await;
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
            ProcessorBuilder::default()
                .with_name("processor_1")
                .with_type("")
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "config_1".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build_arc::<ProcessorMock>()?,
            ProcessorBuilder::default()
                .with_name("error_1")
                .with_type("")
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[TableSubscription::OnUpdateFullTable {
                    table_name: "state_1".to_string(),
                }])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build_arc::<ProcessorError>()?,
        ];
        let state = make_state_tables("state_1", "config_1")?;
        let mut session_context = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .with_max_iter(1)
            .build()?;

        // Run the session context without adding the Error table
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
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(
            session_context.clone(),
        )));
        let response =
            SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input.clone())
                .await;
        assert!(response.is_err());

        // Add the Error table and retry
        session_context.state.insert(
            AvailableSubjects::SessionErrors.to_string(),
            Arc::new(RwLock::new(
                AvailableSubjects::SessionErrors.to_table(None, None)?,
            )),
        );
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await?
            .unwrap();

        assert!(response.is_empty());
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_session_context()
                .get_states()
                .get(AvailableSubjects::SessionErrors.to_string().as_str())
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
