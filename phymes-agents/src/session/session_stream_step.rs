use anyhow::{Result, anyhow};
use arrow::record_batch::RecordBatch;
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuilderTrait, IPCMessage, IPCMessageBuilder, IPCMessageMap, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorSubjectsMap, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, TableBuilder, TableBuilderTrait, TablePublication, TableTrait, TaskTrait, create_error_message_map, create_error_message_map_stream, create_session_tasks_run_log_batch
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, EventBuilderTrait, HashMap, Span, SpanBuilder, TraceBuilderTrait, TraceRecord, create_timestamp_micros
};
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{Level, event};

use crate::{SessionContext, create_message_map, plans::TasksSubscribePublishSession};

/// Traits for running a static or dynamic [SessionStream] step
/// 
/// [SessionStream]: crate::session::session_stream::SessionStream
pub trait SessionStreamStepTrait {

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
    /// based on the outputs of nodes, session_context that can be shared between computational
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
    ///   to the session_context.
    ///
    /// - Subjects: The tables (data) that compose the session_context of the application.
    ///
    /// - Computation: Each task performs a user-defined computation during each
    ///   super-step as defined by the processor network and based on its subscriptions
    ///   that have changed in the previous super-step.
    ///
    /// - Messages: Subset of the session_context tables that are passed to tasks at each super-step.
    ///   Messages are used for communication and coordination between tasks.
    ///
    /// # Usage
    ///
    /// The algorithm follows a sequence of super-step, where each super-step consists
    /// of subscription, computation, and publishing. Tasks perform their computations
    /// in parallel according to which subscriptions were updated.
    /// The computation continues in a series of super-steps until a termination condition is met.
    /// 
    /// # Arguments
    /// 
    /// * `session_context` - [SessionContext] to use while running the current session stream super step
    /// * `messages` - [IPCMessageMap] input messages for the superstep
    /// * `step` - The current step of the session stream supersteps
    ///
    /// # Returns
    ///
    /// [IPCMessageMap] if any of the subscribing session sujects were updated and None otherwise.
    fn run_superstep(
        session_context: Arc<RwLock<SessionContext>>,
        messages: IPCMessageMap,
        step: usize,
    ) -> impl std::future::Future<Output = Result<Option<IPCMessageMap>>> + Send;

    /// Enter the superstep span generating the [Span], [TraceRecord], and [Diagnostics]
    fn enter_span(subject_messages: &IPCMessageMap, session_context: &Arc<RwLock<SessionContext>>, step: usize) -> Result<(Vec<Diagnostics>, Span, TraceRecord)> {
        // Create the span for the session
        let span = SpanBuilder::default()
            .with_span(session_context.read().get_name())
            .build()?;

        // Initialize the channels for collecting the metrics, events, and traces)
        let mut diagnostics_vec = Vec::new();
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        diagnostics_vec.push(diagnostics);

        // Trace the session step
        let trace = diagnostic_builder
            .clone()
            .messages(line!(), file!(), session_context.read().get_name());
        trace.enter(&subject_messages.values().collect::<Vec<_>>());
        let event = diagnostic_builder
            .clone()
            .info(line!(), file!(), session_context.read().get_name());
        event.insert("superstep", &serde_json::Value::Number(step.into()));

        Ok((diagnostics_vec, span, trace))
    }

    /// Exit the span
    fn exit_span(session_context: &Arc<RwLock<SessionContext>>, messages: &IPCMessageMap, diagnostics_vec: Vec<Diagnostics>, trace: TraceRecord) -> Result<()> {
        trace.exit(&messages.values().collect::<Vec<_>>());
        let (_metrics_updated, _traces_updated, _events_updated) = session_context.write().update_metrics_table(&diagnostics_vec)?;
        
        Ok(())
    }

    /// Update the session context subjects from messages including updating the subjects change log
    fn update_subjects_and_changelog_from_messages(session_context: &Arc<RwLock<SessionContext>>, messages: IPCMessageMap) -> Result<()> {
        // Update the session_context and handle any errors
        let mut error_messages = HashMap::<String, IPCMessage>::new();
        let session_context_name = session_context.read().get_name().to_string();
        let update = match session_context.write().update_subjects_from_messages(messages) {
            Ok(update) => update,
            Err(err) => {
                let message_map = create_error_message_map(&err, &session_context_name, true)?;
                error_messages.extend(message_map);
                AvailableSubjects::SubjectsChangeLog.to_table(None, None)?
            }
        };

        let mut messages = vec![IPCMessageBuilder::new()
            .with_subject(update.get_name())
            .with_publisher(&session_context_name)
            .with_update(&TablePublication::Extend {
                table_name: update.get_name().to_string(),
            })
            .with_message(update.to_ipc_stream()?)
            .make_random_name()?
            .build()?
        ];

        // Update the errors
        if !error_messages.is_empty() {
            let errors_update = session_context
                .write()
                .update_subjects_from_messages(error_messages)?;

            messages.push(IPCMessageBuilder::new()
                .with_subject(errors_update.get_name())
                .with_publisher(&session_context_name)
                .with_update(&TablePublication::Extend {
                    table_name: errors_update.get_name().to_string(),
                })
                .with_message(errors_update.to_ipc_stream()?)
                .make_random_name()?
                .build()?
            );
        }

        // Update the subjects change log
        let messages = create_message_map(messages);
        let _ = session_context.write().update_subjects_from_messages(messages)?;

        Ok(())
    }

    /// Update the session context subjects from the ran tasks including the subjects change log
    fn update_subjects_and_changelog_from_tasks(session_context: &Arc<RwLock<SessionContext>>, tasks: HashMap<(String, String), ProcessorSubjectsMap>) -> Result<()> {
        // Create the tasks run log message
        let session_context_name = session_context.read().get_name().to_string();
        let (session_names, (task_names, timestamps)): (Vec<_>, (Vec<_>, Vec<_>)) = tasks
            .into_iter()
            .map(|((task_name, session_name), _)| {
                (session_name, (task_name, create_timestamp_micros()))
            })
            .unzip();
        let tasks_run_log_batch =
            create_session_tasks_run_log_batch(session_names, task_names, timestamps)?;
        let tasks_run_log_table = AvailableSubjects::SessionTasksRunLog
            .to_table(None, Some(vec![tasks_run_log_batch]))?;
        let messages = create_message_map(vec![
            IPCMessageBuilder::new()
                .with_subject(tasks_run_log_table.get_name())
                .with_publisher(&session_context_name)
                .with_update(&TablePublication::Extend {
                    table_name: tasks_run_log_table.get_name().to_string(),
                })
                .with_message(tasks_run_log_table.to_ipc_stream()?)
                .make_random_name()?
                .build()?,
        ]);

        // Update the tasks run log
        let state_update = session_context.write().update_subjects_from_messages(messages)?;

        // Update the subjects change log
        let messages = create_message_map(vec![
            IPCMessageBuilder::new()
                .with_subject(state_update.get_name())
                .with_publisher(&session_context_name)
                .with_update(&TablePublication::Extend {
                    table_name: state_update.get_name().to_string(),
                })
                .with_message(state_update.to_ipc_stream()?)
                .make_random_name()?
                .build()?,
        ]);
        let _ = session_context.write().update_subjects_from_messages(messages)?;

        Ok(())
    }
 
    /// Get the next tasks to run using the [TasksSubscribePublishSession] pre-compiled tasks and [SessionContext] helpers
    fn get_tasks(
        session_context: &Arc<RwLock<SessionContext>>
    ) -> impl std::future::Future<Output = Result<HashMap<(String, String), ProcessorSubjectsMap>>> + Send { async move {
        let num_rows =  session_context.read()
            .get_states()
            .get(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .unwrap_or_else(|| {
                panic!(
                    "Missing table for `{}` in session `{}`.",
                    AvailableSubjects::SessionTasksSubscribePublish,
                    session_context.read().get_name()
                )
            })
            .read()
            .count_rows();
        if num_rows == 0 {
            let tasks_publish_subscribe_messages = TasksSubscribePublishSession::default().tasks_subscribe_publish_messages()?;
            for (step, messages) in tasks_publish_subscribe_messages.into_iter().enumerate() {
                if messages.is_empty() {
                    session_context.read().tasks_subscribe()?;
                } else {
                    let _result = SessionStreamStepMinimal::run_superstep(Arc::clone(&session_context), messages, step).await?;
                }
                // let subjects_reading = session_context.read();
                // let table_reading = subjects_reading
                //     .get_states()
                //     .get(AvailableSubjects::SessionErrors.to_string().as_str())
                //     .unwrap()
                //     .read();
                // println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);        
            }
        }
        session_context.read().tasks_subscribe_publish()
    } }

    /// Run the tasks
    /// 
    /// # Notes
    /// * Any filtering or partitioning of tasks into subject and user should be done before calling this method
    /// 
    /// # Arguments
    /// 
    /// * `session_context` - [SessionContext] to use while running the current session stream super step
    /// * `tasks` - Tasks that are ready to run
    /// * `diagnostics_vec` - Optional vector of [Diagnostics]
    /// * `span` - [Span] for the current session stream super step 
    /// 
    /// # Returns
    /// 
    /// * [SendableRecordBatchStreamMessageMap] - Subject streams from running the task
    /// * [SendableRecordBatchStreamMessageMap] - User streams from the running task
    fn run_tasks(session_context: &Arc<RwLock<SessionContext>>, tasks: &HashMap<(String, String), ProcessorSubjectsMap>, diagnostics_vec: &mut Option<Vec<Diagnostics>>, span: &Option<Span>) -> Result<SendableRecordBatchStreamMessageMap> {
        // Iterate through each task and collect the resulting stream responses
        let mut subject_streams = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for ((task_name, _session_name), processor_subjects_map) in tasks.iter() {
            event!(Level::INFO, "Superstep for task {}", &task_name);

            // Clone the task
            let task = session_context
                .read()
                .get_tasks()
                .get(task_name)
                .unwrap_or_else(|| {
                    panic!(
                        "Missing task `{task_name}` in session `{}`.",
                        session_context.read().get_name()
                    )
                })
                .clone();

            // Create the diagnostics for the task
            let diagnostic_builder = if let (Some(diagnostics_vec), Some(span)) = (diagnostics_vec.as_mut(), span.as_ref()) {
                let diagnostics = Diagnostics::new();
                let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
                diagnostics_vec.push(diagnostics);
                Some(diagnostic_builder)
            } else {
                None
            };

            // Run the task and collect the stream responses
            match task.run(
                diagnostic_builder.as_ref(),
                processor_subjects_map,
                session_context.read().get_states(),
            ) {
                Ok(result) => {
                    for (resp_name, resp) in result.into_iter() {
                        subject_streams.insert(resp_name, resp);
                    }
                }
                Err(err) => {
                    // Intercept the error and wrap into a `SendableRecordBatch` for consumption
                    event!(Level::ERROR, "{} for task {}", err.to_string(), &task_name);
                    let message_map = create_error_message_map_stream(&err, task_name, true)?;
                    subject_streams.extend(message_map);
                }
            }
        }

        Ok(subject_streams)
    }
 
    /// Join the message streams and intercept any errors
    /// 
    /// # Notes
    /// * Handling of intermediate errors in the chains of futures is not implemented yet...
    fn join_message_streams(
        messages: SendableRecordBatchStreamMessageMap,
    ) -> impl std::future::Future<Output = Result<IPCMessageMap>> + Send { async {
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
    } }

}

/// A single step of a [SessionStream]
/// [SessionStream]: crate::session::session_stream::SessionStream
pub struct SessionStreamStep {}

impl SessionStreamStepTrait for SessionStreamStep {
    fn run_superstep(
        session_context: Arc<RwLock<SessionContext>>,
        messages: IPCMessageMap,
        step: usize,
    ) -> impl std::future::Future<Output = Result<Option<IPCMessageMap>>> + Send { async move {
        // Start the diagnostics
        let (mut diagnostics_vec, span, trace) = if session_context.read().get_diagnostics() {
            let (diagnostics_vec, span, trace) = Self::enter_span(&messages, &session_context, step)?;
            (Some(diagnostics_vec), Some(span), Some(trace))
        } else {
            (None, None, None)
        };

        // Update the session context with the incoming messages
        if !messages.is_empty() {
            Self::update_subjects_and_changelog_from_messages(&session_context, messages)?;
        }        

        // Retrieve the task subscriptions and corresponding publications
        let tasks = Self::get_tasks(&session_context).await?;

        // Break if there is nothing to update
        if tasks.is_empty() {
            if let (Some(diagnostics_vec), Some(trace)) = (diagnostics_vec, trace) {
                Self::exit_span(&session_context, &HashMap::<String, IPCMessage>::new(), diagnostics_vec, trace)?;
            }

            Ok(None)
        } else {
            // Iterate through each task and collect the resulting stream responses
            let (subject_tasks, session_tasks) = tasks.into_iter().partition(|((t, s), _v)| t != s);
            let subject_streams = Self::run_tasks(&session_context, &subject_tasks, &mut diagnostics_vec, &span)?;
            let user_streams = Self::run_tasks(&session_context, &session_tasks, &mut diagnostics_vec, &span)?;

            // Update the tasks run log
            Self::update_subjects_and_changelog_from_tasks(&session_context, subject_tasks)?;
            Self::update_subjects_and_changelog_from_tasks(&session_context, session_tasks)?;

            // Join each of the response futures
            let subject_batches =
                match Self::join_message_streams(subject_streams).await {
                    Ok(subject_batches) => subject_batches,
                    Err(err) => create_error_message_map(&err, session_context.read().get_name(), true)?,
                };

            // Update the session context with the incoming messages
            if !subject_batches.is_empty() {
                Self::update_subjects_and_changelog_from_messages(&session_context, subject_batches)?;
            }

            // Join each of the response futures
            let user_batches = Self::join_message_streams(user_streams).await?;            
            if let (Some(diagnostics_vec), Some(trace)) = (diagnostics_vec, trace) {
                Self::exit_span(&session_context, &user_batches, diagnostics_vec, trace)?;
            }

            Ok(Some(user_batches))
        }
    } }
}

/// A single step of a minimal [SessionStream] that does not including logging and diagnostics
/// [SessionStream]: crate::session::session_stream::SessionStream
pub struct SessionStreamStepMinimal {}

impl SessionStreamStepTrait for SessionStreamStepMinimal {
    fn run_superstep(
        session_context: Arc<RwLock<SessionContext>>,
        messages: IPCMessageMap,
        _step: usize,
    ) -> impl std::future::Future<Output = Result<Option<IPCMessageMap>>> + Send { async move {

        // Update the session context with the incoming messages
        if !messages.is_empty() {
            let _ = session_context.write().update_subjects_from_messages(messages)?;
        }

        // Retrieve the task subscriptions and corresponding publications
        let subject_tasks = session_context.read().tasks_subscribe_publish()?;

        if !subject_tasks.is_empty() {
            // Iterate through each task and collect the resulting stream responses
            let subject_streams = Self::run_tasks(&session_context, &subject_tasks, &mut None, &None)?;

            // Join each of the response futures
            let subject_batches =
                match Self::join_message_streams(subject_streams).await {
                    Ok(subject_batches) => subject_batches,
                    Err(err) => create_error_message_map(&err, session_context.read().get_name(), true)?,
                };

            // Update the session context with the incoming messages
            if !subject_batches.is_empty() {
                let _ = session_context.write().update_subjects_from_messages(subject_batches)?;
            }
        }

        Ok(None)
    } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderTrait, test_session_context_builder::{
            make_test_session_context_builder_parallel, make_test_session_context_builder_sequential,
        }
    };
    use phymes_core::{
        AvailableSubjects, AvailableTableSubscribePolicies,
        ProcessorBuilder, ProcessorPlanBuilder, TablePublication, TableSubscription, TaskPlan,
        test_processor::{ProcessorError, ProcessorMock},
        test_task,
    };

    #[tokio::test]
    async fn test_session_run_superstep_no_state_update() -> Result<()> {
        let session_context = make_test_session_context_builder_parallel("session_1", 4)?
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let session_context_arc = Arc::new(RwLock::new(session_context));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_context_arc),
            test_task::make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::None,
                true
            )?,
            0,
        )
        .await?;
        assert!(response.is_none());

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
            3
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
            4
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
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
            session_context_arc
                .try_read()
                .unwrap()
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
            1
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
            2
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
            0
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
            1
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
            1
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
            1
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_extend_state_update_single_task() -> Result<()> {
        let session_context = make_test_session_context_builder_parallel("session_1", 4)?
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let session_context_arc = Arc::new(RwLock::new(session_context));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_context_arc),
            test_task::make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::Extend {
                    table_name: "state_1".to_string(),
                },
                true
            )?,
            0,
        )
        .await?
        .unwrap();
        assert!(response.is_empty());

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
        ); // Originally 3
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
            5
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
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
            session_context_arc
                .try_read()
                .unwrap()
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
            3
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
            5
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
            1
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
            1
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
            1
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

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_single_task() -> Result<()> {
        let session_context = make_test_session_context_builder_parallel("session_1", 4)?
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let session_context_arc = Arc::new(RwLock::new(session_context));
        let response = SessionStreamStep::run_superstep(
            Arc::clone(&session_context_arc),
            test_task::make_test_input_message(
                "task_1",
                "session_1",
                "state_1",
                "state_1",
                &TablePublication::Replace {
                    table_name: "state_1".to_string(),
                },
                true
            )?,
            0,
        )
        .await?
        .unwrap();
        assert!(response.is_empty());

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
            6
        ); // Originally 3
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
            5
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
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
            session_context_arc
                .try_read()
                .unwrap()
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
            3
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
            5
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
            1
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
            1
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
            1
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

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_parallel_tasks() -> Result<()> {
        // Superstep 1
        let session_context = make_test_session_context_builder_parallel("session_1", 4)?
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let mut input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Replace {
                table_name: "state_1".to_string(),
            },
            true,
        )?;
        input.extend(test_task::make_test_input_message(
            "task_2",
            "session_1",
            "state_2",
            "state_2",
            &TablePublication::Replace {
                table_name: "state_2".to_string(),
            },
            true,
        )?);
        input.extend(test_task::make_test_input_message(
            "task_3",
            "session_1",
            "state_3",
            "state_3",
            &TablePublication::Replace {
                table_name: "state_3".to_string(),
            },
            true,
        )?);
        let session_context_arc = Arc::new(RwLock::new(session_context));
        let mut response = SessionStreamStep::run_superstep(Arc::clone(&session_context_arc), input, 0)
            .await?
            .unwrap();
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
        assert_eq!(n_rows, 4);

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
        assert_eq!(n_rows, 4);

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
            6
        ); // Originally 3
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
            5
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
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
            session_context_arc
                .try_read()
                .unwrap()
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
            session_context_arc
                .try_read()
                .unwrap()
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
            session_context_arc
                .try_read()
                .unwrap()
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
            3
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
            5
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
            1
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
            1
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
            1
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

        // Superstep 2
        let mut response = SessionStreamStep::run_superstep(
            Arc::clone(&session_context_arc),
            HashMap::<String, IPCMessage>::new(),
            0,
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
        assert_eq!(n_rows, 5);

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
        assert_eq!(n_rows, 5);

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
        assert_eq!(n_rows, 5);

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
        ); // The same as superstep 1
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
            6
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get("state_2")
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
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            6
        );
        assert_eq!(
            session_context_arc
                .try_read()
                .unwrap()
                .get_states()
                .get("state_3")
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
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            6
        );
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
                .get(AvailableSubjects::SessionEvents.to_string().as_str())
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
                .get(AvailableSubjects::SessionTraces.to_string().as_str())
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

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_sequential_tasks() -> Result<()> {
        // Superstep 1
        let session_context = make_test_session_context_builder_sequential("session_1", 4)?
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let input = test_task::make_test_input_message(
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
        let mut response = SessionStreamStep::run_superstep(Arc::clone(&session_context_arc), input, 0)
            .await?
            .unwrap();

        // Check the response
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
            6
        ); // Originally 3
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
            7
        );
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
            3
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
            5
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
            1
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
            1
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
            1
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

        // Supersteps 2
        let mut response = SessionStreamStep::run_superstep(
            Arc::clone(&session_context_arc),
            HashMap::<String, IPCMessage>::new(),
            1,
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
        assert_eq!(n_rows, 7);

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

    #[tokio::test]
    async fn test_session_run_superstep_schema_mismatch_error() -> Result<()> {
        let session_context = make_test_session_context_builder_sequential("session_1", 4)?
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &TablePublication::Replace {
                table_name: "state_1".to_string(),
            },
            false,
        )?;
        let session_context_arc = Arc::new(RwLock::new(session_context));
        let response =
            SessionStreamStep::run_superstep(Arc::clone(&session_context_arc), input, 0).await;
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
            ProcessorPlanBuilder::default()
                .with_processor(
                    ProcessorBuilder::default()
                        .with_name("processor_1")
                        .with_type(ProcessorMock::get_static_name())
                        .build_arc::<ProcessorMock>()?,
                )
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_1".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    ProcessorBuilder::default()
                        .with_name("error_1")
                        .with_type(ProcessorError::get_static_name())
                        .build_arc::<ProcessorError>()?,
                )
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[TableSubscription::OnUpdateFullTable {
                    table_name: "state_1".to_string(),
                },
                    TableSubscription::AlwaysFullTable {
                        table_name: "error_1".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        ];
        let mut state = test_task::make_state_tables("state_1", "processor_1")?;
        state.push(test_task::make_config_tables("error_1")?);
        let session_context = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_envs(vec![test_task::make_runtime_env("rt_1")?])
            .with_state(state)
            .with_max_iter(1)
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?
            .build_with_tables()?;

        // Run the session context
        let input = test_task::make_test_input_message(
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
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_context_arc), input, 0)
            .await?
            .unwrap();

        assert!(response.is_empty());
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
            3
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
            5
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
            1
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
            1
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
            1
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
            1
        );
        let session_reading = session_context_arc.read();
        let table_reading = session_reading.get_states()
            .get(AvailableSubjects::SessionErrors.to_string().as_str())
            .unwrap()
            .read();
        let errors = table_reading.get_column_as_vec_str("content");
        assert_eq!(errors, ["This is an error!"]);

        Ok(())
    }
}
