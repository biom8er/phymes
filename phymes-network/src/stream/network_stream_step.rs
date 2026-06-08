use anyhow::{Result, anyhow};
use futures::{FutureExt, TryStreamExt};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, EventBuilderTrait, HashMap, Span,
    SpanBuilder, TraceBuilderTrait, TraceRecord, create_timestamp_micros,
};
use phymes_event::{Publication, Subscription};
use phymes_message::{
    IPCMessage, IPCMessageBuilder, IPCMessageMap, MessageBuilderTrait, MessageTrait,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap,
    create_error_message_map, create_error_message_map_stream, create_message_map,
};
use phymes_processor::ProcessorSubjectsMap;
use phymes_schemas::{
    AvailableSubjects, AvailableSubjectsTrait, create_session_tasks_run_log_batch,
};
use phymes_subject::{
    BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_task::{SubscriptionTrait, TaskTrait};
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{Level, event};

use crate::{
    Network,
    core::{NextSuperstepNetwork, NextTaskNetwork},
};

/// Traits for running a static or dynamic [NetworkStream] step
///
/// [NetworkStream]: crate::session::network_stream::NetworkStream
pub trait NetworkStreamStepTrait {
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
    /// The additional complexity that is added by the `Network`
    /// includes dynamical computational graph where edges are conditionally executed
    /// based on the outputs of nodes, network that can be shared between computational
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
    ///   to the network.
    ///
    /// - Subjects: The data that compose the network of the application.
    ///
    /// - Computation: Each task performs a user-defined computation during each
    ///   super-step as defined by the processor network and based on its subscriptions
    ///   that have changed in the previous super-step.
    ///
    /// - Messages: Subset of the network subjects that are passed to tasks at each super-step.
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
    /// * `network` - [Network] to use while running the current session stream super step
    /// * `messages` - [IPCMessageMap] input messages for the superstep
    ///
    /// # Returns
    ///
    /// [IPCMessageMap] if any of the subscribing session sujects were updated and None otherwise.
    fn run_superstep(
        network: Arc<Network>,
        messages: IPCMessageMap,
    ) -> impl std::future::Future<Output = Result<Option<IPCMessageMap>>> + Send;

    /// Enter the superstep span generating the [Span], [TraceRecord], and [Diagnostics]
    fn enter_span(
        subject_messages: &IPCMessageMap,
        network: &Arc<Network>,
        step: u32,
    ) -> Result<(Vec<Diagnostics>, Span, TraceRecord)> {
        // Create the span for the session
        let span = SpanBuilder::default()
            .with_span(network.get_name())
            .build()?;

        // Initialize the channels for collecting the metrics, events, and traces)
        let mut diagnostics_vec = Vec::new();
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        diagnostics_vec.push(diagnostics);

        // Trace the session step
        let trace = diagnostic_builder
            .clone()
            .messages(line!(), file!(), network.get_name());
        trace.enter(&subject_messages.values().collect::<Vec<_>>());
        let event = diagnostic_builder
            .clone()
            .info(line!(), file!(), network.get_name());
        event.insert("superstep", &serde_json::Value::Number(step.into()));

        Ok((diagnostics_vec, span, trace))
    }

    /// Exit the span
    fn exit_span(
        network: &Arc<Network>,
        messages: &IPCMessageMap,
        diagnostics_vec: Vec<Diagnostics>,
        trace: TraceRecord,
        step: u32,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            trace.exit(&messages.values().collect::<Vec<_>>());
            network
                .update_metrics_subjects(&diagnostics_vec, step)
                .await?;

            Ok(())
        }
    }

    /// Update the session context subjects from messages including updating the subjects change log
    fn update_subjects_and_changelog_from_messages(
        network: &Arc<Network>,
        messages: IPCMessageMap,
        step: u32,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            // Update the network and handle any errors
            let network_name = network.get_name().to_string();
            let (changelog, meta, errors) =
                network.update_subjects_from_messages(messages, step).await;

            let mut messages = Vec::new();
            if let Some(subject) = changelog {
                let message = IPCMessageBuilder::new()
                    .with_subject(subject.get_name())
                    .with_publisher(&network_name)
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
                    .with_publisher(&network_name)
                    .with_update(&Publication::Extend {
                        subject_name: subject.get_name().to_string(),
                    })
                    .with_message(subject.to_ipc_stream()?)
                    .make_random_name()?
                    .build()?;
                messages.push(message);
            }

            // Update the errors
            if let Some(subject) = errors {
                let message = IPCMessageBuilder::new()
                    .with_subject(subject.get_name())
                    .with_publisher(&network_name)
                    .with_update(&Publication::Extend {
                        subject_name: AvailableSubjects::SessionErrors.to_string(),
                    })
                    .with_message(subject.to_ipc_stream()?)
                    .make_random_name()?
                    .build()?;
                let mut message_map = HashMap::<String, IPCMessage>::new();
                let _ = message_map.insert(message.get_name().to_string(), message);
                let (update, meta, _errors) = network
                    .update_subjects_from_messages(message_map, step)
                    .await;

                if let Some(subject) = update {
                    let message = IPCMessageBuilder::new()
                        .with_subject(subject.get_name())
                        .with_publisher(&network_name)
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
                        .with_publisher(&network_name)
                        .with_update(&Publication::Extend {
                            subject_name: subject.get_name().to_string(),
                        })
                        .with_message(subject.to_ipc_stream()?)
                        .make_random_name()?
                        .build()?;
                    messages.push(message);
                }
            }

            // Update the subjects change log
            let messages = create_message_map(messages);
            let _ = network.update_subjects_from_messages(messages, step).await;

            Ok(())
        }
    }

    /// Update the session context subjects from the ran tasks including the subjects change log
    fn update_subjects_and_changelog_from_tasks(
        network: &Arc<Network>,
        tasks: HashMap<(String, String), ProcessorSubjectsMap>,
        step: u32,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            // Create the tasks run log message
            let network_name = network.get_name().to_string();
            let (session_names, (task_names, (supersteps, timestamps))): (
                Vec<_>,
                (Vec<_>, (Vec<_>, Vec<_>)),
            ) = tasks
                .into_iter()
                .map(|((task_name, session_name), _)| {
                    (
                        session_name,
                        (task_name, (step as i64, create_timestamp_micros())),
                    )
                })
                .unzip();
            let tasks_run_log_batch = create_session_tasks_run_log_batch(
                session_names,
                task_names,
                supersteps,
                timestamps,
            )?;
            let tasks_run_log_table = AvailableSubjects::SessionTasksRunLog
                .to_subject(None, Some(vec![tasks_run_log_batch]))?;
            let messages = create_message_map(vec![
                IPCMessageBuilder::new()
                    .with_subject(tasks_run_log_table.get_name())
                    .with_publisher(&network_name)
                    .with_update(&Publication::Extend {
                        subject_name: tasks_run_log_table.get_name().to_string(),
                    })
                    .with_message(tasks_run_log_table.to_ipc_stream()?)
                    .make_random_name()?
                    .build()?,
            ]);

            // Update the tasks run log
            let (changelog, meta, errors) =
                network.update_subjects_from_messages(messages, step).await;
            if let Some(table) = errors {
                let error = table.get_column_as_vec_str("content").join("; ");
                return Err(anyhow!(error));
            }

            // Update the subjects change log
            let mut messages = Vec::new();
            if let Some(subject) = changelog {
                let message = IPCMessageBuilder::new()
                    .with_subject(subject.get_name())
                    .with_publisher(&network_name)
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
                    .with_publisher(&network_name)
                    .with_update(&Publication::Extend {
                        subject_name: subject.get_name().to_string(),
                    })
                    .with_message(subject.to_ipc_stream()?)
                    .make_random_name()?
                    .build()?;
                messages.push(message);
            }
            let messages = create_message_map(messages);
            let (_update, _meta, errors) =
                network.update_subjects_from_messages(messages, step).await;
            if let Some(table) = errors {
                let error = table.get_column_as_vec_str("content").join("; ");
                return Err(anyhow!(error));
            }

            Ok(())
        }
        .boxed()
    }

    /// Get the next superstep using the [NextSuperstepNetwork] pre-compiled tasks and [Network] helpers
    fn next_superstep(network: &Arc<Network>) -> impl std::future::Future<Output = u32> + Send {
        async move {
            // Compute the current superstep
            let next_superstep_messages = NextSuperstepNetwork::default()
                .as_task_messages()
                .unwrap_or_else(|_err| {
                    panic!("Missing pre-compiled tasks for `NextSuperstepNetwork`.")
                });
            for messages in next_superstep_messages.into_iter() {
                let _ = NetworkStreamStepMinimal::run_superstep(Arc::clone(network), messages)
                    .await
                    .unwrap_or_else(|err| {
                        panic!(
                            "Error `{err}` running pre-compiled tasks for `NextSuperstepNetwork`."
                        )
                    });
            }

            // Return the next superstep
            network
                .current_superstep()
                .await
                .unwrap_or_else(|err| panic!("Error `{err}` reading the `current_superstep`."))
        }
    }

    /// Get the current superstep handling any initialization
    fn current_superstep(network: &Arc<Network>) -> impl std::future::Future<Output = u32> + Send {
        async move {
            let (step, next) = match network.current_superstep().await {
                Ok(step) => (step, false),
                Err(_err) => (
                    network.increment_superstep().await.unwrap_or_default(),
                    true,
                ),
            };
            if next {
                Self::next_superstep(network).await
            } else {
                step
            }
        }
    }

    /// Increment the superstep
    fn increment_superstep(
        network: &Arc<Network>,
    ) -> impl std::future::Future<Output = u32> + Send {
        async move {
            let _step = network.increment_superstep().await.unwrap_or_default();
            Self::next_superstep(network).await
        }
    }

    /// Get the next tasks to run using the [NextTaskNetwork] pre-compiled tasks and [Network] helpers
    fn next_tasks(
        network: &Arc<Network>,
    ) -> impl std::future::Future<Output = HashMap<(String, String), ProcessorSubjectsMap>> + Send
    {
        async move {
            // Check if there are tasks subscribe and publish available, and determine them if not
            let rt = network.runtime_env.clone();
            let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string() }
                .subscribe_to_subject(&rt, network.get_name()).unwrap()
                .ok_or(anyhow!("Unable to get the subject `{}` from object storage for session `{}` while getting the next task.", 
                    AvailableSubjects::SessionTasksSubscribePublish,
                    network.get_name()
                )).unwrap()
                .try_collect()
                .await.unwrap();
            if subscriptions.is_empty() {
                let next_task_messages = NextTaskNetwork::default()
                    .as_task_messages()
                    .unwrap_or_else(|_err| {
                        panic!("Missing pre-compiled tasks for `NextTaskNetwork`.")
                    });
                for messages in next_task_messages.into_iter() {
                    if messages.is_empty() {
                        if let Err(_err) = network.tasks_subscribe().await {
                            dbg!(&_err);
                            return HashMap::<(String, String), ProcessorSubjectsMap>::new();
                        }
                    } else if let Err(_err) =
                        NetworkStreamStepMinimal::run_superstep(Arc::clone(network), messages).await
                    {
                        dbg!(&_err);
                        return HashMap::<(String, String), ProcessorSubjectsMap>::new();
                    }
                }
            }

            // Return the tasks subscribe and publish if availabe or an empty map
            match network.tasks_subscribe_publish().await {
                Ok(tasks) => tasks,
                Err(_err) => {
                    dbg!(&_err);
                    HashMap::<(String, String), ProcessorSubjectsMap>::new()
                },
            }
        }
    }

    /// Run the tasks
    ///
    /// # Notes
    /// * Any filtering or partitioning of tasks into subject and user should be done before calling this method
    ///
    /// # Arguments
    ///
    /// * `network` - [Network] to use while running the current session stream super step
    /// * `tasks` - Tasks that are ready to run
    /// * `diagnostics_vec` - Optional vector of [Diagnostics]
    /// * `span` - [Span] for the current session stream super step
    ///
    /// # Returns
    ///
    /// * [SendableRecordBatchStreamMessageMap] - Subject streams from running the task
    /// * [SendableRecordBatchStreamMessageMap] - User streams from the running task
    fn run_tasks(
        network: &Arc<Network>,
        tasks: &HashMap<(String, String), ProcessorSubjectsMap>,
        diagnostics_vec: &mut Option<Vec<Diagnostics>>,
        span: &Option<Span>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        // Iterate through each task and collect the resulting stream responses
        let mut subject_streams = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for ((task_name, _session_name), processor_subjects_map) in tasks.iter() {
            event!(Level::INFO, "Superstep for task {}", &task_name);

            // Clone the task
            let task = network
                .tasks()
                .get(task_name)
                .unwrap_or_else(|| {
                    panic!(
                        "Missing task `{task_name}` in session `{}`.",
                        network.get_name()
                    )
                })
                .clone();

            // Create the diagnostics for the task
            let diagnostic_builder = if let (Some(diagnostics_vec), Some(span)) =
                (diagnostics_vec.as_mut(), span.as_ref())
            {
                let diagnostics = Diagnostics::new();
                let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(span);
                diagnostics_vec.push(diagnostics);
                Some(diagnostic_builder)
            } else {
                None
            };

            // Run the task and collect the stream responses
            match task.run(
                diagnostic_builder.as_ref(),
                processor_subjects_map,
                network.runtime_env(),
                network.get_name(),
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
    ) -> impl std::future::Future<Output = Result<IPCMessageMap>> + Send {
        async {
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
                    let result: Result<Vec<_>> = resp.get_message_own().try_collect().await;
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
                            Ok(batches) => match SubjectBuilder::new()
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
                                Err(err) => {
                                    create_error_message_map(&err, "NetworkStreamStep", true)?
                                }
                            },
                            Err(err) => create_error_message_map(&err, "NetworkStreamStep", true)?,
                        };

                        // Add the message to the joined responses
                        response_batches.extend(message_map);
                    }
                    Err(err) => {
                        // Intercept the error and forward to the error subject
                        event!(Level::ERROR, "{err}");
                        let message_map =
                            create_error_message_map(&anyhow!("{err}"), "NetworkStreamStep", true)?;
                        response_batches.extend(message_map);
                    }
                }
            }

            Ok(response_batches)
        }
    }
}

/// A single step of a [NetworkStream]
///
/// [NetworkStream]: crate::NetworkStream
pub struct NetworkStreamStep {}

impl NetworkStreamStepTrait for NetworkStreamStep {
    async fn run_superstep(
        network: Arc<Network>,
        messages: IPCMessageMap,
    ) -> Result<Option<IPCMessageMap>> {
        // Get the next superstep handling any initialization
        let step = Self::current_superstep(&network).await;
        // dbg!(&step);

        // Start the diagnostics
        let (mut diagnostics_vec, span, trace) = if network.get_diagnostics() {
            let (diagnostics_vec, span, trace) = Self::enter_span(&messages, &network, step)?;
            (Some(diagnostics_vec), Some(span), Some(trace))
        } else {
            (None, None, None)
        };

        // Update the session context with the incoming messages
        if !messages.is_empty() {
            Self::update_subjects_and_changelog_from_messages(&network, messages, step).await?;
        }

        // Retrieve the task subscriptions and corresponding publications
        let tasks = Self::next_tasks(&network).await;
        // dbg!(&tasks);

        // Break if there is nothing to update
        if tasks.is_empty() {
            if let (Some(diagnostics_vec), Some(trace)) = (diagnostics_vec, trace) {
                Self::exit_span(
                    &network,
                    &HashMap::<String, IPCMessage>::new(),
                    diagnostics_vec,
                    trace,
                    step,
                )
                .await?;
            }

            Ok(None)
        } else {
            // Iterate through each task and collect the resulting stream responses
            let (subject_tasks, session_tasks) = tasks.into_iter().partition(|((t, s), _v)| t != s);
            let subject_streams =
                Self::run_tasks(&network, &subject_tasks, &mut diagnostics_vec, &span)?;
            let user_streams =
                Self::run_tasks(&network, &session_tasks, &mut diagnostics_vec, &span)?;

            // Update the tasks run log
            Self::update_subjects_and_changelog_from_tasks(&network, subject_tasks, step).await?;
            Self::update_subjects_and_changelog_from_tasks(&network, session_tasks, step).await?;

            // Increment the superstep
            let _step = Self::increment_superstep(&network).await;

            // Join each of the response futures
            let subject_batches = match Self::join_message_streams(subject_streams).await {
                Ok(subject_batches) => subject_batches,
                Err(err) => create_error_message_map(&err, network.get_name(), true)?,
            };

            // Update the session context with the incoming messages
            if !subject_batches.is_empty() {
                Self::update_subjects_and_changelog_from_messages(&network, subject_batches, step)
                    .await?;
            }

            // Join each of the response futures
            let user_batches = Self::join_message_streams(user_streams).await?;
            if let (Some(diagnostics_vec), Some(trace)) = (diagnostics_vec, trace) {
                Self::exit_span(&network, &user_batches, diagnostics_vec, trace, step).await?;
            }

            Ok(Some(user_batches))
        }
    }
}

/// A single step of a minimal [NetworkStream] that does not including logging and diagnostics
///
/// [NetworkStream]: crate::NetworkStream
pub struct NetworkStreamStepMinimal {}

impl NetworkStreamStepTrait for NetworkStreamStepMinimal {
    /// Minimal implementation of `join_message_streams` without intercepting Errors
    async fn join_message_streams(
        messages: SendableRecordBatchStreamMessageMap,
    ) -> Result<IPCMessageMap> {
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
                let result: Result<Vec<_>> = resp.get_message_own().try_collect().await;
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
            // Check the response
            let (resp_name, resp) = response?;
            let batches = resp?;
            let table = SubjectBuilder::new()
                .with_name(resp_name.as_str())
                .with_record_batches(batches)?
                .build()?;
            let message = response_builder
                .remove(resp_name.as_str())
                .unwrap()
                .with_message(table.to_ipc_stream()?)
                .build()?;

            // Add the message to the joined responses
            let message_map = message.to_map()?;
            response_batches.extend(message_map);
        }

        Ok(response_batches)
    }

    /// Minimal implementation of `run_superstep` without error handling and diagnostics
    async fn run_superstep(
        network: Arc<Network>,
        messages: IPCMessageMap,
    ) -> Result<Option<IPCMessageMap>> {
        // Update the session context with the incoming messages
        if !messages.is_empty() {
            let (_update, _meta, errors) = network.update_subjects_from_messages(messages, 0).await;
            if let Some(table) = errors {
                let error = table.get_column_as_vec_str("content").join("; ");
                return Err(anyhow!(error));
            }
        }

        // Retrieve the task subscriptions and corresponding publications
        let subject_tasks = network.tasks_subscribe_publish().await?;

        if !subject_tasks.is_empty() {
            // Iterate through each task and collect the resulting stream responses
            let subject_streams = Self::run_tasks(&network, &subject_tasks, &mut None, &None)?;

            // Join each of the response futures
            let subject_batches = Self::join_message_streams(subject_streams).await?;

            // Update the session context with the incoming messages
            if !subject_batches.is_empty() {
                let (_update, _meta, errors) = network
                    .update_subjects_from_messages(subject_batches, 0)
                    .await;
                if let Some(table) = errors {
                    let error = table.get_column_as_vec_str("content").join("; ");
                    return Err(anyhow!(error));
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderTrait, test_network_builder,
    };
    use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
    use phymes_processor::{
        ProcessorBuilder, ProcessorPlanBuilder,
        test_processor::{ProcessorError, ProcessorMock},
    };
    use phymes_schemas::AvailableSubjects;
    use phymes_subject::{
        BuildableTrait, ObjectStorageBackend, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectPlan, SubjectPlanBuilderTrait, make_store,
    };
    use phymes_task::{TaskPlan, test_task};

    #[tokio::test]
    async fn test_session_run_superstep_no_state_update() -> Result<()> {
        let (network, session_messages) =
            test_network_builder::make_test_network_builder_parallel("session_1", 4)?
                .with_diagnostics(true)
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        let messages = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::None,
            true,
        )?;
        let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages).await?;
        assert!(response.is_none());

        // check the session and network
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        assert_eq!(subscriptions.last().unwrap().num_rows(), 4);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_2".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_3".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_extend_state_update_single_task() -> Result<()> {
        let (network, session_messages) =
            test_network_builder::make_test_network_builder_parallel("session_1", 4)?
                .with_diagnostics(true)
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let messages = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::Extend {
                subject_name: "state_1".to_string(),
            },
            true,
        )?;
        let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages)
            .await?
            .unwrap();
        assert!(response.is_empty());

        // check the session and network
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 12); // Originally 3
        assert_eq!(subscriptions.last().unwrap().num_rows(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_2".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_3".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2); // DM, Check!(): changed from 3
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 4); // DM, Check!(): changed from 5
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_single_task() -> Result<()> {
        let (network, session_messages) =
            test_network_builder::make_test_network_builder_parallel("session_1", 4)?
                .with_diagnostics(true)
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
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
        let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages)
            .await?
            .unwrap();
        assert!(response.is_empty());

        // check the session and network
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6); // Originally 3
        assert_eq!(subscriptions.last().unwrap().num_rows(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_2".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_3".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2); // DM, Check!(): changed from 3
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 4); // DM, Check!(): changed from 5
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_parallel_tasks() -> Result<()> {
        // Superstep 1
        let (network, session_messages) =
            test_network_builder::make_test_network_builder_parallel("session_1", 4)?
                .with_diagnostics(true)
                .add_network_interface(Some(&["state_1", "state_2", "state_3"]))?
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let mut messages = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::Replace {
                subject_name: "state_1".to_string(),
            },
            true,
        )?;
        messages.extend(test_task::make_test_input_message(
            "task_2",
            "session_1",
            "state_2",
            "state_2",
            &Publication::Replace {
                subject_name: "state_2".to_string(),
            },
            true,
        )?);
        messages.extend(test_task::make_test_input_message(
            "task_3",
            "session_1",
            "state_3",
            "state_3",
            &Publication::Replace {
                subject_name: "state_3".to_string(),
            },
            true,
        )?);
        let mut response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages)
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
            Publication::Extend {
                subject_name: "state_1".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = SubjectBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 5); // DM, check!(): changed from 4

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
            Publication::Extend {
                subject_name: "state_2".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_2")
            .unwrap()
            .get_message_own();
        let partitions = SubjectBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 5); // DM, Check!(): changed from 4

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
            Publication::Extend {
                subject_name: "state_3".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_3")
            .unwrap()
            .get_message_own();
        let partitions = SubjectBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 5); // DM, Check!(): changed from 4

        // check the session and network
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6); // Originally 3
        assert_eq!(subscriptions.last().unwrap().num_rows(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_2".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6);
        assert_eq!(subscriptions.last().unwrap().num_rows(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_3".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6);
        assert_eq!(subscriptions.last().unwrap().num_rows(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);

        // Superstep 2
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?;

        assert!(response.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_replace_state_update_sequential_tasks() -> Result<()> {
        // Superstep 1
        let (network, session_messages) =
            test_network_builder::make_test_network_builder_sequential("session_1", 4)?
                .with_diagnostics(true)
                .add_network_interface(Some(&["state_1"]))?
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
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
        let mut response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages)
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
            Publication::Extend {
                subject_name: "state_1".to_string()
            }
        );

        let bytes = response
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = SubjectBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 7); // DM, Check!(): changed from 4

        // check the session and network
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6); // Originally 3
        assert_eq!(subscriptions.last().unwrap().num_rows(), 7);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 5);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);

        // Supersteps 2
        let response = NetworkStreamStep::run_superstep(
            Arc::clone(&network_arc),
            HashMap::<String, IPCMessage>::new(),
        )
        .await?;

        assert!(response.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_schema_mismatch_error() -> Result<()> {
        let (network, session_messages) =
            test_network_builder::make_test_network_builder_sequential("session_1", 4)?
                .with_diagnostics(true)
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
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
            false,
        )?;
        let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages).await?;
        assert!(response.is_none());

        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_run_superstep_processor_error() -> Result<()> {
        // Create an error emitting session plan
        let task_plans = vec![
            TaskPlan {
                task_name: "task_1".to_string(),
                processor_names: vec!["processor_1".to_string()],
            },
            TaskPlan {
                task_name: "task_2".to_string(),
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
                .with_publications(&[Publication::Extend {
                    subject_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: "state_1".to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: "processor_1".to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    ProcessorBuilder::default()
                        .with_name("error_1")
                        .with_type(ProcessorError::get_static_name())
                        .build_arc::<ProcessorError>()?,
                )
                .with_publications(&[Publication::Extend {
                    subject_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: "state_1".to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: "error_1".to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
        ];
        let mut subjects = test_task::make_subject_tables("state_1", "processor_1")?;
        subjects.push(test_task::make_config_tables("error_1")?);
        let subjects_plan = subjects
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();
        let rt = RuntimeEnv::get_builder()
            .with_name("rt_1")
            .with_max_steps(1)
            .with_object_store(make_store(&ObjectStorageBackend::InMemory, None, None)?)
            .build_arc()?;
        let (network, session_messages) = NetworkBuilder::new()
            .with_name("session_1")
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_env(rt)
            .with_subjects(subjects_plan)
            .with_diagnostics(true)
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;

        // Run the session context
        let network_arc = Arc::new(network);
        let _ = network_arc
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
        let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), messages)
            .await?
            .unwrap();

        assert!(response.is_empty());
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2); // DM, Check!(): changed from 3
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 4); // DM, Check!(): changed from 5
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionEvents.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
            .with_record_batches(subscriptions)
            .unwrap()
            .build()
            .unwrap();
        let errors = subject.get_column_as_vec_str("content");
        assert!(errors.first().unwrap().contains("This is an error!"));
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "session_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionSupersteps.to_string().as_str())
            .with_record_batches(subscriptions)
            .unwrap()
            .build()
            .unwrap();
        let columns = subject.get_column_as_vec_primitive::<u32>("superstep")?;
        assert_eq!(columns, [1, 2]);

        Ok(())
    }
}
