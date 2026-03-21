use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use clap::ValueEnum;
use futures::{StreamExt, TryStreamExt};
use parking_lot::RwLock;
use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, AvailableSubjectsTrait, AvailableSubscribeEvents, AvailableUpdateEvents, BuildableTrait, BuilderTrait, IPCMessageBuilder, IPCMessageMap, MappableTrait, MessageBuilderTrait, MessageTrait, ObjectStorageBackend, ProcessorSubjects, ProcessorSubjectsBuilder, ProcessorSubjectsMap, Publication, RuntimeEnv, RuntimeEnvTrait, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait, Subscription, create_chat_record_batch, create_session_supersteps_batch, create_session_tasks_subscribe_batch, create_subjects_change_log_batch, create_subjects_num_rows_batch, create_subjects_object_store_meta_batch, from_diagnostics_to_tables, make_store
};
use phymes_diagnostics::{Diagnostics, HashMap, create_timestamp_micros};
use std::sync::Arc;
use tracing::{Level, event};

use crate::{PublicationTrait, SessionContextBuilder, SubscriptionTrait, TaskMap, clear_subject, create_message_map};

/// The [SessionContext] creates a (dynamic) execution graph based on a [TaskPlan]
///   and manages the running of individual [Task]s and the [Message]s passed between them.
///
/// [TaskPlan]: phymes_core::TaskPlan
/// [Task]: phymes_core::TaskTrait
/// [Message]: phymes_core::MessageTrait
#[derive(Debug, Clone)]
pub struct SessionContext {
    /// A unique UUID that identifies the session
    pub(crate) name: String,
    /// The list of available tasks that can be run during the session
    pub(crate) tasks: TaskMap,
    /// Cache of subjects and their schemas
    pub(crate) subjects: HashMap<String, SchemaRef>,
    /// Runtime environment configuration to use during task runs
    pub(crate) runtime_env: Arc<RuntimeEnv>,
    /// Whether to gather diagnostic information or not
    pub(crate) diagnostics: bool,
    /// Optional messages to initialize the session with
    pub(crate) messages: Option<IPCMessageMap>,
}

impl Default for SessionContext {
    fn default() -> Self {
        Self { 
            name: Default::default(), 
            tasks: Default::default(), 
            subjects: Default::default(), 
            runtime_env: Default::default(), 
            diagnostics: Default::default(), 
            messages: Default::default(),
        }
    }
}

impl SessionContext {
    pub fn new(
        name: String,
        tasks: TaskMap,
        subjects: HashMap<String, SchemaRef>,
        runtime_env: Arc<RuntimeEnv>,
        diagnostics: bool,
        messages: Option<IPCMessageMap>,
    ) -> SessionContext {
        Self {
            name,
            tasks,
            subjects,
            runtime_env,
            diagnostics,
            messages
        }
    }

    pub fn tasks(&self) -> &TaskMap {
        &self.tasks
    }

    pub fn subjects(&self) -> &HashMap<String, SchemaRef> {
        &self.subjects
    }

    pub fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        &self.runtime_env
    }

    pub fn get_max_steps(&self) -> usize {
        self.runtime_env().max_steps()
    }

    pub fn get_diagnostics(&self) -> bool {
        self.diagnostics
    }

    /// Compute the next tasks to subscribe
    pub async fn tasks_subscribe(&self) -> Result<()> {
        // Get the subject
        let batches: Vec<RecordBatch> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionTasksSubscribeAggregate.to_string() }
            .subscribe_to_subject(self.runtime_env())?
            .ok_or(anyhow!("Unable to get the subject `{}` from object storage for session `{}` during tasks_subscribe.", 
                AvailableSubjects::SessionTasksSubscribeAggregate,
                self.get_name()
            ))?
            .try_collect()
            .await?;
        let subject = SubjectBuilder::default()
            .with_name(
                AvailableSubjects::SessionTasksSubscribeAggregate
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(batches)?
            .build()?;

        // Clear the subject
        let _locations: Vec<RecordBatch> = clear_subject(self.runtime_env(), &AvailableSubjects::SessionTasksSubscribeAggregate.to_string(), false)?
            .try_collect()
            .await?;

        // Extract out the columns
        let session_names = subject.get_column_as_vec_str("session_name");
        let task_names = subject.get_column_as_vec_str("task_name");
        let processor_names = subject.get_column_as_vec_str("processor_name");
        let processor_types = subject.get_column_as_vec_str("processor_type");
        let subscription_names =
            subject.get_column_as_vec_nested_nonprimitive::<String>("subscription_name-List")?;
        let subscription_table_names = subject
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_name-List")?;
        let subscribe_types = subject.get_column_as_vec_str("subscribe_type-Last");
        let update_types = subject.get_column_as_vec_str("update_type-Last");
        let supersteps = subject.get_column_as_vec_nested_primitive::<i64>("superstep-List")?;
        let superstep_lasts =
            subject.get_column_as_vec_nested_primitive::<i64>("superstep-Max-List")?;

        // Determine the processor subscriptions
        let processors_subscribe = session_names
            .into_iter()
            .zip(task_names)
            .zip(processor_names)
            .zip(processor_types)
            .zip(subscription_names)
            .zip(subscription_table_names)
            .zip(subscribe_types)
            .zip(update_types)
            .zip(supersteps)
            .zip(superstep_lasts)
            .map(
                |(
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            ((session_name, task_name), processor_name),
                                            processor_type,
                                        ),
                                        subscription_names,
                                    ),
                                    subscription_table_names,
                                ),
                                subscribe_type,
                            ),
                            update_type,
                        ),
                        supersteps,
                    ),
                    supersteps_lasts,
                )| {
                    let subscriptions = subscription_names
                        .iter()
                        .zip(subscription_table_names.iter())
                        .map(|(name, subject)| {
                            Subscription::from_str_fuzzy(name, subject).unwrap()
                        })
                        .collect::<Vec<_>>();
                    let update_policy = AvailableUpdateEvents::from_str(update_type, false)
                        .unwrap()
                        .build();
                    let subjects_change_log = subscription_table_names
                        .iter()
                        .zip(supersteps_lasts.iter())
                        .map(|(subject_name, superstep)| {
                            (subject_name.to_string(), superstep.to_owned())
                        })
                        .collect::<HashMap<_, _>>();
                    let updates = update_policy.determine_updates(
                        &subscriptions,
                        supersteps.last().unwrap(),
                        &subjects_change_log,
                    );
                    let subscribe_policy =
                        AvailableSubscribeEvents::from_str_fuzzy(subscribe_type)
                            .unwrap()
                            .build();
                    let subscribe = subscribe_policy.check_subscriptions(
                        &subscriptions,
                        &updates,
                        &HashMap::<String, SchemaRef>::new(),
                    );
                    (
                        session_name,
                        task_name,
                        processor_name,
                        processor_type,
                        subscription_names,
                        subscription_table_names,
                        subscribe_type,
                        update_type,
                        supersteps,
                        supersteps_lasts,
                        subscribe,
                    )
                },
            )
            .collect::<Vec<_>>();

        // Determine the task subscriptions
        let mut tasks_subscribe = HashMap::<(String, String), bool>::new();
        for (
            session_name,
            task_name,
            _processor_name,
            _processor_type,
            _subscription_names,
            _subscription_table_names,
            _subscribe_type,
            _update_type,
            _supersteps,
            _supersteps_lasts,
            subscribe,
        ) in processors_subscribe.iter()
        {
            if let Some(subscribe_t) =
                tasks_subscribe.get_mut(&(session_name.to_string(), task_name.to_string()))
            {
                *subscribe_t &= subscribe;
            } else {
                let _ = tasks_subscribe.insert(
                    (session_name.to_string(), task_name.to_string()),
                    *subscribe,
                );
            }
        }
        let (session_names_subscribe, task_names_subscribe): (String, String) = tasks_subscribe
            .into_iter()
            .filter_map(|(k, v)| if v { Some(k) } else { None })
            .unzip();

        // Determine the subjects to subscribe to
        let (
            ((((session_names, task_names), processor_names), processor_types), subscription_names),
            subscription_table_names,
        ) = processors_subscribe
            .into_iter()
            .filter_map(
                |(
                    session_name,
                    task_name,
                    processor_name,
                    processor_type,
                    subscription_names,
                    subscription_table_names,
                    subscribe_type,
                    update_type,
                    supersteps,
                    supersteps_lasts,
                    _subscribe,
                )| {
                    if session_names_subscribe.contains(session_name)
                        && task_names_subscribe.contains(task_name)
                    {
                        let subscribe = subscription_names
                            .into_iter()
                            .zip(subscription_table_names)
                            .zip(supersteps)
                            .zip(supersteps_lasts)
                            .filter_map(|(((name, subject), superstep), superstep_last)| {
                                let subscriptions = vec![
                                    Subscription::from_str_fuzzy(&name, &subject).unwrap(),
                                ];
                                let update_policy =
                                    AvailableUpdateEvents::from_str(update_type, false)
                                        .unwrap()
                                        .build();
                                let mut subjects_change_log = HashMap::<String, i64>::new();
                                let _ =
                                    subjects_change_log.insert(subject.to_string(), superstep_last);
                                let updates = update_policy.determine_updates(
                                    &subscriptions,
                                    &superstep,
                                    &subjects_change_log,
                                );
                                let subscribe_policy =
                                    AvailableSubscribeEvents::from_str_fuzzy(subscribe_type)
                                        .unwrap()
                                        .build();
                                let subscribe = subscribe_policy.check_subscriptions(
                                    &subscriptions,
                                    &updates,
                                    &HashMap::<String, SchemaRef>::new(),
                                );
                                if subscribe {
                                    Some((
                                        (
                                            (
                                                (
                                                    (
                                                        session_name.to_string(),
                                                        task_name.to_string(),
                                                    ),
                                                    processor_name.to_string(),
                                                ),
                                                processor_type.to_string(),
                                            ),
                                            name,
                                        ),
                                        subject,
                                    ))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();
                        Some(subscribe)
                    } else {
                        None
                    }
                },
            )
            .flatten()
            .unzip();

        // Create the table
        let batch = create_session_tasks_subscribe_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
        )?;
        let table = SubjectBuilder::default()
            .with_name(
                AvailableSubjects::SessionTasksSubscribe
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the table
        let message = IPCMessageBuilder::default()
            .with_subject(table.get_name())
            .with_publisher(self.get_name())
            .with_update(&Publication::Replace {
                subject_name: table.get_name().to_string(),
            })
            .with_message(table.to_ipc_stream()?)
            .make_name()?
            .build()?;
        let messages = create_message_map(vec![message]);
        let (_update, _meta, errors) = self.update_subjects_from_messages(messages).await;
        if let Some(table) = errors {
            let error = table.get_column_as_vec_str("content").join("; ");
            return Err(anyhow!(error));
        }

        Ok(())
    }

    /// Take the task subscriptions and publications that are ready to subscribe and publish
    ///
    /// # Notes
    /// * See schema at [AvailableSubjects::SessionTasksSubscribePublish]
    /// * The columns are taken to prevent infinite loops of the same tasks
    /// 
    /// # Todo
    /// * Update the schema to include the backend, bucket, and additional storage config information
    pub async fn tasks_subscribe_publish(
        &self,
    ) -> Result<HashMap<(String, String), ProcessorSubjectsMap>> {
        // Get the subject
        let batches: Vec<RecordBatch> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string() }
            .subscribe_to_subject(self.runtime_env())?
            .ok_or(anyhow!("Unable to get the subject `{}` from object storage for session `{}` during tasks_subscribe.", 
                AvailableSubjects::SessionTasksSubscribePublish,
                self.get_name()
            ))?
            .try_collect()
            .await?;

        // Return if there are no tasks
        if batches.is_empty() {
            return Ok(HashMap::<(String, String), ProcessorSubjectsMap>::new());
        }

        let subject = SubjectBuilder::default()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(batches)?
            .build()?;

        // Clear the subject
        let _locations: Vec<RecordBatch> = clear_subject(self.runtime_env(), &AvailableSubjects::SessionTasksSubscribePublish.to_string(), false)?
            .try_collect()
            .await?;

        // Extract the columns
        let task_names = subject.get_column_as_vec_nonprimitive::<String>("task_name")?;
        let processor_names = subject.get_column_as_vec_nonprimitive::<String>("processor_name")?;
        let processor_types = subject.get_column_as_vec_nonprimitive::<String>("processor_type")?;
        let subscription_names =
            subject.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let subscription_table_names =
            subject.get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let publication_names =
            subject.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let publication_table_names =
            subject.get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let session_names = subject.get_column_as_vec_nonprimitive::<String>("session_name")?;

        // Map to objects
        let combined = task_names
            .into_iter()
            .zip(subscription_names)
            .zip(subscription_table_names)
            .zip(publication_names)
            .zip(publication_table_names)
            .zip(processor_names)
            .zip(processor_types)
            .zip(session_names)
            .map(
                |(
                    (
                        (
                            (
                                (
                                    ((task_name, subscription_names), subscription_table_names),
                                    publication_names,
                                ),
                                publication_table_names,
                            ),
                            processor_name,
                        ),
                        processor_type,
                    ),
                    session_name,
                )| {
                    let subscriptions = subscription_names
                        .iter()
                        .zip(subscription_table_names.iter())
                        .map(|(subscription_name, subscription_table_name)| {
                            Subscription::from_str_fuzzy(
                                subscription_name,
                                subscription_table_name,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let publications = publication_names
                        .iter()
                        .zip(publication_table_names.iter())
                        .map(|(publication_name, publication_table_name)| {
                            Publication::from_str_fuzzy(
                                publication_name,
                                publication_table_name,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let processor_subjects = ProcessorSubjectsBuilder::default()
                        .with_name(&processor_name)
                        .with_subscriptions(&subscriptions)
                        .with_publications(&publications)
                        .build()
                        .unwrap();
                    (
                        task_name,
                        processor_subjects,
                        processor_name,
                        processor_type,
                        session_name,
                    )
                },
            )
            .collect::<Vec<_>>();

        // Aggregate processors
        // DM: not possible to have two-levels of nesting with Arrow RecordBatches
        let mut tasks = HashMap::<(String, String), ProcessorSubjectsMap>::new();
        for (task_name, processor_subjects, processor_name, _processor_type, session_name) in
            combined
        {
            if let Some(task) = tasks.get_mut(&(task_name.to_string(), session_name.to_string())) {
                let _ = task.insert(processor_name, processor_subjects);
            } else {
                let mut processor = HashMap::<String, ProcessorSubjects>::new();
                let _ = processor.insert(processor_name, processor_subjects);
                let _ = tasks.insert((task_name.to_string(), session_name.to_string()), processor);
            }
        }

        Ok(tasks)
    }

    /// Create the metrics table if it does not exist or update with the new metrics
    pub async fn update_metrics_subjects(
        &mut self,
        diagnostics_vec: &[Diagnostics],
    ) -> Result<()> {
        // create the pivot table and clear the metrics
        let (metrics_subject, traces_subject, events_subject) =
            from_diagnostics_to_tables(diagnostics_vec)?;
        
        // DM: does it make sense to cache the current step?
        let step = self.current_superstep().await?;

        // update the state with the metrics
        if let Some(metrics_subject) = metrics_subject {
            if !self.subjects().contains_key(AvailableSubjects::SessionMetrics.to_string().as_str()) {
                let _ = self.subjects.insert(AvailableSubjects::SessionMetrics.to_string(), AvailableSubjects::SessionMetrics.to_schema());
            }
            let _publication: Vec<RecordBatch> = Publication::Extend { subject_name: AvailableSubjects::SessionMetrics.to_string() }
                .publish_to_subject(self.runtime_env(), metrics_subject.get_record_batches_own(), step)?
                .ok_or(anyhow!("Unable to put the subject `{}` into object storage for session `{}` while updating the metrics tables.",
                    AvailableSubjects::SessionMetrics,
                    self.get_name()))?
                .try_collect()
                .await?;
        }

        // update the state with the traces
        if let Some(traces_subject) = traces_subject {
            if !self.subjects().contains_key(AvailableSubjects::SessionTraces.to_string().as_str()) {
                let _ = self.subjects.insert(AvailableSubjects::SessionTraces.to_string(), AvailableSubjects::SessionTraces.to_schema());
            }
            let _publication: Vec<RecordBatch> = Publication::Extend { subject_name: AvailableSubjects::SessionTraces.to_string() }
                .publish_to_subject(self.runtime_env(), traces_subject.get_record_batches_own(), step)?
                .ok_or(anyhow!("Unable to put the subject `{}` into object storage for session `{}` while updating the metrics tables.",
                    AvailableSubjects::SessionTraces,
                    self.get_name()))?
                .try_collect()
                .await?;
        }

        // update the state with the events
        if let Some(events_subject) = events_subject {
            if !self.subjects().contains_key(AvailableSubjects::SessionEvents.to_string().as_str()) {
                let _ = self.subjects.insert(AvailableSubjects::SessionEvents.to_string(), AvailableSubjects::SessionEvents.to_schema());
            }
            let _publication: Vec<RecordBatch> = Publication::Extend { subject_name: AvailableSubjects::SessionEvents.to_string() }
                .publish_to_subject(self.runtime_env(), events_subject.get_record_batches_own(), step)?
                .ok_or(anyhow!("Unable to put the subject `{}` into object storage for session `{}` while updating the metrics tables.",
                    AvailableSubjects::SessionEvents,
                    self.get_name()))?
                .try_collect()
                .await?;
        }

        Ok(())
    }

    /// Increment the session superstep
    pub async fn increment_superstep(&self) -> Result<u32> {
        // Increment the superstep
        let next_superstep = self.current_superstep().await.unwrap_or_default() + 1;
        let session_names = [self.get_name()]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let supersteps = vec![next_superstep];
        let batch = create_session_supersteps_batch(session_names, supersteps)?;
        let table = Subject::get_builder()
            .with_name(AvailableSubjects::SessionSupersteps.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;

        // If this is the first increment then replace the empty batch
        let superstep_message = if next_superstep <= 1 {
            IPCMessageBuilder::default()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionSupersteps.to_string().as_str())
                .with_update(&Publication::Replace {
                    subject_name: AvailableSubjects::SessionSupersteps.to_string(),
                })
                .with_publisher(self.get_name())
                .make_name()?
                .build()?
        } else {
            IPCMessageBuilder::default()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionSupersteps.to_string().as_str())
                .with_update(&Publication::Extend {
                    subject_name: AvailableSubjects::SessionSupersteps.to_string(),
                })
                .with_publisher(self.get_name())
                .make_name()?
                .build()?
        };
        let messages = create_message_map(vec![superstep_message]);
        let (_update, _meta, errors) = self.update_subjects_from_messages(messages).await;
        if let Some(table) = errors {
            let error = table.get_column_as_vec_str("content").join("; ");
            return Err(anyhow!(error));
        }
        Ok(next_superstep)
    }

    /// Get the current session superstep
    pub async fn current_superstep(&self) -> Result<u32> {
        let batches: Vec<RecordBatch> = Subscription::AlwaysAllRecordBatches { subject_name: AvailableSubjects::SessionSuperstepMax.to_string() }
            .subscribe_to_subject(self.runtime_env())?
            .ok_or(anyhow!("Unable to get the subject `{}` from object storage for session `{}` while getting the current superstep.", 
                AvailableSubjects::SessionSuperstepMax,
                self.get_name()
            ))?
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SessionSuperstepMax.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let current_superstep = subject
            .get_column_as_vec_primitive::<u32>("superstep-Max")?
            .last()
            .ok_or(anyhow!(
                "Missing rows for `{}` in session `{}`.",
                AvailableSubjects::SessionSuperstepMax,
                self.get_name()
            ))?
            .to_owned();
        Ok(current_superstep)
    }

    /// Update the row counts for the subjects
    pub async fn update_subject_num_rows(&mut self) -> Result<()> {
        let mut subject_names = Vec::new();
        let mut num_rows = Vec::new();

        // DM: migrate to using `CountSubjectRowsSession` 
        // // Sort the hashmap
        // let mut sorted_map = self.subjects.iter().collect::<Vec<_>>();
        // sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        // for (_name, state) in sorted_map.iter() {
        //     let name = state.read().get_name().to_string();
        //     let num_row = state.read().count_rows() as i64;
        //     subject_names.push(name.clone());
        //     num_rows.push(num_row);
        // }

        // create the record batch
        let batch = create_subjects_num_rows_batch(subject_names, num_rows)?;

        // create the table
        let subject_num_rows = Subject::get_builder()
            .with_name(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;

        // Add the subjects num rows table to the state or update
        if !self.subjects().contains_key(AvailableSubjects::SubjectsNumRows.to_string().as_str()) {
            let _ = self.subjects.insert(AvailableSubjects::SubjectsNumRows.to_string(), AvailableSubjects::SubjectsNumRows.to_schema());
        }
        let step = self.current_superstep().await?;
        let _publication: Vec<RecordBatch> = Publication::Extend { subject_name: AvailableSubjects::SubjectsNumRows.to_string() }
            .publish_to_subject(self.runtime_env(), subject_num_rows.get_record_batches_own(), step)?
            .ok_or(anyhow!("Unable to put the subject `{}` into object storage for session `{}` while updating the subject number rows.",
                AvailableSubjects::SessionMetrics,
                self.get_name()))?
            .try_collect()
            .await?;
        Ok(())
    }

    /// Find the table by matching schemas
    pub fn get_subject_name_by_schema(&self, schema: &SchemaRef) -> Option<&str> {
        let mut sorted_map = self.subjects.iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        for (name, subject) in sorted_map.iter() {
            if schema.eq(subject) {
                return Some(name);
            }
        }
        None
    }

    /// Update the state from the published messages
    ///   and return a table of subject change logs and any errors
    pub async fn update_subjects_from_messages(
        &self,
        messages: IPCMessageMap,
    ) -> (Option<Subject>, Option<Subject>, Option<Subject>) {
        let mut change_log_subject_names = Vec::new();
        let mut change_log_task_names = Vec::new();
        let mut change_log_session_names = Vec::new();
        let mut change_log_num_rows = Vec::new();
        let mut change_log_supersteps = Vec::new();
        let mut store_meta_subject_names = Vec::new();
        let mut store_meta_task_names = Vec::new();
        let mut store_meta_session_names = Vec::new();
        let mut store_meta_num_rows = Vec::new();
        let mut store_meta_supersteps = Vec::new();
        let mut store_meta_location = Vec::new();
        let mut store_meta_bucket = Vec::new();
        let mut store_meta_e_tag = Vec::new();
        let mut store_meta_version = Vec::new();
        let mut store_meta_size = Vec::new();
        let mut store_meta_last_modified = Vec::new();
        let mut errors = Vec::new();

        // Update the subjects with each of the messages
        let step = self.current_superstep().await.unwrap_or_default();
        for (_name, message) in messages.into_iter() {
            // Should the subject be updated?
            let update = message.get_update().clone();
            if update == Publication::None {
                continue;
            }

            // Try to update the state with the new record batches
            let subject_name = message.get_update().subject_name().to_string();
            if let Some(schema) = self.subjects().get(subject_name.as_str()) {
                let publisher = message.get_publisher().to_string();

                // Handle any inconsistencies in the message
                // DM, todo!(): Mostly an issue with empty batches which should be ignored anyway
                match SubjectBuilder::new_from_ipc_stream(&message.get_message_own()) {
                    Ok(builder) => {
                        let subject = builder.with_name(subject_name.as_str()).build().unwrap();
                        let num_rows = subject.count_rows(); // DM: not used currently...

                        // Check for a mismatch in the schema and intercept any errors
                        if schema.ne(&subject.get_schema()) {                            
                            let error = format!(
                                "Schema `{}` for Subject `{subject_name}` from publisher `{publisher}` does match the cached Subject Schema `{}`",
                                    subject.get_schema(), schema
                            );
                            errors.push(error);
                            continue;
                        }

                        // Publish to the object store
                        let mut object_store_metadata = Vec::new();
                        if let Ok(Some(mut stream)) = update
                            .publish_to_subject(self.runtime_env(), subject.get_record_batches_own(), step) {
                            while let Some(batch) = stream.next().await {
                                match batch {
                                    Ok(metadata) => {
                                        // Record the subject object store metadata
                                        object_store_metadata.push(metadata);
                                    },
                                    Err(err) => {
                                        // Record the error
                                        let error = format!(
                                            "Subject `{subject_name}` from publisher `{publisher}` failed to update the target subject with error `{err:?}`"
                                        );
                                        errors.push(error);
                                    },
                                }
                            }
                        } else {
                            // Record the error
                            let error = format!(
                                "Subject `{subject_name}` from publisher `{publisher}` failed to put to the object store"
                            );
                            errors.push(error);
                        }

                        if !object_store_metadata.is_empty() {
                            // Record the subject object store metadata
                            let metadata_subject = Subject::get_builder()
                                .with_name("object_store_metadata")
                                .with_record_batches(object_store_metadata).unwrap()
                                .build().unwrap();
                            let n_rows = metadata_subject.count_rows();
                            store_meta_subject_names.extend((0..n_rows).map(|_| subject_name.clone()).collect::<Vec<_>>());
                            store_meta_task_names.extend((0..n_rows).map(|_| publisher.clone()).collect::<Vec<_>>());
                            store_meta_session_names.extend((0..n_rows).map(|_| self.get_name().to_string()).collect::<Vec<_>>());
                            store_meta_num_rows.extend((0..n_rows).map(|_| num_rows as i64).collect::<Vec<_>>());
                            store_meta_supersteps.extend((0..n_rows).map(|_| step as i64).collect::<Vec<_>>());
                            store_meta_location.extend(metadata_subject.get_column_as_vec_nonprimitive::<String>("location").unwrap());
                            store_meta_bucket.extend(metadata_subject.get_column_as_vec_nonprimitive::<String>("bucket").unwrap());
                            store_meta_e_tag.extend(metadata_subject.get_column_as_vec_nonprimitive::<String>("e_tag").unwrap());
                            store_meta_version.extend(metadata_subject.get_column_as_vec_nonprimitive::<String>("version").unwrap());
                            store_meta_size.extend(metadata_subject.get_column_as_vec_primitive::<u32>("size").unwrap());
                            store_meta_last_modified.extend(metadata_subject.get_column_as_vec_primitive::<i64>("last_modified").unwrap());

                            // Record the subject change log information
                            change_log_subject_names.push(subject_name);
                            change_log_task_names.push(publisher);
                            change_log_session_names.push(self.get_name().to_string());
                            change_log_num_rows.push(num_rows as i64);
                            change_log_supersteps.push(step as i64);
                        }
                        
                    }
                    Err(err) => {
                        let error = format!(
                            "Subject `{subject_name}` with update `{update:?}` from publisher `{publisher}` failed to build the object store with error `{err:?}`"
                        );
                        errors.push(error);
                    }
                }
            } else {
                // Mismatch in subject names of the update and cache
                let error = format!(
                    "Subject `{subject_name}` with update `{update:?}` is not in the session subject schema cache! Cached subjects are {:?}",
                    self.subjects().keys()
                );
                errors.push(error);
            }
        }

        // Prepare the change log subject
        let change_log = if !change_log_subject_names.is_empty() {
            let batch = create_subjects_change_log_batch(
                change_log_subject_names,
                change_log_task_names,
                change_log_session_names,
                change_log_num_rows,
                change_log_supersteps,
            )
            .unwrap();
            Some(
                AvailableSubjects::SubjectsChangeLog
                    .to_subject(None, Some(vec![batch]))
                    .unwrap(),
            )
        } else {
            None
        };

        // Prepare the object metadata subject
        let store_meta = if !store_meta_subject_names.is_empty() {
            let batch = create_subjects_object_store_meta_batch(
                store_meta_subject_names,
                store_meta_task_names,
                store_meta_session_names,
                store_meta_num_rows,
                store_meta_supersteps,
                store_meta_location,
                store_meta_bucket,
                store_meta_e_tag,
                store_meta_version,
                store_meta_size,
                store_meta_last_modified,
            )
            .unwrap();
            Some(
                AvailableSubjects::SubjectsObjectStoreMeta
                    .to_subject(None, Some(vec![batch]))
                    .unwrap(),
            )
        } else {
            None
        };

        // Prepare the errors subject
        let error_log = if !errors.is_empty() {
            let tools = errors
                .iter()
                .map(|_| "tool".to_string())
                .collect::<Vec<_>>();
            let timestamps = errors
                .iter()
                .map(|_| create_timestamp_micros())
                .collect::<Vec<_>>();
            let batch = create_chat_record_batch(tools, errors, timestamps).unwrap();
            Some(
                AvailableSubjects::SessionErrors
                    .to_subject(None, Some(vec![batch]))
                    .unwrap(),
            )
        } else {
            None
        };

        (change_log, store_meta, error_log)
    }
}

impl MappableTrait for SessionContext {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for SessionContext {
    type T = SessionContextBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_session_context_builder, Task, test_task
    };
    use arrow::array::Int64Array;
    use phymes_core::{
        IPCMessage, create_session_tasks_subscribe_aggregate_batch,
        create_session_tasks_subscribe_publish_batch,
        test_subject,
    };
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempdir;

    #[test]
    fn test_session_get_subject_name_by_schema() -> Result<()> {
        let session_context =
            test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?.build()?;

        // table should be found
        let schema = test_subject::make_test_subject_schema(8)?;
        let name = session_context.get_table_name_by_schema(&schema).unwrap();
        assert_eq!(name, "state_1");

        // table should not be found
        let schema = test_subject::make_test_subject_schema(2)?;
        let name = session_context.get_table_name_by_schema(&schema);
        assert!(name.is_none());
        Ok(())
    }

    #[test]
    fn test_session_update_subject_num_rows_subject() -> Result<()> {
        let mut session_context =
            test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?.build()?;
        session_context.update_subject_num_rows();
        let info = session_context
            .subjects()
            .get(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .unwrap()
            .read();

        assert_eq!(
            info.get_column_as_vec_str("subject_name"),
            [
                "processor_1",
                "processor_2",
                "processor_3",
                "state_1",
                "state_2",
                "state_3",
            ]
        );
        let num_rows = info
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("num_rows")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(num_rows, [1, 1, 1, 12, 12, 12]);

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_session_read_write_state() -> Result<()> {
        // Create the session
        let session_context =
            test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?.build()?;

        // Write the session to disk
        let tmp_dir = tempdir()?;
        session_context.write_state(tmp_dir.path().to_str().unwrap(), "tag")?;

        // Read the state
        let mut session_context_empty =
            test_session_context_builder::make_test_session_context_builder_parallel_empty("session_1", 25)?.build()?;
        session_context_empty.read_state(tmp_dir.path().to_str().unwrap(), "tag")?;

        for subject in session_context.subjects().keys() {
            assert_eq!(
                session_context
                    .subjects()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_record_batches(),
                session_context_empty
                    .subjects()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_record_batches()
            );
            assert_eq!(
                session_context
                    .subjects()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_schema(),
                session_context_empty
                    .subjects()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_schema()
            );
            assert_eq!(
                session_context
                    .subjects()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_name(),
                session_context_empty
                    .subjects()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_name()
            );
        }
        tmp_dir.close()?;
        Ok(())
    }

    #[test]
    fn test_session_update_subjects_from_messages() -> Result<()> {
        // Case 1: no state update
        let session_context =
            test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?.build()?;
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::None,
            true,
        )?;
        let (updates, errors) = session_context.update_subjects_from_messages(input);

        // check the updates
        assert!(updates.is_none());
        assert!(errors.is_none());

        // check the session
        assert_eq!(
            session_context
                .subjects()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .subjects()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .subjects()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .subjects()
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

        // Case 2: update state
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::Extend {
                subject_name: "state_1".to_string(),
            },
            true,
        )?;
        let (updates, errors) = session_context.update_subjects_from_messages(input);

        // check the updates
        assert_eq!(updates.as_ref().unwrap().count_rows(), 1);
        assert!(errors.is_none());
        let col = updates
            .as_ref()
            .unwrap()
            .get_column_as_vec_str("subject_name");
        assert_eq!(col, ["state_1"]);
        let col = updates.as_ref().unwrap().get_column_as_vec_str("task_name");
        assert_eq!(col, ["session_1"]);
        let col = updates
            .as_ref()
            .unwrap()
            .get_column_as_vec_str("session_name");
        assert_eq!(col, ["session_1"]);
        let col = updates
            .as_ref()
            .unwrap()
            .get_column_as_vec_primitive::<i64>("num_rows")?;
        assert_eq!(col, [12]);

        // check the session context
        assert_eq!(
            session_context
                .subjects()
                .get("state_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            6
        ); // Originally 3
        assert_eq!(
            session_context
                .subjects()
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
            session_context
                .subjects()
                .get("state_2")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );
        assert_eq!(
            session_context
                .subjects()
                .get("state_3")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .len(),
            3
        );

        // Case 3: Error due to mismatching schemas
        let input = test_task::make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &Publication::Extend {
                subject_name: "state_1".to_string(),
            },
            false,
        )?;
        let (updates, errors) = session_context.update_subjects_from_messages(input);
        assert!(updates.is_none());
        assert!(errors.is_some());
        assert_eq!(errors.unwrap().count_rows(), 1);

        // Case 4: Error due to mismatching table names
        let message = IPCMessage::new(
            "task_1",
            "state_1",
            "session_1",
            Some(test_subject::make_test_subject("state_1", 4, 8, 3)?.to_ipc_stream()?),
            Some(Publication::Extend {
                subject_name: "NotFound".to_string(),
            }),
        );
        let mut input = HashMap::<String, IPCMessage>::new();
        input.insert(message.get_name().to_string(), message);
        let (updates, errors) = session_context.update_subjects_from_messages(input);
        assert!(updates.is_none());
        assert!(errors.is_some());
        assert_eq!(errors.unwrap().count_rows(), 1);

        Ok(())
    }

    #[test]
    fn test_session_tasks_subscribe_all_subscribe() -> Result<()> {
        // Make the test data
        let session_names = ["session_1", "session_1", "session_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let task_names = ["session_1", "task_1", "task_1", "task_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = ["session_1", "processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = [
            "ProcessorEcho",
            "ProcessorMock",
            "ProcessorMock",
            "ProcessorMock",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let subscription_names = vec![
            ["OnUpdateLastRecordBatch"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscription_table_names = vec![
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_1", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_2", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_3", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscribe_type = ["Any", "All", "All", "All"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let update_types = [
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let supersteps = vec![vec![0], vec![0, 0], vec![0, 0], vec![0, 0]];
        // let timestamps = vec![vec![1768954478778611],
        //     vec![1768954478778609,1768954478778609],
        //     vec![1768954478778609,1768954478778609],
        //     vec![1768954478778609,1768954478778609]];
        let superstep_lasts = vec![vec![1], vec![0, 1], vec![0, 1], vec![0, 1]];
        // let timestamp_lasts = vec![vec![1768954478786822],
        //     vec![1768954478776320,1768954478786822],
        //     vec![1768954478776344,1768954478786822],
        //     vec![1768954478776354,1768954478786822]];

        let batch = create_session_tasks_subscribe_aggregate_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscribe_type,
            update_types,
            subscription_names,
            subscription_table_names,
            supersteps,
            superstep_lasts,
        )?;
        let table_tasks_subscribe_aggregate =
            AvailableSubjects::SessionTasksSubscribeAggregate.to_subject(None, Some(vec![batch]))?;
        let table_tasks_subscribe =
            AvailableSubjects::SessionTasksSubscribe.to_subject(None, None)?;

        // Make the session context for testing
        let mut state = HashMap::<String, Arc<RwLock<Subject>>>::new();
        let _ = state.insert(
            table_tasks_subscribe_aggregate.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe_aggregate)),
        );
        let _ = state.insert(
            table_tasks_subscribe.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe)),
        );
        let session_context = SessionContext::new(
            "session_1".to_string(),
            HashMap::<String, Arc<Task>>::new(),
            state,
            HashMap::<String, Arc<RuntimeEnv>>::new(),
            true,
            make_store(&ObjectStorageBackend::default(), None, None)?
        );

        // Run and check the updated state
        session_context.tasks_subscribe()?;
        let table_reading = session_context
            .subjects()
            .get("SessionTasksSubscribe")
            .unwrap()
            .read();
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(
            column,
            [
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1"
            ]
        );
        let column = table_reading.get_column_as_vec_str("task_name");
        assert_eq!(
            column,
            [
                "session_1",
                "task_1",
                "task_1",
                "task_1",
                "task_1",
                "task_1",
                "task_1"
            ]
        );
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(
            column,
            [
                "session_1",
                "processor_1",
                "processor_1",
                "processor_2",
                "processor_2",
                "processor_3",
                "processor_3"
            ]
        );
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(
            column,
            [
                "ProcessorEcho",
                "ProcessorMock",
                "ProcessorMock",
                "ProcessorMock",
                "ProcessorMock",
                "ProcessorMock",
                "ProcessorMock"
            ]
        );
        let column = table_reading.get_column_as_vec_str("subscription_name");
        assert_eq!(
            column,
            [
                "OnUpdateLastRecordBatch",
                "AlwaysAllRecordBatches",
                "OnUpdateAllRecordBatches",
                "AlwaysAllRecordBatches",
                "OnUpdateAllRecordBatches",
                "AlwaysAllRecordBatches",
                "OnUpdateAllRecordBatches"
            ]
        );
        let column = table_reading.get_column_as_vec_str("subscription_table_name");
        assert_eq!(
            column,
            [
                "state_1",
                "processor_1",
                "state_1",
                "processor_2",
                "state_1",
                "processor_3",
                "state_1"
            ]
        );
        Ok(())
    }

    #[test]
    fn test_session_tasks_subscribe_none_subscribe() -> Result<()> {
        // Make the test data
        let session_names = ["session_1", "session_1", "session_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let task_names = ["session_1", "task_1", "task_1", "task_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = ["session_1", "processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = [
            "ProcessorEcho",
            "ProcessorMock",
            "ProcessorMock",
            "ProcessorMock",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let subscription_names = vec![
            ["OnUpdateLastRecordBatch"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscription_table_names = vec![
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_1", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_2", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_3", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscribe_type = ["Any", "All", "All", "All"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let update_types = [
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let supersteps = vec![vec![0], vec![0, 0], vec![0, 0], vec![0, 0]];
        // let timestamps = vec![vec![1768954478778611],
        //     vec![1768954478778609,1768954478778609],
        //     vec![1768954478778609,1768954478778609],
        //     vec![1768954478778609,1768954478778609]];
        let superstep_lasts = vec![vec![0], vec![0, 0], vec![0, 0], vec![0, 0]];

        let batch = create_session_tasks_subscribe_aggregate_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscribe_type,
            update_types,
            subscription_names,
            subscription_table_names,
            supersteps,
            superstep_lasts,
        )?;
        let table_tasks_subscribe_aggregate =
            AvailableSubjects::SessionTasksSubscribeAggregate.to_subject(None, Some(vec![batch]))?;
        let table_tasks_subscribe =
            AvailableSubjects::SessionTasksSubscribe.to_subject(None, None)?;

        // Make the session context for testing
        let mut state = HashMap::<String, Arc<RwLock<Subject>>>::new();
        let _ = state.insert(
            table_tasks_subscribe_aggregate.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe_aggregate)),
        );
        let _ = state.insert(
            table_tasks_subscribe.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe)),
        );
        let session_context = SessionContext::new(
            "session_1".to_string(),
            HashMap::<String, Arc<Task>>::new(),
            state,
            HashMap::<String, Arc<RuntimeEnv>>::new(),
            true,
            make_store(&ObjectStorageBackend::default(), None, None)?
        );

        // Run and check the updated state
        session_context.tasks_subscribe()?;
        let table_reading = session_context
            .subjects()
            .get("SessionTasksSubscribe")
            .unwrap()
            .read();
        assert_eq!(table_reading.count_rows(), 0);
        Ok(())
    }

    #[test]
    fn test_session_tasks_subscribe_some_subscribe() -> Result<()> {
        // Make the test data
        let session_names = ["session_1", "session_1", "session_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let task_names = ["session_1", "task_1", "task_1", "task_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = ["session_1", "processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = [
            "ProcessorEcho",
            "ProcessorMock",
            "ProcessorMock",
            "ProcessorMock",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let subscription_names = vec![
            ["OnUpdateLastRecordBatch"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscription_table_names = vec![
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_1", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_2", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_3", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscribe_type = ["Any", "All", "All", "All"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let update_types = [
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
            "SubjectChangedSinceLastRunUpdate",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let supersteps = vec![vec![0], vec![0, 0], vec![0, 0], vec![0, 0]];
        // let timestamps = vec![vec![1768954478778611],
        //     vec![1768954478778609,1768954478778609],
        //     vec![1768954478778609,1768954478778609],
        //     vec![1768954478778609,1768954478778609]];
        let superstep_lasts = vec![vec![1], vec![0, 0], vec![0, 1], vec![0, 1]];
        // let timestamp_lasts = vec![vec![1768954478786822],
        //     vec![1768954478776320,0],
        //     vec![1768954478776344,1768954478786822],
        //     vec![1768954478776354,1768954478786822]];

        let batch = create_session_tasks_subscribe_aggregate_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscribe_type,
            update_types,
            subscription_names,
            subscription_table_names,
            supersteps,
            superstep_lasts,
        )?;
        let table_tasks_subscribe_aggregate =
            AvailableSubjects::SessionTasksSubscribeAggregate.to_subject(None, Some(vec![batch]))?;
        let table_tasks_subscribe =
            AvailableSubjects::SessionTasksSubscribe.to_subject(None, None)?;

        // Make the session context for testing
        let mut state = HashMap::<String, Arc<RwLock<Subject>>>::new();
        let _ = state.insert(
            table_tasks_subscribe_aggregate.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe_aggregate)),
        );
        let _ = state.insert(
            table_tasks_subscribe.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe)),
        );
        let session_context = SessionContext::new(
            "session_1".to_string(),
            HashMap::<String, Arc<Task>>::new(),
            state,
            HashMap::<String, Arc<RuntimeEnv>>::new(),
            true,
            make_store(&ObjectStorageBackend::default(), None, None)?
        );

        // Run and check the updated state
        session_context.tasks_subscribe()?;
        let table_reading = session_context
            .subjects()
            .get("SessionTasksSubscribe")
            .unwrap()
            .read();
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, ["session_1"]);
        let column = table_reading.get_column_as_vec_str("task_name");
        assert_eq!(column, ["session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, ["session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, ["ProcessorEcho"]);
        let column = table_reading.get_column_as_vec_str("subscription_name");
        assert_eq!(column, ["OnUpdateLastRecordBatch"]);
        let column = table_reading.get_column_as_vec_str("subscription_table_name");
        assert_eq!(column, ["state_1"]);
        Ok(())
    }

    #[test]
    fn test_session_tasks_subscribe_publish() -> Result<()> {
        // Make the test data
        let session_names = ["session_1", "session_1", "session_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let task_names = ["task_1", "task_1", "task_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = ["processor_1", "processor_2", "processor_3", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = [
            "ProcessorMock",
            "ProcessorMock",
            "ProcessorMock",
            "ProcessorEcho",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let subscription_names = vec![
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["AlwaysAllRecordBatches", "OnUpdateAllRecordBatches"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["OnUpdateLastRecordBatch"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let subscription_table_names = vec![
            ["processor_1", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_2", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["processor_3", "state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let publication_names = vec![
            ["Extend"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["Extend"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["Extend"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["Extend"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];
        let publication_table_names = vec![
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ["state_1"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ];

        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table_tasks_subscribe_publish =
            AvailableSubjects::SessionTasksSubscribePublish.to_subject(None, Some(vec![batch]))?;

        // Make the session context for testing
        let mut state = HashMap::<String, Arc<RwLock<Subject>>>::new();
        let _ = state.insert(
            table_tasks_subscribe_publish.get_name().to_string(),
            Arc::new(RwLock::new(table_tasks_subscribe_publish)),
        );
        let session_context = SessionContext::new(
            "session_1".to_string(),
            HashMap::<String, Arc<Task>>::new(),
            state,
            HashMap::<String, Arc<RuntimeEnv>>::new(),
            true,
            make_store(&ObjectStorageBackend::default(), None, None)?
        );

        // Run and check the updated state
        let tasks = session_context.tasks_subscribe_publish()?;
        let mut expected_tasks =
            HashMap::<(String, String), HashMap<String, ProcessorSubjects>>::new();
        let mut processor_subjects_map = HashMap::<String, ProcessorSubjects>::new();
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name("session_1")
            .with_subscriptions(&[Subscription::OnUpdateLastRecordBatch {
                subject_name: "state_1".to_string(),
            }])
            .with_publications(&[Publication::Extend {
                subject_name: "state_1".to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert("session_1".to_string(), processor_subjects);
        let _ = expected_tasks.insert(
            ("session_1".to_string(), "session_1".to_string()),
            processor_subjects_map,
        );
        let mut processor_subjects_map = HashMap::<String, ProcessorSubjects>::new();
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name("processor_1")
            .with_subscriptions(&[
                Subscription::AlwaysAllRecordBatches {
                    subject_name: "processor_1".to_string(),
                },
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: "state_1".to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: "state_1".to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert("processor_1".to_string(), processor_subjects);
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name("processor_2")
            .with_subscriptions(&[
                Subscription::AlwaysAllRecordBatches {
                    subject_name: "processor_2".to_string(),
                },
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: "state_1".to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: "state_1".to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert("processor_2".to_string(), processor_subjects);
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name("processor_3")
            .with_subscriptions(&[
                Subscription::AlwaysAllRecordBatches {
                    subject_name: "processor_3".to_string(),
                },
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: "state_1".to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: "state_1".to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert("processor_3".to_string(), processor_subjects);
        let _ = expected_tasks.insert(
            ("task_1".to_string(), "session_1".to_string()),
            processor_subjects_map,
        );

        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks, expected_tasks);
        Ok(())
    }
}
