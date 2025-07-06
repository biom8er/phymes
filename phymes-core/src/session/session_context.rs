use anyhow::Result;
use arrow::array::{ArrayRef, UInt64Array};
use arrow::array::{BooleanArray, StringArray};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use futures::{FutureExt, Stream, TryStreamExt};
use parking_lot::{Mutex, RwLock};
use std::fs::File;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use tokio::task::JoinSet;
use tracing::{Level, event, instrument};

use super::common_traits::PubSubTrait;
use super::{
    common_traits::{
        BuildableTrait, BuilderTrait, IncomingMessageMap, MappableTrait, OutgoingMessageMap,
        RunnableTrait, StateMap, TaskMap,
    },
    runtime_env::RuntimeEnv,
    session_context_builder::SessionContextBuilder,
};
use crate::metrics::{ArrowTaskMetricsSet, HashMap};
use crate::table::arrow_table_publish::ArrowTablePublish;
use crate::table::{
    arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
    arrow_table_publish::ArrowTableUpdateTrait,
};
use crate::task::{
    arrow_message::{
        ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
        ArrowIncomingMessageTrait, ArrowMessageBuilderTrait, ArrowMessageTrait,
        ArrowOutgoingMessage, ArrowOutgoingMessageTrait,
    },
    arrow_task::ArrowTaskTrait,
};

/// The `SessionContext` creates an execution graph based on a
/// `SessionPlan` and manages the running of individual tasks
/// and the messages passed between tasks.
#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct SessionContext {
    /// A unique UUID that identifies the session
    pub(crate) name: String,
    /// The list of available tasks that can be run during the session
    pub(crate) tasks: TaskMap,
    /// Data that should be persisted between queries
    pub(crate) state: StateMap,
    /// Metrics tracked during task runs
    pub(crate) metrics: ArrowTaskMetricsSet,
    /// Runtime environment configuration to use during task runs
    pub(crate) runtime_envs: HashMap<String, Arc<Mutex<RuntimeEnv>>>,
    /// The maximum number of iterations before stopping
    pub(crate) max_iter: usize,
}

impl SessionContext {
    /// Get a task
    pub(crate) fn get_tasks(&self) -> &TaskMap {
        &self.tasks
    }

    /// Get state
    pub fn get_states(&self) -> &StateMap {
        &self.state
    }

    /// Get the metrics for the session
    pub fn get_metrics_info_as_table(&self, table_name: &str) -> Result<ArrowTable> {
        // extract out values from metrics
        let mut task_names_vec = Vec::<String>::new();
        let mut metric_names_vec = Vec::<String>::new();
        let mut metric_values_vec = Vec::<u64>::new();
        // let mut metrics_sorted = self.metrics.clone_inner().iter().map(|m| Arc::clone(m)).collect::<Vec<_>>();
        // metrics_sorted.sort_by(|a, b| a.task().as_ref().unwrap().cmp(b.task().as_ref().unwrap()));
        // metrics_sorted.sort_by(|a, b| a.value().name().to_string().cmp(&b.value().name().to_string()));
        for metric in self.metrics.clone_inner().iter() {
            task_names_vec.push(metric.task().as_ref().unwrap().to_string());
            metric_names_vec.push(metric.value().name().to_string());
            metric_values_vec.push(metric.value().as_usize() as u64);
        }

        if let Some(val) = self.metrics.clone_inner().elapsed_compute() {
            task_names_vec.push("All".to_string());
            metric_names_vec.push("elapsed_compute".to_string());
            metric_values_vec.push(val as u64);
        }

        if let Some(val) = self.metrics.clone_inner().output_rows() {
            task_names_vec.push("All".to_string());
            metric_names_vec.push("output_rows".to_string());
            metric_values_vec.push(val as u64);
        }

        // create the record batch
        let task_names: ArrayRef = Arc::new(StringArray::from(task_names_vec));
        let metric_names: ArrayRef = Arc::new(StringArray::from(metric_names_vec));
        let metric_values: ArrayRef = Arc::new(UInt64Array::from(metric_values_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("task_name", task_names),
            ("metric_name", metric_names),
            ("metric_value", metric_values),
        ])?;

        // create the table
        ArrowTable::get_builder()
            .with_name(table_name)
            .with_record_batches(vec![batch])?
            .build()
    }

    /// Get the max iterations
    pub(crate) fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Get the subject schema
    pub fn get_subjects_info_as_table(&self, table_name: &str) -> Result<ArrowTable> {
        let mut subject_names = Vec::new();
        let mut cols_names = Vec::new();
        let mut type_names = Vec::new();
        let mut num_rows = Vec::new();

        // Sort the hashmap
        let mut sorted_map = self.state.iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        for (_name, state) in sorted_map.iter() {
            let fields = state.try_read().unwrap().get_schema().fields().clone();
            let name = state.try_read().unwrap().get_name().to_string();
            let num_row = state.try_read().unwrap().count_rows() as u64;
            for field in fields.iter() {
                subject_names.push(name.clone());
                cols_names.push(field.name().to_string());
                type_names.push(field.data_type().to_string());
                num_rows.push(num_row);
            }
        }

        // create the record batch
        let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
        let cols_names: ArrayRef = Arc::new(StringArray::from(cols_names));
        let type_names: ArrayRef = Arc::new(StringArray::from(type_names));
        let num_rows: ArrayRef = Arc::new(UInt64Array::from(num_rows));
        let batch = RecordBatch::try_from_iter(vec![
            ("subject_names", subject_names),
            ("column_names", cols_names),
            ("type_names", type_names),
            ("num_rows", num_rows),
        ])?;

        // create the table
        ArrowTable::get_builder()
            .with_name(table_name)
            .with_record_batches(vec![batch])?
            .build()
    }

    /// Get the session tasks information
    /// as a list of task_names, processor_names, subject_names, and pub_or_sub
    ///   where publications are + and subscriptions are -
    pub fn get_tasks_info_as_table(&self, table_name: &str) -> Result<ArrowTable> {
        let mut task_names = Vec::new();
        let mut processor_names = Vec::new();
        let mut subject_names = Vec::new();
        let mut pub_or_sub = Vec::new();

        // Sort the hashmap
        let mut sorted_map = self.tasks.iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        for (name, task) in sorted_map.iter() {
            for p in task.get_processors().iter() {
                // Get the sub and pub
                for sub in p.get_subscriptions().iter() {
                    subject_names.push(sub.get_table_name().to_string());
                    pub_or_sub.push("-".to_string());
                    task_names.push(name.to_string());
                    processor_names.push(p.get_name().to_string());
                }
                for publications in p.get_publications().iter() {
                    subject_names.push(publications.get_table_name().to_string());
                    pub_or_sub.push("+".to_string());
                    task_names.push(name.to_string());
                    processor_names.push(p.get_name().to_string());
                }
            }
        }

        // create the record batch
        let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
        let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
        let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
        let pub_or_sub: ArrayRef = Arc::new(StringArray::from(pub_or_sub));
        let batch = RecordBatch::try_from_iter(vec![
            ("task_names", task_names),
            ("processor_names", processor_names),
            ("subject_names", subject_names),
            ("pub_or_sub", pub_or_sub),
        ])?;

        // create the table
        ArrowTable::get_builder()
            .with_name(table_name)
            .with_record_batches(vec![batch])?
            .build()
    }

    /// Find the table by matching schemas
    pub fn get_table_name_by_schema(&self, schema: &SchemaRef) -> Option<&str> {
        let mut sorted_map = self.state.iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(b.0));
        for (name, table) in sorted_map.iter() {
            if schema.eq(&table.try_read().unwrap().get_schema()) {
                return Some(name);
            }
        }
        None
    }

    /// Get the subject as a csv string
    pub fn get_subject_as_csv_str(
        &self,
        name: &str,
        delimiter: u8,
        header: bool,
    ) -> Result<String> {
        let csv = self
            .state
            .get(name)
            .unwrap()
            .try_read()
            .unwrap()
            .to_csv(delimiter, header)?;
        let csv_str = String::from_utf8_lossy(csv.as_ref()).into_owned();
        Ok(csv_str)
    }

    /// Initialize the superstep_update with all tasks and their update subscriptions
    pub fn init_superstep_updates(&self) -> HashMap<String, HashMap<String, bool>> {
        let mut init = HashMap::<String, HashMap<String, bool>>::new();
        for (task_name, task) in self.tasks.iter() {
            let mut subscriptions = HashMap::<String, bool>::new();
            for subscription in task.get_subscriptions() {
                if subscription.is_update() {
                    subscriptions.insert(subscription.get_table_name().to_string(), false);
                }
            }
            init.insert(task_name.to_string(), subscriptions);
        }
        init
    }

    /// Save the current state to disk
    pub fn write_state(&self, path: &str, tag: &str) -> Result<()> {
        for (name, subject) in self.state.iter() {
            let pathname = format!("{path}/{tag}-{}-{name}", self.get_name());
            let mut file = std::fs::File::create(pathname)?;
            match subject.try_read().unwrap().to_ipc_file(&mut file) {
                Ok(()) => (),
                Err(e) => event!(Level::ERROR, "Error writing state: {e:?}"),
            };
        }
        Ok(())
    }

    /// Read state
    pub fn read_state(&mut self, path: &str, tag: &str) -> Result<()> {
        for (name, subject) in self.state.iter() {
            let pathname = format!("{path}/{tag}-{}-{name}", self.get_name());
            let file = std::fs::File::open(pathname)?;
            match ArrowTableBuilder::new_from_ipc_file(&file) {
                Ok(table_builder) => {
                    let table = table_builder.with_name(name).build()?;
                    let update = ArrowTablePublish::Replace {
                        table_name: name.to_string(),
                    };
                    subject
                        .try_write()
                        .unwrap()
                        .update_table(table.get_record_batches_own(), update)?;
                }
                Err(e) => event!(Level::ERROR, "Error reading state: {e:?}"),
            };
        }
        Ok(())
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

/// State tracked during the course of running a [`SessionStream`]
#[derive(Default, Debug)]
pub struct SessionStreamState {
    /// The session context
    session_context: SessionContext,
    /// The current iteration
    iter: usize,
    /// The changes from the last superstep
    /// where keys are tasks and values are subjects
    superstep_updates: HashMap<String, HashMap<String, bool>>,
}

impl SessionStreamState {
    pub fn new(session_context: SessionContext) -> Self {
        let init = session_context.init_superstep_updates();
        Self {
            session_context,
            iter: 0,
            superstep_updates: init,
        }
    }

    /// Get the session context
    pub fn get_session_context(&self) -> &SessionContext {
        &self.session_context
    }

    /// Get the session context
    pub fn get_session_context_own(self) -> SessionContext {
        self.session_context
    }

    /// Get the session context
    pub fn get_session_context_mut(&mut self) -> &mut SessionContext {
        &mut self.session_context
    }

    /// Get the current iteration
    pub fn get_iter(&self) -> usize {
        self.iter
    }

    /// Update the current iteration
    pub fn set_iter(&mut self, iter: usize) {
        self.iter = iter;
    }

    /// Get the superstep update
    pub fn get_superstep_updates(&self) -> &HashMap<String, HashMap<String, bool>> {
        &self.superstep_updates
    }

    /// Get all tasks whose subscribed subjects have all been updated
    pub fn get_tasks_with_all_subjects_updated(&self) -> Vec<String> {
        let mut task_names = Vec::new();
        for (task_name, v) in self.superstep_updates.iter() {
            let mut updated = true;
            for (_subject, is_updated) in v.iter() {
                if !*is_updated {
                    // DM: useful for debugging
                    //println!("get_tasks_with_all_subjects_updated: tasks: {}, subject: {}, is_updated: {}", task_name, _subject, is_updated);
                    updated = false;
                    break;
                }
            }
            if updated {
                task_names.push(task_name.to_string());
            }
        }
        task_names.sort();
        task_names
    }

    /// Extend the superstep update
    ///
    /// # Notes
    ///
    /// We assume that all tasks available have already been added
    ///   upon initialization of `superstep_updates`
    ///
    /// # Arguments
    ///
    /// * `updates` - Map where keys are subjects and values are publishers
    pub fn extend_superstep_updates(&mut self, updates: HashMap<String, Vec<String>>) {
        for (task_name, subjects) in self.superstep_updates.iter_mut() {
            for (subject, publishers) in updates.iter() {
                for publisher in publishers.iter() {
                    if publisher != task_name && subjects.contains_key(subject) {
                        // DM: Useful for debugging
                        //println!("extend_superstep_updates: tasks: {}, subject: {}, publisher: {}", task_name, subject, publisher);
                        *subjects.get_mut(subject).unwrap() = true;
                    }
                }
            }
        }
    }

    /// Set the last superstep update
    pub fn set_superstep_updates(&mut self, updates: HashMap<String, HashMap<String, bool>>) {
        self.superstep_updates = updates;
    }

    /// Clear task from superstep update
    pub fn clear_subjects_from_task_for_superstep_updates(&mut self, task_name: &str) {
        let task_update = self.superstep_updates.get(task_name).unwrap();
        let task_update = task_update
            .iter()
            .map(|(s, _)| (s.to_string(), false))
            .collect::<HashMap<_, _>>();
        self.superstep_updates
            .insert(task_name.to_string(), task_update);
    }

    /// Update the state from the published messages
    /// and return a map of changed subscriptions along with their publishers
    #[instrument(skip(self, messages))]
    fn update_state_from_messages(
        &self,
        messages: IncomingMessageMap,
    ) -> HashMap<String, Vec<String>> {
        let mut subjects_updated = HashMap::<String, Vec<String>>::new();
        event!(Level::DEBUG, "Message updates {:?}.", &messages.keys());
        for (_name, message) in messages.into_iter() {
            // Try to update the state with the new record batches
            if let Some(state) = self
                .session_context
                .get_states()
                .get(message.get_update().get_table_name())
            {
                let update = message.get_update().clone();
                let publisher = message.get_publisher().to_string();
                state
                    .try_write()
                    .unwrap()
                    .update_table(message.get_message_own().get_record_batches_own(), update)
                    .unwrap();

                // Record the table name that was updated and the pubisher who updated it
                if let Some(v) = subjects_updated.get_mut(state.try_read().unwrap().get_name()) {
                    v.push(publisher);
                } else {
                    subjects_updated.insert(
                        state.try_read().unwrap().get_name().to_string(),
                        vec![publisher],
                    );
                }
            }
        }
        subjects_updated
    }

    /// Update the state from serde_json::Value
    /// and return a map of changed subscriptions along with their publishers
    ///
    /// # Notes
    ///
    /// We assume that the caller is often a server that will
    ///   intercept error calls e.g., from bad JSON strings
    ///
    /// # Arguments
    ///
    /// * `schema` - the RecordBatch schema
    /// * `json_str` - JSON string
    /// * `publish` - the publication protocol
    pub fn update_state_from_json_str(
        &self,
        schema: &SchemaRef,
        json_str: &str,
        publish: &ArrowTablePublish,
    ) -> Result<HashMap<String, Vec<String>>> {
        // Convert to json value
        let json_value: Vec<serde_json::Value> = serde_json::from_str(json_str)?;

        // Create the incoming message
        let table = ArrowTableBuilder::new()
            .with_schema(schema.clone())
            .with_name(publish.get_table_name())
            .with_json_values(&json_value)?
            .build()
            .unwrap();
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(publish.get_table_name())
            .with_subject(publish.get_table_name())
            .with_publisher(self.session_context.get_name())
            .with_message(table)
            .with_update(publish)
            .build()
            .unwrap();
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Update the state
        Ok(self.update_state_from_messages(incoming_message_map))
    }

    /// Update the state from string formated as a CSV
    /// and return a map of changed subscriptions along with their publishers
    ///
    /// # Notes
    ///
    /// We assume that the caller is often a server that will
    ///   intercept error calls e.g., from badly formatted CSV
    ///
    /// # Arguments
    ///
    /// * `schema` - the RecordBatch schema
    /// * `csv_str` - CSV string
    /// * `publish` - the publication protocol
    pub fn update_state_from_csv_str(
        &self,
        schema: &SchemaRef,
        csv_str: &str,
        publish: &ArrowTablePublish,
        delimiter: u8,
        header: bool,
        batch_size: usize,
    ) -> Result<HashMap<String, Vec<String>>> {
        // Create the incoming message
        let table = ArrowTableBuilder::new()
            .with_schema(schema.clone())
            .with_name(publish.get_table_name())
            .with_csv(csv_str.as_bytes(), delimiter, header, batch_size)?
            .build()
            .unwrap();
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(publish.get_table_name())
            .with_subject(publish.get_table_name())
            .with_publisher(self.session_context.get_name())
            .with_message(table)
            .with_update(publish)
            .build()
            .unwrap();
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Update the state
        Ok(self.update_state_from_messages(incoming_message_map))
    }

    /// Write superstep updates to file
    pub fn write_superstep_updates(&self, file: &mut File) -> Result<()> {
        // Convert the superstep updates to a record batch
        let mut task_vec = Vec::new();
        let mut subject_vec = Vec::new();
        let mut status_vec = Vec::new();
        for (task_name, subjects) in self.superstep_updates.iter() {
            for (subject_name, status) in subjects.iter() {
                task_vec.push(task_name.to_string());
                subject_vec.push(subject_name.to_string());
                status_vec.push(status.to_owned());
            }
        }
        let task_names: ArrayRef = Arc::new(StringArray::from(task_vec));
        let subject_names: ArrayRef = Arc::new(StringArray::from(subject_vec));
        let status_vec: ArrayRef = Arc::new(BooleanArray::from(status_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("task_name", task_names),
            ("subject_name", subject_names),
            ("status_value", status_vec),
        ])?;

        // Write to IPC file
        let table = ArrowTable::get_builder()
            .with_name("superstep_updates")
            .with_record_batches(vec![batch])?
            .build()?;
        table.to_ipc_file(file)
    }

    /// Read superstep updates to file
    pub fn read_superstep_updates(&mut self, file: &File) -> Result<()> {
        // Read in the IPC file
        let table = ArrowTableBuilder::new_from_ipc_file(file)?
            .with_name("superstep_updates")
            .build()?;

        // Extract out the data
        let task_vec = table.get_column_as_str_vec("task_name");
        let subject_vec = table.get_column_as_str_vec("subject_name");
        let status_vec = table
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("status_value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.superstep_updates = HashMap::<String, HashMap<String, bool>>::new();
        for iter in 0..task_vec.len() {
            let mut superstep_update = HashMap::<String, bool>::new();
            superstep_update.insert(
                subject_vec.get(iter).unwrap().to_string(),
                status_vec.get(iter).unwrap().to_owned(),
            );
            if self
                .superstep_updates
                .contains_key(task_vec.get(iter).unwrap().to_owned())
            {
                self.superstep_updates
                    .get_mut(task_vec.get(iter).unwrap().to_owned())
                    .unwrap()
                    .insert(
                        subject_vec.get(iter).unwrap().to_string(),
                        status_vec.get(iter).unwrap().to_owned(),
                    );
            } else {
                self.superstep_updates
                    .insert(task_vec.get(iter).unwrap().to_string(), superstep_update);
            }
        }
        Ok(())
    }

    /// Write the session stream state to disk
    pub fn write_state(&self, path: &str, tag: &str) -> Result<()> {
        // write the session context state
        match self.session_context.write_state(path, tag) {
            Ok(()) => (),
            Err(_e) => (),
        }

        // Prepare the file
        let pathname = format!(
            "{path}/{tag}-{}-superstep_updates",
            self.get_session_context().get_name()
        );
        let mut file = std::fs::File::create(pathname)?;

        // write the session context state
        match self.write_superstep_updates(&mut file) {
            Ok(()) => (),
            Err(e) => event!(Level::ERROR, "Error writing superstep updates: {e:?}"),
        }
        Ok(())
    }

    /// Read the session state from disk
    pub fn read_state(&mut self, path: &str, tag: &str) -> Result<()> {
        // write the session context state
        match self.session_context.read_state(path, tag) {
            Ok(()) => (),
            Err(_e) => (),
        }

        // Prepare the file
        let pathname = format!(
            "{path}/{tag}-{}-superstep_updates",
            self.get_session_context().get_name()
        );
        let file = std::fs::File::open(pathname)?;

        // write the session context state
        match self.read_superstep_updates(&file) {
            Ok(()) => (),
            Err(e) => event!(Level::ERROR, "Error reading superstep updates: {e:?}"),
        }
        Ok(())
    }
}

/// A single step of a [`SessionStream`]
pub struct SessionStreamStep {}

impl SessionStreamStep {
    /// Join the message streams using JointSet
    async fn join_message_streams(messages: OutgoingMessageMap) -> Result<IncomingMessageMap> {
        event!(Level::DEBUG, "Messages to join: {:?}.", &messages.keys());
        // Inspect each of the response futures
        let mut response_builder = HashMap::<String, ArrowIncomingMessageBuilder>::new();
        let mut join_set = JoinSet::new();
        messages.into_iter().for_each(|(resp_name, resp)| {
            // Copy over name, source, destination for later building of the complete response
            let message = ArrowIncomingMessageBuilder::new()
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
        let mut response_batches = HashMap::<String, ArrowIncomingMessage>::new();
        // Note that currently this doesn't identify the thread that panicked
        //
        // TODO: Replace with [join_next_with_id](https://docs.rs/tokio/latest/tokio/task/struct.JoinSet.html#method.join_next_with_id
        // once it is stable
        while let Some(response) = join_set.join_next().await {
            match response {
                Ok((resp_name, resp)) => {
                    // Complete the input message with the processed stream
                    let table = ArrowTableBuilder::new()
                        .with_name(resp_name.as_str())
                        .with_record_batches(resp?)?
                        .build()?;
                    let message = response_builder
                        .remove(resp_name.as_str())
                        .unwrap()
                        .with_message(table)
                        .build()?;
                    let message_map = message.to_map()?;
                    response_batches.extend(message_map);
                }
                Err(e) => {
                    if e.is_panic() {
                        std::panic::resume_unwind(e.into_panic());
                    } else {
                        unreachable!();
                    }
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
    /// Returns:
    ///
    /// `OutgoingMessageMap` streams if the the `Session` subsject was updated and None otherwise.
    #[instrument(skip(state, messages))]
    pub async fn run_superstep(
        state: Arc<RwLock<SessionStreamState>>,
        messages: IncomingMessageMap,
    ) -> Result<Option<IncomingMessageMap>> {
        // Update the state
        let update = state
            .try_write()
            .unwrap()
            .update_state_from_messages(messages);
        state.try_write().unwrap().extend_superstep_updates(update);

        // Iterate through each task and collect the resulting stream responses
        let mut session_streams = HashMap::<String, ArrowOutgoingMessage>::new();
        let mut response_streams = HashMap::<String, ArrowOutgoingMessage>::new();
        let tasks_update = state
            .try_read()
            .unwrap()
            .get_tasks_with_all_subjects_updated();
        for (task_name, task) in state.try_read().unwrap().session_context.get_tasks().iter() {
            // Continue to the next task all subscribed subjects are not updated
            if !tasks_update.contains(task_name) {
                continue;
            }
            event!(Level::DEBUG, "Superstep for task {}", &task_name);

            // Run the task and collect the stream responses
            let messages = task.get_subscriptions_from_state(
                state.try_read().unwrap().session_context.get_states(),
            );
            for (resp_name, resp) in task.run(messages)?.into_iter() {
                if task_name == state.try_read().unwrap().session_context.get_name() {
                    session_streams.insert(resp_name, resp);
                } else {
                    response_streams.insert(resp_name, resp);
                }
            }
        }

        // Break if there is nothing to update
        if session_streams.is_empty() && response_streams.is_empty() {
            let init = state
                .try_read()
                .unwrap()
                .get_session_context()
                .init_superstep_updates();
            state.try_write().unwrap().set_superstep_updates(init);
            return Ok(None);
        }

        // Remove the ran tasks from the update
        for task_name in tasks_update.iter() {
            state
                .try_write()
                .unwrap()
                .clear_subjects_from_task_for_superstep_updates(task_name.as_str());
        }

        // Join each of the response futures
        let response_batches = SessionStreamStep::join_message_streams(response_streams).await?;

        // Update the state
        let update = state
            .try_write()
            .unwrap()
            .update_state_from_messages(response_batches);
        state.try_write().unwrap().extend_superstep_updates(update);

        // Increment the step
        let iter = state.try_read().unwrap().get_iter() + 1;
        state.try_write().unwrap().set_iter(iter);

        // Return the session stream if any
        if session_streams.is_empty() {
            return Ok(Some(HashMap::<String, ArrowIncomingMessage>::new()));
        } else {
            // Join each of the session futures
            let session_batches = SessionStreamStep::join_message_streams(session_streams).await?;
            return Ok(Some(session_batches));
        }
    }
}

pub struct SessionStream {
    /// The state
    state: Arc<RwLock<SessionStreamState>>,
    /// The next result
    #[allow(clippy::type_complexity)]
    next: Option<Pin<Box<dyn Future<Output = Result<Option<IncomingMessageMap>>> + Send>>>,
}

impl SessionStream {
    pub fn new(input: IncomingMessageMap, state: Arc<RwLock<SessionStreamState>>) -> Self {
        #[allow(clippy::type_complexity)]
        let next: Option<
            Pin<Box<dyn Future<Output = Result<Option<IncomingMessageMap>>> + Send>>,
        > = Some(Box::pin(SessionStreamStep::run_superstep(
            Arc::clone(&state),
            input,
        )));
        Self { state, next }
    }
}

impl Stream for SessionStream {
    type Item = Result<IncomingMessageMap>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Get the current iter
        let mut iter = self.state.try_read().unwrap().get_iter();
        let max_iter = self
            .state
            .try_read()
            .unwrap()
            .get_session_context()
            .get_max_iter();
        while iter < max_iter {
            // Poll the next item
            let res = if let Some(fut) = self.next.as_mut() {
                match ready!(fut.poll_unpin(cx)) {
                    Ok(Some(res)) => res,
                    Ok(None) => return Poll::Ready(None),
                    _ => HashMap::<String, ArrowIncomingMessage>::new(),
                }
            } else {
                return Poll::Ready(None);
            };

            // Prepare the next itme
            self.next = Some(Box::pin(SessionStreamStep::run_superstep(
                Arc::clone(&self.state),
                HashMap::<String, ArrowIncomingMessage>::new(),
            )));
            iter = self.state.try_read().unwrap().get_iter();

            // Return the poll
            if res.is_empty() {
                // Skip empty results
                continue;
            } else {
                return Poll::Ready(Some(Ok(res)));
            }
        }
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            1,
            Some(
                self.state
                    .try_read()
                    .unwrap()
                    .get_session_context()
                    .get_max_iter(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::HashSet;
    use crate::table::arrow_table::test_table::make_test_table_schema;
    use crate::{
        session::session_context_builder::test_session_context_builder::{
            make_test_session_context_parallel_task, make_test_session_context_parallel_task_empty,
            make_test_session_context_sequential_task,
        },
        table::arrow_table_publish::ArrowTablePublish,
        task::arrow_task::test_task::make_test_input_message,
    };
    #[cfg(not(target_family = "wasm"))]
    use tempfile::{tempdir, tempfile};

    #[test]
    fn test_session_update_state() -> Result<()> {
        // Case 1: no state update
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &ArrowTablePublish::None,
        )?;
        let session_stream_step = SessionStreamState::new(session_context);
        let updates = session_stream_step.update_state_from_messages(input);

        // check the response
        assert!(updates.is_empty());
        assert_eq!(
            session_stream_step
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
            session_stream_step
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
            session_stream_step
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
            session_stream_step
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

        // Case 2: update state
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &ArrowTablePublish::Extend {
                table_name: "state_1".to_string(),
            },
        )?;
        let updates = session_stream_step.update_state_from_messages(input);

        // check the response
        assert_eq!(updates.len(), 1);
        assert_eq!(updates.get("state_1").unwrap().len(), 1);
        assert_eq!(
            updates.get("state_1").unwrap().first().unwrap(),
            "session_1"
        );
        assert_eq!(
            session_stream_step
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
            session_stream_step
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
            session_stream_step
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
            session_stream_step
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

        Ok(())
    }

    #[test]
    fn test_session_update_state_from_json_str() -> Result<()> {
        // make the session state
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;
        let session_stream_state = SessionStreamState::new(session_context);

        // Case 1: valid json str
        let json_str = r#"[{"a": "a", "b": 0, "c": 0}]"#;
        let schema = session_stream_state
            .get_session_context()
            .get_states()
            .get("config_1")
            .unwrap()
            .try_read()
            .unwrap()
            .get_schema();
        let updates = session_stream_state.update_state_from_json_str(
            &schema,
            json_str,
            &ArrowTablePublish::Extend {
                table_name: "config_1".to_string(),
            },
        )?;

        // check the response
        assert_eq!(updates.len(), 1);
        assert_eq!(updates.get("config_1").unwrap().len(), 1);
        assert_eq!(
            updates.get("config_1").unwrap().first().unwrap(),
            "session_1"
        );
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .count_rows(),
            2
        ); // Originally 1
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            1
        ); // Unchanged

        // Case 2: invalid json str
        let json_str = r#"{"a": "a", "b": 0, "c": 1}"#;
        let updates = session_stream_state.update_state_from_json_str(
            &schema,
            json_str,
            &ArrowTablePublish::Extend {
                table_name: "config_1".to_string(),
            },
        );
        assert!(updates.is_err());

        Ok(())
    }

    #[test]
    fn test_session_update_state_from_csv_str() -> Result<()> {
        // make the session state
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;
        let session_stream_state = SessionStreamState::new(session_context);

        // Case 1: valid json str
        let csv_str = r#"
            a,b,c
            a,0,0
            a,0,0"#;
        let csv_str = csv_str.trim();
        let schema = session_stream_state
            .get_session_context()
            .get_states()
            .get("config_1")
            .unwrap()
            .try_read()
            .unwrap()
            .get_schema();
        let updates = session_stream_state.update_state_from_csv_str(
            &schema,
            csv_str,
            &ArrowTablePublish::Extend {
                table_name: "config_1".to_string(),
            },
            b',',
            true,
            2,
        )?;

        // check the response
        assert_eq!(updates.len(), 1);
        assert_eq!(updates.get("config_1").unwrap().len(), 1);
        assert_eq!(
            updates.get("config_1").unwrap().first().unwrap(),
            "session_1"
        );
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .count_rows(),
            3
        ); // Originally 1
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            2
        ); // Unchanged

        // Case 2: valid csv string more headers than in the schema
        let csv_str = r#"
            a,b,c,d
            a,0,0,a
            a,0,0,a"#;
        let updates = session_stream_state.update_state_from_csv_str(
            &schema,
            csv_str,
            &ArrowTablePublish::Extend {
                table_name: "config_1".to_string(),
            },
            b',',
            true,
            4,
        )?;

        // check the response
        assert_eq!(updates.len(), 1);
        assert_eq!(updates.get("config_1").unwrap().len(), 1);
        assert_eq!(
            updates.get("config_1").unwrap().first().unwrap(),
            "session_1"
        );
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .count_rows(),
            3
        ); // Originally 1
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            2
        ); // Unchanged

        // Case 3: valid csv string but mismatched schema
        let csv_str = r#"
            a,b,
            a,0,
            a,0,"#;
        let updates = session_stream_state.update_state_from_csv_str(
            &schema,
            csv_str,
            &ArrowTablePublish::Extend {
                table_name: "config_1".to_string(),
            },
            b',',
            true,
            4,
        )?;

        // check the response
        assert_eq!(updates.len(), 1);
        assert_eq!(updates.get("config_1").unwrap().len(), 1);
        assert_eq!(
            updates.get("config_1").unwrap().first().unwrap(),
            "session_1"
        );
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .count_rows(),
            3
        ); // Originally 1
        assert_eq!(
            session_stream_state
                .get_session_context()
                .get_states()
                .get("config_1")
                .unwrap()
                .try_read()
                .unwrap()
                .get_record_batches()
                .last()
                .unwrap()
                .num_rows(),
            2
        ); // Unchanged

        Ok(())
    }

    #[test]
    fn test_session_get_table_name_by_schema() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;

        // table should be found
        let schema = make_test_table_schema(8)?;
        let name = session_context.get_table_name_by_schema(&schema).unwrap();
        assert_eq!(name, "state_1");

        // table should not be found
        let schema = make_test_table_schema(2)?;
        let name = session_context.get_table_name_by_schema(&schema);
        assert!(name.is_none());
        Ok(())
    }

    #[test]
    fn test_session_get_tasks_info_as_table() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;
        let info = session_context.get_tasks_info_as_table("table")?;
        assert_eq!(info.get_name(), "table");
        assert_eq!(
            info.get_column_as_str_vec("task_names"),
            [
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "task_1",
                "task_1",
                "task_1",
                "task_2",
                "task_2",
                "task_2",
                "task_3",
                "task_3",
                "task_3"
            ]
        );
        assert_eq!(
            info.get_column_as_str_vec("processor_names"),
            [
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "session_1",
                "processor_1",
                "processor_1",
                "processor_1",
                "processor_2",
                "processor_2",
                "processor_2",
                "processor_3",
                "processor_3",
                "processor_3"
            ]
        );
        assert_eq!(
            info.get_column_as_str_vec("subject_names"),
            [
                "state_1", "state_2", "state_3", "state_1", "state_2", "state_3", "state_1",
                "config_1", "state_1", "state_2", "config_2", "state_2", "state_3", "config_3",
                "state_3"
            ]
        );
        assert_eq!(
            info.get_column_as_str_vec("pub_or_sub"),
            [
                "-", "-", "-", "+", "+", "+", "-", "-", "+", "-", "-", "+", "-", "-", "+"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_session_get_subjects_info_as_table() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;
        let info = session_context.get_subjects_info_as_table("table")?;
        assert_eq!(info.get_name(), "table");
        assert_eq!(
            info.get_column_as_str_vec("subject_names"),
            [
                "config_1", "config_1", "config_1", "config_2", "config_2", "config_2", "config_3",
                "config_3", "config_3", "state_1", "state_1", "state_1", "state_1", "state_1",
                "state_1", "state_2", "state_2", "state_2", "state_2", "state_2", "state_2",
                "state_3", "state_3", "state_3", "state_3", "state_3", "state_3"
            ]
        );
        assert_eq!(
            info.get_column_as_str_vec("column_names"),
            [
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "ids",
                "collection",
                "title",
                "text",
                "metadata",
                "embedding",
                "ids",
                "collection",
                "title",
                "text",
                "metadata",
                "embedding",
                "ids",
                "collection",
                "title",
                "text",
                "metadata",
                "embedding"
            ]
        );
        assert_eq!(
            info.get_column_as_str_vec("type_names"),
            [
                "Utf8",
                "UInt32",
                "UInt16",
                "Utf8",
                "UInt32",
                "UInt16",
                "Utf8",
                "UInt32",
                "UInt16",
                "UInt32",
                "Utf8",
                "Utf8",
                "Utf8",
                "Utf8",
                "FixedSizeList(Field { name: \"item\", data_type: UInt32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, 8)",
                "UInt32",
                "Utf8",
                "Utf8",
                "Utf8",
                "Utf8",
                "FixedSizeList(Field { name: \"item\", data_type: UInt32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, 8)",
                "UInt32",
                "Utf8",
                "Utf8",
                "Utf8",
                "Utf8",
                "FixedSizeList(Field { name: \"item\", data_type: UInt32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, 8)"
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
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            num_rows,
            [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                12, 12, 12, 12
            ]
        );

        Ok(())
    }

    #[test]
    fn test_session_init_superstep_updates() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;
        let init = session_context.init_superstep_updates();
        assert_eq!(init.len(), 4);
        assert_eq!(
            init.keys().map(|k| k.as_str()).collect::<HashSet<_>>(),
            ["task_1", "task_2", "task_3", "session_1"]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        assert_eq!(
            init.get("task_1")
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect::<Vec<_>>(),
            &["state_1"]
        );
        for (_k, v) in init.get("task_1").unwrap() {
            assert!(!v);
        }
        assert_eq!(
            init.get("task_2")
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect::<Vec<_>>(),
            &["state_2"]
        );
        for (_k, v) in init.get("task_2").unwrap() {
            assert!(!v);
        }
        assert_eq!(
            init.get("task_3")
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect::<Vec<_>>(),
            &["state_3"]
        );
        for (_k, v) in init.get("task_3").unwrap() {
            assert!(!v);
        }
        assert_eq!(
            init.get("session_1")
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect::<HashSet<_>>(),
            ["state_1", "state_2", "state_3"]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        for (_k, v) in init.get("session_1").unwrap() {
            assert!(!v);
        }

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_session_read_write_state() -> Result<()> {
        // Create the session

        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 25)?;

        // Write the session to disk
        let tmp_dir = tempdir()?;
        session_context.write_state(tmp_dir.path().to_str().unwrap(), "tag")?;

        // Read the state
        let mut session_context_empty =
            make_test_session_context_parallel_task_empty("session_1", metrics.clone(), 25)?;
        session_context_empty.read_state(tmp_dir.path().to_str().unwrap(), "tag")?;

        for subject in session_context.get_states().keys() {
            assert_eq!(
                session_context
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_record_batches(),
                session_context_empty
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_record_batches()
            );
            assert_eq!(
                session_context
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_schema(),
                session_context_empty
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_schema()
            );
            assert_eq!(
                session_context
                    .get_states()
                    .get(subject)
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_name(),
                session_context_empty
                    .get_states()
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

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_session_read_write_superstep_update() -> Result<()> {
        // initialize the session stream state
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_parallel_task("session_1", metrics.clone(), 4)?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));

        // write the session stream state to file
        let mut file = tempfile()?;
        session_stream_state
            .try_read()
            .unwrap()
            .write_superstep_updates(&mut file)?;

        // read the session stream state back to file
        let session_context =
            make_test_session_context_sequential_task("session_1", metrics.clone(), 4)?;
        let session_stream_state_test =
            Arc::new(RwLock::new(SessionStreamState::new(session_context)));

        assert_ne!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates(),
            session_stream_state_test
                .try_read()
                .unwrap()
                .get_superstep_updates()
        );
        println!("session_stream_state_test.try_write()");
        session_stream_state_test
            .try_write()
            .unwrap()
            .read_superstep_updates(&file)?;
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates(),
            session_stream_state_test
                .try_read()
                .unwrap()
                .get_superstep_updates()
        );

        Ok(())
    }

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
                &ArrowTablePublish::None,
            )?,
        )
        .await?;
        assert!(response.is_none());

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 0);
        assert!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_tasks_with_all_subjects_updated()
                .is_empty()
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
                &ArrowTablePublish::Extend {
                    table_name: "state_1".to_string(),
                },
            )?,
        )
        .await?
        .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 1);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_tasks_with_all_subjects_updated()
                .is_empty()
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
                &ArrowTablePublish::Replace {
                    table_name: "state_1".to_string(),
                },
            )?,
        )
        .await?
        .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 1);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_tasks_with_all_subjects_updated()
                .is_empty()
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
            &ArrowTablePublish::Replace {
                table_name: "state_1".to_string(),
            },
        )?;
        input.extend(make_test_input_message(
            "task_2",
            "session_1",
            "state_2",
            "state_2",
            &ArrowTablePublish::Replace {
                table_name: "state_2".to_string(),
            },
        )?);
        input.extend(make_test_input_message(
            "task_3",
            "session_1",
            "state_3",
            "state_3",
            &ArrowTablePublish::Replace {
                table_name: "state_3".to_string(),
            },
        )?);
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await?
            .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 1);
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
                .get_tasks_with_all_subjects_updated(),
            ["session_1"]
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
            HashMap::<String, ArrowIncomingMessage>::new(),
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
            ArrowTablePublish::Extend {
                table_name: "state_1".to_string()
            }
        );

        let partitions = response
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
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
            ArrowTablePublish::Extend {
                table_name: "state_2".to_string()
            }
        );

        let partitions = response
            .remove("from_session_1_on_state_2")
            .unwrap()
            .get_message_own();
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
            ArrowTablePublish::Extend {
                table_name: "state_3".to_string()
            }
        );

        let partitions = response
            .remove("from_session_1_on_state_3")
            .unwrap()
            .get_message_own();
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 6);

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 2);
        assert_eq!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_superstep_updates()
                .len(),
            4
        );
        assert!(
            session_stream_state
                .try_read()
                .unwrap()
                .get_tasks_with_all_subjects_updated()
                .is_empty()
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
            &ArrowTablePublish::Replace {
                table_name: "state_1".to_string(),
            },
        )?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_stream_state), input)
            .await?
            .unwrap();
        assert!(response.is_empty());

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 1);
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
                .get_tasks_with_all_subjects_updated(),
            ["session_1", "task_1", "task_2", "task_3"]
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
            HashMap::<String, ArrowIncomingMessage>::new(),
        )
        .await?;
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 2);
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
                .get_tasks_with_all_subjects_updated(),
            ["session_1", "task_1", "task_2", "task_3"]
        );
        let _ = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            HashMap::<String, ArrowIncomingMessage>::new(),
        )
        .await?;
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 3);
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
                .get_tasks_with_all_subjects_updated(),
            ["session_1", "task_1", "task_2", "task_3"]
        );
        let mut response = SessionStreamStep::run_superstep(
            Arc::clone(&session_stream_state),
            HashMap::<String, ArrowIncomingMessage>::new(),
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
            ArrowTablePublish::Extend {
                table_name: "state_1".to_string()
            }
        );

        let partitions = response
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 8);

        // check the session and state
        assert_eq!(session_stream_state.try_read().unwrap().get_iter(), 4);
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
                .get_tasks_with_all_subjects_updated(),
            ["session_1", "task_1", "task_2", "task_3"]
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
    async fn test_session_stream_replace_state_update_sequential_tasks() -> Result<()> {
        // session -> task_1: add a row
        //         -> task_2: add a row
        //         -> task_3: add a row
        //         -> session
        let metrics = ArrowTaskMetricsSet::new();
        let session_context =
            make_test_session_context_sequential_task("session_1", metrics.clone(), 4)?;
        let input = make_test_input_message(
            "task_1",
            "session_1",
            "state_1",
            "state_1",
            &ArrowTablePublish::Replace {
                table_name: "state_1".to_string(),
            },
        )?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
        let session_stream = SessionStream::new(input, session_stream_state.clone());
        let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
            session_stream.try_collect().await?;

        // check the response
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
            ArrowTablePublish::Extend {
                table_name: "state_1".to_string()
            }
        );

        // Check the metrics
        let partitions = response
            .get_mut(0)
            .unwrap()
            .remove("from_session_1_on_state_1")
            .unwrap()
            .get_message_own();
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 6);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 5385);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);
        let _info = session_stream_state
            .try_read()
            .unwrap()
            .get_session_context()
            .get_metrics_info_as_table("")?;

        Ok(())
    }
}
