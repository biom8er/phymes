use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Field, Schema},
};
use clap::ValueEnum;
use phymes_diagnostics::{HashSet, create_timestamp_micros};
use phymes_event::{AvailableSubscribeEvents, AvailableUpdateEvents, Publication, Subscription};
use phymes_processor::{AvailableProcessors, ProcessorPlanBuilder};
use phymes_schemas::{
    AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, create_network_mermaid_batch, create_network_processors_batch, create_network_runtime_envs_batch, create_network_subject_schemas_batch, create_network_tasks_batch, create_network_tasks_run_log_batch, create_subjects_change_log_batch, create_subjects_num_rows_batch, create_subjects_object_store_meta_batch, from_data_type_to_str, from_str_to_data_type,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, ObjectStorageBackend, RuntimeEnv,
    RuntimeEnvBuilderTrait, Subject, SubjectBuilderTrait, SubjectFilePartition,
    SubjectFolderPartition, SubjectPlan, SubjectPlanBuilderTrait, SubjectPlanTrait, SubjectTrait,
    make_store,
};
use phymes_task::TaskPlanBuilder;
use serde_json::{Map, Value};

use crate::{
    InvokeTaskNetworkBuilder, NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait, TaskResponseNetworkBuilder, core::{CountSubjectRowsNetwork, NextSuperstepNetwork, NextTaskNetwork},
};

/// Trait extension for [NetworkBuilderTrait] to enable exporting to and importing from tabular format
pub trait NetworkBuilderTabularTrait {
    /// Convert the network into tables
    ///
    /// # Notes
    ///
    /// * All subjects that are a part of the subjects are included
    /// * Additional meta tables describing the Network schema are included
    /// * Mermaid_js scripts are also included
    ///
    /// # Arguments
    ///
    /// * `include_mermaid` - whether to include the mermaid flowchart and erDiagrams or not
    /// * `include_errors` - whether to include the errors table or not
    /// * `include_diagnostics` - whether to include the diagnostics tables or not
    /// * `include_tasks_run_log` - whether to include the initialized `TasksRunLog`
    /// * `include_subjects_change_log` - whether to include the initialized `SubjectsNumRows` and `SubjectsChangeLog`
    ///
    /// # Returns
    ///
    /// * `Vec<SubjectPlan>` with the Network in tabular format and Optional with the subject data
    fn to_subject_plans(
        &self,
        include_mermaid: bool,
        include_errors: bool,
        include_diagnostics: bool,
        include_tasks_run_log: bool,
        include_subjects_change_log: bool,
    ) -> Result<Vec<SubjectPlan>>;

    /// Get the subjects in tabular form
    ///
    /// # Arguments
    /// * `additional_tables` - Additional tables to include in addition to what is in the subjects
    fn get_subjects_as_subject_plan(
        &self,
        additional_subjects: &[SubjectPlan],
    ) -> Result<SubjectPlan>;

    /// Get the tasks in tabular form
    fn get_tasks_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Get the processors in tabular form
    ///
    /// # Note
    ///
    /// * No sorting is performed when generating the table
    ///   so that order of processors is maintained
    fn get_processors_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Get the runtime environments in tabular form
    fn get_runtime_envs_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Get mermaid js chart strings
    fn get_mermaid_js_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Create the initial `TasksRunLog` table
    fn get_tasks_run_log_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Create the initial `SubjectsNumRows` table
    fn get_subjects_num_rows_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Create the `SubjectsNumRows` and `SubjectsChangeLog` tables
    fn get_subjects_change_log_as_subject_plan(&self) -> Result<SubjectPlan>;

    fn get_subjects_object_store_meta_as_subject_plan(&self) -> Result<SubjectPlan>;

    /// Create the network from subjects
    ///
    /// # Notes
    ///
    /// * Minimally, the meta subject describing the [Network] schema must be included
    /// * Optionally, the subject will be populated with data if the subjects data are included
    /// * Mermaid_js scripts are ignored
    ///
    /// # Arguments
    ///
    /// * `subject_plans` - List of [SubjectPlan]s describing the [Network] schema with
    ///   optional subject tables with the actual data
    /// * `subjects` - Optionally the subject data. If none the subject tables will be initialized.
    ///
    /// [Network]: crate::Network
    fn from_subject_plans(
        subject_plans: &[&SubjectPlan],
        subjects: Option<Vec<SubjectPlan>>,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Create empty subjects from `NetworkSubjectSchemas`
    fn with_subjects_as_subject_plans(self, subjects: &SubjectPlan) -> Result<Self>
    where
        Self: Sized;
    fn with_tasks_as_subject_plans(self, tasks: &SubjectPlan) -> Result<Self>
    where
        Self: Sized;
    fn with_processors_as_subject_plans(self, processors: &SubjectPlan) -> Result<Self>
    where
        Self: Sized;
    fn with_runtime_envs_as_subject_plans(self, runtime_envs: &SubjectPlan) -> Result<Self>
    where
        Self: Sized;

    /// Tasks to exclude from the tables
    fn tasks_to_exclude(&self) -> Result<HashSet<String>>;

    /// Processors to exclude from the tables
    fn processors_to_exclude(&self) -> Result<HashSet<String>>;

    /// Subjects to exclude from the tables
    fn subjects_to_exclude(&self) -> Result<HashSet<String>>;
}

impl NetworkBuilderTabularTrait for NetworkBuilder {
    fn to_subject_plans(
        &self,
        include_mermaid: bool,
        include_errors: bool,
        include_diagnostics: bool,
        include_tasks_run_log: bool,
        include_subjects_change_log: bool,
    ) -> Result<Vec<SubjectPlan>> {
        let mut tables = Vec::new();
        if include_mermaid {
            tables.push(self.get_mermaid_js_as_subject_plan()?);
        }
        if include_errors {
            tables.push(AvailableSubjects::NetworkErrors.to_subject_plan(None, None)?);
        }
        if include_diagnostics {
            tables.push(AvailableSubjects::NetworkMetrics.to_subject_plan(None, None)?);
            tables.push(AvailableSubjects::NetworkTraces.to_subject_plan(None, None)?);
            tables.push(AvailableSubjects::NetworkEvents.to_subject_plan(None, None)?);
        }
        // include_tasks_run_log is after include_subjects_change_log
        // so that the timestamp of all tasks is greater than the timestamp of all subjects
        if include_subjects_change_log {
            tables.push(self.get_subjects_num_rows_as_subject_plan()?);
            tables.push(self.get_subjects_change_log_as_subject_plan()?);
            tables.push(self.get_subjects_object_store_meta_as_subject_plan()?);
        }
        if include_tasks_run_log {
            tables.push(self.get_tasks_run_log_as_subject_plan()?);
        }
        tables.extend([
            self.get_subjects_as_subject_plan(&tables)?,
            self.get_tasks_as_subject_plan()?,
            self.get_processors_as_subject_plan()?,
            self.get_runtime_envs_as_subject_plan()?,
        ]);
        Ok(tables)
    }

    fn from_subject_plans(
        subject_plans: &[&SubjectPlan],
        mut subjects: Option<Vec<SubjectPlan>>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        // initialize the builder
        let mut builder = Self::new();

        // extract the schema
        for subject_plan in subject_plans {
            if subject_plan.get_name()
                == AvailableSubjects::NetworkSubjectSchemas
                    .to_string()
                    .as_str()
                && subjects.is_none()
            {
                builder = builder.with_subjects_as_subject_plans(subject_plan)?;
            } else if subject_plan.get_name()
                == AvailableSubjects::NetworkTasks.to_string().as_str()
            {
                builder = builder.with_tasks_as_subject_plans(subject_plan)?;
            } else if subject_plan.get_name()
                == AvailableSubjects::NetworkProcessors.to_string().as_str()
            {
                builder = builder.with_processors_as_subject_plans(subject_plan)?;
            } else if subject_plan.get_name()
                == AvailableSubjects::NetworkRuntimeEnvs.to_string().as_str()
            {
                builder = builder.with_runtime_envs_as_subject_plans(subject_plan)?;
            } else if subject_plan.get_name() == AvailableSubjects::NetworkMermaid.to_string().as_str()
                // Diagnostic tables
                || subject_plan.get_name() == AvailableSubjects::NetworkErrors.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTraces.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkMetrics.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkEvents.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::MetricPivot.to_string().as_str()
                // Subjects change log tables
                || subject_plan.get_name() == AvailableSubjects::SubjectsNumRows.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::SubjectsChangeLog.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::SubjectsObjectStoreMeta.to_string().as_str()
                // Tasks run log tables
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksCheck.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksPublish.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksPublishAggregate.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksRunLog.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksSubscribe.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksSubscribeAggregate.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkTasksSubscribePublish.to_string().as_str()
                // Network superstep tables
                || subject_plan.get_name() == AvailableSubjects::NetworkSupersteps.to_string().as_str()
                || subject_plan.get_name() == AvailableSubjects::NetworkSuperstepMax.to_string().as_str()
            {
                // These tables are created on the fly so we do not want to duplicate them.
                // If the user wishes to continue already generated tables they can do so
                // by passing them as optional subjects.
                continue;
            } else {
                return Err(anyhow!(
                    "Unrecognized subject {} found when creating NetworkBuilder",
                    subject_plan.get_name()
                ));
            }
        }

        // Add the optional subjects
        if subjects.is_some() {
            Ok(builder.with_subjects(subjects.take().unwrap()))
        } else {
            Ok(builder)
        }
    }

    fn get_subjects_as_subject_plan(
        &self,
        additional_subjects: &[SubjectPlan],
    ) -> Result<SubjectPlan> {
        // Check that the subjects exists
        if self.subjects.is_none() {
            return Err(anyhow!("Add subjects before making the subject tables."));
        }
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the subject tables."
            ));
        };

        // Tables to exclude
        let exclusion_set = self.subjects_to_exclude()?;

        // initialize the table columns
        let mut network_names = Vec::<String>::new();
        let mut subject_names = Vec::<String>::new();
        let mut cols_names = Vec::<String>::new();
        let mut type_names = Vec::<String>::new();

        // Sort the hashmap
        let mut sorted_subjects = self
            .subjects
            .as_ref()
            .unwrap()
            .iter()
            .chain(additional_subjects)
            .filter(|t| !exclusion_set.contains(t.get_name()))
            .collect::<Vec<_>>();
        sorted_subjects.sort_by(|a, b| a.get_name().cmp(b.get_name()));
        for subject in sorted_subjects.iter() {
            if !subject_names.contains(&subject.get_name().to_string()) {
                let fields = subject.subject().get_schema().fields().clone();
                for field in fields.iter() {
                    let type_name = from_data_type_to_str(field.data_type());
                    network_names.push(network_name.to_string());
                    subject_names.push(subject.get_name().to_string());
                    cols_names.push(field.name().to_string());
                    type_names.push(type_name);
                }
            }
        }

        // create the record batch
        let batch = create_network_subject_schemas_batch(
            network_names,
            subject_names,
            cols_names,
            type_names,
        )?;

        // create the subject
        let subject = Subject::get_builder()
            .with_name(
                AvailableSubjects::NetworkSubjectSchemas
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_tasks_as_subject_plan(&self) -> Result<SubjectPlan> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the tasks table."));
        }
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the subject tables."
            ));
        };

        let exclusion_set = self.tasks_to_exclude()?;

        // extract the tasks in order
        #[allow(clippy::type_complexity)]
        let ((network_names, task_names), processor_names): (
            (Vec<String>, Vec<String>),
            Vec<String>,
        ) = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|t| !exclusion_set.contains(&t.task_name))
            .flat_map(|t| {
                t.processor_names
                    .iter()
                    .map(|p| {
                        (
                            (network_name.to_string(), t.task_name.to_string()),
                            p.to_string(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unzip();

        // create the record batch
        let batch = create_network_tasks_batch(network_names, task_names, processor_names)?;

        // create the table
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkTasks.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_processors_as_subject_plan(&self) -> Result<SubjectPlan> {
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processors before making the Mermaid Flowchart."
            ));
        }
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the subject tables."
            ));
        };

        // extract the processors in order
        let exclusion_set = self.processors_to_exclude()?;
        let (
            (
                (
                    (
                        (((network_names, processor_names), processor_types), pub_sub_name),
                        pub_sub_table_names,
                    ),
                    is_sub,
                ),
                subscribe_types,
            ),
            update_types,
        ) = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter(|p| !exclusion_set.contains(p.get_name()))
            .flat_map(|p| {
                let subs = p
                    .get_subscriptions()
                    .iter()
                    .map(|s| {
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    network_name.to_string(),
                                                    p.get_name().to_string(),
                                                ),
                                                p.get_type().to_string(),
                                            ),
                                            s.get_name().to_string(),
                                        ),
                                        s.subject_name().to_string(),
                                    ),
                                    1,
                                ),
                                p.get_subscribe_policy().get_name().to_string(),
                            ),
                            p.get_update_policy().get_name().to_string(),
                        )
                    })
                    .collect::<Vec<_>>();
                let pubs = p
                    .get_publications()
                    .iter()
                    .map(|s| {
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    network_name.to_string(),
                                                    p.get_name().to_string(),
                                                ),
                                                p.get_type().to_string(),
                                            ),
                                            s.get_name().to_string(),
                                        ),
                                        s.subject_name().to_string(),
                                    ),
                                    0,
                                ),
                                p.get_subscribe_policy().get_name().to_string(),
                            ),
                            p.get_update_policy().get_name().to_string(),
                        )
                    })
                    .collect::<Vec<_>>();
                subs.into_iter().chain(pubs).collect::<Vec<_>>()
            })
            .unzip();

        // create the record batch
        let batch = create_network_processors_batch(
            network_names,
            processor_names,
            processor_types,
            pub_sub_name,
            pub_sub_table_names,
            subscribe_types,
            update_types,
            is_sub,
        )?;

        // create the table
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkProcessors.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_runtime_envs_as_subject_plan(&self) -> Result<SubjectPlan> {
        if self.runtime_env.is_none() {
            return Err(anyhow!(
                "Add runtime environments before making the subject tables."
            ));
        }
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the subject tables."
            ));
        };

        // create the record batch
        let network_names = vec![network_name.to_string()];
        let runtime_env_names = vec![self.runtime_env.as_ref().unwrap().get_name().to_string()];
        let object_store_backend = vec![ObjectStorageBackend::InMemory.to_string()]; // DM: need to find a way to get the backend from the store...
        let object_store_bucket = vec![String::new()]; // DM: need to find a way to get the backend from the store...
        let object_store_config = vec![
            serde_json::to_string(&self.runtime_env.as_ref().unwrap().object_store_config).unwrap(),
        ];
        let subject_folder_partitioning = vec![
            self.runtime_env
                .as_ref()
                .unwrap()
                .subject_folder_partitioning
                .to_string(),
        ];
        let subject_file_partitioning = vec![
            self.runtime_env
                .as_ref()
                .unwrap()
                .subject_file_partitioning
                .to_string(),
        ];
        let max_memory = vec![self.runtime_env.as_ref().unwrap().max_memory as u32];
        let max_time = vec![self.runtime_env.as_ref().unwrap().max_time as u32];
        let max_steps = vec![self.runtime_env.as_ref().unwrap().max_steps as u32];
        let max_tasks = vec![self.runtime_env.as_ref().unwrap().max_tasks as u32];
        let batch = create_network_runtime_envs_batch(
            network_names,
            runtime_env_names,
            object_store_backend,
            object_store_bucket,
            object_store_config,
            subject_folder_partitioning,
            subject_file_partitioning,
            max_memory,
            max_time,
            max_steps,
            max_tasks,
        )?;

        // create the table
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkRuntimeEnvs.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_mermaid_js_as_subject_plan(&self) -> Result<SubjectPlan> {
        let flowchart_diagram = self.to_mermaid_flowchart(false, false)?;
        let er_diagram = self.to_mermaid_erdiagram(false, true)?;
        let network_name = self.name.as_ref().unwrap().to_string();
        let timestamp = create_timestamp_micros();

        // create the record batch
        let batch = create_network_mermaid_batch(
            vec![network_name],
            vec![flowchart_diagram],
            vec![er_diagram],
            vec![timestamp],
        )?;

        // create the table
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkMermaid.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_tasks_run_log_as_subject_plan(&self) -> Result<SubjectPlan> {
        if self.tasks.is_none() {
            return Err(anyhow!(
                "Add task plans before making the tasks run log table."
            ));
        }
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the tasks run log table."
            ));
        };

        // Tasks to exclude
        let exclusion_set = self.tasks_to_exclude()?;

        // Create the table
        let (((network_names, task_names), supersteps), timestamps) = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|task| {
                if exclusion_set.contains(&task.task_name) {
                    None
                } else {
                    Some((
                        (
                            (network_name.to_string(), task.task_name.to_string()),
                            0_i64,
                        ),
                        create_timestamp_micros(),
                    ))
                }
            })
            .unzip();
        let batch =
            create_network_tasks_run_log_batch(network_names, task_names, supersteps, timestamps)?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::NetworkTasksRunLog.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_subjects_num_rows_as_subject_plan(&self) -> Result<SubjectPlan> {
        if self.subjects.is_none() {
            return Err(anyhow!(
                "Add subjects before making the subjects num rows table."
            ));
        }

        // Create the table
        let exclusion_set = self.subjects_to_exclude()?;
        let (subject_names, num_rows): (Vec<String>, Vec<i64>) = self
            .subjects
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|t| {
                if exclusion_set.contains(t.get_name()) {
                    None
                } else {
                    Some((t.get_name().to_string(), t.subject().count_rows() as i64))
                }
            })
            .unzip();
        let batch = create_subjects_num_rows_batch(subject_names, num_rows)?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_subjects_change_log_as_subject_plan(&self) -> Result<SubjectPlan> {
        if self.subjects.is_none() {
            return Err(anyhow!(
                "Add subjects before making the subjects change log table."
            ));
        }
        if self.tasks.is_none() {
            return Err(anyhow!(
                "Add task plans before making subjects change log table."
            ));
        }
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processor plans before making subjects change log table."
            ));
        }
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the subjects change log table."
            ));
        };

        // Create the table
        let exclusion_set = self.tasks_to_exclude()?;
        let ((((subject_names, task_names), network_names), num_rows), supersteps) = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|task| {
                if exclusion_set.contains(&task.task_name) {
                    None
                } else {
                    let subjects = self
                        .subjects
                        .as_ref()
                        .unwrap()
                        .iter()
                        .filter_map(|table| {
                            let (subscriptions, publications) =
                                self.get_sub_pub_for_task(&task.task_name);
                            let publication_names = publications
                                .into_iter()
                                .map(|p| p.subject_name())
                                .collect::<Vec<_>>();
                            let subscription_names = subscriptions
                                .into_iter()
                                .map(|p| p.subject_name())
                                .collect::<Vec<_>>();
                            let table_names = publication_names
                                .into_iter()
                                .chain(subscription_names)
                                .collect::<HashSet<_>>();
                            if table_names.contains(&table.get_name()) {
                                Some((
                                    (
                                        (
                                            (
                                                table.get_name().to_string(),
                                                task.task_name.to_string(),
                                            ),
                                            network_name.to_string(),
                                        ),
                                        0_i64,
                                    ),
                                    0_i64,
                                ))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    Some(subjects)
                }
            })
            .flatten()
            .unzip();
        let batch = create_subjects_change_log_batch(
            subject_names,
            task_names,
            network_names,
            num_rows,
            supersteps,
        )?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn get_subjects_object_store_meta_as_subject_plan(&self) -> Result<SubjectPlan> {
        if self.subjects.is_none() {
            return Err(anyhow!(
                "Add subjects before making the subjects object store meta table."
            ));
        }
        if self.tasks.is_none() {
            return Err(anyhow!(
                "Add task plans before making subjects object store meta table."
            ));
        }
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processor plans before making subjects object store meta table."
            ));
        }
        let bucket = if let Some(rt) = self.runtime_env.as_ref() {
            rt.object_store_bucket.as_str()
        } else {
            return Err(anyhow!(
                "Add runtime env before making subjects object store meta table."
            ));
        };
        let network_name = if let Some(network_name) = self.name.as_ref() {
            network_name
        } else {
            return Err(anyhow!(
                "Add network name before making the subjects object store meta table."
            ));
        };

        // Create the table
        let exclusion_set = self.tasks_to_exclude()?;
        let (
            (
                (
                    (
                        (
                            (
                                (
                                    (((subject_names, task_names), network_names), num_rows),
                                    supersteps,
                                ),
                                location,
                            ),
                            bucket,
                        ),
                        e_tag,
                    ),
                    version,
                ),
                size,
            ),
            last_modified,
        ) = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|task| {
                if exclusion_set.contains(&task.task_name) {
                    None
                } else {
                    let subjects = self
                        .subjects
                        .as_ref()
                        .unwrap()
                        .iter()
                        .filter_map(|table| {
                            let (subscriptions, publications) =
                                self.get_sub_pub_for_task(&task.task_name);
                            let publication_names = publications
                                .into_iter()
                                .map(|p| p.subject_name())
                                .collect::<Vec<_>>();
                            let subscription_names = subscriptions
                                .into_iter()
                                .map(|p| p.subject_name())
                                .collect::<Vec<_>>();
                            let table_names = publication_names
                                .into_iter()
                                .chain(subscription_names)
                                .collect::<HashSet<_>>();
                            if table_names.contains(&table.get_name()) {
                                Some((
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        table
                                                                            .get_name()
                                                                            .to_string(),
                                                                        task.task_name.to_string(),
                                                                    ),
                                                                    network_name.to_string(),
                                                                ),
                                                                0_i64,
                                                            ),
                                                            0_i64,
                                                        ),
                                                        String::new(),
                                                    ),
                                                    bucket.to_string(),
                                                ),
                                                String::new(),
                                            ),
                                            String::new(),
                                        ),
                                        0_u32,
                                    ),
                                    create_timestamp_micros(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    Some(subjects)
                }
            })
            .flatten()
            .unzip();
        let batch = create_subjects_object_store_meta_batch(
            subject_names,
            task_names,
            network_names,
            num_rows,
            supersteps,
            location,
            bucket,
            e_tag,
            version,
            size,
            last_modified,
        )?;
        let subject = Subject::get_builder()
            .with_name(
                AvailableSubjects::SubjectsObjectStoreMeta
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }

    fn with_subjects_as_subject_plans(self, subjects: &SubjectPlan) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let subjects_vec_str = subjects.subject().get_column_as_vec_str("subject_name");
        let columns_vec_str = subjects.subject().get_column_as_vec_str("column_name");
        let types_vec_str = subjects.subject().get_column_as_vec_str("type_name");

        // get unique subjects
        let subjects_unique = subjects_vec_str.iter().collect::<HashSet<_>>();
        let combined = subjects_vec_str
            .iter()
            .zip(columns_vec_str.iter())
            .zip(types_vec_str.iter())
            .map(|((x, y), z)| (x, y, z))
            .collect::<Vec<_>>();

        // build the subjects tables
        let mut subjects = Vec::new();
        for subject in subjects_unique {
            let mut fields = Vec::new();
            for (s, c, t) in combined.iter() {
                if s == &subject {
                    let data_type = from_str_to_data_type(t)?;
                    fields.push(Field::new(**c, data_type, false));
                }
            }
            let batch = RecordBatch::new_empty(Arc::new(Schema::new(fields)));
            let subject = Subject::get_builder()
                .with_record_batches(vec![batch])?
                .with_name(subject)
                .build()?;
            let plan = SubjectPlan::get_builder().with_subject(subject).build()?;
            subjects.push(plan);
        }

        Ok(self.with_subjects(subjects))
    }

    fn with_tasks_as_subject_plans(self, tasks: &SubjectPlan) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let tasks_vec_str = tasks.subject().get_column_as_vec_str("task_name");
        let processors_vec_str = tasks.subject().get_column_as_vec_str("processor_name");

        // get unique tasks while preserving order
        let mut tasks_unique = tasks_vec_str.iter().collect::<HashSet<_>>();
        let mut sort_tasks = Vec::new();
        for task_name in tasks_vec_str.iter() {
            if tasks_unique.contains(task_name) && tasks_unique.remove(task_name) {
                sort_tasks.push(task_name);
            }
        }
        let combined = tasks_vec_str
            .iter()
            .zip(processors_vec_str.iter())
            .collect::<Vec<_>>();

        // build the task plans
        let mut tasks = Vec::new();
        for task in sort_tasks {
            let builder = TaskPlanBuilder::default().with_name(task);
            let mut processor_names = Vec::new();
            for (t, p) in combined.iter() {
                if t == &task {
                    processor_names.push(p);
                }
            }
            let task_plan = builder
                .with_processor_names(&processor_names.iter().map(|&&&s| s).collect::<Vec<_>>())
                .build()?;
            tasks.push(task_plan);
        }

        Ok(self.with_tasks(tasks))
    }

    fn with_processors_as_subject_plans(self, procesors: &SubjectPlan) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let processor_vec_str = procesors.subject().get_column_as_vec_str("processor_name");
        let type_vec_str = procesors.subject().get_column_as_vec_str("processor_type");
        let subscribe_vec_str = procesors.subject().get_column_as_vec_str("subscribe_type");
        let update_vec_str = procesors.subject().get_column_as_vec_str("update_type");
        let pub_sub_vec_str = procesors
            .subject()
            .get_column_as_vec_str("publication_subscription_name");
        let pub_sub_tab_name_vec_str = procesors
            .subject()
            .get_column_as_vec_str("publication_subscription_table_name");
        let is_sub_vec = procesors
            .subject()
            .get_column_as_vec_primitive::<u8>("is_subscription")?;

        // get unique processors while preserving order
        let mut processors_unique = processor_vec_str.iter().collect::<HashSet<_>>();
        let sort_processors = processor_vec_str
            .iter()
            .filter(|p| processors_unique.remove(p))
            .collect::<Vec<_>>();
        let combined = processor_vec_str
            .iter()
            .zip(type_vec_str.iter())
            .zip(subscribe_vec_str.iter())
            .zip(update_vec_str.iter())
            .zip(pub_sub_vec_str.iter())
            .zip(pub_sub_tab_name_vec_str.iter())
            .zip(is_sub_vec.iter())
            .map(|((((((a, b), c), d), e), f), g)| (a, b, c, d, e, f, g))
            .collect::<Vec<_>>();

        // build the processors in order
        let mut processors = Vec::new();
        for processor_name in sort_processors {
            let mut processor = None;
            let mut subscriptions = Vec::new();
            let mut publications = Vec::new();
            let mut subscribe_policy = None;
            let mut update_policy = None;
            for (name, t, s_t, u_t, sub, sub_tab, is_sub) in combined.iter() {
                if name == &processor_name {
                    if **is_sub == 1 {
                        let subscription = Subscription::from_str_fuzzy(sub, sub_tab)?;
                        subscriptions.push(subscription);
                    } else {
                        let publication = Publication::from_str_fuzzy(sub, sub_tab)?;
                        publications.push(publication);
                    }
                    // DM: a short name is used for better front-end aesthetics
                    let subscribe = AvailableSubscribeEvents::from_str_fuzzy(s_t)
                        .map_err(|e| anyhow!("{e:?}"))?
                        .build();
                    subscribe_policy.replace(subscribe);
                    let update = AvailableUpdateEvents::from_str(u_t, false)
                        .map_err(|e| anyhow!("{e:?}"))?
                        .build();
                    update_policy.replace(update);
                    let p = AvailableProcessors::from_str(t, false)
                        .map_err(|e| anyhow!("{e:?}",))?
                        .build_arc(processor_name);
                    processor.replace(p);
                }
            }
            let processor_plan = ProcessorPlanBuilder::default()
                .with_processor(
                    processor
                        .take()
                        .unwrap_or(AvailableProcessors::default().build_arc(processor_name)),
                )
                .with_subscriptions(&subscriptions)
                .with_publications(&publications)
                .with_subscribe_policy(
                    subscribe_policy
                        .take()
                        .unwrap_or(AvailableSubscribeEvents::default().build()),
                )
                .with_update_policy(
                    update_policy
                        .take()
                        .unwrap_or(AvailableUpdateEvents::default().build()),
                )
                .build()?;
            processors.push(processor_plan);
        }

        Ok(self.with_processors(processors))
    }

    fn with_runtime_envs_as_subject_plans(self, runtime_envs: &SubjectPlan) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let runtime_envs_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_str("runtime_env_name");
        let object_store_backend_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_str("object_store_backend");
        let object_store_bucket_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_str("object_store_bucket");
        let object_store_backend_config_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_str("object_store_config");
        let subject_folder_partitioning_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_str("subject_folder_partitioning");
        let subject_file_partitioning_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_str("subject_file_partitioning");
        let mex_memory_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_primitive::<u32>("max_memory")?;
        let max_time_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_primitive::<u32>("max_time")?;
        let max_steps_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_primitive::<u32>("max_steps")?;
        let max_tasks_vec_str = runtime_envs
            .subject()
            .get_column_as_vec_primitive::<u32>("max_tasks")?;

        // get unique subjects
        let runtime_envs_unique = runtime_envs_vec_str.iter().collect::<HashSet<_>>();
        let combined = runtime_envs_vec_str
            .iter()
            .zip(object_store_backend_vec_str.iter())
            .zip(object_store_bucket_vec_str.iter())
            .zip(object_store_backend_config_vec_str.iter())
            .zip(subject_folder_partitioning_vec_str.iter())
            .zip(subject_file_partitioning_vec_str.iter())
            .zip(mex_memory_vec_str.iter())
            .zip(max_time_vec_str.iter())
            .zip(max_steps_vec_str.iter())
            .zip(max_tasks_vec_str.iter())
            .map(|(((((((((a, b), c), d), e), f), g), h), i), j)| (a, b, c, d, e, f, g, h, i, j))
            .collect::<Vec<_>>();

        // build the task plans
        let mut runtime_envs = Vec::new();
        for rt_name in runtime_envs_unique {
            let mut rt = RuntimeEnv::get_builder().with_name(rt_name);
            for (name, os_backend, os_bucket, os_config, folder, file, mem, time, steps, tasks) in
                combined.iter()
            {
                if name == &rt_name {
                    let backend = ObjectStorageBackend::from_str(os_backend, false)
                        .map_err(|err| anyhow!("{err}"))?;
                    let config = serde_json::from_str::<Map<String, Value>>(os_config)?;
                    let store_bucket = if os_bucket.is_empty() {
                        None
                    } else {
                        Some(os_bucket.to_string())
                    };
                    let store_config = if config.is_empty() {
                        None
                    } else {
                        Some(config.clone())
                    };
                    let store = make_store(&backend, store_bucket.as_ref(), store_config.as_ref())?;
                    rt = rt
                        .with_object_store(store)
                        .with_object_store_config(&config)
                        .with_subject_folder_partitioning(
                            &SubjectFolderPartition::from_str(folder, false)
                                .map_err(|err| anyhow!("{err}"))?,
                        )
                        .with_subject_file_partitioning(
                            &SubjectFilePartition::from_str(file, false)
                                .map_err(|err| anyhow!("{err}"))?,
                        )
                        .with_max_memory(**mem as usize)
                        .with_max_time(**time as usize)
                        .with_max_steps(**steps as usize)
                        .with_max_tasks(**tasks as usize);
                }
            }
            runtime_envs.push(rt.build()?);
        }

        let runtime_env = runtime_envs.pop().ok_or(anyhow!(
            "Missing runtime env when building from subject plans."
        ))?;
        Ok(self.with_runtime_env(Arc::new(runtime_env)))
    }

    fn subjects_to_exclude(&self) -> Result<HashSet<String>> {
        // Exclude subjects from `NextTaskNetwork`
        let next_task_network = NextTaskNetwork::default();
        let tables_publish_subscribe = if let Some(network_name) = self.name.as_ref() {
            if network_name != next_task_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    next_task_network.as_mermaid_flowchart(),
                    false,
                )?
                .with_subjects_from_mermaid_erdiagram(
                    next_task_network.as_mermaid_erdiagram(),
                    false,
                    false,
                )?
                .with_name(next_task_network.network_name)
                .add_processor_subjects()?
                .subjects
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `NextSuperstepNetwork`
        let next_superstep = NextSuperstepNetwork::default();
        let tables_next_superstep = if let Some(network_name) = self.name.as_ref() {
            if network_name != next_superstep.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    next_superstep.as_mermaid_flowchart(),
                    false,
                )?
                .with_name(next_superstep.network_name)
                .add_processor_subjects()?
                .subjects
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `SubjectsNumRowsNetwork`
        let subjects_network = CountSubjectRowsNetwork::default();
        let tables_subjects = if let Some(network_name) = self.name.as_ref() {
            if network_name != subjects_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    subjects_network.as_mermaid_flowchart(),
                    false,
                )?
                .with_subjects_from_mermaid_erdiagram(
                    subjects_network.as_mermaid_erdiagram(),
                    false,
                    false,
                )?
                .with_name(subjects_network.network_name)
                .add_processor_subjects()?
                .subjects
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `InvokeTaskNetworkBuilder`
        let invoke_task_network = InvokeTaskNetworkBuilder::default();
        let tables_invoke_task = if let Some(network_name) = self.name.as_ref() {
            if network_name != invoke_task_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    &invoke_task_network.as_mermaid_flowchart(),
                    false,
                )?
                .with_subjects_from_mermaid_erdiagram(
                    &invoke_task_network.as_mermaid_erdiagram()?,
                    false,
                    false,
                )?
                .with_name(invoke_task_network.network_name)
                .add_processor_subjects()?
                .subjects
                .unwrap()
                .into_iter()
                .filter_map(|t| if t.get_name() == "Bytes" {
                    None
                } else {
                    Some(t.get_name().to_string())
                })
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `TaskResponseNetworkBuilder`
        let task_response_network = TaskResponseNetworkBuilder::default();
        let tables_task_response = if let Some(network_name) = self.name.as_ref() {
            if network_name != task_response_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    &task_response_network.as_mermaid_flowchart(),
                    false,
                )?
                .with_subjects_from_mermaid_erdiagram(
                    &task_response_network.as_mermaid_erdiagram(),
                    false,
                    false,
                )?
                .with_name(task_response_network.network_name)
                .add_processor_subjects()?
                .subjects
                .unwrap()
                .into_iter()
                .filter_map(|t| if t.get_name() == AvailableSubjects::Bytes.to_string().as_str() || t.get_name() == AvailableInterfaceSubjects::ToolMessages.to_string().as_str() {
                    None
                } else {
                    Some(t.get_name().to_string())
                })
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Wrap into a HashSet
        let exclusion_set = tables_publish_subscribe
            .into_iter()
            .chain(tables_next_superstep)
            .chain(tables_subjects)
            .chain(tables_invoke_task)
            .chain(tables_task_response)
            .collect::<HashSet<_>>();
        Ok(exclusion_set)
    }

    fn tasks_to_exclude(&self) -> Result<HashSet<String>> {
        // Exclude subjects from `NextTaskNetwork`
        let next_task_network = NextTaskNetwork::default();
        let tasks_publish_subscribe = if let Some(network_name) = self.name.as_ref() {
            if network_name != next_task_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    next_task_network.as_mermaid_flowchart(),
                    false,
                )?
                .tasks
                .unwrap()
                .into_iter()
                .map(|t| t.task_name)
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `NextSuperstepNetwork`
        let next_superstep = NextSuperstepNetwork::default();
        let tasks_next_superstep = if let Some(network_name) = self.name.as_ref() {
            if network_name != next_superstep.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    next_superstep.as_mermaid_flowchart(),
                    false,
                )?
                .tasks
                .unwrap()
                .into_iter()
                .map(|t| t.task_name)
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `SubjectsNumRowsNetwork`
        let subjects_network = CountSubjectRowsNetwork::default();
        let tasks_subjects = if let Some(network_name) = self.name.as_ref() {
            if network_name != subjects_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    subjects_network.as_mermaid_flowchart(),
                    false,
                )?
                .tasks
                .unwrap()
                .into_iter()
                .map(|t| t.task_name)
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `InvokeTaskNetworkBuilder`
        let invoke_task_network = InvokeTaskNetworkBuilder::default();
        let tasks_invoke_task = if let Some(network_name) = self.name.as_ref() {
            if network_name != invoke_task_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    &invoke_task_network.as_mermaid_flowchart(),
                    false,
                )?
                .tasks
                .unwrap()
                .into_iter()
                .map(|t| t.task_name)
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `TaskResponseNetworkBuilder`
        let task_response_network = TaskResponseNetworkBuilder::default();
        let tasks_task_response = if let Some(network_name) = self.name.as_ref() {
            if network_name != task_response_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    &task_response_network.as_mermaid_flowchart(),
                    false,
                )?
                .tasks
                .unwrap()
                .into_iter()
                .map(|t| t.task_name)
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Wrap into a HashSet
        let exclusion_set = tasks_publish_subscribe
            .into_iter()
            .chain(tasks_next_superstep)
            .chain(tasks_subjects)
            .chain(tasks_invoke_task)
            .chain(tasks_task_response)
            .collect::<HashSet<_>>();
        Ok(exclusion_set)
    }

    fn processors_to_exclude(&self) -> Result<HashSet<String>> {
        // Exclude subjects from `NextTaskNetwork`
        let next_task_network = NextTaskNetwork::default();
        let processors_task_network = if let Some(network_name) = self.name.as_ref() {
            if network_name != next_task_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    next_task_network.as_mermaid_flowchart(),
                    false,
                )?
                .processors
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `NextSuperstepNetwork`
        let next_superstep = NextSuperstepNetwork::default();
        let processors_next_superstep = if let Some(network_name) = self.name.as_ref() {
            if network_name != next_superstep.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    next_superstep.as_mermaid_flowchart(),
                    false,
                )?
                .processors
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `SubjectsNumRowsNetwork`
        let subjects_network = CountSubjectRowsNetwork::default();
        let processors_subjects = if let Some(network_name) = self.name.as_ref() {
            if network_name != subjects_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    subjects_network.as_mermaid_flowchart(),
                    false,
                )?
                .processors
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `InvokeTaskNetworkBuilder`
        let invoke_task_network = InvokeTaskNetworkBuilder::default();
        let processors_invoke_task = if let Some(network_name) = self.name.as_ref() {
            if network_name != invoke_task_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    &invoke_task_network.as_mermaid_flowchart(),
                    false,
                )?
                .processors
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Exclude subjects from `TaskResponseNetworkBuilder`
        let task_response_network = TaskResponseNetworkBuilder::default();
        let processors_task_response = if let Some(network_name) = self.name.as_ref() {
            if network_name != task_response_network.network_name {
                NetworkBuilder::from_mermaid_flowchart(
                    &task_response_network.as_mermaid_flowchart(),
                    false,
                )?
                .processors
                .unwrap()
                .into_iter()
                .map(|t| t.get_name().to_string())
                .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Wrap into a HashSet
        let exclusion_set = processors_task_network
            .into_iter()
            .chain(processors_next_superstep)
            .chain(processors_subjects)
            .chain(processors_invoke_task)
            .chain(processors_task_response)
            .collect::<HashSet<_>>();
        Ok(exclusion_set)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_network_builder;
    use phymes_task::test_task;

    use super::*;

    #[test]
    fn test_to_from_arrow_tables() -> Result<()> {
        // Init runtime env
        let runtime_env = test_task::make_runtime_env("rt_1")?;

        // Init subjects
        let mut subjects = test_task::make_subject_tables("subjects_1", "config_1")?;
        subjects.extend(test_task::make_subject_tables("subjects_2", "config_2")?);
        subjects.extend(test_task::make_subject_tables("subjects_3", "config_3")?);
        let subject_plans = subjects
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        // Make the builder
        let builder = test_network_builder::make_test_network_builder_parallel_processors()
            .with_name("")
            .with_runtime_env(runtime_env)
            .with_subjects(subject_plans);

        // Test to tables
        let tables = builder.to_subject_plans(true, true, true, true, true)?;

        // Check the tables
        assert_eq!(
            tables.first().unwrap().get_name(),
            AvailableSubjects::NetworkMermaid.to_string().as_str()
        );
        assert_eq!(
            tables.get(1).unwrap().get_name(),
            AvailableSubjects::NetworkErrors.to_string().as_str()
        );
        assert_eq!(
            tables.get(2).unwrap().get_name(),
            AvailableSubjects::NetworkMetrics.to_string().as_str()
        );
        assert_eq!(
            tables.get(3).unwrap().get_name(),
            AvailableSubjects::NetworkTraces.to_string().as_str()
        );
        assert_eq!(
            tables.get(4).unwrap().get_name(),
            AvailableSubjects::NetworkEvents.to_string().as_str()
        );
        assert_eq!(
            tables.get(5).unwrap().get_name(),
            AvailableSubjects::SubjectsNumRows.to_string().as_str()
        );
        assert_eq!(
            tables.get(6).unwrap().get_name(),
            AvailableSubjects::SubjectsChangeLog.to_string().as_str()
        );
        assert_eq!(
            tables.get(7).unwrap().get_name(),
            AvailableSubjects::SubjectsObjectStoreMeta
                .to_string()
                .as_str()
        );
        assert_eq!(
            tables.get(8).unwrap().get_name(),
            AvailableSubjects::NetworkTasksRunLog.to_string().as_str()
        );
        assert_eq!(
            tables.get(9).unwrap().get_name(),
            AvailableSubjects::NetworkSubjectSchemas
                .to_string()
                .as_str()
        );
        assert_eq!(
            tables.get(10).unwrap().get_name(),
            AvailableSubjects::NetworkTasks.to_string().as_str()
        );
        assert_eq!(
            tables.get(11).unwrap().get_name(),
            AvailableSubjects::NetworkProcessors.to_string().as_str()
        );
        assert_eq!(
            tables.get(12).unwrap().get_name(),
            AvailableSubjects::NetworkRuntimeEnvs.to_string().as_str()
        );

        // Test from tables
        let tables_test =
            NetworkBuilder::from_subject_plans(&tables.iter().collect::<Vec<_>>(), None)?
                .with_name("")
                .to_subject_plans(true, true, true, true, true)?;

        // Check the tables
        assert_eq!(
            tables_test.first().unwrap().get_name(),
            tables.first().unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name"),
            tables
                .first()
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name")
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .subject()
                .get_column_as_vec_str("flowchart_diagram"),
            tables
                .first()
                .unwrap()
                .subject()
                .get_column_as_vec_str("flowchart_diagram")
        );
        // Contains the added subjects
        assert_ne!(
            tables_test
                .first()
                .unwrap()
                .subject()
                .get_column_as_vec_str("er_diagram"),
            tables
                .first()
                .unwrap()
                .subject()
                .get_column_as_vec_str("er_diagram")
        );
        assert_eq!(
            tables_test.get(1).unwrap().get_name(),
            tables.get(1).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(1)
                .unwrap()
                .subject()
                .get_column_as_vec_str("error"),
            tables
                .get(1)
                .unwrap()
                .subject()
                .get_column_as_vec_str("error")
        );

        assert_eq!(
            tables_test.get(2).unwrap().get_name(),
            tables.get(2).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("metric_name"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("metric_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("metric_value")?,
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("metric_value")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("labels"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("labels")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("id")?,
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("id")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("span_name"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("span_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("parent_name"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("parent_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("span_id")?,
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("span_id")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("parent_id")?,
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("parent_id")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("file"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("file")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("thread"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("thread")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("function"),
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_str("function")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("line")?,
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("line")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("timestamp")?,
            tables
                .get(2)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("timestamp")?
        );
        assert_eq!(
            tables_test.get(3).unwrap().get_name(),
            tables.get(3).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("tracer_type"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("tracer_type")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("tracer_event"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("tracer_event")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("message_name"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("message_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_name"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("labels"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("labels")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("id")?,
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("id")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("span_name"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("span_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("parent_name"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("parent_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("span_id")?,
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("span_id")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("parent_id")?,
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("parent_id")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("file"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("file")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("thread"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("thread")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("function"),
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_str("function")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("line")?,
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("line")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("timestamp")?,
            tables
                .get(3)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("timestamp")?
        );
        assert_eq!(
            tables_test.get(4).unwrap().get_name(),
            tables.get(4).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("event_level"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("event_level")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("record_name"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("record_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("record_value"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("record_value")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("labels"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("labels")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("id")?,
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("id")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("span_name"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("span_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("parent_name"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("parent_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("span_id")?,
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("span_id")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("parent_id")?,
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("parent_id")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("file"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("file")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("thread"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("thread")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("function"),
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_str("function")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("line")?,
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("line")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("timestamp")?,
            tables
                .get(4)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("timestamp")?
        );
        assert_eq!(
            tables_test.get(5).unwrap().get_name(),
            tables.get(5).unwrap().get_name()
        );
        // DM: need to check why this test is failing
        // let tables_test_set = tables_test
        //     .get(5)
        //     .unwrap().subject()
        //     .get_column_as_vec_str("subject_name")
        //     .into_iter()
        //     .collect::<HashSet<_>>();
        // let tables_set = tables
        //     .get(5)
        //     .unwrap().subject()
        //     .get_column_as_vec_str("subject_name")
        //     .into_iter()
        //     .collect::<HashSet<_>>();
        // left: {"NetworkTasksRunLog", "NetworkEvents", "subjects_1", "config_2", "NetworkMetrics", "NetworkTraces", "SubjectsNumRows", "SubjectsChangeLog", "subjects_3", "NetworkErrors", "subjects_2", "config_3", "config_1", "NetworkMermaid"}
        // right: {"config_2", "subjects_3", "subjects_1", "config_1", "subjects_2", "config_3"}
        // assert_eq!(tables_test_set, tables_set);
        assert_eq!(
            tables_test.get(6).unwrap().get_name(),
            tables.get(6).unwrap().get_name()
        );
        let tables_test_set = tables_test
            .get(6)
            .unwrap()
            .subject()
            .get_column_as_vec_str("subject_name")
            .into_iter()
            .collect::<HashSet<_>>();
        let tables_set = tables
            .get(6)
            .unwrap()
            .subject()
            .get_column_as_vec_str("subject_name")
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(tables_test_set, tables_set);
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name"),
            tables
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name")
        );
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name"),
            tables
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name")
        );
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("num_rows")?,
            tables
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("num_rows")?
        );
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("superstep")?,
            tables
                .get(6)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("superstep")?
        );
        assert_eq!(
            tables_test.get(7).unwrap().get_name(),
            tables.get(7).unwrap().get_name()
        );
        let tables_test_set = tables_test
            .get(7)
            .unwrap()
            .subject()
            .get_column_as_vec_str("subject_name")
            .into_iter()
            .collect::<HashSet<_>>();
        let tables_set = tables
            .get(7)
            .unwrap()
            .subject()
            .get_column_as_vec_str("subject_name")
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(tables_test_set, tables_set);
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("num_rows")?,
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("num_rows")?
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("superstep")?,
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<i64>("superstep")?
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("location"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("location")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("bucket"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("bucket")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("e_tag"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("e_tag")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("version"),
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_str("version")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("size")?,
            tables
                .get(7)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("size")?
        );
        // assert_eq!(
        //     tables_test
        //         .get(7)
        //         .unwrap().subject()
        //         .get_column_as_vec_primitive::<i64>("last_modified")?,
        //     tables.get(7).unwrap().subject().get_column_as_vec_primitive::<i64>("last_modified")?
        // );
        assert_eq!(
            tables_test.get(8).unwrap().get_name(),
            tables.get(8).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(8)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name"),
            tables
                .get(8)
                .unwrap()
                .subject()
                .get_column_as_vec_str("network_name")
        );
        assert_eq!(
            tables_test
                .get(8)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name"),
            tables
                .get(8)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name")
        );
        // assert_eq!(
        //     tables_test
        //         .get(8)
        //         .unwrap().subject()
        //         .get_column_as_vec_primitive::<i64>("timestamp")?,
        //     tables
        //         .get(8)
        //         .unwrap().subject()
        //         .get_column_as_vec_primitive::<i64>("timestamp")?
        // );
        assert_eq!(
            tables_test.get(9).unwrap().get_name(),
            tables.get(9).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(9)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_name"),
            tables
                .get(9)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_name")
        );
        assert_eq!(
            tables_test
                .get(9)
                .unwrap()
                .subject()
                .get_column_as_vec_str("column_name"),
            tables
                .get(9)
                .unwrap()
                .subject()
                .get_column_as_vec_str("column_name")
        );
        assert_eq!(
            tables_test
                .get(9)
                .unwrap()
                .subject()
                .get_column_as_vec_str("type_name"),
            tables
                .get(9)
                .unwrap()
                .subject()
                .get_column_as_vec_str("type_name")
        );
        assert_eq!(
            tables_test.get(1).unwrap().get_name(),
            tables.get(1).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(10)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name"),
            tables
                .get(10)
                .unwrap()
                .subject()
                .get_column_as_vec_str("task_name")
        );
        assert_eq!(
            tables_test
                .get(10)
                .unwrap()
                .subject()
                .get_column_as_vec_str("processor_name"),
            tables
                .get(10)
                .unwrap()
                .subject()
                .get_column_as_vec_str("processor_name")
        );
        assert_eq!(
            tables_test.get(11).unwrap().get_name(),
            tables.get(11).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("processor_name"),
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("processor_name")
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("processor_type"),
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("processor_type")
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("publication_subscription_name"),
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("publication_subscription_name")
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("publication_subscription_table_name"),
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("publication_subscription_table_name")
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u8>("is_subscription")?,
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u8>("is_subscription")?
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subscribe_type"),
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subscribe_type")
        );
        assert_eq!(
            tables_test
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("update_type"),
            tables
                .get(11)
                .unwrap()
                .subject()
                .get_column_as_vec_str("update_type")
        );
        assert_eq!(
            tables_test.get(12).unwrap().get_name(),
            tables.get(12).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("runtime_env_name"),
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("runtime_env_name")
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("object_store_backend"),
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("object_store_backend")
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("object_store_bucket"),
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("object_store_bucket")
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("object_store_config"),
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("object_store_config")
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_folder_partitioning"),
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_folder_partitioning")
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_file_partitioning"),
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_str("subject_file_partitioning")
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_memory")?,
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_memory")?
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_time")?,
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_time")?
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_steps")?,
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_steps")?
        );
        assert_eq!(
            tables_test
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_tasks")?,
            tables
                .get(12)
                .unwrap()
                .subject()
                .get_column_as_vec_primitive::<u32>("max_tasks")?
        );

        Ok(())
    }
}
