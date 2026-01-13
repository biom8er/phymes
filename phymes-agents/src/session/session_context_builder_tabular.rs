use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Field, Schema},
};
use clap::ValueEnum;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableTableSubscribePolicies, AvailableTableUpdatePolicies, BuildableTrait, BuilderTrait, MappableTrait, ProcessorPlanBuilder, RuntimeEnv, RuntimeEnvTrait, Table, TableBuilderTrait, TablePublication, TableSubscription, TableTrait, TaskPlanBuilder, create_session_mermaid_batch, create_session_processors_batch, create_session_runtime_envs_batch, create_session_subjects_batch, create_session_tasks_batch, create_session_tasks_run_log_batch, create_subjects_change_log_batch, create_subjects_num_rows_batch, from_data_type_to_str, from_str_to_data_type
};
use phymes_diagnostics::{HashSet, create_timestamp_micros};

use crate::{
    AvailableProcessors, SessionContextBuilder, SessionContextBuilderMermaidTrait,
    SessionContextBuilderTrait,
};

/// Trait extension for [SessionContextBuilderTrait] to enable exporting to and importing from tabular format
pub trait SessionContextBuilderTabularTrait {
    /// Convert the session into tables
    ///
    /// # Notes
    ///
    /// * All subjects that are a part of the state are included
    /// * Additional meta tables describing the SessionContext schema are included
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
    /// * `Vec<ArrowTable>` with the SessionContext in tabular format and Optional `Vec<ArrowTable>` with the state
    fn to_arrow_tables(
        &self,
        include_mermaid: bool,
        include_errors: bool,
        include_diagnostics: bool,
        include_tasks_run_log: bool,
        include_subjects_change_log: bool,
    ) -> Result<Vec<Table>>;

    /// Get the subjects in tabular form
    ///
    /// # Arguments
    /// * `additional_tables` - Additional tables to include in addition to what is in the state
    fn get_subjects_as_table(&self, additional_tables: &[Table]) -> Result<Table>;

    /// Get the tasks in tabular form
    fn get_tasks_as_table(&self) -> Result<Table>;

    /// Get the processors in tabular form
    ///
    /// # Note
    ///
    /// * No sorting is performed when generating the table
    ///   so that order of processors is maintained
    fn get_processors_as_table(&self) -> Result<Table>;

    /// Get the runtime environments in tabular form
    fn get_runtime_envs_as_table(&self) -> Result<Table>;

    /// Get mermaid js chart strings
    fn get_mermaid_js_as_table(&self) -> Result<Table>;

    /// Create the initial `TasksRunLog` table
    fn get_tasks_run_log_as_table(&self) -> Result<Table>;

    /// Create the initial `SubjectsNumRows` table
    fn get_subjects_num_rows_as_table(&self) -> Result<Table>;

    /// Create the `SubjectsNumRows` and `SubjectsChangeLog` tables
    fn get_subjects_change_log_as_table(&self) -> Result<Table>;

    /// Create the session from tables
    ///
    /// # Notes
    ///
    /// * Minimally, the meta tables describing the [SessionContext] schema must be included
    /// * Optionally, the subject tables will be populated with data if the state tables are included
    /// * Mermaid_js scripts are ignored
    ///
    /// # Arguments
    ///
    /// * `tables` - List of [Table]s describing the [SessionContext] schema with
    ///   optional subject tables with the actual data
    /// * `state` - Optionally the subject data. If none the subject tables will be initialized.
    ///
    /// [SessionContext]: phymes_core::SessionContext
    fn from_arrow_tables(tables: &[&Table], state: Option<Vec<Table>>) -> Result<Self>
    where
        Self: Sized;

    /// Create empty subject tables from `SessionSubjects`
    fn with_subjects_as_tables(self, subjects: &Table) -> Result<Self>
    where
        Self: Sized;
    fn with_tasks_as_tables(self, tasks: &Table) -> Result<Self>
    where
        Self: Sized;
    fn with_processors_as_tables(self, processors: &Table) -> Result<Self>
    where
        Self: Sized;
    fn with_runtime_envs_as_tables(self, runtime_envs: &Table) -> Result<Self>
    where
        Self: Sized;
}

impl SessionContextBuilderTabularTrait for SessionContextBuilder {
    fn to_arrow_tables(
        &self,
        include_mermaid: bool,
        include_errors: bool,
        include_diagnostics: bool,
        include_tasks_run_log: bool,
        include_subjects_change_log: bool,
    ) -> Result<Vec<Table>> {
        let mut tables = Vec::new();
        if include_mermaid {
            tables.push(self.get_mermaid_js_as_table()?);
        }
        if include_errors {
            tables.push(AvailableSubjects::SessionErrors.to_table(None, None)?);
        }
        if include_diagnostics {
            tables.push(AvailableSubjects::SessionMetrics.to_table(None, None)?);
            tables.push(AvailableSubjects::SessionTraces.to_table(None, None)?);
            tables.push(AvailableSubjects::SessionEvents.to_table(None, None)?);
        }
        // include_tasks_run_log is before include_subjects_change_log
        // so that the timestamp of all tasks is less than the timestamp of all subjects
        if include_tasks_run_log {
            tables.push(self.get_tasks_run_log_as_table()?);
        }
        if include_subjects_change_log {
            tables.push(self.get_subjects_num_rows_as_table()?);
            tables.push(self.get_subjects_change_log_as_table()?);
        }
        tables.extend([
            self.get_subjects_as_table(&tables)?,
            self.get_tasks_as_table()?,
            self.get_processors_as_table()?,
            self.get_runtime_envs_as_table()?,
        ]);
        Ok(tables)
    }

    fn from_arrow_tables(tables: &[&Table], mut state: Option<Vec<Table>>) -> Result<Self>
    where
        Self: Sized,
    {
        // initialize the builder
        let mut builder = Self::new();

        // extract the schema
        for table in tables {
            if table.get_name() == AvailableSubjects::SessionSubjects.to_string().as_str()
                && state.is_none()
            {
                builder = builder.with_subjects_as_tables(table)?;
            } else if table.get_name() == AvailableSubjects::SessionTasks.to_string().as_str() {
                builder = builder.with_tasks_as_tables(table)?;
            } else if table.get_name() == AvailableSubjects::SessionProcessors.to_string().as_str()
            {
                builder = builder.with_processors_as_tables(table)?;
            } else if table.get_name() == AvailableSubjects::SessionRuntimeEnvs.to_string().as_str()
            {
                builder = builder.with_runtime_envs_as_tables(table)?;
            } else if table.get_name() == AvailableSubjects::SessionMermaid.to_string().as_str()
                // Diagnostic tables
                || table.get_name() == AvailableSubjects::SessionErrors.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTraces.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionMetrics.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionEvents.to_string().as_str()
                || table.get_name() == AvailableSubjects::MetricPivot.to_string().as_str()
                // Subjects change log tables
                || table.get_name() == AvailableSubjects::SubjectsNumRows.to_string().as_str()
                || table.get_name() == AvailableSubjects::SubjectsChangeLog.to_string().as_str()
                // Tasks run log tables
                || table.get_name() == AvailableSubjects::SessionTasksCheck.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTasksPublish.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTasksPublishAggregate.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTasksRunLog.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTasksSubscribe.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTasksSubscribeAggregate.to_string().as_str()
                || table.get_name() == AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str()
            {
                // These tables are created on the fly so we do not want to duplicate them.
                // If the user wishes to continue already generated tables they can do so
                // by passing them as optional state.
                continue;
            } else {
                return Err(anyhow!(
                    "Unrecognized table {} found when creating SessionContextBuilder",
                    table.get_name()
                ));
            }
        }

        // Add the optional state
        if state.is_some() {
            Ok(builder.with_state(state.take().unwrap()))
        } else {
            Ok(builder)
        }
    }

    fn get_subjects_as_table(&self, additional_tables: &[Table]) -> Result<Table> {
        // Check that the state exists
        if self.state.is_none() {
            return Err(anyhow!(
                "Add state subjects before making the subject tables."
            ));
        }
        let session_name = if let Some(session_name) = self.name.as_ref() {
            session_name
        } else {
            return Err(anyhow!(
                "Add session name before making the subject tables."
            ));
        };

        let mut session_names = Vec::<String>::new();
        let mut subject_names = Vec::<String>::new();
        let mut cols_names = Vec::<String>::new();
        let mut type_names = Vec::<String>::new();

        // Sort the hashmap
        let mut sorted_state = self
            .state
            .as_ref()
            .unwrap()
            .iter()
            .chain(additional_tables)
            .collect::<Vec<_>>();
        sorted_state.sort_by(|a, b| a.get_name().cmp(b.get_name()));
        for subject in sorted_state.iter() {
            if !subject_names.contains(&subject.get_name().to_string()) {
                let fields = subject.get_schema().fields().clone();
                for field in fields.iter() {
                    let type_name = from_data_type_to_str(field.data_type());
                    session_names.push(session_name.to_string());
                    subject_names.push(subject.get_name().to_string());
                    cols_names.push(field.name().to_string());
                    type_names.push(type_name);
                }
            }
        }

        // create the record batch
        let batch =
            create_session_subjects_batch(session_names, subject_names, cols_names, type_names)?;

        // create the table
        Table::get_builder()
            .with_name(AvailableSubjects::SessionSubjects.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_tasks_as_table(&self) -> Result<Table> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the tasks table."));
        }
        let session_name = if let Some(session_name) = self.name.as_ref() {
            session_name
        } else {
            return Err(anyhow!(
                "Add session name before making the subject tables."
            ));
        };

        let mut session_names = Vec::<String>::new();
        let mut task_names = Vec::new();
        let mut processor_names = Vec::new();
        let mut runtime_env_names = Vec::new();

        // extract the tasks in order
        for task in self.tasks.as_ref().unwrap().iter() {
            for p in task.processor_names.iter() {
                session_names.push(session_name.to_string());
                task_names.push(task.task_name.to_owned());
                processor_names.push(p.to_string());
                runtime_env_names.push(task.runtime_env_name.to_owned());
            }
        }

        // create the record batch
        let batch = create_session_tasks_batch(
            session_names,
            task_names,
            processor_names,
            runtime_env_names,
        )?;

        // create the table
        Table::get_builder()
            .with_name(AvailableSubjects::SessionTasks.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_processors_as_table(&self) -> Result<Table> {
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processors before making the Mermaid Flowchart."
            ));
        }
        let session_name = if let Some(session_name) = self.name.as_ref() {
            session_name
        } else {
            return Err(anyhow!(
                "Add session name before making the subject tables."
            ));
        };
        let mut session_names = Vec::<String>::new();
        let mut processor_names = Vec::new();
        let mut processor_types = Vec::new();
        let mut pub_sub_name = Vec::new();
        let mut pub_sub_table_names = Vec::new();
        let mut is_sub = Vec::new();
        let mut subscribe_types = Vec::new();
        let mut update_types = Vec::new();

        // extract the processors in order
        for processor in self.processors.as_ref().unwrap().iter() {
            for sub in processor.get_subscriptions() {
                pub_sub_name.push(sub.get_name().to_string());
                pub_sub_table_names.push(sub.get_table_name().to_string());
                is_sub.push(1);
                session_names.push(session_name.to_string());
                processor_names.push(processor.get_name().to_string());
                processor_types.push(processor.get_type().to_string());
                subscribe_types.push(processor.get_subscribe_policy().get_name().to_string());
                update_types.push(processor.get_update_policy().get_name().to_string());
            }
            for p in processor.get_publications() {
                pub_sub_name.push(p.get_name().to_string());
                pub_sub_table_names.push(p.get_table_name().to_string());
                is_sub.push(0);
                session_names.push(session_name.to_string());
                processor_names.push(processor.get_name().to_string());
                processor_types.push(processor.get_type().to_string());
                subscribe_types.push(processor.get_subscribe_policy().get_name().to_string());
                update_types.push(processor.get_update_policy().get_name().to_string());
            }
        }

        // create the record batch
        let batch = create_session_processors_batch(
            session_names,
            processor_names,
            processor_types,
            pub_sub_name,
            pub_sub_table_names,
            subscribe_types,
            update_types,
            is_sub,
        )?;

        // create the table
        Table::get_builder()
            .with_name(AvailableSubjects::SessionProcessors.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_runtime_envs_as_table(&self) -> Result<Table> {
        if self.runtime_envs.is_none() {
            return Err(anyhow!(
                "Add runtime environments before making the Mermaid Flowchart."
            ));
        }
        let session_name = if let Some(session_name) = self.name.as_ref() {
            session_name
        } else {
            return Err(anyhow!(
                "Add session name before making the subject tables."
            ));
        };
        let mut session_names = Vec::<String>::new();
        let mut runtime_env_names = Vec::new();
        let mut memory_limits = Vec::new();
        let mut time_limits = Vec::new();

        // sort the runtime environments
        let mut sorted_rts = self
            .runtime_envs
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        sorted_rts.sort_by(|a, b| a.name.cmp(&b.name));
        for rt in sorted_rts.iter() {
            session_names.push(session_name.to_string());
            runtime_env_names.push(rt.get_name().to_string());
            let memory_limit = rt.memory_limit.unwrap_or_default();
            memory_limits.push(memory_limit as u32);
            let time_limit = rt.time_limit.unwrap_or_default();
            time_limits.push(time_limit as u32);
        }

        // create the record batch
        let batch = create_session_runtime_envs_batch(
            session_names,
            runtime_env_names,
            memory_limits,
            time_limits,
        )?;

        // create the table
        Table::get_builder()
            .with_name(AvailableSubjects::SessionRuntimeEnvs.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_mermaid_js_as_table(&self) -> Result<Table> {
        let flowchart_diagram = self.to_mermaid_flowchart(false, false)?;
        let er_diagram = self.to_mermaid_erdiagram(false, true)?;
        let session_context_name = self.name.as_ref().unwrap().to_string();
        let timestamp = create_timestamp_micros();

        // create the record batch
        let batch = create_session_mermaid_batch(
            vec![session_context_name],
            vec![flowchart_diagram],
            vec![er_diagram],
            vec![timestamp],
        )?;

        // create the table
        Table::get_builder()
            .with_name(AvailableSubjects::SessionMermaid.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_tasks_run_log_as_table(&self) -> Result<Table> {
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the tasks run log table."));
        }
        let session_name = if let Some(session_name) = self.name.as_ref() {
            session_name
        } else {
            return Err(anyhow!(
                "Add session name before making the tasks run log table."
            ));
        };
        let ((session_names, task_names), timestamps) = self.tasks.as_ref().unwrap().iter()
            .map(|task| ((session_name.to_string(), task.task_name.to_string()), create_timestamp_micros()))
            .unzip();
        let batch = create_session_tasks_run_log_batch(session_names, task_names, timestamps)?;
        Table::get_builder().with_name(AvailableSubjects::SessionTasksRunLog.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_subjects_num_rows_as_table(&self) -> Result<Table> {
        if self.state.is_none() {
            return Err(anyhow!("Add subjects before making the subjects num rows table."));
        }
        let (subject_names, num_rows): (Vec<String>, Vec<i64>) = self.state.as_ref().unwrap().iter()
            .map(|t| (t.get_name().to_string(), t.count_rows() as i64))
            .unzip();
        let batch = create_subjects_num_rows_batch(subject_names, num_rows)?;
        Table::get_builder().with_name(AvailableSubjects::SubjectsNumRows.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_subjects_change_log_as_table(&self) -> Result<Table> {
        if self.state.is_none() {
            return Err(anyhow!("Add subjects before making the subjects change log table."));
        }
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making subjects change log table."));
        }
        if self.processors.is_none() {
            return Err(anyhow!("Add processor plans before making subjects change log table."));
        }
        let session_name = if let Some(session_name) = self.name.as_ref() {
            session_name
        } else {
            return Err(anyhow!(
                "Add session name before making the subjects change log table."
            ));
        };
        let ((((subject_names, task_names), session_names), num_rows_delta), timestamps) = self.tasks.as_ref().unwrap().iter()
            .map(|task| self.state.as_ref().unwrap().iter()
                .filter_map(|table| {
                    let (_subscriptions, publications) = self.get_sub_pub_for_task(&task.task_name);
                    let table_names = publications.into_iter().map(|p| p.get_table_name()).collect::<Vec<_>>();
                    if table_names.contains(&table.get_name()) {
                        Some(((((table.get_name().to_string(), task.task_name.to_string()), session_name.to_string()), 0 as i64), create_timestamp_micros()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>())
            .flatten()
            .unzip();
        let batch = create_subjects_change_log_batch(subject_names, task_names, session_names, num_rows_delta, timestamps)?;
        Table::get_builder().with_name(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn with_subjects_as_tables(self, subjects: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let subjects_vec_str = subjects.get_column_as_vec_str("subject_name");
        let columns_vec_str = subjects.get_column_as_vec_str("column_name");
        let types_vec_str = subjects.get_column_as_vec_str("type_name");

        // get unique subjects
        let subjects_unique = subjects_vec_str.iter().collect::<HashSet<_>>();
        let combined = subjects_vec_str
            .iter()
            .zip(columns_vec_str.iter())
            .zip(types_vec_str.iter())
            .map(|((x, y), z)| (x, y, z))
            .collect::<Vec<_>>();

        // build the state tables
        let mut state = Vec::new();
        for subject in subjects_unique {
            let mut fields = Vec::new();
            for (s, c, t) in combined.iter() {
                if s == &subject {
                    let data_type = from_str_to_data_type(t)?;
                    fields.push(Field::new(**c, data_type, false));
                }
            }
            let batch = RecordBatch::new_empty(Arc::new(Schema::new(fields)));
            let table = Table::get_builder()
                .with_record_batches(vec![batch])?
                .with_name(subject)
                .build()?;
            state.push(table);
        }

        Ok(self.with_state(state))
    }

    fn with_tasks_as_tables(self, tasks: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let tasks_vec_str = tasks.get_column_as_vec_str("task_name");
        let processors_vec_str = tasks.get_column_as_vec_str("processor_name");
        let runtime_envs_vec_str = tasks.get_column_as_vec_str("runtime_env_name");

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
            .zip(runtime_envs_vec_str.iter())
            .map(|((x, y), z)| (x, y, z))
            .collect::<Vec<_>>();

        // build the task plans
        let mut tasks = Vec::new();
        for task in sort_tasks {
            let mut builder = TaskPlanBuilder::default().with_name(task);
            let mut processor_names = Vec::new();
            for (t, p, r) in combined.iter() {
                if t == &task {
                    processor_names.push(p);
                    builder = builder.with_runtime_env_name(r);
                }
            }
            let task_plan = builder
                .with_processor_names(&processor_names.iter().map(|&&&s| s).collect::<Vec<_>>())
                .build()?;
            tasks.push(task_plan);
        }

        Ok(self.with_tasks(tasks))
    }

    fn with_processors_as_tables(self, procesors: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let processor_vec_str = procesors.get_column_as_vec_str("processor_name");
        let type_vec_str = procesors.get_column_as_vec_str("processor_type");
        let subscribe_vec_str = procesors.get_column_as_vec_str("subscribe_type");
        let update_vec_str = procesors.get_column_as_vec_str("update_type");
        let pub_sub_vec_str = procesors.get_column_as_vec_str("publication_subscription_name");
        let pub_sub_tab_name_vec_str =
            procesors.get_column_as_vec_str("publication_subscription_table_names");
        let is_sub_vec = procesors.get_column_as_vec_primitive::<u8>("is_subscription")?;

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
                        let subscription = TableSubscription::from_str_fuzzy(sub, sub_tab)?;
                        subscriptions.push(subscription);
                    } else {
                        let publication = TablePublication::from_str_fuzzy(sub, sub_tab)?;
                        publications.push(publication);
                    }
                    let subscribe = AvailableTableSubscribePolicies::from_str(s_t, false)
                        .map_err(|e| anyhow!("{e:?}"))?
                        .build();
                    subscribe_policy.replace(subscribe);
                    let update = AvailableTableUpdatePolicies::from_str(u_t, false)
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
                        .unwrap_or(AvailableTableSubscribePolicies::default().build()),
                )
                .with_update_policy(
                    update_policy
                        .take()
                        .unwrap_or(AvailableTableUpdatePolicies::default().build()),
                )
                .build()?;
            processors.push(processor_plan);
        }

        Ok(self.with_processors(processors))
    }

    fn with_runtime_envs_as_tables(self, runtime_envs: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // extract arrays
        let runtime_envs_vec_str = runtime_envs.get_column_as_vec_str("runtime_env_name");
        let memory_limits_vec_str =
            runtime_envs.get_column_as_vec_primitive::<u32>("memory_limit")?;
        let time_limits_vec_str = runtime_envs.get_column_as_vec_primitive::<u32>("time_limit")?;

        // get unique subjects
        let runtime_envs_unique = runtime_envs_vec_str.iter().collect::<HashSet<_>>();
        let combined = runtime_envs_vec_str
            .iter()
            .zip(memory_limits_vec_str.iter())
            .zip(time_limits_vec_str.iter())
            .map(|((x, y), z)| (x, y, z))
            .collect::<Vec<_>>();

        // build the task plans
        let mut runtime_envs = Vec::new();
        for rt_name in runtime_envs_unique {
            let mut rt = RuntimeEnv::new().with_name(rt_name);
            for (name, mem, time) in combined.iter() {
                if name == &rt_name {
                    let memory_limit = if **mem == u32::default() {
                        None
                    } else {
                        Some(**mem as usize)
                    };
                    let time_limit = if **time == u32::default() {
                        None
                    } else {
                        Some(**time as usize)
                    };
                    rt.memory_limit = memory_limit;
                    rt.time_limit = time_limit;
                }
            }
            runtime_envs.push(rt);
        }

        Ok(self.with_runtime_envs(runtime_envs))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_session_context_builder::make_test_session_builder_parallel_task;
    use phymes_core::test_task::{make_runtime_env, make_state_tables};

    use super::*;

    #[test]
    fn test_to_from_arrow_tables() -> Result<()> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);

        // Make the builder
        let builder = make_test_session_builder_parallel_task()
            .with_name("")
            .with_runtime_envs(runtime_envs)
            .with_state(state);

        // Test to tables
        let tables = builder.to_arrow_tables(true, true, true, true, true)?;

        // Check the tables
        assert_eq!(
            tables.first().unwrap().get_name(),
            AvailableSubjects::SessionMermaid.to_string().as_str()
        );
        assert_eq!(
            tables.get(1).unwrap().get_name(),
            AvailableSubjects::SessionErrors.to_string().as_str()
        );
        assert_eq!(
            tables.get(2).unwrap().get_name(),
            AvailableSubjects::SessionMetrics.to_string().as_str()
        );
        assert_eq!(
            tables.get(3).unwrap().get_name(),
            AvailableSubjects::SessionTraces.to_string().as_str()
        );
        assert_eq!(
            tables.get(4).unwrap().get_name(),
            AvailableSubjects::SessionEvents.to_string().as_str()
        );
        assert_eq!(
            tables.get(5).unwrap().get_name(),
            AvailableSubjects::SessionSubjects.to_string().as_str()
        );
        assert_eq!(
            tables.get(6).unwrap().get_name(),
            AvailableSubjects::SessionTasks.to_string().as_str()
        );
        assert_eq!(
            tables.get(7).unwrap().get_name(),
            AvailableSubjects::SessionProcessors.to_string().as_str()
        );
        assert_eq!(
            tables.get(8).unwrap().get_name(),
            AvailableSubjects::SessionRuntimeEnvs.to_string().as_str()
        );
        assert_eq!(
            tables.get(9).unwrap().get_name(),
            AvailableSubjects::SessionTasksRunLog.to_string().as_str()
        );
        assert_eq!(
            tables.get(10).unwrap().get_name(),
            AvailableSubjects::SubjectsNumRows.to_string().as_str()
        );
        assert_eq!(
            tables.get(11).unwrap().get_name(),
            AvailableSubjects::SubjectsChangeLog.to_string().as_str()
        );

        // Test from tables
        let tables_test =
            SessionContextBuilder::from_arrow_tables(&tables.iter().collect::<Vec<_>>(), None)?
                .with_name("")
                .to_arrow_tables(true, true, true, true, true)?;

        // Check the tables
        assert_eq!(
            tables_test.first().unwrap().get_name(),
            tables.first().unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .get_column_as_vec_str("session_context_name"),
            tables
                .first()
                .unwrap()
                .get_column_as_vec_str("session_context_name")
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .get_column_as_vec_str("flowchart_diagram"),
            tables
                .first()
                .unwrap()
                .get_column_as_vec_str("flowchart_diagram")
        );
        // Contains the added subjects
        assert_ne!(
            tables_test
                .first()
                .unwrap()
                .get_column_as_vec_str("er_diagram"),
            tables.first().unwrap().get_column_as_vec_str("er_diagram")
        );
        assert_eq!(
            tables_test.get(1).unwrap().get_name(),
            tables.get(1).unwrap().get_name()
        );
        assert_eq!(
            tables_test.get(1).unwrap().get_column_as_vec_str("error"),
            tables.get(1).unwrap().get_column_as_vec_str("error")
        );

        assert_eq!(
            tables_test.get(2).unwrap().get_name(),
            tables.get(2).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("metric_name"),
            tables.get(2).unwrap().get_column_as_vec_str("metric_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("metric_value")?,
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("metric_value")?
        );
        assert_eq!(
            tables_test.get(2).unwrap().get_column_as_vec_str("labels"),
            tables.get(2).unwrap().get_column_as_vec_str("labels")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("id")?,
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("id")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("span_name"),
            tables.get(2).unwrap().get_column_as_vec_str("span_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("parent_name"),
            tables.get(2).unwrap().get_column_as_vec_str("parent_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("span_id")?,
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("span_id")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("parent_id")?,
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("parent_id")?
        );
        assert_eq!(
            tables_test.get(2).unwrap().get_column_as_vec_str("file"),
            tables.get(2).unwrap().get_column_as_vec_str("file")
        );
        assert_eq!(
            tables_test.get(2).unwrap().get_column_as_vec_str("thread"),
            tables.get(2).unwrap().get_column_as_vec_str("thread")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("function"),
            tables.get(2).unwrap().get_column_as_vec_str("function")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("line")?,
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("line")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("timestamp")?,
            tables
                .get(2)
                .unwrap()
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
                .get_column_as_vec_str("tracer_type"),
            tables.get(3).unwrap().get_column_as_vec_str("tracer_type")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("tracer_event"),
            tables.get(3).unwrap().get_column_as_vec_str("tracer_event")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("message_name"),
            tables.get(3).unwrap().get_column_as_vec_str("message_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("subject_name"),
            tables.get(3).unwrap().get_column_as_vec_str("subject_name")
        );
        assert_eq!(
            tables_test.get(3).unwrap().get_column_as_vec_str("labels"),
            tables.get(3).unwrap().get_column_as_vec_str("labels")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("id")?,
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("id")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("span_name"),
            tables.get(3).unwrap().get_column_as_vec_str("span_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("parent_name"),
            tables.get(3).unwrap().get_column_as_vec_str("parent_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("span_id")?,
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("span_id")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("parent_id")?,
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("parent_id")?
        );
        assert_eq!(
            tables_test.get(3).unwrap().get_column_as_vec_str("file"),
            tables.get(3).unwrap().get_column_as_vec_str("file")
        );
        assert_eq!(
            tables_test.get(3).unwrap().get_column_as_vec_str("thread"),
            tables.get(3).unwrap().get_column_as_vec_str("thread")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("function"),
            tables.get(3).unwrap().get_column_as_vec_str("function")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("line")?,
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("line")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("timestamp")?,
            tables
                .get(3)
                .unwrap()
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
                .get_column_as_vec_str("event_level"),
            tables.get(4).unwrap().get_column_as_vec_str("event_level")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("record_name"),
            tables.get(4).unwrap().get_column_as_vec_str("record_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("record_value"),
            tables.get(4).unwrap().get_column_as_vec_str("record_value")
        );
        assert_eq!(
            tables_test.get(4).unwrap().get_column_as_vec_str("labels"),
            tables.get(4).unwrap().get_column_as_vec_str("labels")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("id")?,
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("id")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("span_name"),
            tables.get(4).unwrap().get_column_as_vec_str("span_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("parent_name"),
            tables.get(4).unwrap().get_column_as_vec_str("parent_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("span_id")?,
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("span_id")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("parent_id")?,
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("parent_id")?
        );
        assert_eq!(
            tables_test.get(4).unwrap().get_column_as_vec_str("file"),
            tables.get(4).unwrap().get_column_as_vec_str("file")
        );
        assert_eq!(
            tables_test.get(4).unwrap().get_column_as_vec_str("thread"),
            tables.get(4).unwrap().get_column_as_vec_str("thread")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("function"),
            tables.get(4).unwrap().get_column_as_vec_str("function")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("line")?,
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("line")?
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("timestamp")?,
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_primitive::<i64>("timestamp")?
        );
        assert_eq!(
            tables_test.get(5).unwrap().get_name(),
            tables.get(5).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(5)
                .unwrap()
                .get_column_as_vec_str("subject_name"),
            tables.get(5).unwrap().get_column_as_vec_str("subject_name")
        );
        assert_eq!(
            tables_test
                .get(5)
                .unwrap()
                .get_column_as_vec_str("column_name"),
            tables.get(5).unwrap().get_column_as_vec_str("column_name")
        );
        assert_eq!(
            tables_test
                .get(5)
                .unwrap()
                .get_column_as_vec_str("type_name"),
            tables.get(5).unwrap().get_column_as_vec_str("type_name")
        );
        assert_eq!(
            tables_test.get(1).unwrap().get_name(),
            tables.get(1).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .get_column_as_vec_str("task_name"),
            tables.get(6).unwrap().get_column_as_vec_str("task_name")
        );
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .get_column_as_vec_str("processor_name"),
            tables
                .get(6)
                .unwrap()
                .get_column_as_vec_str("processor_name")
        );
        assert_eq!(
            tables_test
                .get(6)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name"),
            tables
                .get(6)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name")
        );
        assert_eq!(
            tables_test.get(7).unwrap().get_name(),
            tables.get(7).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_str("processor_name"),
            tables
                .get(7)
                .unwrap()
                .get_column_as_vec_str("processor_name")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_str("processor_type"),
            tables
                .get(7)
                .unwrap()
                .get_column_as_vec_str("processor_type")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_name"),
            tables
                .get(7)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_name")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_table_names"),
            tables
                .get(7)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_table_names")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_primitive::<u8>("is_subscription")?,
            tables
                .get(7)
                .unwrap()
                .get_column_as_vec_primitive::<u8>("is_subscription")?
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_str("subscribe_type"),
            tables
                .get(7)
                .unwrap()
                .get_column_as_vec_str("subscribe_type")
        );
        assert_eq!(
            tables_test
                .get(7)
                .unwrap()
                .get_column_as_vec_str("update_type"),
            tables.get(7).unwrap().get_column_as_vec_str("update_type")
        );
        assert_eq!(
            tables_test.get(8).unwrap().get_name(),
            tables.get(8).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(8)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name"),
            tables
                .get(8)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name")
        );
        assert_eq!(
            tables_test
                .get(8)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("memory_limit")?,
            tables
                .get(8)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("memory_limit")?
        );
        assert_eq!(
            tables_test
                .get(8)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("time_limit")?,
            tables
                .get(8)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("time_limit")?
        );

        Ok(())
    }
}
