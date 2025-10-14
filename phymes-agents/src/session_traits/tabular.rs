use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt8Array, UInt32Array},
    datatypes::{Field, Schema},
};
use clap::ValueEnum;
use phymes_core::{
    schemas::{available_subjects::{AvailableSubjects, AvailableSubjectsTrait}, mermaid::create_mermaid_batch}, session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context::SessionContextTableNames,
        session_context_builder::{
            SessionContextBuilder, SessionContextBuilderTrait, TaskPlanBuilder,
        },
    }, table::{
        data_types::{from_data_type_to_str, from_str_to_data_type}, table_publish::TablePublish, table_subscribe::{from_str_to_subscribe, TableSubscribe}, table_trait::{Table, TableBuilderTrait, TableTrait}
    }, task::processor::ProcessorBuilder
};
use phymes_diagnostics::{create_timestamp_micros, HashSet};

use crate::{
    session_plans::available_processors::AvailableProcessors,
    session_traits::mermaid::SessionContextBuilderMermaidTrait,
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
    /// * `include_subjects` - whether to include the subject data or not
    /// * `include_mermaid` - whether to include the mermaid flowchart and erDiagrams or not
    ///
    /// # Returns
    ///
    /// * `Vec<ArrowTable>` with the SessionContext in tabular format and Optional `Vec<ArrowTable>` with the state
    fn to_arrow_tables(
        &self,
        include_subjects: bool,
        include_mermaid: bool,
        include_errors: bool,
    ) -> Result<(Vec<Table>, Option<Vec<Table>>)>;

    /// Get the subjects in tabular form
    fn get_subjects_as_table(&self) -> Result<Table>;

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

    /// Create the session from tables
    ///
    /// # Notes
    ///
    /// * Minimally, the meta tables describing the SessionContext schema must be included
    /// * Optionally, the subject tables will be populated with data if the state tables are included
    /// * Mermaid_js scripts are ignored
    ///
    /// # Arguments
    ///
    /// * `tables` - List of [ArrowTable]s describing the [SessionContext] schema with
    ///   optional subject tables with the actual data
    /// * `state` - Optionally the subject data. If none the subject tables will be initialized.
    fn from_arrow_tables(tables: &[&Table], state: Option<Vec<Table>>) -> Result<Self>
    where
        Self: Sized;

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
        include_subjects: bool,
        include_mermaid: bool,
        include_errors: bool,
    ) -> Result<(Vec<Table>, Option<Vec<Table>>)> {
        let mut tables = vec![
            self.get_subjects_as_table()?,
            self.get_tasks_as_table()?,
            self.get_processors_as_table()?,
            self.get_runtime_envs_as_table()?,
        ];
        if include_mermaid {
            tables.push(self.get_mermaid_js_as_table()?);
        }
        if include_errors {
            tables.push(AvailableSubjects::Errors.to_table(None, None)?);
        }
        let state = if include_subjects {
            self.state.clone()
        } else {
            None
        };
        Ok((tables, state))
    }

    fn from_arrow_tables(tables: &[&Table], mut state: Option<Vec<Table>>) -> Result<Self>
    where
        Self: Sized,
    {
        // initialize the builder
        let mut builder = Self::new();

        // extract the schema
        for table in tables {
            if table.get_name() == SessionContextTableNames::Subjects.to_string().as_str() && state.is_none()
            {
                builder = builder.with_subjects_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::Tasks.to_string().as_str() {
                builder = builder.with_tasks_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::Processors.to_string().as_str() {
                builder = builder.with_processors_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::RuntimeEnvironments.to_string().as_str() {
                builder = builder.with_runtime_envs_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::MermaidJS.to_string().as_str() {
                continue;
            } else if table.get_name() == SessionContextTableNames::Errors.to_string().as_str() {
                continue;
            } else if table.get_name() == SessionContextTableNames::Traces.to_string().as_str() {
                continue;
            } else if table.get_name() == SessionContextTableNames::Metrics.to_string().as_str() {
                continue;
            } else if table.get_name() == SessionContextTableNames::Events.to_string().as_str() {
                continue;
            } else if table.get_name() == SessionContextTableNames::MetricPivot.to_string().as_str() {
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

    fn get_subjects_as_table(&self) -> Result<Table> {
        // Check that the state exists
        if self.state.is_none() {
            return Err(anyhow!(
                "Add state subjects before making the subject tables."
            ));
        }

        let mut subject_names = Vec::new();
        let mut cols_names = Vec::new();
        let mut type_names = Vec::new();

        // Sort the hashmap
        let mut sorted_state = self.state.as_ref().unwrap().iter().collect::<Vec<_>>();
        sorted_state.sort_by(|a, b| a.get_name().cmp(b.get_name()));
        for subject in sorted_state.iter() {
            let fields = subject.get_schema().fields().clone();
            for field in fields.iter() {
                let type_name = from_data_type_to_str(field.data_type());
                subject_names.push(subject.get_name().to_string());
                cols_names.push(field.name().to_string());
                type_names.push(type_name);
            }
        }

        // create the record batch
        let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
        let cols_names: ArrayRef = Arc::new(StringArray::from(cols_names));
        let type_names: ArrayRef = Arc::new(StringArray::from(type_names));
        let batch = RecordBatch::try_from_iter(vec![
            ("subject_name", subject_names),
            ("column_name", cols_names),
            ("type_name", type_names),
        ])?;

        // create the table
        Table::get_builder()
            .with_name(SessionContextTableNames::Subjects.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_tasks_as_table(&self) -> Result<Table> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the tasks table."));
        }

        let mut task_names = Vec::new();
        let mut processor_names = Vec::new();
        let mut runtime_env_names = Vec::new();

        // extract the tasks in order
        for task in self.tasks.as_ref().unwrap().iter() {
            for p in task.processor_names.iter() {
                task_names.push(task.task_name.to_owned());
                processor_names.push(p.to_string());
                runtime_env_names.push(task.runtime_env_name.to_owned());
            }
        }

        // create the record batch
        let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
        let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
        let runtime_env_names: ArrayRef = Arc::new(StringArray::from(runtime_env_names));
        let batch = RecordBatch::try_from_iter(vec![
            ("task_name", task_names),
            ("processor_name", processor_names),
            ("runtime_env_name", runtime_env_names),
        ])?;

        // create the table
        Table::get_builder()
            .with_name(SessionContextTableNames::Tasks.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_processors_as_table(&self) -> Result<Table> {
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processors before making the Mermaid Flowchart."
            ));
        }
        let mut processor_names = Vec::new();
        let mut processor_types = Vec::new();
        let mut pub_sub_name = Vec::new();
        let mut pub_sub_table_names = Vec::new();
        let mut is_sub = Vec::new();
        let mut subscribe_types = Vec::new();

        // extract the processors in order
        for processor in self.processors.as_ref().unwrap().iter() {
            for sub in processor.get_subscriptions() {
                pub_sub_name.push(sub.get_name());
                pub_sub_table_names.push(sub.get_table_name());
                is_sub.push(1);
                processor_names.push(processor.get_name());
                processor_types.push(processor.get_type());
                subscribe_types.push(processor.get_subscribe().get_name());
            }
            for p in processor.get_publications() {
                pub_sub_name.push(p.get_name());
                pub_sub_table_names.push(p.get_table_name());
                is_sub.push(0);
                processor_names.push(processor.get_name());
                processor_types.push(processor.get_type());
                subscribe_types.push(processor.get_subscribe().get_name());
            }
        }

        // create the record batch
        let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
        let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
        let pub_sub_name: ArrayRef = Arc::new(StringArray::from(pub_sub_name));
        let pub_sub_table_names: ArrayRef = Arc::new(StringArray::from(pub_sub_table_names));
        let is_sub: ArrayRef = Arc::new(UInt8Array::from(is_sub));
        let subscribe_types: ArrayRef = Arc::new(StringArray::from(subscribe_types));
        let batch = RecordBatch::try_from_iter(vec![
            ("processor_name", processor_names),
            ("processor_type", processor_types),
            ("publication_subscription_name", pub_sub_name),
            ("publication_subscription_table_names", pub_sub_table_names),
            ("is_subscription", is_sub),
            ("subscribe_type", subscribe_types),
        ])?;

        // create the table
        Table::get_builder()
            .with_name(SessionContextTableNames::Processors.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_runtime_envs_as_table(&self) -> Result<Table> {
        if self.runtime_envs.is_none() {
            return Err(anyhow!(
                "Add runtime environments before making the Mermaid Flowchart."
            ));
        }
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
            runtime_env_names.push(rt.get_name());
            let memory_limit = rt.memory_limit.unwrap_or_default();
            memory_limits.push(memory_limit as u32);
            let time_limit = rt.time_limit.unwrap_or_default();
            time_limits.push(time_limit as u32);
        }

        // create the record batch
        let runtime_env_names: ArrayRef = Arc::new(StringArray::from(runtime_env_names));
        let memory_limits: ArrayRef = Arc::new(UInt32Array::from(memory_limits));
        let time_limits: ArrayRef = Arc::new(UInt32Array::from(time_limits));
        let batch = RecordBatch::try_from_iter(vec![
            ("runtime_env_name", runtime_env_names),
            ("memory_limit", memory_limits),
            ("time_limit", time_limits),
        ])?;

        // create the table
        Table::get_builder()
            .with_name(SessionContextTableNames::RuntimeEnvironments.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_mermaid_js_as_table(&self) -> Result<Table> {
        let flowchart_diagram = self.to_mermaid_flowchart()?;
        let er_diagram = self.to_mermaid_erdiagram()?;
        let session_context_name = self.name.as_ref().unwrap().to_string();
        let timestamp = create_timestamp_micros();

        // create the record batch
        let batch = create_mermaid_batch(
            vec![session_context_name], 
            vec![flowchart_diagram], 
            vec![er_diagram], 
            vec![timestamp])?;

        // create the table
        Table::get_builder()
            .with_name(SessionContextTableNames::MermaidJS.to_string().as_str())
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
            let mut builder = TaskPlanBuilder::default();
            builder.task_name.replace(task.to_string());
            builder.processor_names.replace(Vec::new());
            for (t, p, r) in combined.iter() {
                if t == &task {
                    builder
                        .processor_names
                        .as_mut()
                        .unwrap()
                        .push(p.to_string());
                    builder.runtime_env_name.replace(r.to_string());
                }
            }
            tasks.push(builder.build()?);
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
        let pub_sub_vec_str = procesors.get_column_as_vec_str("publication_subscription_name");
        let pub_sub_tab_name_vec_str =
            procesors.get_column_as_vec_str("publication_subscription_table_names");
        let is_sub_vec = procesors.get_column_as_vec_primitive::<u8>("is_subscription")?;

        // get unique processors while preserving order
        let mut processors_unique = processor_vec_str.iter().collect::<HashSet<_>>();
        let mut sort_processors = Vec::new();
        for processor_name in processor_vec_str.iter() {
            if processors_unique.contains(processor_name)
                && processors_unique.remove(processor_name)
            {
                sort_processors.push(processor_name);
            }
        }
        let combined = processor_vec_str
            .iter()
            .zip(type_vec_str.iter())
            .zip(subscribe_vec_str.iter())
            .zip(pub_sub_vec_str.iter())
            .zip(pub_sub_tab_name_vec_str.iter())
            .zip(is_sub_vec.iter())
            .map(|(((((a, b), c), d), e), f)| (a, b, c, d, e, f))
            .collect::<Vec<_>>();

        // build the processors in order
        let mut processors = Vec::new();
        for processor_name in sort_processors {
            let mut builder = ProcessorBuilder::default();
            builder.processor_name.replace(processor_name.to_string());
            builder.subscriptions.replace(Vec::new());
            builder.publications.replace(Vec::new());
            for (name, t, s_t, sub, sub_tab, is_sub) in combined.iter() {
                if name == &processor_name {
                    if **is_sub == 1 {
                        let subscription = TableSubscribe::from_str(sub, sub_tab)?;
                        builder.subscriptions.as_mut().unwrap().push(subscription);
                    } else {
                        let publication = TablePublish::from_str(sub, sub_tab)?;
                        builder.publications.as_mut().unwrap().push(publication);
                    }
                    let subscribe = from_str_to_subscribe(s_t)?;
                    builder.processor_type.replace(t.to_string());
                    builder.subscribe.replace(subscribe);
                }
            }
            let available_processor = AvailableProcessors::from_str(
                builder.processor_type.as_ref().unwrap().as_str(),
                false
            )
            .unwrap();
            let processor = available_processor.build_with_builder(builder)?;
            processors.push(processor);
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
    use phymes_core::{
        session::session_context_builder::test_session_context_builder::make_test_session_builder_parallel_task,
        task::task_trait::test_task::{make_runtime_env, make_state_tables},
    };

    use super::*;

    // DM: need to check for table values in the first test section
    // DM: need to test with state
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
        let (tables, state) = builder.to_arrow_tables(false, true, true)?;

        // Check the tables
        assert_eq!(
            tables.first().unwrap().get_name(),
            SessionContextTableNames::Subjects.to_string().as_str()
        );
        assert_eq!(
            tables.get(1).unwrap().get_name(),
            SessionContextTableNames::Tasks.to_string().as_str()
        );
        assert_eq!(
            tables.get(2).unwrap().get_name(),
            SessionContextTableNames::Processors.to_string().as_str()
        );
        assert_eq!(
            tables.get(3).unwrap().get_name(),
            SessionContextTableNames::RuntimeEnvironments.to_string().as_str()
        );
        assert_eq!(
            tables.get(4).unwrap().get_name(),
            SessionContextTableNames::MermaidJS.to_string().as_str()
        );
        assert_eq!(
            tables.get(5).unwrap().get_name(),
            SessionContextTableNames::Errors.to_string().as_str()
        );

        // Test from tables
        let (tables_test, _state_test) =
            SessionContextBuilder::from_arrow_tables(&tables.iter().collect::<Vec<_>>(), state)?
                .with_name("")
                .to_arrow_tables(false, true, true)?;

        // Check the tables
        assert_eq!(
            tables_test.first().unwrap().get_name(),
            tables.first().unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .get_column_as_vec_str("subject_name"),
            tables
                .first()
                .unwrap()
                .get_column_as_vec_str("subject_name")
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .get_column_as_vec_str("column_name"),
            tables.first().unwrap().get_column_as_vec_str("column_name")
        );
        assert_eq!(
            tables_test
                .first()
                .unwrap()
                .get_column_as_vec_str("type_name"),
            tables.first().unwrap().get_column_as_vec_str("type_name")
        );
        assert_eq!(
            tables_test.get(1).unwrap().get_name(),
            tables.get(1).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(1)
                .unwrap()
                .get_column_as_vec_str("task_name"),
            tables.get(1).unwrap().get_column_as_vec_str("task_name")
        );
        assert_eq!(
            tables_test
                .get(1)
                .unwrap()
                .get_column_as_vec_str("processor_name"),
            tables
                .get(1)
                .unwrap()
                .get_column_as_vec_str("processor_name")
        );
        assert_eq!(
            tables_test
                .get(1)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name"),
            tables
                .get(1)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name")
        );
        assert_eq!(
            tables_test.get(2).unwrap().get_name(),
            tables.get(2).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("processor_name"),
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_str("processor_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("processor_type"),
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_str("processor_type")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_name"),
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_name")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_table_names"),
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_str("publication_subscription_table_names")
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<u8>("is_subscription")?,
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_primitive::<u8>("is_subscription")?
        );
        assert_eq!(
            tables_test
                .get(2)
                .unwrap()
                .get_column_as_vec_str("subscribe_type"),
            tables
                .get(2)
                .unwrap()
                .get_column_as_vec_str("subscribe_type")
        );
        assert_eq!(
            tables_test.get(3).unwrap().get_name(),
            tables.get(3).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name"),
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_str("runtime_env_name")
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("memory_limit")?,
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("memory_limit")?
        );
        assert_eq!(
            tables_test
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("time_limit")?,
            tables
                .get(3)
                .unwrap()
                .get_column_as_vec_primitive::<u32>("time_limit")?
        );
        assert_eq!(
            tables_test.get(4).unwrap().get_name(),
            tables.get(4).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("session_context_name"),
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_str("session_context_name")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("flowchart_diagram"),
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_str("flowchart_diagram")
        );
        assert_eq!(
            tables_test
                .get(4)
                .unwrap()
                .get_column_as_vec_str("er_diagram"),
            tables
                .get(4)
                .unwrap()
                .get_column_as_vec_str("er_diagram")
        );        
        assert_eq!(
            tables_test.get(5).unwrap().get_name(),
            tables.get(5).unwrap().get_name()
        );
        assert_eq!(
            tables_test
                .get(5)
                .unwrap()
                .get_column_as_vec_str("error"),
            tables
                .get(5)
                .unwrap()
                .get_column_as_vec_str("error")
        );

        Ok(())
    }
}