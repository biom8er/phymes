use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow::{array::{ArrayRef, RecordBatch, StringArray, UInt32Array}, datatypes::{DataType, Field, Schema}};
use phymes_core::{metrics::HashSet, session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, session_context_builder::{SessionContextBuilder, SessionContextBuilderTrait, TaskPlanBuilder}}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};

use crate::session_traits::mermaid_js::{from_data_type_to_str, from_str_to_data_type, SessionContextBuilderMermaidTrait};

/// Reserved table names for the [SessionContext]
pub enum SessionContextTableNames {
    Metrics,
    Tasks,
    Processors,
    Subjects,
    RuntimeEnvironments,
    MermaidJS,
}

impl MappableTrait for SessionContextTableNames {
    fn get_name(&self) -> &str {
        match self {
            Self::Metrics => "METRICS",
            Self::Tasks => "TASKS",
            Self::Processors => "PROCESSORS",
            Self::Subjects => "SUBJECTS",
            Self::RuntimeEnvironments => "RUNTIME_ENVIRONMENTS",
            Self::MermaidJS => "MERMAID_JS"
        }
    }
}

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
    /// * `Vec<ArrowTable>` with the SessionContext in tabular format
    fn to_arrow_tables(&self, include_subjects: bool, include_mermaid: bool) -> Result<Vec<ArrowTable>>;

    /// Get the subjects in tabular form
    fn get_subjects_as_table(&self) -> Result<ArrowTable>;

    /// Get the tasks in tabular form
    fn get_tasks_as_table(&self) -> Result<ArrowTable>;

    /// Get the processors in tabular form
    /// 
    /// # Note
    /// 
    /// * No sorting is performed when generating the table
    ///   so that order of processors is maintained
    fn get_processors_as_table(&self) -> Result<ArrowTable>;

    /// Get the runtime environments in tabular form
    fn get_runtime_envs_as_table(&self) -> Result<ArrowTable>;

    /// Get mermaid js chart strings
    fn get_mermaid_js_as_table(&self) -> Result<ArrowTable>;

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
    /// * `make_subjects` - Whether to make empty subject tables or not
    ///   Should be set to false if subject data is provided in addition to schema tables
    fn from_arrow_tables(tables: Vec<ArrowTable>, make_subjects: bool) -> Result<Self> where Self: Sized;

    fn with_subjects_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized;
    fn with_tasks_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized;
    fn with_processors_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized;
    fn with_runtime_envs_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized;

    
}

impl SessionContextBuilderTabularTrait for SessionContextBuilder {
    fn to_arrow_tables(&self, include_subjects: bool, include_mermaid: bool) -> Result<Vec<ArrowTable>> {
        let mut tables = vec![
            self.get_tasks_as_table()?,
            self.get_processors_as_table()?,
            self.get_runtime_envs_as_table()?,
        ];
        if include_subjects {
            tables.push(self.get_subjects_as_table()?);
        }
        if include_mermaid {
            tables.push(self.get_mermaid_js_as_table()?);
        }        
        Ok(tables)
    }

    fn from_arrow_tables(tables: Vec<ArrowTable>, make_subjects: bool) -> Result<Self> where Self: Sized {
        // initialize the builder
        let mut builder = Self::new();

        // extract the schema
        let mut state = Vec::new();
        for table in tables.into_iter() {
            if table.get_name() == SessionContextTableNames::Subjects.get_name() && make_subjects {
                builder = builder.with_subjects_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::Tasks.get_name() {
                builder = builder.with_tasks_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::Processors.get_name() {
                builder = builder.with_processors_as_tables(table)?;
            } else if table.get_name() == SessionContextTableNames::RuntimeEnvironments.get_name() {
                builder = builder.with_runtime_envs_as_tables(table)?;
            } else {
                state.push(table);
            }
        }

        // assume the rest is state
        if !state.is_empty() && !make_subjects {
            Ok(builder.with_state(state))
        } else {
            Ok(builder)
        }
    }    

    fn get_subjects_as_table(&self) -> Result<ArrowTable> {
        // Check that the state exists        
        if self.state.is_none() {
            return Err(anyhow!("Add state subjects before making the subject tables."));
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
                let type_name = from_data_type_to_str(&field.data_type());
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
        ArrowTable::get_builder()
            .with_name(SessionContextTableNames::Subjects.get_name())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_tasks_as_table(&self) -> Result<ArrowTable> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the tasks table."));
        }

        let mut task_names = Vec::new();
        let mut processor_names = Vec::new();
        let mut runtime_env_names = Vec::new();

        // Sort the tasks
        let mut sorted_tasks = self.tasks.as_ref().unwrap().iter().collect::<Vec<_>>();
        sorted_tasks.sort_by(|a, b| a.task_name.cmp(&b.task_name));
        for task in sorted_tasks.iter() {
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
        ArrowTable::get_builder()
            .with_name(SessionContextTableNames::Tasks.get_name())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn get_processors_as_table(&self) -> Result<ArrowTable> {
        if self.processors.is_none() {
            return Err(anyhow!("Add processors before making the Mermaid Flowchart."));
        }
        let mut processor_names = Vec::new();
        let mut subscription_names = Vec::new();
        let mut subscription_table_names = Vec::new();
        let mut publication_names = Vec::new();
        let mut publication_table_names = Vec::new();
        let mut subscribe_types = Vec::new();

        // Get the processors in order
        for processor in self.processors.as_ref().unwrap().iter() {
            let mut sub_tmp = Vec::new();
            let mut sub_tab_tmp = Vec::new();
            let mut pub_tmp = Vec::new();
            let mut pub_tab_tmp = Vec::new();
            for sub in processor.get_subscriptions() {
                sub_tmp.push(sub.get_name());
                sub_tab_tmp.push(sub.get_table_name());
            }
            for p in processor.get_publications() {
                pub_tmp.push(p.get_name());
                pub_tab_tmp.push(p.get_table_name());
            }
            processor_names.push(processor.get_name());
            subscribe_types.push(processor.get_subscribe().get_name());
            subscription_names.extend(sub_tmp);
            subscription_table_names.extend(sub_tab_tmp);
            publication_names.extend(pub_tmp);
            publication_table_names.extend(pub_tab_tmp);
        }

        // create the record batch
        let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
        let subscription_names: ArrayRef = Arc::new(StringArray::from(subscription_names));
        let subscription_table_names: ArrayRef = Arc::new(StringArray::from(subscription_table_names));
        let publication_names: ArrayRef = Arc::new(StringArray::from(publication_names));
        let publication_table_names: ArrayRef = Arc::new(StringArray::from(publication_table_names));
        let subscribe_types: ArrayRef = Arc::new(StringArray::from(subscribe_types));
        let batch = RecordBatch::try_from_iter(vec![
            ("processor_name", processor_names),
            ("subscription_name", subscription_names),
            ("subscription_subject_name", subscription_table_names),
            ("publication_name", publication_names),
            ("publication_subject_name", publication_table_names),
            ("subscribe_type", subscribe_types),
        ])?;

        // create the table
        ArrowTable::get_builder()
            .with_name(SessionContextTableNames::Processors.get_name())
            .with_record_batches(vec![batch])?
            .build()        
    }

    fn get_runtime_envs_as_table(&self) -> Result<ArrowTable> {
        if self.runtime_envs.is_none() {
            return Err(anyhow!("Add runtime environments before making the Mermaid Flowchart."));
        }
        let mut runtime_env_names = Vec::new();
        let mut memory_limits = Vec::new();
        let mut time_limits = Vec::new();

        // sort the runtime environments
        let mut sorted_rts = self.runtime_envs.as_ref().unwrap().iter().collect::<Vec<_>>();
        sorted_rts.sort_by(|a, b| a.name.cmp(&b.name));
        for rt in sorted_rts.iter() {
            runtime_env_names.push(rt.get_name());
            memory_limits.push(rt.memory_limit.clone().map(|v| v as u32));
            time_limits.push(rt.time_limit.clone().map(|v| v as u32));
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
        ArrowTable::get_builder()
            .with_name(SessionContextTableNames::RuntimeEnvironments.get_name())
            .with_record_batches(vec![batch])?
            .build()    
        
    }

    fn get_mermaid_js_as_table(&self) -> Result<ArrowTable> {
        let flowchart = self.to_mermaid_flowchart()?;
        let erdiagram = self.to_mermaid_erdiagram()?;

        // create the record batch
        let flowchart: ArrayRef = Arc::new(StringArray::from(vec![flowchart]));
        let erdiagram: ArrayRef = Arc::new(StringArray::from(vec![erdiagram]));
        let batch = RecordBatch::try_from_iter(vec![
            ("mermaid_js_flowchart", flowchart),
            ("mermaid_js_erdiagram", erdiagram),
        ])?;

        // create the table
        ArrowTable::get_builder()
            .with_name(SessionContextTableNames::MermaidJS.get_name())
            .with_record_batches(vec![batch])?
            .build()
    }

    fn with_subjects_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized {
        // extract arrays
        let subjects_vec_str = subjects.get_column_as_vec_str("subject_name");
        let columns_vec_str = subjects.get_column_as_vec_str("column_name");
        let types_vec_str = subjects.get_column_as_vec_str("type_name");

        // get unique subjects
        let subjects_unique = subjects_vec_str.iter().collect::<HashSet<_>>();
        let combined = subjects_vec_str.iter()
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
            let table = ArrowTable::get_builder().with_record_batches(vec![batch])?.with_name(&subject).build()?;
            state.push(table);
        }

        Ok(self.with_state(state))
    }

    fn with_tasks_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized {
        // extract arrays
        let tasks_vec_str = subjects.get_column_as_vec_str("task_name");
        let processors_vec_str = subjects.get_column_as_vec_str("processor_name");
        let runtime_envs_vec_str = subjects.get_column_as_vec_str("runtime_env_name");

        // get unique subjects
        let tasks_unique = tasks_vec_str.iter().collect::<HashSet<_>>();
        let combined = tasks_vec_str.iter()
            .zip(processors_vec_str.iter())
            .zip(runtime_envs_vec_str.iter())
            .map(|((x, y), z)| (x, y, z))
            .collect::<Vec<_>>();
        
        // build the task plans
        let mut tasks = Vec::new();
        for task in tasks_unique {
            let mut builder = TaskPlanBuilder::default();
            builder.task_name.replace(task.to_string());
            builder.processor_names.replace(Vec::new());
            for (t, p, r) in combined.iter() {
                if t == &task {
                    builder.processor_names.as_mut().unwrap().push(p.to_string());
                    builder.runtime_env_name.replace(r.to_string());
                }
            }
            tasks.push(builder.build()?);
        }

        Ok(self.with_tasks(tasks))
    }

    fn with_processors_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized {
        let _ = subjects;
        todo!()
    }

    fn with_runtime_envs_as_tables(self, subjects: ArrowTable) -> Result<Self> where Self: Sized {
        let _ = subjects;
        todo!()
    }
}