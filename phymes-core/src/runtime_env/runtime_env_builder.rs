use crate::{
    BuilderTrait, ObjectStorageBackend, RuntimeEnv, SubjectFilePartition, SubjectFolderPartition,
    make_store,
};
use anyhow::{Result, anyhow};
use object_store::ObjectStore;
use serde_json::{Map, Value};
use std::{fmt::Debug, sync::Arc};

pub trait RuntimeEnvBuilderTrait: BuilderTrait + Debug + Send + Sync {
    fn with_max_memory(self, max: usize) -> Self;
    fn with_max_time(self, max: usize) -> Self;
    fn with_max_steps(self, max: usize) -> Self;
    fn with_max_tasks(self, max: usize) -> Self;
    fn with_object_store_config(self, config: &Map<String, Value>) -> Self;
    fn add_object_store_config(self, key: &str, value: &Value) -> Self;
    fn with_object_store(self, store: Arc<dyn ObjectStore>) -> Self;
    fn with_object_store_backend(self, backend: &ObjectStorageBackend) -> Self;
    fn with_object_store_bucket(self, bucket: &str) -> Self;
    fn with_subject_folder_partitioning(self, partitioning: &SubjectFolderPartition) -> Self;
    fn with_subject_file_partitioning(self, partitioning: &SubjectFilePartition) -> Self;
}

#[derive(Default, Debug, Clone)]
pub struct RuntimeEnvBuilder {
    /// Runtime environment name
    pub name: Option<String>,
    /// The max allowable memory
    pub max_memory: Option<usize>,
    /// the max allowable compute time
    pub max_time: Option<usize>,
    /// the max number of superstep iterations
    pub max_steps: Option<usize>,
    /// the max number of concurrent tasks
    pub max_tasks: Option<usize>,
    /// The object store
    pub object_store: Option<Arc<dyn ObjectStore>>,
    /// Copy of the backend for the object store
    pub object_store_backend: Option<ObjectStorageBackend>,
    /// copy of the bucket for the object store
    pub object_store_bucket: Option<String>,
    /// Additional object store configuration options not in the environmental variables
    pub object_store_config: Option<Map<String, Value>>,
    /// The subject folder partitioning
    pub subject_folder_partitioning: Option<SubjectFolderPartition>,
    /// The subject folder partitioning
    pub subject_file_partitioning: Option<SubjectFilePartition>,
}

impl PartialEq for RuntimeEnvBuilder {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.max_memory == other.max_memory
            && self.max_time == other.max_time
            && self.max_steps == other.max_steps
            && self.max_tasks == other.max_tasks
            && self.object_store_backend == other.object_store_backend
            && self.object_store_bucket == other.object_store_bucket
            && self.object_store_config == other.object_store_config
            && self.subject_folder_partitioning == other.subject_folder_partitioning
            && self.subject_file_partitioning == other.subject_file_partitioning
    }
}

impl BuilderTrait for RuntimeEnvBuilder {
    type T = RuntimeEnv;

    fn new() -> Self {
        Self {
            name: None,
            max_memory: None,
            max_time: None,
            max_steps: None,
            max_tasks: None,
            object_store: None,
            object_store_backend: None,
            object_store_bucket: None,
            object_store_config: None,
            subject_folder_partitioning: None,
            subject_file_partitioning: None,
        }
    }

    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        let t = Self::T {
            name: self.name.ok_or(anyhow!(
                "Please define the name before trying to build the runtime Env!"
            ))?,
            max_memory: self.max_memory.unwrap_or_default(),
            max_time: self.max_time.unwrap_or_default(),
            max_steps: self.max_steps.unwrap_or(25),
            max_tasks: self.max_tasks.unwrap_or(8),
            object_store: self.object_store.unwrap_or(make_store(
                &ObjectStorageBackend::default(),
                None,
                None,
            )?),
            object_store_backend: self.object_store_backend.unwrap_or_default(),
            object_store_bucket: self.object_store_bucket.unwrap_or_default(),
            object_store_config: self.object_store_config.unwrap_or_default(),
            subject_folder_partitioning: self.subject_folder_partitioning.unwrap_or_default(),
            subject_file_partitioning: self.subject_file_partitioning.unwrap_or_default(),
        };
        Ok(t)
    }
}

impl RuntimeEnvBuilderTrait for RuntimeEnvBuilder {
    fn with_object_store_config(mut self, config: &Map<String, Value>) -> Self {
        self.object_store_config = Some(config.to_owned());
        self
    }

    fn add_object_store_config(mut self, k: &str, v: &Value) -> Self {
        let mut config = if let Some(config) = self.object_store_config {
            config
        } else {
            Map::<String, Value>::new()
        };
        let _ = config.insert(k.to_string(), v.to_owned());
        self.object_store_config = Some(config);
        self
    }

    fn with_max_memory(mut self, max: usize) -> Self {
        self.max_memory = Some(max);
        self
    }

    fn with_max_time(mut self, max: usize) -> Self {
        self.max_time = Some(max);
        self
    }

    fn with_subject_folder_partitioning(mut self, partitioning: &SubjectFolderPartition) -> Self {
        self.subject_folder_partitioning = Some(partitioning.to_owned());
        self
    }

    fn with_subject_file_partitioning(mut self, partitioning: &SubjectFilePartition) -> Self {
        self.subject_file_partitioning = Some(partitioning.to_owned());
        self
    }

    fn with_max_steps(mut self, max: usize) -> Self {
        self.max_steps = Some(max);
        self
    }

    fn with_max_tasks(mut self, max: usize) -> Self {
        self.max_tasks = Some(max);
        self
    }

    fn with_object_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.object_store = Some(store);
        self
    }

    fn with_object_store_backend(mut self, backend: &ObjectStorageBackend) -> Self {
        self.object_store_backend = Some(backend.to_owned());
        self
    }

    fn with_object_store_bucket(mut self, bucket: &str) -> Self {
        self.object_store_bucket = Some(bucket.to_string());
        self
    }
}
