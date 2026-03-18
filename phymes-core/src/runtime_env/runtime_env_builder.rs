use std::fmt::Debug;
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use crate::{BuilderTrait, ObjectStorageBackend, RuntimeEnv, SubjectFilePartition, SubjectFolderPartition};

pub trait RuntimeEnvBuilderTrait: BuilderTrait + Debug + Send + Sync {
    fn with_memory_limit(self, limit: usize) -> Self;
    fn with_time_limit(self, limit: usize) -> Self;
    fn with_object_store_backend(self, backend: &ObjectStorageBackend) -> Self;
    fn with_object_store_bucket(self, bucket: &str) -> Self;
    fn with_object_store_backend_config(self, config: &Map<String, Value>) -> Self;
    fn add_object_store_backend_config(self, key: &str, value: &Value) -> Self;
    fn with_subject_folder_partitioning(self, partitioning: &SubjectFolderPartition) -> Self;
    fn with_subject_file_partitioning(self, partitioning: &SubjectFilePartition) -> Self;
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct RuntimeEnvBuilder {
    /// Runtime environment name
    pub name: Option<String>,
    /// The max allowable memory
    pub memory_limit: Option<usize>,
    /// the max allowable compute time
    pub time_limit: Option<usize>,
    /// Subject object store backend [ObjectStorageBackend]
    pub object_store_backend: Option<ObjectStorageBackend>,
    /// The object store bucket (or container or root)
    pub object_store_bucket: Option<String>,
    /// Additional backend configuration options not in the environmental variables
    pub object_store_backend_config: Option<Map<String,Value>>,
    /// The subject folder partitioning
    pub subject_folder_partitioning: Option<SubjectFolderPartition>,
    /// The subject folder partitioning
    pub subject_file_partitioning: Option<SubjectFilePartition>,
}

impl BuilderTrait for RuntimeEnvBuilder {
    type T = RuntimeEnv;

    fn new() -> Self {
        Self {
            name: None,
            memory_limit: None,
            time_limit: None,
            object_store_backend: None,
            object_store_bucket: None,
            object_store_backend_config: None,
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
            name: self.name.ok_or(anyhow!("Please define the name before trying to build the runtime Env!"))?,
            memory_limit: self.memory_limit.unwrap_or_default(),
            time_limit: self.time_limit.unwrap_or_default(),
            object_store_backend: self.object_store_backend.unwrap_or_default(),
            object_store_bucket: self.object_store_bucket.unwrap_or_default(),
            object_store_backend_config: self.object_store_backend_config.unwrap_or_default(),
            subject_folder_partitioning: self.subject_folder_partitioning.unwrap_or_default(),
            subject_file_partitioning: self.subject_file_partitioning.unwrap_or_default(),
        };
        Ok(t)
    }
}

impl RuntimeEnvBuilderTrait for RuntimeEnvBuilder {

    fn with_object_store_backend(mut self, backend: &ObjectStorageBackend) -> Self {
        self.object_store_backend = Some(backend.to_owned());
        self
    }

    fn with_object_store_bucket(mut self, bucket: &str) -> Self {
        self.object_store_bucket = Some(bucket.to_string());
        self
    }

    fn with_object_store_backend_config(mut self, config: &Map<String, Value>) -> Self {
        self.object_store_backend_config = Some(config.to_owned());
        self
    }

    fn add_object_store_backend_config(mut self, k: &str, v: &Value) -> Self {
        let mut config = if let Some(config) = self.object_store_backend_config {
            config
        } else {
            Map::<String, Value>::new()
        };
        let _ = config.insert(k.to_string(), v.to_owned());
        self.object_store_backend_config = Some(config);
        self
    }
    
    fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }
    
    fn with_time_limit(mut self, limit: usize) -> Self {
        self.time_limit = Some(limit);
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
}