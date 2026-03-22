use std::{fmt::Debug, sync::Arc};

use object_store::ObjectStore;
use serde_json::{Map, Value};

use crate::{BuildableTrait, MappableTrait, ObjectStorageBackend, RuntimeEnvBuilder, SubjectFilePartition, SubjectFolderPartition, make_store};

/// # Notes
/// * A work in progress...
/// * Missing methods for specifying the device or number of devices
/// * Missing methods for disk usage and access
pub trait RuntimeEnvTrait: BuildableTrait + MappableTrait + Send + Sync {
    fn new(name: &str, max_memory: usize, max_time: usize, max_steps: usize, max_tasks: usize, object_store: Arc<dyn ObjectStore>, object_store_backend: &ObjectStorageBackend, object_store_bucket: &str, object_store_config: &Map<String,Value>, subject_folder_partitioning: &SubjectFolderPartition, subject_file_partitioning: &SubjectFilePartition) -> Self;
    fn max_steps(&self) -> usize;
    fn max_tasks(&self) -> usize;
    fn object_store(&self) -> &Arc<dyn ObjectStore>;
}

/// The runtime environment for the session
#[derive(Debug)]
pub struct RuntimeEnv {
    /// name for the runtime environment config
    pub name: String,
    /// the max allowable memory
    pub max_memory: usize,
    /// the max allowable compute time
    pub max_time: usize,
    /// the max number of superstep iterations
    pub max_steps: usize,
    /// the max number of concurrent tasks
    pub max_tasks: usize,
    /// The object store
    pub object_store: Arc<dyn ObjectStore>,
    /// Copy of the backend for the object store
    pub object_store_backend: ObjectStorageBackend,
    /// copy of the bucket for the object store
    pub object_store_bucket: String,
    /// Additional backend configuration options not in the environmental variables
    pub object_store_config: Map<String,Value>,
    /// The subject folder partitioning
    pub subject_folder_partitioning: SubjectFolderPartition,
    /// The subject folder partitioning
    pub subject_file_partitioning: SubjectFilePartition,
}

impl Default for RuntimeEnv {
    fn default() -> Self {
        Self { 
            name: Default::default(), 
            max_memory: Default::default(), 
            max_time: Default::default(), 
            max_steps: 25, 
            max_tasks: 8, 
            object_store: make_store(&ObjectStorageBackend::default(), None, None).unwrap(), 
            object_store_backend: ObjectStorageBackend::default(),
            object_store_bucket: String::new(),
            object_store_config: Default::default(), 
            subject_folder_partitioning: Default::default(), 
            subject_file_partitioning: Default::default() 
        }
    }
}

impl PartialEq for RuntimeEnv {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.max_memory == other.max_memory && self.max_time == other.max_time && self.max_steps == other.max_steps && self.max_tasks == other.max_tasks && self.object_store_backend == other.object_store_backend && self.object_store_bucket == other.object_store_bucket && self.object_store_config == other.object_store_config && self.subject_folder_partitioning == other.subject_folder_partitioning && self.subject_file_partitioning == other.subject_file_partitioning
    }
}

impl MappableTrait for RuntimeEnv {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for RuntimeEnv {
    type T = RuntimeEnvBuilder;

    fn get_builder() -> Self::T
    where
        Self: Sized {
        Self::T::default()
    }
}

impl RuntimeEnvTrait for RuntimeEnv {
    fn new(name: &str, max_memory: usize, max_time: usize, max_steps: usize, max_tasks: usize, object_store: Arc<dyn ObjectStore>, object_store_backend: &ObjectStorageBackend, object_store_bucket: &str, object_store_config: &Map<String,Value>, subject_folder_partitioning: &SubjectFolderPartition, subject_file_partitioning: &SubjectFilePartition) -> Self {
        Self { name: name.to_string(), 
            max_memory, max_time, max_steps, max_tasks,
            object_store,
            object_store_config: object_store_config.to_owned(), 
            object_store_backend: object_store_backend.to_owned(),
            object_store_bucket: object_store_bucket.to_owned(),
            subject_folder_partitioning: subject_folder_partitioning.to_owned(), 
            subject_file_partitioning: subject_file_partitioning.to_owned() }
    }
    
    fn max_steps(&self) -> usize {
        self.max_steps
    }
    
    fn max_tasks(&self) -> usize {
        self.max_tasks
    }
    
    fn object_store(&self) -> &Arc<dyn ObjectStore> {
        &self.object_store
    }
}
