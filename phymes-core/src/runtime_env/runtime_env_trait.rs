use std::fmt::Debug;

use serde_json::{Map, Value};

use crate::{BuildableTrait, MappableTrait, ObjectStorageBackend, RuntimeEnvBuilder, SubjectFilePartition, SubjectFolderPartition};

/// `BuidableTrait` + `BuilderTraint` - `get_builder` - `build`
///
/// # Notes
/// * A work in progress...
/// * Missing methods for specifying the device or number of devices
/// * Missing methods for disk usage and access
pub trait RuntimeEnvTrait: BuildableTrait + MappableTrait + Send + Sync {
    fn new(name: &str, memory_limit: usize, time_limit: usize, object_store_backend: &ObjectStorageBackend, object_store_bucket: &str, object_store_backend_config: &Map<String,Value>, subject_folder_partitioning: &SubjectFolderPartition, subject_file_partitioning: &SubjectFilePartition) -> Self;
}

#[derive(Default, Debug, PartialEq)]
pub struct RuntimeEnv {
    /// name for the runtime environment config
    pub name: String,
    /// the max allowable memory
    pub memory_limit: usize,
    /// the max allowable compute time
    pub time_limit: usize,
    /// Subject object store backend [ObjectStorageBackend]
    pub object_store_backend: ObjectStorageBackend,
    /// The object store bucket (or container or root)
    pub object_store_bucket: String,
    /// Additional backend configuration options not in the environmental variables
    pub object_store_backend_config: Map<String,Value>,
    /// The subject folder partitioning
    pub subject_folder_partitioning: SubjectFolderPartition,
    /// The subject folder partitioning
    pub subject_file_partitioning: SubjectFilePartition,
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
    fn new(name: &str, memory_limit: usize, time_limit: usize, object_store_backend: &ObjectStorageBackend, object_store_bucket: &str, object_store_backend_config: &Map<String,Value>, subject_folder_partitioning: &SubjectFolderPartition, subject_file_partitioning: &SubjectFilePartition) -> Self {
        Self { name: name.to_string(), 
            memory_limit, time_limit, 
            object_store_backend: object_store_backend.to_owned(), 
            object_store_bucket: object_store_bucket.to_owned(), 
            object_store_backend_config: object_store_backend_config.to_owned(), 
            subject_folder_partitioning: subject_folder_partitioning.to_owned(), 
            subject_file_partitioning: subject_file_partitioning.to_owned() }
    }
}
