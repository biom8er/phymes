use std::fmt::Debug;
use arrow::datatypes::SchemaRef;
use serde_json::{Map, Value};
use crate::{BuildableTrait, MappableTrait, ObjectStorageBackend, SubjectConstraint, SubjectPlanBuilder};

pub trait SubjectPlanTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
    fn schema(&self) -> &SchemaRef;
    fn backend(&self) -> &ObjectStorageBackend;
    fn locations(&self) -> &Vec<String>;
    fn bucket(&self) -> &str;
    fn metadata(&self) -> &Map<String, Value>;
    fn constraints(&self) -> &Vec<SubjectConstraint>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubjectPlan {
    /// Subject name
    pub(crate) name: String,
    /// Subject schema
    pub(crate) schema: SchemaRef,
    /// Subject object store backend [ObjectStorageBackend]
    pub(crate) backend: ObjectStorageBackend,
    /// Location within the bucket that the subject data partitions are
    /// 
    /// # Note
    /// * Not a one to one mapping: RecordBatch != location
    /// * Many to many mapping: Vec<RecordBatch> -> Multiple locations
    pub(crate) locations: Vec<String>,
    /// The object store bucket (or container or root)
    pub(crate) bucket: String,
    /// Metadata for each location/partition including e_tag, version, size, last_modified, etc.
    pub(crate) metadata: Map<String, Value>,
    /// Constraints on the subject
    pub(crate) constraints: Vec<SubjectConstraint>,
}

impl MappableTrait for SubjectPlan {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for SubjectPlan {
    type T = SubjectPlanBuilder;

    fn get_builder() -> Self::T
    where
        Self: Sized {
        Self::T::default()
    }
}

impl SubjectPlanTrait for SubjectPlan {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn backend(&self) -> &ObjectStorageBackend {
        &self.backend
    }

    fn locations(&self) -> &Vec<String> {
        &self.locations
    }

    fn bucket(&self) -> &str {
        &self.bucket
    }

    fn metadata(&self) -> &Map<String, Value> {
        &self.metadata
    }

    fn constraints(&self) -> &Vec<SubjectConstraint> {
        &self.constraints
    }
}