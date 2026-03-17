use std::fmt::Debug;
use arrow::datatypes::SchemaRef;
use serde_json::{Map, Value};
use crate::{BuildableTrait, MappableTrait, ObjectStorageBackend, SubjectPlanBuilder};

pub trait SubjectPlanTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
    fn schema(&self) -> &SchemaRef;
    fn backend(&self) -> &ObjectStorageBackend;
    fn location(&self) -> &str;
    fn bucket(&self) -> &str;
    fn metadata(&self) -> &Map<String, Value>;
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
    ///   (defaults to the name of the subject)
    pub(crate) location: String,
    /// The object store bucket (or container or root)
    pub(crate) bucket: String,
    /// Metadata including e_tag, version, size, last_modified, etc.
    pub(crate) metadata: Map<String, Value>,
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

    fn location(&self) -> &str {
        &self.location
    }

    fn bucket(&self) -> &str {
        &self.bucket
    }

    fn metadata(&self) -> &Map<String, Value> {
        &self.metadata
    }
}