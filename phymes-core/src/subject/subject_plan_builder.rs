use std::fmt::Debug;
use anyhow::{anyhow, Result};
use arrow::datatypes::SchemaRef;
use serde_json::{Map, Value};
use crate::{BuilderTrait, ObjectStorageBackend, SubjectConstraint, SubjectPlan};

pub trait SubjectPlanBuilderTrait: BuilderTrait + Debug + Send + Sync {
    fn with_schema(self, schema: SchemaRef) -> Self;
    fn with_backend(self, backend: &ObjectStorageBackend) -> Self;
    fn with_locations(self, locations: &[&str]) -> Self;
    fn with_bucket(self, bucket: &str) -> Self;
    fn with_metadata(self, metadata: &Map<String, Value>) -> Self;
    fn add_metadata(self, key: &str, value: &Value) -> Self;
    fn with_constraints(self, constraints: &[SubjectConstraint]) -> Self;
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct SubjectPlanBuilder {
    /// Subject name
    pub name: Option<String>,
    /// Subject schema
    pub schema: Option<SchemaRef>,
    /// Subject object store backend [ObjectStorageBackend]
    pub backend: Option<ObjectStorageBackend>,
    /// Location within the bucket that the subject data partitions are
    ///   (defaults to the name of the subject with an "ipc" extension)
    pub locations: Option<Vec<String>>,
    /// The object store bucket (or container or root)
    pub bucket: Option<String>,
    /// Object store metadata including e_tag, version, size, last_modified, etc.
    pub metadata: Option<Map<String, Value>>,
    /// Constraints on the subject
    pub constraints: Option<Vec<SubjectConstraint>>,
}

impl BuilderTrait for SubjectPlanBuilder {
    type T = SubjectPlan;

    fn new() -> Self {
        Self {
            name: None,
            schema: None,
            backend: None,
            locations: None,
            bucket: None,
            metadata: None,
            constraints: None,
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
            name: self.name.ok_or(anyhow!("Please define the name before trying to build the subject plan!"))?,
            schema: self.schema.ok_or(anyhow!("Please define the schema before trying to build the subject plan!"))?,
            backend: self.backend.ok_or(anyhow!("Please define the backend before trying to build the subject plan!"))?,
            locations: self.locations.unwrap_or_default(),
            bucket: self.bucket.ok_or(anyhow!("Please define the bucket before trying to build the subject plan!"))?,
            metadata: self.metadata.unwrap_or_default(),
            constraints: self.constraints.unwrap_or_default(),
        };
        Ok(t)
    }
}

impl SubjectPlanBuilderTrait for SubjectPlanBuilder {
    fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    fn with_backend(mut self, backend: &ObjectStorageBackend) -> Self {
        self.backend = Some(backend.to_owned());
        self
    }

    fn with_locations(mut self, locations: &[&str]) -> Self {
        self.locations = Some(locations.into_iter().map(|s| s.to_string()).collect::<Vec<_>>());
        self
    }

    fn with_bucket(mut self, bucket: &str) -> Self {
        self.bucket = Some(bucket.to_string());
        self
    }

    fn with_metadata(mut self, metadata: &Map<String, Value>) -> Self {
        self.metadata = Some(metadata.to_owned());
        self
    }

    fn add_metadata(mut self, k: &str, v: &Value) -> Self {
        let mut metadata = if let Some(metadata) = self.metadata {
            metadata
        } else {
            Map::<String, Value>::new()
        };
        let _ = metadata.insert(k.to_string(), v.to_owned());
        self.metadata = Some(metadata);
        self
    }

    fn with_constraints(mut self, constraints: &[SubjectConstraint]) -> Self {
        self.constraints = Some(constraints.to_owned());
        self
    }
}