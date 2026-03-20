use std::{fmt::Debug, sync::Arc};
use anyhow::{anyhow, Result};
use arrow::datatypes::Schema;
use crate::{BuildableTrait, BuilderTrait, IndexType, MappableTrait, Subject, SubjectBuilderTrait, SubjectConstraintType, SubjectPlan, SubjectSequenceType};

pub trait SubjectPlanBuilderTrait: BuilderTrait + Debug + Send + Sync {
    fn with_subject(self, subject: Subject) -> Self;
    fn with_constraints(self, constraints: &[SubjectConstraintType]) -> Self;
    fn with_indices(self, indices: &[IndexType]) -> Self;
    fn with_sequences(self, sequences: &[SubjectSequenceType]) -> Self;
    fn get_name(&self) -> Result<String>;
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct SubjectPlanBuilder {
    /// Subject subject
    pub subject: Option<Subject>,
    /// Constraints on the subject
    pub constraints: Option<Vec<SubjectConstraintType>>,
    /// Indexes on the subject
    pub indices: Option<Vec<IndexType>>,
    /// Sequences on the subject
    pub sequences: Option<Vec<SubjectSequenceType>>,
}

impl BuilderTrait for SubjectPlanBuilder {
    type T = SubjectPlan;

    fn new() -> Self {
        Self {
            subject: None,
            constraints: None,
            indices: None,
            sequences: None,
        }
    }

    fn with_name(mut self, name: &str) -> Self {
        let subject = Subject::get_builder()
            .with_name(name)
            .with_schema(Arc::new(Schema::empty()))
            .with_record_batches(Vec::new()).unwrap()
            .build().unwrap();
        self.subject = Some(subject);
        self
    }

    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        let t = Self::T {
            subject: self.subject.ok_or(anyhow!("Please define the subject before trying to build the subject plan!"))?,
            constraints: self.constraints.unwrap_or_default(),
            indices: self.indices.unwrap_or_default(),
            sequences: self.sequences.unwrap_or_default(),
        };
        Ok(t)
    }
}

impl SubjectPlanBuilderTrait for SubjectPlanBuilder {
    fn with_subject(mut self, subject: Subject) -> Self {
        self.subject = Some(subject);
        self
    }

    fn with_constraints(mut self, constraints: &[SubjectConstraintType]) -> Self {
        self.constraints = Some(constraints.to_owned());
        self
    }
    
    fn with_indices(mut self, indices: &[IndexType]) -> Self {
        self.indices = Some(indices.to_owned());
        self
    }
    
    fn with_sequences(mut self, sequences: &[SubjectSequenceType]) -> Self {
        self.sequences = Some(sequences.to_owned());
        self
    }
    
    fn get_name(&self) -> Result<String> {
        let name = self.subject.as_ref().ok_or(anyhow!("Please define the subject before trying to get the subject name!"))?.get_name().to_string();
        Ok(name)
    }
}