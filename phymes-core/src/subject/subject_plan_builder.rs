use std::fmt::Debug;
use anyhow::{anyhow, Result};
use crate::{BuilderTrait, IndexType, SubjectConstraintType, SubjectPlan, SubjectSequenceType, Table};

pub trait SubjectPlanBuilderTrait: BuilderTrait + Debug + Send + Sync {
    fn with_table(self, table: Table) -> Self;
    fn with_constraints(self, constraints: &[SubjectConstraintType]) -> Self;
    fn with_indices(self, indices: &[IndexType]) -> Self;
    fn with_sequences(self, sequences: &[SubjectSequenceType]) -> Self;
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct SubjectPlanBuilder {
    /// Subject name
    pub name: Option<String>,
    /// Subject table
    pub table: Option<Table>,
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
            name: None,
            table: None,
            constraints: None,
            indices: None,
            sequences: None,
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
            table: self.table.ok_or(anyhow!("Please define the table before trying to build the subject plan!"))?,
            constraints: self.constraints.unwrap_or_default(),
            indices: self.indices.unwrap_or_default(),
            sequences: self.sequences.unwrap_or_default(),
        };
        Ok(t)
    }
}

impl SubjectPlanBuilderTrait for SubjectPlanBuilder {
    fn with_table(mut self, table: Table) -> Self {
        self.table = Some(table);
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
}