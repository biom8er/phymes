use std::fmt::Debug;
use crate::{BuildableTrait, IndexType, MappableTrait, SubjectConstraintType, SubjectPlanBuilder, SubjectSequenceType, Table};

pub trait SubjectPlanTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
    fn table(&self) -> &Table;
    fn constraints(&self) -> &Vec<SubjectConstraintType>;
    fn indices(&self) -> &Vec<IndexType>;
    fn sequences(&self) -> &Vec<SubjectSequenceType>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubjectPlan {
    /// Subject name
    pub(crate) name: String,
    /// Initial table with optional data
    pub(crate) table: Table,
    /// Constraints on the subject
    pub(crate) constraints: Vec<SubjectConstraintType>,
    /// Indexes on the subject
    pub(crate) indices: Vec<IndexType>,
    /// Sequences on the subject
    pub(crate) sequences: Vec<SubjectSequenceType>,
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
    fn table(&self) -> &Table {
        &self.table
    }

    fn constraints(&self) -> &Vec<SubjectConstraintType> {
        &self.constraints
    }
    
    fn indices(&self) -> &Vec<IndexType> {
        &self.indices
    }
    
    fn sequences(&self) -> &Vec<SubjectSequenceType> {
        &self.sequences
    }
}