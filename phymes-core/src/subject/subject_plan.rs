use crate::{
    BuildableTrait, IndexType, MappableTrait, Subject, SubjectConstraintType, SubjectPlanBuilder,
    SubjectSequenceType,
};
use std::fmt::Debug;

pub trait SubjectPlanTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
    fn subject(&self) -> &Subject;
    fn subject_own(self) -> Subject;
    fn constraints(&self) -> &Vec<SubjectConstraintType>;
    fn indices(&self) -> &Vec<IndexType>;
    fn sequences(&self) -> &Vec<SubjectSequenceType>;
    /// Create the additional subjects for the constraints
    /// DM: missing operator to join the columns of RecordBatches
    fn constraints_subjects(&self) -> Vec<Subject>;
    /// Create the additional subjects for the indices
    fn indices_subjects(&self) -> Vec<Subject>;
    /// Create the additional subjects for the sequences
    fn sequences_subjects(&self) -> Vec<Subject>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubjectPlan {
    /// Initial table with optional data
    pub(crate) subject: Subject,
    /// Constraints on the subject
    pub(crate) constraints: Vec<SubjectConstraintType>,
    /// Indexes on the subject
    pub(crate) indices: Vec<IndexType>,
    /// Sequences on the subject
    pub(crate) sequences: Vec<SubjectSequenceType>,
}

impl MappableTrait for SubjectPlan {
    fn get_name(&self) -> &str {
        &self.subject.get_name()
    }
}

impl BuildableTrait for SubjectPlan {
    type T = SubjectPlanBuilder;

    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl SubjectPlanTrait for SubjectPlan {
    fn subject(&self) -> &Subject {
        &self.subject
    }

    fn subject_own(self) -> Subject {
        self.subject
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

    fn constraints_subjects(&self) -> Vec<Subject> {
        todo!()
    }

    fn indices_subjects(&self) -> Vec<Subject> {
        todo!()
    }

    fn sequences_subjects(&self) -> Vec<Subject> {
        todo!()
    }
}
