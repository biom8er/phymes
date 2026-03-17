use serde::{Deserialize, Serialize};

use crate::subject::index::{ConstraintCharacteristics, ConstraintReferenceMatchKind, IndexType, NullsDistinctOption, ReferentialAction};

/// A subject-level constraint
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum SubjectConstraint {
    Unique(UniqueConstraint),
    PrimaryKey(PrimaryKeyConstraint),
    ForeignKey(ForeignKeyConstraint),
    Check(CheckConstraint),
    Index(IndexConstraint),
    Custom(String),
}

impl From<UniqueConstraint> for SubjectConstraint {
    fn from(constraint: UniqueConstraint) -> Self {
        SubjectConstraint::Unique(constraint)
    }
}

impl From<PrimaryKeyConstraint> for SubjectConstraint {
    fn from(constraint: PrimaryKeyConstraint) -> Self {
        SubjectConstraint::PrimaryKey(constraint)
    }
}

impl From<ForeignKeyConstraint> for SubjectConstraint {
    fn from(constraint: ForeignKeyConstraint) -> Self {
        SubjectConstraint::ForeignKey(constraint)
    }
}

impl From<CheckConstraint> for SubjectConstraint {
    fn from(constraint: CheckConstraint) -> Self {
        SubjectConstraint::Check(constraint)
    }
}

impl From<IndexConstraint> for SubjectConstraint {
    fn from(constraint: IndexConstraint) -> Self {
        SubjectConstraint::Index(constraint)
    }
}

impl std::fmt::Display for SubjectConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SubjectConstraint::Unique(constraint) => write!(f, "Unique: {constraint:?}"),
            SubjectConstraint::PrimaryKey(constraint) => write!(f, "PrimaryKey: {constraint:?}"),
            SubjectConstraint::ForeignKey(constraint) => write!(f, "ForeignKey: {constraint:?}"),
            SubjectConstraint::Check(constraint) => write!(f, "Check: {constraint:?}"),
            SubjectConstraint::Index(constraint) => write!(f, "Index: {constraint:?}"),
            SubjectConstraint::Custom(c) => write!(f, "Custom: {c}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
/// A `CHECK` constraint
pub struct CheckConstraint {
    /// Optional constraint name.
    pub name: Option<String>,
    /// The boolean expression the CHECK constraint enforces.
    pub expr: String, // DM: todo!() as serializable DataConfig
    /// `ENFORCED` / `NOT ENFORCED` flag.
    pub enforced: Option<bool>,
}

/// A referential integrity constraint
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct ForeignKeyConstraint {
    /// Optional constraint name.
    pub name: Option<String>,
    /// MySQL-specific index name associated with the foreign key.
    /// <https://dev.mysql.com/doc/refman/8.4/en/create-table-foreign-keys.html>
    pub index_name: Option<String>,
    /// Columns in the local table that participate in the foreign key.
    pub columns: Vec<String>,
    /// Referenced foreign table name.
    pub foreign_table: String,
    /// Columns in the referenced table.
    pub referred_columns: Vec<String>,
    /// Action to perform `ON DELETE`.
    pub on_delete: Option<ReferentialAction>,
    /// Action to perform `ON UPDATE`.
    pub on_update: Option<ReferentialAction>,
    /// Optional `MATCH` kind (FULL | PARTIAL | SIMPLE).
    pub match_kind: Option<ConstraintReferenceMatchKind>,
    /// Optional characteristics (e.g., `DEFERRABLE`).
    pub characteristics: Option<ConstraintCharacteristics>,
}

/// Index constraint
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct IndexConstraint {
    pub name: Option<String>,
    /// IndexType
    pub index_type: Option<IndexType>,
    /// Referred column identifier list.
    pub columns: Vec<String>,
}

/// `PRIMARY KEY` constraints statements
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct PrimaryKeyConstraint {
    /// Constraint name.
    ///
    /// Can be not the same as `index_name`
    pub name: Option<String>,
    /// Index name
    pub index_name: Option<String>,
    /// Optional `USING` of [index type][1] statement before columns.
    ///
    /// [1]: IndexType
    pub index_type: Option<IndexType>,
    /// Identifiers of the columns that form the primary key.
    pub columns: Vec<String>,
    /// Optional characteristics like `DEFERRABLE`.
    pub characteristics: Option<ConstraintCharacteristics>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
/// Unique constraint definition.
pub struct UniqueConstraint {
    /// Constraint name.
    ///
    /// Can be not the same as `index_name`
    pub name: Option<String>,
    /// Index name
    pub index_name: Option<String>,
    /// Optional `USING` of [index type][1] statement before columns.
    ///
    /// [1]: IndexType
    pub index_type: Option<IndexType>,
    /// Identifiers of the columns that are unique.
    pub columns: Vec<String>,
    /// Optional characteristics like `DEFERRABLE`.
    pub characteristics: Option<ConstraintCharacteristics>,
    /// Optional Postgres nulls handling: `[ NULLS [ NOT ] DISTINCT ]`
    pub nulls_distinct: NullsDistinctOption,
}