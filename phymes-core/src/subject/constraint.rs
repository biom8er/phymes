use serde::{Deserialize, Serialize};

use crate::subject::index_type::IndexType;

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

/// Unique index nulls handling option
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum NullsDistinctOption {
    /// Not specified
    None,
    /// NULLS DISTINCT
    Distinct,
    /// NULLS NOT DISTINCT
    NotDistinct,
}

impl std::fmt::Display for NullsDistinctOption {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Distinct => write!(f, " NULLS DISTINCT"),
            Self::NotDistinct => write!(f, " NULLS NOT DISTINCT"),
        }
    }
}

/// Used in UNIQUE and foreign key constraints. The individual settings may occur in any order.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct ConstraintCharacteristics {
    /// `[ DEFERRABLE | NOT DEFERRABLE ]`
    pub deferrable: Option<bool>,
    /// `[ INITIALLY DEFERRED | INITIALLY IMMEDIATE ]`
    pub initially: Option<DeferrableInitial>,
    /// `[ ENFORCED | NOT ENFORCED ]`
    pub enforced: Option<bool>,
}

/// Initial setting for deferrable constraints (`INITIALLY IMMEDIATE` or `INITIALLY DEFERRED`).
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum DeferrableInitial {
    /// `INITIALLY IMMEDIATE`
    Immediate,
    /// `INITIALLY DEFERRED`
    Deferred,
}

impl ConstraintCharacteristics {
    fn deferrable_text(&self) -> Option<&'static str> {
        self.deferrable.map(|deferrable| {
            if deferrable {
                "DEFERRABLE"
            } else {
                "NOT DEFERRABLE"
            }
        })
    }

    fn initially_immediate_text(&self) -> Option<&'static str> {
        self.initially
            .map(|initially_immediate| match initially_immediate {
                DeferrableInitial::Immediate => "INITIALLY IMMEDIATE",
                DeferrableInitial::Deferred => "INITIALLY DEFERRED",
            })
    }

    fn enforced_text(&self) -> Option<&'static str> {
        self.enforced.map(
            |enforced| {
                if enforced {
                    "ENFORCED"
                } else {
                    "NOT ENFORCED"
                }
            },
        )
    }
}

impl std::fmt::Display for ConstraintCharacteristics {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let deferrable = self.deferrable_text();
        let initially_immediate = self.initially_immediate_text();
        let enforced = self.enforced_text();

        match (deferrable, initially_immediate, enforced) {
            (None, None, None) => Ok(()),
            (None, None, Some(enforced)) => write!(f, "{enforced}"),
            (None, Some(initial), None) => write!(f, "{initial}"),
            (None, Some(initial), Some(enforced)) => write!(f, "{initial} {enforced}"),
            (Some(deferrable), None, None) => write!(f, "{deferrable}"),
            (Some(deferrable), None, Some(enforced)) => write!(f, "{deferrable} {enforced}"),
            (Some(deferrable), Some(initial), None) => write!(f, "{deferrable} {initial}"),
            (Some(deferrable), Some(initial), Some(enforced)) => {
                write!(f, "{deferrable} {initial} {enforced}")
            }
        }
    }
}

/// Used in foreign key constraints in `ON UPDATE` and `ON DELETE` options.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum ReferentialAction {
    /// `RESTRICT` - disallow action if it would break referential integrity.
    Restrict,
    /// `CASCADE` - propagate the action to referencing rows.
    Cascade,
    /// `SET NULL` - set referencing columns to NULL.
    SetNull,
    /// `NO ACTION` - no action at the time; may be deferred.
    NoAction,
    /// `SET DEFAULT` - set referencing columns to their default values.
    SetDefault,
}

impl std::fmt::Display for ReferentialAction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            ReferentialAction::Restrict => "RESTRICT",
            ReferentialAction::Cascade => "CASCADE",
            ReferentialAction::SetNull => "SET NULL",
            ReferentialAction::NoAction => "NO ACTION",
            ReferentialAction::SetDefault => "SET DEFAULT",
        })
    }
}

/// Used in `DROP` statements.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum DropBehavior {
    /// `RESTRICT` - refuse to drop if there are any dependent objects.
    Restrict,
    /// `CASCADE` - automatically drop objects that depend on the object being dropped.
    Cascade,
}

impl std::fmt::Display for DropBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            DropBehavior::Restrict => "RESTRICT",
            DropBehavior::Cascade => "CASCADE",
        })
    }
}

/// `MATCH` type for constraint references
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum ConstraintReferenceMatchKind {
    /// `MATCH FULL`
    Full,
    /// `MATCH PARTIAL`
    Partial,
    /// `MATCH SIMPLE`
    Simple,
}

impl std::fmt::Display for ConstraintReferenceMatchKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "MATCH FULL"),
            Self::Partial => write!(f, "MATCH PARTIAL"),
            Self::Simple => write!(f, "MATCH SIMPLE"),
        }
    }
}