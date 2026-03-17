use serde::{Deserialize, Serialize};

/// [PostgreSQL] unique index nulls handling option: `[ NULLS [ NOT ] DISTINCT ]`
///
/// [PostgreSQL]: https://www.postgresql.org/docs/17/sql-altertable.html
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

/// `<constraint_characteristics> = [ DEFERRABLE | NOT DEFERRABLE ] [ INITIALLY DEFERRED | INITIALLY IMMEDIATE ] [ ENFORCED | NOT ENFORCED ]`
///
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

/// `<referential_action> =
/// { RESTRICT | CASCADE | SET NULL | NO ACTION | SET DEFAULT }`
///
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

/// `<drop behavior> ::= CASCADE | RESTRICT`.
///
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
///
/// See: <https://www.postgresql.org/docs/current/sql-createtable.html#SQL-CREATETABLE-PARMS-REFERENCES>
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

/// Indexing method used by that index
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum IndexType {
    /// B-Tree index (commonly default for many databases).
    BTree,
    /// Hash index.
    Hash,
    /// Generalized Inverted Index (GIN).
    GIN,
    /// Generalized Search Tree (GiST) index.
    GiST,
    /// Space-partitioned GiST (SPGiST) index.
    SPGiST,
    /// Block Range Index (BRIN).
    BRIN,
    /// Bloom filter based index.
    Bloom,
    /// Users may define their own index types, which would
    /// not be covered by the above variants.
    Custom(String),
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::BTree => write!(f, "BTREE"),
            Self::Hash => write!(f, "HASH"),
            Self::GIN => write!(f, "GIN"),
            Self::GiST => write!(f, "GIST"),
            Self::SPGiST => write!(f, "SPGIST"),
            Self::BRIN => write!(f, "BRIN"),
            Self::Bloom => write!(f, "BLOOM"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}