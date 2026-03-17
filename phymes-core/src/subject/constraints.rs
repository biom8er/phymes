use serde::{Deserialize, Serialize};

/// A subject-level constraint
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum SubjectConstraint {
    Unique(UniqueConstraint),
    PrimaryKey(PrimaryKeyConstraint),
    ForeignKey(ForeignKeyConstraint),
    Check(CheckConstraint),
    Index(IndexConstraint),
    FulltextOrSpatial(FullTextOrSpatialConstraint),
    PrimaryKeyUsingIndex(ConstraintUsingIndex),
    UniqueUsingIndex(ConstraintUsingIndex),
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

impl From<FullTextOrSpatialConstraint> for SubjectConstraint {
    fn from(constraint: FullTextOrSpatialConstraint) -> Self {
        SubjectConstraint::FulltextOrSpatial(constraint)
    }
}

impl std::fmt::Display for SubjectConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SubjectConstraint::Unique(constraint) => constraint.fmt(f),
            SubjectConstraint::PrimaryKey(constraint) => constraint.fmt(f),
            SubjectConstraint::ForeignKey(constraint) => constraint.fmt(f),
            SubjectConstraint::Check(constraint) => constraint.fmt(f),
            SubjectConstraint::Index(constraint) => constraint.fmt(f),
            SubjectConstraint::FulltextOrSpatial(constraint) => constraint.fmt(f),
            SubjectConstraint::PrimaryKeyUsingIndex(c) => c.fmt_with_keyword(f, "PRIMARY KEY"),
            SubjectConstraint::UniqueUsingIndex(c) => c.fmt_with_keyword(f, "UNIQUE"),
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

impl std::fmt::Display for CheckConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}CHECK ({})",
            display_constraint_name(&self.name),
            self.expr
        )?;
        if let Some(b) = self.enforced {
            write!(f, " {}", if b { "ENFORCED" } else { "NOT ENFORCED" })
        } else {
            Ok(())
        }
    }
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

impl std::fmt::Display for ForeignKeyConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}FOREIGN KEY{} ({}) REFERENCES {}",
            display_constraint_name(&self.name),
            display_option_spaced(&self.index_name),
            display_comma_separated(&self.columns),
            self.foreign_table,
        )?;
        if !self.referred_columns.is_empty() {
            write!(f, "({})", display_comma_separated(&self.referred_columns))?;
        }
        if let Some(match_kind) = &self.match_kind {
            write!(f, " {match_kind}")?;
        }
        if let Some(action) = &self.on_delete {
            write!(f, " ON DELETE {action}")?;
        }
        if let Some(action) = &self.on_update {
            write!(f, " ON UPDATE {action}")?;
        }
        if let Some(characteristics) = &self.characteristics {
            write!(f, " {characteristics}")?;
        }
        Ok(())
    }
}

/// MySQLs [fulltext][1] definition. Since the [`SPATIAL`][2] definition is exactly the same,
/// and MySQL displays both the same way, it is part of this definition as well.
///
/// Supported syntax:
///
/// ```markdown
/// {FULLTEXT | SPATIAL} [INDEX | KEY] [index_name] (key_part,...)
///
/// key_part: col_name
/// ```
///
/// [1]: https://dev.mysql.com/doc/refman/8.0/en/fulltext-natural-language.html
/// [2]: https://dev.mysql.com/doc/refman/8.0/en/spatial-types.html
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct FullTextOrSpatialConstraint {
    /// Whether this is a `FULLTEXT` (true) or `SPATIAL` (false) definition.
    pub fulltext: bool,
    /// Optional index name.
    pub opt_index_name: Option<String>,
    /// Referred column identifier list.
    pub columns: Vec<String>,
}

impl std::fmt::Display for FullTextOrSpatialConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.fulltext {
            write!(f, "FULLTEXT")?;
        } else {
            write!(f, "SPATIAL")?;
        }

        if let Some(name) = &self.opt_index_name {
            write!(f, " {name}")?;
        }

        write!(f, " ({})", display_comma_separated(&self.columns))?;

        Ok(())
    }
}

/// Index constraint
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct IndexConstraint {
    pub name: Option<String>,
    /// Optional [index type][1].
    ///
    /// [1]: IndexType
    pub index_type: Option<IndexType>,
    /// Referred column identifier list.
    pub columns: Vec<String>,
}

impl std::fmt::Display for IndexConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(f, " {name}")?;
        }
        if let Some(index_type) = &self.index_type {
            write!(f, " USING {index_type}")?;
        }
        Ok(())
    }
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

impl std::fmt::Display for PrimaryKeyConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}PRIMARY KEY{}{} ({})",
            display_constraint_name(&self.name),
            display_option_spaced(&self.index_name),
            display_option(" USING ", "", &self.index_type),
            display_comma_separated(&self.columns),
        )?;

        write!(f, "{}", display_option_spaced(&self.characteristics))?;
        Ok(())
    }
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

impl std::fmt::Display for UniqueConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}UNIQUE{}{:>}{} ({})",
            display_constraint_name(&self.name),
            self.nulls_distinct,
            display_option_spaced(&self.index_name),
            display_option(" USING ", "", &self.index_type),
            display_comma_separated(&self.columns),
        )?;

        write!(f, "{}", display_option_spaced(&self.characteristics))?;
        Ok(())
    }
}

/// PostgreSQL constraint that promotes an existing unique index to a table constraint
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct ConstraintUsingIndex {
    /// Optional constraint name.
    pub name: Option<String>,
    /// The name of the existing unique index to promote.
    pub index_name: String,
    /// Optional characteristics like `DEFERRABLE`.
    pub characteristics: Option<ConstraintCharacteristics>,
}

impl ConstraintUsingIndex {
    /// Format as `[CONSTRAINT name] <keyword> USING INDEX index_name [characteristics]`.
    pub fn fmt_with_keyword(&self, f: &mut std::fmt::Formatter, keyword: &str) -> std::fmt::Result {
        write!(
            f,
            "{}{} USING INDEX {}",
            display_constraint_name(&self.name),
            keyword,
            self.index_name,
        )?;
        write!(f, "{}", display_option_spaced(&self.characteristics))?;
        Ok(())
    }
}

/// === helper.rs

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



/// Indexing method used by that index.
///
/// This structure isn't present on ANSI, but is found at least in [`MySQL` CREATE TABLE][1],
/// [`MySQL` CREATE INDEX][2], and [Postgresql CREATE INDEX][3] statements.
///
/// [1]: https://dev.mysql.com/doc/refman/8.0/en/create-table.html
/// [2]: https://dev.mysql.com/doc/refman/8.0/en/create-index.html
/// [3]: https://www.postgresql.org/docs/14/sql-createindex.html
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

/// Helper used to format a slice using a separator string (e.g., `", "`).
pub struct DisplaySeparated<'a, T>
where
    T: std::fmt::Display,
{
    slice: &'a [T],
    sep: &'static str,
}

impl<T> std::fmt::Display for DisplaySeparated<'_, T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut delim = "";
        for t in self.slice {
            f.write_str(delim)?;
            delim = self.sep;
            t.fmt(f)?;
        }
        Ok(())
    }
}

pub(crate) fn display_separated<'a, T>(slice: &'a [T], sep: &'static str) -> DisplaySeparated<'a, T>
where
    T: std::fmt::Display,
{
    DisplaySeparated { slice, sep }
}

pub(crate) fn display_comma_separated<T>(slice: &[T]) -> DisplaySeparated<'_, T>
where
    T: std::fmt::Display,
{
    DisplaySeparated { slice, sep: ", " }
}

pub(crate) fn display_constraint_name(name: &'_ Option<String>) -> impl std::fmt::Display + '_ {
    struct ConstraintName<'a>(&'a Option<String>);
    impl std::fmt::Display for ConstraintName<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            if let Some(name) = self.0 {
                write!(f, "CONSTRAINT {name} ")?;
            }
            Ok(())
        }
    }
    ConstraintName(name)
}

/// If `option` is
/// * `Some(inner)` => create display struct for `"{prefix}{inner}{postfix}"`
/// * `_` => do nothing
#[must_use]
pub(crate) fn display_option<'a, T: std::fmt::Display>(
    prefix: &'a str,
    postfix: &'a str,
    option: &'a Option<T>,
) -> impl std::fmt::Display + 'a {
    struct OptionDisplay<'a, T>(&'a str, &'a str, &'a Option<T>);
    impl<T: std::fmt::Display> std::fmt::Display for OptionDisplay<'_, T> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            if let Some(inner) = self.2 {
                let (prefix, postfix) = (self.0, self.1);
                write!(f, "{prefix}{inner}{postfix}")?;
            }
            Ok(())
        }
    }
    OptionDisplay(prefix, postfix, option)
}

/// If `option` is
/// * `Some(inner)` => create display struct for `" {inner}"`
/// * `_` => do nothing
#[must_use]
pub(crate) fn display_option_spaced<T: std::fmt::Display>(option: &Option<T>) -> impl std::fmt::Display + '_ {
    display_option(" ", "", option)
}