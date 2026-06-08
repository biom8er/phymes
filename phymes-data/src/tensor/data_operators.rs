use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataStreamManager {
    /// Accumulate the record batches before streaming operations for each record batch
    #[default]
    #[value(name = "Accumulate")]
    Accumulate,
    /// Stream the record batches
    #[value(name = "Stream")]
    Stream,
}

impl Display for DataStreamManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Accumulate => write!(f, "Accumulate"),
            Self::Stream => write!(f, "Stream"),
        }
    }
}

/// Data Aggregation (Reduction) operators
///
/// # Notes
/// - `Max`, `Min`, `Sum`, `Mean`, and `Var` can only be applied to non-nested primitive [DataType]s
/// - `Count` can be applied to all [DataType]s and generates a UInt32Array
/// - `Concat` can only be applied to Utf8 [DataType] to generate a new Utf8 [DataType] by joining the [String]s together
/// - `List` and `Set` can be applied to all primitive [DataType]s except floats.
///   Non-nested primitive and non-primitive [DataType]s will generate a nested primitive or non-primitive Array.
///   Nested primitive or non-primitive [DataType]s will maintain the nested primitive or non-primitive Array through extension of the list or set.
/// - `First` and `Last` can be applied to all [DataType]s.
///
/// [DataType]: arrow::datatypes::DataType
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataAggregatorOperator {
    #[value(name = "Max")]
    Max,
    #[value(name = "Min")]
    Min,
    #[value(name = "Sum")]
    Sum,
    #[value(name = "Mean")]
    Mean,
    #[value(name = "Var")]
    Var,
    #[default]
    #[value(name = "Count")]
    Count,
    #[value(name = "Concat")]
    Concat,
    #[value(name = "ConcatSemicolonSeperator")]
    ConcatSemicolonSeperator,
    #[value(name = "List")]
    List,
    #[value(name = "Set")]
    Set,
    #[value(name = "First")]
    First,
    #[value(name = "Last")]
    Last,
}

impl Display for DataAggregatorOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
            Self::Sum => write!(f, "Sum"),
            Self::Mean => write!(f, "Mean"),
            Self::Var => write!(f, "Var"),
            Self::Count => write!(f, "Count"),
            Self::Concat => write!(f, "Concat"),
            Self::ConcatSemicolonSeperator => write!(f, "ConcatSemicolonSeperator"),
            Self::List => write!(f, "List"),
            Self::Set => write!(f, "Set"),
            Self::First => write!(f, "First"),
            Self::Last => write!(f, "Last"),
        }
    }
}

/// Data Comparison operators
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataComparatorOperator {
    /// Primitive
    #[value(name = "Equals")]
    Equals,
    #[value(name = "NotEquals")]
    NotEquals,
    #[value(name = "LessThanOrEqualTo")]
    LessThanOrEqualTo,
    #[value(name = "GreaterThanOrEqualTo")]
    GreaterThanOrEqualTo,
    #[value(name = "LessThan")]
    LessThan,
    #[value(name = "GreaterThan")]
    GreaterThan,
    /// Nested and non-primitive
    #[value(name = "Contains")]
    Contains,
    #[value(name = "EndsWith")]
    EndsWith,
    #[value(name = "CaseInsensitiveLike")]
    CaseInsensitiveLike,
    #[value(name = "Like")]
    Like,
    #[value(name = "CaseInsensitiveNotLike")]
    CaseInsensitiveNotLike,
    #[value(name = "NotLike")]
    NotLike,
    #[value(name = "InList")]
    InList,
    #[value(name = "InListUtf8")]
    InListUtf8,
    #[value(name = "NotInList")]
    NotInList,
    #[value(name = "NotInListUtf8")]
    NotInListUtf8,
    #[value(name = "RegExpIsMatch")]
    RegExpIsMatch,
    #[value(name = "StartsWith")]
    StartsWith,
}

impl Display for DataComparatorOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Equals => write!(f, "Equals"),
            Self::NotEquals => write!(f, "NotEquals"),
            Self::LessThanOrEqualTo => write!(f, "LessThanOrEqualTo"),
            Self::GreaterThanOrEqualTo => write!(f, "GreaterThanOrEqualTo"),
            Self::LessThan => write!(f, "LessThan"),
            Self::GreaterThan => write!(f, "GreaterThan"),
            Self::Contains => write!(f, "Contains"),
            Self::EndsWith => write!(f, "EndsWith"),
            Self::CaseInsensitiveLike => write!(f, "CaseInsensitiveLike"),
            Self::Like => write!(f, "Like"),
            Self::CaseInsensitiveNotLike => write!(f, "CaseInsensitiveNotLike"),
            Self::NotLike => write!(f, "NotLike"),
            Self::InList => write!(f, "InList"),
            Self::InListUtf8 => write!(f, "InListUtf8"),
            Self::NotInList => write!(f, "NotInList"),
            Self::NotInListUtf8 => write!(f, "NotInListUtf8"),
            Self::RegExpIsMatch => write!(f, "RegExpIsMatch"),
            Self::StartsWith => write!(f, "StartsWith"),
        }
    }
}

/// Data Comparison predicates to evaluate parenthetic groups
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataComparatorPredicate {
    #[default]
    #[value(name = "All")]
    All,
    #[value(name = "Any")]
    Any,
}

impl Display for DataComparatorPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "All"),
            Self::Any => write!(f, "Any"),
        }
    }
}

/// Data distance operators between two equal length vectors
#[derive(Default, Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataDistanceOperator {
    #[default]
    #[value(name = "NormalizedDotProduct")]
    NormalizedDotProduct,
    #[value(name = "NormalizedDifference")]
    NormalizedDifference,
}

impl Display for DataDistanceOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NormalizedDotProduct => write!(f, "NormalizedDotProduct"),
            Self::NormalizedDifference => write!(f, "NormalizedDifference"),
        }
    }
}

/// Data cast operators work in conjunction with DataCastAs to change the column name,
///   DataCastDataType to change the data type, and DataCastTemplate to apply a template
///
/// # Notes
///
/// Casting uses the [arrow_cast] crate to convert between [DataType]s
///
/// [DataType]: arrow::datatypes::DataType
///
/// 1. Check if conversion is possible <https://arrow.apache.org/rust/arrow_cast/cast/fn.can_cast_types.html>
/// 2. Convert between types <https://arrow.apache.org/rust/arrow_cast/cast/fn.cast_with_options.html>
/// 3. Encode/Decode Base64 <https://arrow.apache.org/rust/arrow_cast/base64/index.html> with BASE64_URL_SAFE_NO_PAD engine
/// 4. Convert a List-UInt8 (Bytes) to Utf8
///
/// Casting allows for applying a [String] template for formatting
///
/// Casting allows for converting between DateTime strings and numeric Timestamps
///
/// [arrow_cast]: <https://arrow.apache.org/rust/arrow_cast/index.html>
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataCastOperator {
    // #[value(name = "Base64Encode")]
    // Base64Encode,
    // #[value(name = "Base64Decode")]
    // Base64Decode,
    #[value(name = "Cast")]
    Cast,
    #[value(name = "BytesToString")]
    BytesToString,
    #[value(name = "Hash")]
    Hash,
    #[default]
    #[value(name = "None")]
    None,
}

impl Display for DataCastOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Self::Base64Encode => write!(f, "Base64Encode"),
            // Self::Base64Decode => write!(f, "Base64Decode"),
            Self::Cast => write!(f, "Cast"),
            Self::BytesToString => write!(f, "BytesToString"),
            Self::Hash => write!(f, "Hash"),
            Self::None => write!(f, "None"),
        }
    }
}

/// Data Column operators between one (unary) or two (binary) columns
///
/// # Notes on binary operators
/// - `Max`, `Min`, `Add`, `Sub`, `Mult`, `Div`, and `Var` can only be applied to non-nested primitive [DataType]s
/// - `And`, `Or`, `XOr`, `LeftShift`, and `RightShift` can only be applied to non-nested primitive [DataType]s
/// - `Concat` can only be applied to Utf8 [DataType] to generate a new Utf8 [DataType] by joining the [String]s together
/// - `List` and `Set` can be applied to all primitive [DataType]s except floats.
///   Non-nested primitive and non-primitive [DataType]s will generate a nested primitive or non-primitive Array.
///   Nested primitive or non-primitive [DataType]s will maintain the nested primitive or non-primitive Array through extension of the list or set.
///
/// # Notes on unary operators
/// - `Not`can only be applied to non-nested primitive [DataType]s
/// - `Len` can be applied to all [DataType]s and generates a UInt32Array
///
/// # Notes on intialization operators
/// - `Zeros` and `Ones` will create a new column filled with primitive zero's or one's
/// - `String` will create a new column filled with an empty Utf8
/// - `Value` will create a new column filled with a specified primitive or non-primitive value  
///
/// [DataType]: arrow::datatypes::DataType
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataColumnOperator {
    #[value(name = "And")]
    And,
    #[value(name = "AndNot")]
    AndNot,
    #[value(name = "Or")]
    Or,
    #[value(name = "XOr")]
    XOr,
    #[value(name = "Not")]
    Not,
    #[value(name = "LeftShift")]
    LeftShift,
    #[value(name = "RightShift")]
    RightShift,
    #[value(name = "Max")]
    Max,
    #[value(name = "Min")]
    Min,
    #[value(name = "Add")]
    Add,
    #[value(name = "Sub")]
    Sub,
    #[value(name = "Mult")]
    Mult,
    #[value(name = "Div")]
    Div,
    #[value(name = "Rem")]
    Rem,
    #[value(name = "List")]
    List,
    #[value(name = "Set")]
    Set,
    #[value(name = "Concat")]
    Concat,
    #[value(name = "Len")]
    Len,
    #[default]
    #[value(name = "None")]
    None,
    #[value(name = "Zeros")]
    Zeros,
    #[value(name = "Ones")]
    Ones,
    // #[value(name = "Rand")]
    // Rand,
    #[value(name = "String")]
    String,
    #[value(name = "Value")]
    Value,
    #[value(name = "BroadcastMax")]
    BroadcastMax,
    #[value(name = "BroadcastMin")]
    BroadcastMin,
    #[value(name = "BroadcastMean")]
    BroadcastMean,
    #[value(name = "BroadcastVar")]
    BroadcastVar,
    #[value(name = "BroadcastCount")]
    BroadcastCount,
    #[value(name = "BroadcastList")]
    BroadcastList,
    #[value(name = "BroadcastSet")]
    BroadcastSet,
    #[value(name = "CumSum")]
    CumSum,
}

impl DataColumnOperator {
    /// Can the operator be applied to two columns
    pub fn is_binary(&self) -> bool {
        match self {
            Self::And
            | Self::AndNot
            | Self::Or
            | Self::XOr
            | Self::Not
            | Self::LeftShift
            | Self::RightShift
            | Self::Max
            | Self::Min
            | Self::Add
            | Self::Sub
            | Self::Mult
            | Self::Div
            | Self::Rem
            | Self::List
            | Self::Set
            | Self::Concat
            | Self::Len
            | Self::None => true,
            Self::Zeros
            | Self::Ones
            | Self::String
            | Self::Value
            | Self::BroadcastMax
            | Self::BroadcastMin
            | Self::BroadcastMean
            | Self::BroadcastVar
            | Self::BroadcastCount
            | Self::BroadcastList
            | Self::BroadcastSet
            | Self::CumSum => false,
        }
    }

    /// Can the operator be applied to one columns
    pub fn is_unary(&self) -> bool {
        match self {
            Self::And
            | Self::AndNot
            | Self::Or
            | Self::XOr
            | Self::LeftShift
            | Self::RightShift
            | Self::Max
            | Self::Min
            | Self::Add
            | Self::Sub
            | Self::Mult
            | Self::Div
            | Self::Rem
            | Self::List
            | Self::Set
            | Self::Concat
            | Self::Zeros
            | Self::Ones
            | Self::String
            | Self::Value => false,
            Self::BroadcastMax
            | Self::BroadcastMin
            | Self::BroadcastMean
            | Self::BroadcastVar
            | Self::BroadcastCount
            | Self::BroadcastList
            | Self::BroadcastSet
            | Self::CumSum
            | Self::Not
            | Self::Len
            | Self::None => true,
        }
    }

    /// Can the operator initialize a new column
    pub fn is_init(&self) -> bool {
        match self {
            Self::And
            | Self::AndNot
            | Self::Or
            | Self::XOr
            | Self::Not
            | Self::LeftShift
            | Self::RightShift
            | Self::Max
            | Self::Min
            | Self::Add
            | Self::Sub
            | Self::Mult
            | Self::Div
            | Self::Rem
            | Self::List
            | Self::Set
            | Self::Concat
            | Self::Len
            | Self::BroadcastMax
            | Self::BroadcastMin
            | Self::BroadcastMean
            | Self::BroadcastVar
            | Self::BroadcastCount
            | Self::BroadcastList
            | Self::BroadcastSet
            | Self::CumSum
            | Self::None => false,
            Self::Zeros | Self::Ones | Self::String | Self::Value => true,
        }
    }
}

impl Display for DataColumnOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::And => write!(f, "And"),
            Self::AndNot => write!(f, "AndNot"),
            Self::Or => write!(f, "Or"),
            Self::XOr => write!(f, "XOr"),
            Self::Not => write!(f, "Not"),
            Self::LeftShift => write!(f, "LeftShift"),
            Self::RightShift => write!(f, "RightShift"),
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
            Self::Add => write!(f, "Add"),
            Self::Sub => write!(f, "Sub"),
            Self::Mult => write!(f, "Mult"),
            Self::Div => write!(f, "Div"),
            Self::Rem => write!(f, "Rem"),
            Self::List => write!(f, "List"),
            Self::Set => write!(f, "Set"),
            Self::Concat => write!(f, "Concat"),
            Self::Len => write!(f, "Len"),
            Self::None => write!(f, "None"),
            Self::Zeros => write!(f, "Zeros"),
            Self::Ones => write!(f, "Ones"),
            Self::String => write!(f, "String"),
            Self::Value => write!(f, "Value"),
            Self::BroadcastMax => write!(f, "BroadcastMax"),
            Self::BroadcastMin => write!(f, "BroadcastMin"),
            Self::BroadcastMean => write!(f, "BroadcastMean"),
            Self::BroadcastVar => write!(f, "BroadcastVar"),
            Self::BroadcastCount => write!(f, "BroadcastCount"),
            Self::BroadcastList => write!(f, "BroadcastList"),
            Self::BroadcastSet => write!(f, "BroadcastSet"),
            Self::CumSum => write!(f, "CumSum"),
        }
    }
}

/// Data Join Operators
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataJoinOperator {
    /// Combines rows from two tables where there is a match in the specified column(s) of both tables. Only matching rows are included in the result.
    #[default]
    #[value(name = "Inner")]
    Inner,
    /// Returns all rows from the left table and the matching rows from the right table. If no match exists, NULL values are returned for the right table's columns
    #[value(name = "LeftOuter")]
    LeftOuter,
    /// Returns all rows from the right table and the matching rows from the left table. If no match exists, NULL values are returned for the left table's columns.
    #[value(name = "RightOuter")]
    RightOuter,
    /// Combines the results of both LEFT JOIN and RIGHT JOIN. Returns all rows from both tables, with NULLs for non-matching rows in either table.
    #[value(name = "FullOuter")]
    FullOuter,
    /// Produces a Cartesian product of two tables, pairing each row from the first table with every row from the second table.
    /// Not yet supported.
    #[value(name = "Cross")]
    Cross,
    /// Automatically joins tables based on columns with the same name and data type, including only rows with matching values in those columns.
    /// Not yet supported.
    #[value(name = "Natural")]
    Natural,
}

impl Display for DataJoinOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inner => write!(f, "Inner"),
            Self::LeftOuter => write!(f, "LeftOuter"),
            Self::RightOuter => write!(f, "RightOuter"),
            Self::FullOuter => write!(f, "FullOuter"),
            Self::Cross => write!(f, "Cross"),
            Self::Natural => write!(f, "Natural"),
        }
    }
}
