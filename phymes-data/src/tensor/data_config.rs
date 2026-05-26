use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_diagnostics::HashSet;
use phymes_schemas::{
    AvailableSubjects, DataEncoding, DataFormat, create_bytes_fields, create_values_fields,
};
use phymes_subject::{MappableTrait, Subject, SubjectTrait};
use serde::{Deserialize, Serialize};

use crate::{AvailableJinja2Templates, AvailableOperators, AvailableParsers, CodeCompletionType, DataAggregatorOperator, DataCastOperator, DataColumnOperator, DataComparatorOperator, DataComparatorPredicate, DataDistanceOperator, DataJoinOperator, DataStreamManager, DiffType};

/// Document Filter Type
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DocumentFilterType {
    /// No filtering
    #[value(name = "None")]
    None,
    /// Remove all keys and object except those for text
    #[value(name = "Text")]
    Text,
    /// Remove all keys and object except those for graphics
    #[value(name = "Graphics")]
    Graphics,
    /// Minimal number of keys and objects
    #[default]
    #[value(name = "Default")]
    #[serde(other)]
    Default,
}

impl std::fmt::Display for DocumentFilterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Text => write!(f, "Text"),
            Self::Graphics => write!(f, "Graphics"),
            Self::Default => write!(f, "Default"),
        }
    }
}

/// Document Extraction Type
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DocumentExtractType {
    /// All text including operator metadata
    #[value(name = "Text")]
    Text,
    /// All text after applying heuristics optimized for text embeddings
    #[value(name = "TextEmbeddings")]
    TextEmbeddings,
    /// All graphics include operator metadata
    #[value(name = "Graphics")]
    Graphics,
    /// All images after applying heuristics optimized for text embeddings
    #[value(name = "ImageEmbeddings")]
    ImageEmbeddings,
    /// Default extraction; all text excluding operator metadata
    #[default]
    #[value(name = "Default")]
    #[serde(other)]
    Default,
}

impl std::fmt::Display for DocumentExtractType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "Text"),
            Self::TextEmbeddings => write!(f, "TextEmbeddings"),
            Self::Graphics => write!(f, "Graphics"),
            Self::ImageEmbeddings => write!(f, "ImageEmbeddings"),
            Self::Default => write!(f, "Default"),
        }
    }
}

/// Traits for all configs
pub trait DataConfigTrait: MappableTrait {
    /// Create an example and serialize to JSON
    ///
    /// # Notes
    /// - example implementation: `serde_json::to_vec(&DataConfig)`
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error>;

    /// Build the config from a [Subject] with options
    ///   for JSON Values or Bytes schemas
    fn from_subject(subject: &Subject) -> Result<Self>
    where
        Self: Sized;

    /// Build the config from JSON Values or Bytes
    fn from_subject_as_bytes(subject: &Subject) -> Option<Vec<u8>>
    where
        Self: Sized,
    {
        if subject
            .get_schema()
            .fields()
            .contains(&create_values_fields())
        {
            let mut config_str = subject
                .get_column_as_vec_nonprimitive::<String>("values")
                .unwrap();
            if let Some(last) = config_str.pop() {
                let bytes = last.into_bytes();
                Some(bytes)
            } else {
                None
            }
        } else if subject
            .get_schema()
            .fields()
            .contains(&create_bytes_fields())
        {
            let mut config_str = subject
                .get_column_as_vec_nested_primitive::<u8>("bytes")
                .unwrap();
            config_str.pop()
        } else {
            None
        }
    }

    /// Check required fields for the config
    fn check_required_fields(
        subject_name: &str,
        subject_fields: &HashSet<String>,
        required_fields: &[&str],
    ) -> Result<()>
    where
        Self: Sized,
    {
        for required_field in required_fields {
            if !subject_fields.contains(*required_field) {
                return Err(anyhow!(
                    "Subject `{subject_name}` is missing required field for `{required_field}` in `{}`. Required fields are `{required_fields:?}`",
                    Self::get_static_name()
                ));
            }
        }
        Ok(())
    }

    /// Check required variants in the config
    fn check_required_members(&self, subject_name: &str) -> Result<()>;
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataConfig {
    /// Run on CPU rather than GPU even if a GPU is available.
    #[arg(long)]
    pub cpu: bool,

    /// The left hand side table name
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_name: Option<String>,

    /// The right hand side table name
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_name: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_pk: Option<String>,

    /// The right hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_pk: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_fk: Option<String>,

    /// The right hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_fk: Option<String>,

    /// The left hand side values column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_values: Option<Vec<String>>,

    /// The right hand side values column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_values: Option<Vec<String>>,

    /// The left hand side arguments to the operator
    /// JSONized vector of record batches
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_args: Option<String>,

    /// The right hand side arguments to the operator
    /// JSONized vector of record batches
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_args: Option<String>,

    /// Operator keyword arguments in JSON format
    /// that can be deserialized on the fly
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_kwargs: Option<String>,

    /// The streaming strategy to use for the LHS
    #[arg(long)]
    pub lhs_stream: DataStreamManager,

    /// The streaming strategy to use for the RHS if different than the LHS
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_stream: Option<DataStreamManager>,

    /// The operator to invoke
    #[arg(long)]
    pub operator: AvailableOperators,

    /// [DataJoinOperator] specifying the join operator to apply between tables
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub join_operators: Option<DataJoinOperator>,

    /// Minijinja [String] template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_template: Option<AvailableJinja2Templates>,

    /// [DiffType] specifying the Diff format to use
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<DiffType>,

    /// [CodeCompletionType] specifying the code completion format to use
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_completion: Option<CodeCompletionType>,

    /// Universal Diff or V4a Diff in a serialized JSON `Value` representing
    ///   a `Vec<WorkspacePatchSubject>>`
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_patch: Option<String>,

    /// The name of the resulting document after applying the minijinja template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_name: Option<String>,

    /// A serialized JSON [Value] representing the input for the template beyond the table_expression
    ///   where the table_expression will be inserted into to complete the input for the template
    ///
    /// [Value]: serde_json::Value
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_input: Option<String>,

    /// The length of the document chunks
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parser: Option<AvailableParsers>,

    /// The data format to extract
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<DataFormat>,

    /// The data encoding to extract
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<DataEncoding>,

    /// The data schema to extract with
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<AvailableSubjects>,

    /// Vec of Strings for the comparator columns
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp_columns: Option<Vec<String>>,

    /// Vec of [DataComparatorOperator]s specifying the comparator operator to apply to each cmp_column
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp_operators: Option<Vec<DataComparatorOperator>>,

    /// Data Comparison predicates to evaluate parenthetic groups
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp_predicate: Option<DataComparatorPredicate>,

    /// Vec of Strings for the aggregation columns
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_columns: Option<Vec<String>>,

    /// Vec of [DataAggregatorOperator]s specifying the aggregator operator to apply to each agg_column
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_operators: Option<Vec<DataAggregatorOperator>>,

    /// Vec of [String]s for the columns to rename to
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub as_columns: Option<Vec<String>>,

    /// Vec of [String]s for the columns to reorder and include in the schema
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reorder_columns: Option<Vec<String>>,

    /// Vec of of [DataColumnOperator]s specifying the column transformation operator to apply to each lhs_values and optionally rhs_values
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column_operators: Option<Vec<DataColumnOperator>>,

    /// Vec of of [DataCastOperator]s specifying the cast operator to apply to each lhs_values
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_operators: Option<Vec<DataCastOperator>>,

    /// Vec of [DataType]s cast to [String]s specifying the data type to cast each lhs_values to
    ///
    /// [DataType]: arrow::datatypes::DataType
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_datatypes: Option<Vec<String>>,

    /// Vec of Slice of [String]s specifying the template to use when casting each lhs_value to a [String] representation
    ///   where the template is a simple minijinja template with a single expression for the column
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_templates: Option<Vec<String>>,

    /// true for ascending and false for descending
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asc: Option<bool>,

    /// Vec of [String]s for the pivot table columns
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pvt_columns: Option<Vec<String>>,

    /// Vec of [String]s for default values when missing values are encountered
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_values: Option<Vec<String>>,

    /// Data distance operator to apply between vectors
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dist_operator: Option<DataDistanceOperator>,
}

impl DataConfigTrait for DataConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_subject(subject: &Subject) -> Result<Self>
    where
        Self: Sized,
    {
        if let Some(bytes) = Self::from_subject_as_bytes(subject) {
            // Try to build the config
            match serde_json::from_slice::<DataConfig>(&bytes) {
                Ok(config) => {
                    config.check_required_members(subject.get_name())?;
                    Ok(config)
                }
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(),
                    subject.get_name()
                )),
            }
        } else {
            // Check for the required fields
            let required_fields = &["operator", "cpu", "lhs_stream"];
            let column_names = subject
                .get_schema()
                .fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect::<HashSet<_>>();
            Self::check_required_fields(subject.get_name(), &column_names, required_fields)?;

            // Try to build the config
            match subject.to_struct::<DataConfig>() {
                Ok(mut config_vec) => match config_vec.pop() {
                    Some(config) => Ok(config),
                    None => Err(anyhow!(
                        "No config data found for `{}` with subject {}",
                        Self::get_static_name(),
                        subject.get_name()
                    )),
                },
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(),
                    subject.get_name()
                )),
            }
        }
    }

    fn check_required_members(&self, _subject_name: &str) -> Result<()> {
        Ok(())
    }
}

impl MappableTrait for DataConfig {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}
