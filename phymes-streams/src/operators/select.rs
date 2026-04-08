use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    ops::{BitAnd, BitOr, BitXor, Not},
    str::FromStr,
    sync::Arc,
};

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayRef, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array, ListArray, PrimitiveBuilder, RecordBatch, StringArray, UInt8Array, UInt32Array
    },
    compute::{
        cast,
        kernels::{
            bitwise::{
                bitwise_and, bitwise_and_not, bitwise_not, bitwise_or, bitwise_shift_left,
                bitwise_shift_right, bitwise_xor,
            },
            numeric::rem,
        },
    },
    datatypes::{
        ArrowNativeType, DataType, Float32Type, Float64Type, Int64Type, UInt8Type, UInt32Type,
    },
};
use candle_core::{Device, Tensor, WithDType};
use num_traits::{Bounded, Num, NumCast, WrappingShl, WrappingShr};
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectScript, SubjectTrait, from_str_to_data_type};
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use phymes_diagnostics::HashSet;
use serde_json::json;
use tracing::instrument;

use crate::{
    DataColumnOperator, ToolTrait,
    data::{DataCastOperator, DataConfig},
    operators::{
        DataOperatorTrait,
        group_by::{
            build_aggregator_column_fixed_size_list, build_aggregator_column_list_nonprimitive,
            build_aggregator_column_list_primitive,
        },
    },
};

/// Select and cast the [RecordBatch]es based on the [DataCastOperator] and [DataType] with optional column renaming and template injection
/// Transform one or more columns of the [RecordBatch]es by chaining sequential unary or binary [DataColumnOperator]s
#[derive(Debug, Default)]
pub struct Select {
    lhs_values: Vec<String>,
    rhs_values: Vec<String>,
    as_columns: Vec<String>,
    reorder_columns: Vec<String>,
    column_operators: Vec<DataColumnOperator>,
    cast_operators: Vec<DataCastOperator>,
    cast_datatypes: Vec<DataType>,
    cast_templates: Vec<String>,
}

impl MappableTrait for Select {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Select {
    fn get_description(&self) -> String {
        "Cast specified columns using a specified cast operator and cast data type with optional column renaming and template injection."
            .to_string()
    }
    fn to_json_tool_schema(&self) -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_values".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::Array),
                description: Some(
                    "A list of value column identifiers for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "op_kwargs".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "DataCastOperator and DataType with optional column renaming and template injection in the form of a JSON object".to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = Function {
            name: Self::get_static_name().to_string(),
            description: Some(self.get_description()),
            parameters: FunctionParameters {
                schema_type: JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_values".to_string(),
                    "op_kwargs".to_string(),
                ]),
            },
        };
        let tool = Tool {
            r#type: ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

impl DataOperatorTrait for Select {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        let lhs_values = self
            .lhs_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let rhs_values = self
            .rhs_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let as_columns = self
            .as_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let reorder_columns = self
            .reorder_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let cast_templates = self
            .cast_templates
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        select(
            &lhs_values,
            lhs_args,
            &rhs_values,
            &as_columns,
            &reorder_columns,
            &self.column_operators,
            &self.cast_operators,
            &self.cast_datatypes,
            &cast_templates,
            device,
        )
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let rhs_values = config
            .rhs_values
            .as_ref()
            .cloned()
            .unwrap_or(lhs_values.iter().map(|_| String::new()).collect::<Vec<_>>());
        let as_columns = config
            .as_columns
            .as_ref()
            .cloned()
            .unwrap_or(lhs_values.iter().map(|_| String::new()).collect::<Vec<_>>());
        let reorder_columns_default = lhs_values
            .iter()
            .zip(as_columns.iter())
            .map(|(v, a)| {
                if a.is_empty() {
                    v.to_string()
                } else {
                    a.to_string()
                }
            })
            .collect::<Vec<_>>();
        let reorder_columns = config
            .reorder_columns
            .as_ref()
            .cloned()
            .unwrap_or(reorder_columns_default.clone());
        let column_operators = config.column_operators.as_ref().cloned().unwrap_or(
            lhs_values
                .iter()
                .map(|_| DataColumnOperator::default())
                .collect::<Vec<_>>(),
        );
        let cast_operators = config.cast_operators.as_ref().cloned().unwrap_or(
            lhs_values
                .iter()
                .map(|_| DataCastOperator::default())
                .collect::<Vec<_>>(),
        );
        let cast_datatypes_str = config.cast_datatypes.as_ref().cloned().unwrap_or(
            lhs_values
                .iter()
                .map(|_| "Utf8".to_string())
                .collect::<Vec<_>>(),
        );
        let mut cast_datatypes = Vec::new();
        for s in cast_datatypes_str.into_iter() {
            cast_datatypes.push(from_str_to_data_type(&s)?);
        }
        let cast_templates = config
            .cast_templates
            .as_ref()
            .cloned()
            .unwrap_or(lhs_values.iter().map(|_| String::new()).collect::<Vec<_>>());

        // Ensure that the array lengths match
        if lhs_values.len() != as_columns.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the as_columns length {}",
                lhs_values.len(),
                as_columns.len()
            ));
        } else if lhs_values.len() != rhs_values.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the rhs_values length {}",
                lhs_values.len(),
                rhs_values.len()
            ));
        } else if lhs_values.len() != column_operators.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the column_operators length {}",
                lhs_values.len(),
                column_operators.len()
            ));
        } else if lhs_values.len() != cast_operators.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the cast_operators length {}",
                lhs_values.len(),
                cast_operators.len()
            ));
        } else if lhs_values.len() != cast_datatypes.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the cast_datatypes length {}",
                lhs_values.len(),
                cast_datatypes.len()
            ));
        } else if lhs_values.len() != cast_templates.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the with_templates length {}",
                lhs_values.len(),
                cast_templates.len()
            ));
        } else if lhs_values.len() < reorder_columns.len() {
            return Err(anyhow!(
                "lhs_values length {} is less than the reorder_columns length {}",
                lhs_values.len(),
                cast_templates.len()
            ));
        }

        {
            // Check that the reorder_columns are in the as_columns and lhs_values
            let as_columns_set = reorder_columns_default
                .iter()
                .map(|v| v.as_str())
                .collect::<HashSet<&str>>();
            let reorder_columns_set = reorder_columns
                .iter()
                .map(|v| v.as_str())
                .collect::<HashSet<&str>>();
            if !reorder_columns_set.is_subset(&as_columns_set) {
                return Err(anyhow!(
                    "reorder_columns {reorder_columns_set:?} is not a subset of as_columns and lhs_values {as_columns_set:?}",
                ));
            }
        }

        Ok(Select {
            lhs_values,
            rhs_values,
            as_columns,
            reorder_columns,
            column_operators,
            cast_operators,
            cast_datatypes,
            cast_templates,
        })
    }
}

/// Helper function to compute the column binary operator for tensors
fn column_binary_operator_tensor<T>(
    lhs_column: &str,
    rhs_column: &str,
    column_operator: &DataColumnOperator,
    lhs_arr: &ArrayRef,
    rhs_arr: &ArrayRef,
    device: &Device,
) -> Result<Tensor>
where
    T: Num + Bounded + NumCast + Send + Sync + WithDType + 'static,
{
    let lhs_vec = Subject::get_array_as_vec_primitive::<T>(lhs_arr, lhs_column)?;
    let lhs_tensor = Tensor::from_iter(lhs_vec, device)?;
    let rhs_vec = Subject::get_array_as_vec_primitive::<T>(rhs_arr, rhs_column)?;
    let rhs_tensor = Tensor::from_iter(rhs_vec, device)?;
    let tensor = match column_operator {
        DataColumnOperator::Add => (lhs_tensor + rhs_tensor)?,
        DataColumnOperator::Sub => (lhs_tensor - rhs_tensor)?,
        DataColumnOperator::Mult => (lhs_tensor * rhs_tensor)?,
        DataColumnOperator::Div => (lhs_tensor / rhs_tensor)?,
        DataColumnOperator::Min => lhs_tensor.minimum(&rhs_tensor)?,
        DataColumnOperator::Max => lhs_tensor.maximum(&rhs_tensor)?,
        _ => {
            return Err(anyhow!(
                "Unsupported column operator {column_operator} for lhs column {lhs_column} and rhs column {rhs_column}"
            ));
        }
    };
    Ok(tensor)
}

/// Helper function to compute the column binary operator for tensors
fn column_unary_operator_tensor<T>(
    lhs_column: &str,
    column_operator: &DataColumnOperator,
    lhs_arr: &ArrayRef,
    device: &Device,
) -> Result<Tensor>
where
    T: Num + Bounded + NumCast + Send + Sync + WithDType + 'static,
{
    let lhs_vec = Subject::get_array_as_vec_primitive::<T>(lhs_arr, lhs_column)?;
    let lhs_tensor = Tensor::from_iter(lhs_vec, device)?;
    let shape = lhs_tensor.shape().to_owned();
    let tensor = match column_operator {
        DataColumnOperator::BroadcastMin => lhs_tensor.min_all()?.broadcast_as(shape)?,
        DataColumnOperator::BroadcastMax => lhs_tensor.max_all()?.broadcast_as(shape)?,
        DataColumnOperator::BroadcastMean => lhs_tensor.mean_all()?.broadcast_as(shape)?,
        DataColumnOperator::BroadcastVar => lhs_tensor.var(0)?.broadcast_as(shape)?,
        DataColumnOperator::CumSum => lhs_tensor.cumsum(0)?.broadcast_as(shape)?,
        _ => {
            return Err(anyhow!(
                "Unsupported column operator {column_operator} for lhs column {lhs_column}.",
            ));
        }
    };
    Ok(tensor)
}

/// Helper function to compute the column binary operator for arrow
fn column_binary_operator_arrow<T, D>(
    lhs_column: &str,
    rhs_column: &str,
    column_operator: &DataColumnOperator,
    lhs_arr: &ArrayRef,
    rhs_arr: &ArrayRef,
) -> Result<ArrayRef>
where
    T: ArrowNativeType + Num + Bounded + NumCast + Send + Sync + WithDType + 'static,
    D: ArrowPrimitiveType<Native = T> + 'static,
    <D as ArrowPrimitiveType>::Native: Not<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: BitAnd<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: BitXor<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: BitOr<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: WrappingShl<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: WrappingShr<Output = <D as ArrowPrimitiveType>::Native>,
{
    let lhs_vec = Subject::get_array_as_vec_primitive::<T>(lhs_arr, lhs_column)?;
    let mut builder = PrimitiveBuilder::<D>::new();
    builder.append_slice(&lhs_vec);
    let lhs_arr = builder.finish();
    let rhs_vec = Subject::get_array_as_vec_primitive::<T>(rhs_arr, rhs_column)?;
    let mut builder = PrimitiveBuilder::<D>::new();
    builder.append_slice(&rhs_vec);
    let rhs_arr = builder.finish();
    let arr = match column_operator {
        DataColumnOperator::And => bitwise_and::<D>(&lhs_arr, &rhs_arr)?,
        DataColumnOperator::AndNot => bitwise_and_not::<D>(&lhs_arr, &rhs_arr)?,
        DataColumnOperator::Or => bitwise_or::<D>(&lhs_arr, &rhs_arr)?,
        DataColumnOperator::XOr => bitwise_xor::<D>(&lhs_arr, &rhs_arr)?,
        DataColumnOperator::LeftShift => bitwise_shift_left::<D>(&lhs_arr, &rhs_arr)?,
        DataColumnOperator::RightShift => bitwise_shift_right::<D>(&lhs_arr, &rhs_arr)?,
        _ => {
            return Err(anyhow!(
                "Unsupported column operator {column_operator} for lhs column {lhs_column} and rhs column {rhs_column}"
            ));
        }
    };
    Ok(Arc::new(arr))
}

/// Helper function to compute the column unary operator for arrow
fn column_unary_operator_arrow<T, D>(
    lhs_column: &str,
    column_operator: &DataColumnOperator,
    lhs_arr: &ArrayRef,
) -> Result<ArrayRef>
where
    T: ArrowNativeType + Num + Bounded + NumCast + Send + Sync + WithDType + 'static,
    D: ArrowPrimitiveType<Native = T> + 'static,
    <D as ArrowPrimitiveType>::Native: Not<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: BitAnd<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: BitXor<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: BitOr<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: WrappingShl<Output = <D as ArrowPrimitiveType>::Native>,
    <D as ArrowPrimitiveType>::Native: WrappingShr<Output = <D as ArrowPrimitiveType>::Native>,
{
    let lhs_vec = Subject::get_array_as_vec_primitive::<T>(lhs_arr, lhs_column)?;
    let mut builder = PrimitiveBuilder::<D>::new();
    builder.append_slice(&lhs_vec);
    let lhs_arr = builder.finish();
    let arr = match column_operator {
        DataColumnOperator::Not => bitwise_not::<D>(&lhs_arr)?,
        _ => {
            return Err(anyhow!(
                "Unsupported column operator {column_operator} for lhs column {lhs_column}."
            ));
        }
    };
    Ok(Arc::new(arr))
}

/// Helper function to choose the source of the column
///
/// # Notes
/// * The lhs_batches (new columns) are searched first to enable chaining up column updates
fn find_column(
    lhs_table: &Subject,
    lhs_batches: &[&(&&str, ArrayRef)],
    column_name: &str,
) -> Result<ArrayRef> {
    let mut lhs_filtered = lhs_batches
        .iter()
        .filter_map(|(name, arr)| {
            if &&column_name == name {
                Some(arr)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if let Some(arr) = lhs_filtered.pop() {
        Ok(arr.clone())
    } else if let Ok(_field) = lhs_table.get_schema().field_with_name(column_name) {
        Ok(lhs_table.get_column_as_array(column_name)?)
    } else {
        Err(anyhow!(
            "Unable to find column {column_name} in the provided lhs_args nor in the new lhs batches for `Select` Operator."
        ))
    }
}

/// Helper function to get the rhs
fn rhs_helper(
    rhs_values: &[&str],
    lhs_table: &Subject,
    lhs_batches: &[&(&&str, ArrayRef)],
    index: usize,
) -> Result<(Option<String>, Option<ArrayRef>)> {
    if let Some(rhs_column) = rhs_values.get(index) {
        if rhs_column.is_empty() {
            Ok((None, None))
        } else {
            let rhs_arr = find_column(lhs_table, lhs_batches, rhs_column)?;
            Ok((Some(rhs_column.to_string()), Some(rhs_arr)))
        }
    } else {
        Ok((None, None))
    }
}

/// Hashes a string into an integer using Rust's DefaultHasher if it cannot be parsed into an integer directly.
/// # Notes
/// - This is NOT cryptographically secure — use for non-security purposes only.
fn hash_string<T>(s: &str) -> Result<T>
where
    T: Num + Bounded + NumCast + Send + Sync + WithDType + FromStr + 'static,
{
    match s.parse::<T>() {
        Ok(parsed) => Ok(parsed),
        Err(_) => {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            T::from(hasher.finish()).ok_or(anyhow!("Could not Hash {s} to type {:?}", T::DTYPE))
        }
    }
}

/// Reorder the columns in a pre-[RecordBatch] vec
pub fn reorder_batch_vec_columns(
    batch_vec: &[(&str, ArrayRef)],
    reorder_columns: &[&str],
) -> Vec<(String, ArrayRef)> {
    let mut batch_vec_index = Vec::with_capacity(reorder_columns.len());
    for (column, arr) in batch_vec.iter() {
        for (iter, reorder) in reorder_columns.iter().enumerate() {
            if column == reorder {
                batch_vec_index.push((iter, column, arr));
                break;
            }
        }
    }
    batch_vec_index.sort_by_key(|k| k.0);
    batch_vec_index
        .into_iter()
        .map(|(_iter, col, arr)| (col.to_string(), arr.to_owned()))
        .collect::<Vec<_>>()
}

/// Cast and transform specified columns using a specified cast operator and cast data type with optional column renaming and template injection
///
/// # Notes
/// * Order of operations are as follows:
///   0. Missing column with defaults
///   1. Transform
///   2. Cast
///   3. Template
///   4. Rename
///
/// # Usage
///
/// * An SQL equivalent would be the following at the row level, e.g., SELECT CAST('100' AS INTEGER);
/// * An SQL equivalent would be the following at the table level, e.g., SELECT COUNT(COL1) AS count...
///   `lhs_values` = ["COL1", ...]
///   `as_columns` = ["count", ...]
///   `cast_operators` = [DataCastOperator::Cast, ...]
///   `cast_datatypes` = [DataType::UInt32, ...]
///   `cast_templates` = ["", ...] ignored since this is not a cast to DataType::Utf8
/// * An SQL equivalent to add two columns together and assign the values to a new column would be SELECT COL1 + COL2 as COL3 ...
///   `lhs_values` = ["COL1", ...]
///   `rhs_values` = ["COL2", ...]
///   `as_columns` = ["COL3", ...]
///   `column_operators` = [DataColumnOperator::Add, ...]
/// * An SQL equivalent to add two columns together and update an existing column in place would be SELECT COL1 = COL1 + COL2 ...
///   `lhs_values` = ["COL1", ...]
///   `rhs_values` = ["COL2", ...]
///   `as_columns` = ["COL1", ...]
///   `column_operators` = [DataColumnOperator::Add, ...]
/// * An SQL equivalent to create a copy of a column would be SELECT COL3=COL1 ...
///   `lhs_values` = ["COL1", "COL1", ...]
///   `rhs_values` = ["", "", ...]
///   `as_columns` = ["COL1", "COL3", ...]
///   `column_operators` = [DataColumnOperator::None, DataColumnOperator::None, ...]
/// * An SQL equivalent to chain multiple column operations sequentially would be SELECT COL4=(COL1 + COL2 - COL3) ...
///   `lhs_values` = ["COL1", "COL4", ...]
///   `rhs_values` = ["COL2", "COL3", ...]
///   `as_columns` = ["COL4", "COL4", ...]
///   `column_operators` = [DataColumnOperator::Add, DataColumnOperator::Sub, ...]
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the left-hand side columns to apply the unary or binary transformation to
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `rhs_values` - Optional Slice of Strings for the right-hand side columns to apply the binary transformation to
/// * `as_columns` - Slice of [String]s for the columns to rename to
/// * `reorder_columns` - Slice of [String]s for designating the order of columns as they will appear in the generated schema
///   omitting column names will remove the column from inclusion in the batch
/// * `column_operators` - Slice of [DataColumnOperator]s specifying the transformation between two columns
/// * `cast_operators` - Slice of [DataCastOperator]s specifying the cast operator to apply to each lhs_values
/// * `cast_datatypes` - Slice of [DataType]s specifying the data type to cast each lhs_values to
/// * `cast_templates` - Slice of [String]s specifying the template to use when casting each lhs_value to a [String] representation
///   where the template is a simple minijinja template with a single expression for the column
///   e.g., "Hello {{ COL1 }}"
/// * `device` - The compute device
#[allow(clippy::too_many_arguments)]
#[instrument(skip(
    lhs_values,
    lhs_args,
    rhs_values,
    as_columns,
    reorder_columns,
    column_operators,
    cast_operators,
    cast_datatypes,
    cast_templates,
    device
))]
pub fn select(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    rhs_values: &[&str],
    as_columns: &[&str],
    reorder_columns: &[&str],
    column_operators: &[DataColumnOperator],
    cast_operators: &[DataCastOperator],
    cast_datatypes: &[DataType],
    cast_templates: &[&str],
    device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs into an ArrowTable
    let lhs_table = Subject::get_builder()
        .with_name("select")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;

    // Local mutable copy of `cast_template`
    let mut cast_templates = cast_templates
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    // Apply the cast and optional column renaming and template injection based on the lhs_values
    let mut missing_vec: Vec<(&&str, ArrayRef)> = Vec::new();
    let mut batch_vec: Vec<(&&str, ArrayRef)> = Vec::new();
    for (index, column_name) in lhs_values.iter().enumerate() {
        // Initialize missing columns with default values
        if let Err(_err) = find_column(
            &lhs_table,
            &batch_vec.iter().collect::<Vec<_>>(),
            column_name,
        ) {
            match column_operators.get(index).unwrap() {
                DataColumnOperator::Value => match cast_datatypes.get(index).unwrap() {
                    DataType::Boolean => {
                        let value = if let Some(template) = cast_templates.get_mut(index) {
                            if template.is_empty() {
                                Default::default()
                            } else {
                                let value = template.parse::<bool>()?;
                                *template = String::new();
                                value
                            }
                        } else {
                            Default::default()
                        };
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| value)
                            .collect::<Vec<bool>>();
                        let arr: ArrayRef = Arc::new(BooleanArray::from(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::UInt8 => {
                        let value = if let Some(template) = cast_templates.get_mut(index) {
                            if template.is_empty() {
                                Default::default()
                            } else {
                                let value = template.parse::<u8>()?;
                                *template = String::new();
                                value
                            }
                        } else {
                            Default::default()
                        };
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| value)
                            .collect::<Vec<u8>>();
                        let arr: ArrayRef = Arc::new(UInt8Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::UInt32 => {
                        let value = if let Some(template) = cast_templates.get_mut(index) {
                            if template.is_empty() {
                                Default::default()
                            } else {
                                let value = template.parse::<u32>()?;
                                *template = String::new();
                                value
                            }
                        } else {
                            Default::default()
                        };
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| value)
                            .collect::<Vec<u32>>();
                        let arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Int64 => {
                        let value = if let Some(template) = cast_templates.get_mut(index) {
                            if template.is_empty() {
                                Default::default()
                            } else {
                                let value = template.parse::<i64>()?;
                                *template = String::new();
                                value
                            }
                        } else {
                            Default::default()
                        };
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| value)
                            .collect::<Vec<i64>>();
                        let arr: ArrayRef = Arc::new(Int64Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Float32 => {
                        let value = if let Some(template) = cast_templates.get_mut(index) {
                            if template.is_empty() {
                                Default::default()
                            } else {
                                let value = template.parse::<f32>()?;
                                *template = String::new();
                                value
                            }
                        } else {
                            Default::default()
                        };
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| value)
                            .collect::<Vec<f32>>();
                        let arr: ArrayRef = Arc::new(Float32Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Float64 => {
                        let value = if let Some(template) = cast_templates.get_mut(index) {
                            if template.is_empty() {
                                Default::default()
                            } else {
                                let value = template.parse::<f64>()?;
                                *template = String::new();
                                value
                            }
                        } else {
                            Default::default()
                        };
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| value)
                            .collect::<Vec<f64>>();
                        let arr: ArrayRef = Arc::new(Float64Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Utf8 => {
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| String::new())
                            .collect::<Vec<String>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported column operator {} and data type {} for missing column {column_name}",
                            column_operators.get(index).unwrap(),
                            cast_datatypes.get(index).unwrap()
                        ));
                    }
                },
                DataColumnOperator::Zeros => match cast_datatypes.get(index).unwrap() {
                    DataType::UInt8 => {
                        let default_vec =
                            (0..lhs_table.count_rows()).map(|_| 0).collect::<Vec<u8>>();
                        let arr: ArrayRef = Arc::new(UInt8Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::UInt32 => {
                        let default_vec =
                            (0..lhs_table.count_rows()).map(|_| 0).collect::<Vec<u32>>();
                        let arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Int64 => {
                        let default_vec =
                            (0..lhs_table.count_rows()).map(|_| 0).collect::<Vec<i64>>();
                        let arr: ArrayRef = Arc::new(Int64Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Float32 => {
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| 0f32)
                            .collect::<Vec<f32>>();
                        let arr: ArrayRef = Arc::new(Float32Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Float64 => {
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| 0f64)
                            .collect::<Vec<f64>>();
                        let arr: ArrayRef = Arc::new(Float64Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported column operator {} and data type {} for missing column {column_name}",
                            column_operators.get(index).unwrap(),
                            cast_datatypes.get(index).unwrap()
                        ));
                    }
                },
                DataColumnOperator::Ones => match cast_datatypes.get(index).unwrap() {
                    DataType::UInt8 => {
                        let default_vec =
                            (0..lhs_table.count_rows()).map(|_| 1).collect::<Vec<u8>>();
                        let arr: ArrayRef = Arc::new(UInt8Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::UInt32 => {
                        let default_vec =
                            (0..lhs_table.count_rows()).map(|_| 1).collect::<Vec<u32>>();
                        let arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Int64 => {
                        let default_vec =
                            (0..lhs_table.count_rows()).map(|_| 1).collect::<Vec<i64>>();
                        let arr: ArrayRef = Arc::new(Int64Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Float32 => {
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| 1f32)
                            .collect::<Vec<f32>>();
                        let arr: ArrayRef = Arc::new(Float32Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    DataType::Float64 => {
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| 1f64)
                            .collect::<Vec<f64>>();
                        let arr: ArrayRef = Arc::new(Float64Array::from_iter_values(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported column operator {} and data type {} for missing column {column_name}",
                            column_operators.get(index).unwrap(),
                            cast_datatypes.get(index).unwrap()
                        ));
                    }
                },
                DataColumnOperator::String => match cast_datatypes.get(index).unwrap() {
                    DataType::Utf8 => {
                        let default_vec = (0..lhs_table.count_rows())
                            .map(|_| String::new())
                            .collect::<Vec<String>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(default_vec));
                        missing_vec.push((column_name, arr));
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported column operator {} and data type {} for missing column {column_name}",
                            column_operators.get(index).unwrap(),
                            cast_datatypes.get(index).unwrap()
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported column operator {} for missing column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            }
        }

        // Combine the missing with the new columns
        let batch_missing_vec = batch_vec
            .iter()
            .chain(missing_vec.iter())
            .collect::<Vec<_>>();

        // Transform the column
        let lhs_arr = find_column(&lhs_table, &batch_missing_vec, column_name)?;
        let column_data_type = lhs_arr.data_type();
        let column_cast: ArrayRef = match column_operators.get(index).unwrap() {
            DataColumnOperator::Add
            | DataColumnOperator::Sub
            | DataColumnOperator::Mult
            | DataColumnOperator::Div
            | DataColumnOperator::Max
            | DataColumnOperator::Min => match column_data_type {
                DataType::UInt8 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    let tensor = column_binary_operator_tensor::<u8>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                        device,
                    )?;
                    Arc::new(UInt8Array::from_iter_values(tensor.to_vec1::<u8>()?))
                }
                DataType::UInt32 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    let tensor = column_binary_operator_tensor::<u32>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                        device,
                    )?;
                    Arc::new(UInt32Array::from_iter_values(tensor.to_vec1::<u32>()?))
                }
                DataType::Int64 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    let tensor = column_binary_operator_tensor::<i64>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                        device,
                    )?;
                    Arc::new(Int64Array::from_iter_values(tensor.to_vec1::<i64>()?))
                }
                DataType::Float32 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    let tensor = column_binary_operator_tensor::<f32>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                        device,
                    )?;
                    Arc::new(Float32Array::from_iter_values(tensor.to_vec1::<f32>()?))
                }
                DataType::Float64 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    let tensor = column_binary_operator_tensor::<f64>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                        device,
                    )?;
                    Arc::new(Float64Array::from_iter_values(tensor.to_vec1::<f64>()?))
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::BroadcastMax
            | DataColumnOperator::BroadcastMin
            | DataColumnOperator::BroadcastMean
            | DataColumnOperator::BroadcastVar
            | DataColumnOperator::CumSum => match column_data_type {
                DataType::UInt8 => {
                    let tensor = column_unary_operator_tensor::<u8>(
                        column_name,
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        device,
                    )?;
                    Arc::new(UInt8Array::from_iter_values(tensor.to_vec1::<u8>()?))
                }
                DataType::UInt32 => {
                    let tensor = column_unary_operator_tensor::<u32>(
                        column_name,
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        device,
                    )?;
                    Arc::new(UInt32Array::from_iter_values(tensor.to_vec1::<u32>()?))
                }
                DataType::Int64 => {
                    let tensor = column_unary_operator_tensor::<i64>(
                        column_name,
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        device,
                    )?;
                    Arc::new(Int64Array::from_iter_values(tensor.to_vec1::<i64>()?))
                }
                DataType::Float32 => {
                    let tensor = column_unary_operator_tensor::<f32>(
                        column_name,
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        device,
                    )?;
                    Arc::new(Float32Array::from_iter_values(tensor.to_vec1::<f32>()?))
                }
                DataType::Float64 => {
                    let tensor = column_unary_operator_tensor::<f64>(
                        column_name,
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        device,
                    )?;
                    Arc::new(Float64Array::from_iter_values(tensor.to_vec1::<f64>()?))
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::Rem => {
                let lhs_arr = find_column(&lhs_table, &batch_missing_vec, column_name)?;
                let rhs_arr = find_column(
                    &lhs_table,
                    &batch_missing_vec,
                    rhs_values.get(index).unwrap(),
                )?;
                rem(&lhs_arr, &rhs_arr)?
            }
            DataColumnOperator::List => match column_data_type {
                DataType::UInt8 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u8>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<u8>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| vec![l, r])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::UInt32 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u32>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<u32>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| vec![l, r])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Int64 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<i64>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<i64>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| vec![l, r])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Float32 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<f32>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<f32>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| vec![l, r])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Float64 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<f64>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<f64>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| vec![l, r])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Utf8 => {
                    let lhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| vec![l, r])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u8>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<u8>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u32>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<u32>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<i64>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<i64>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<f32>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<f32>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<f64>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<f64>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_nonprimitive::<String>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u8>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u32>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<i64>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<f32>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<f32, Float32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<f64>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<f64, Float64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_nonprimitive::<String>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(mut l, r)| {
                                l.extend(r);
                                l
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::Set => match column_data_type {
                DataType::UInt8 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u8>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<u8>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| {
                            [l, r]
                                .into_iter()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::UInt32 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u32>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<u32>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| {
                            [l, r]
                                .into_iter()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Int64 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<i64>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_primitive::<i64>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| {
                            [l, r]
                                .into_iter()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Utf8 => {
                    let lhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| {
                            [l, r]
                                .into_iter()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u8>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u32>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<i64>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_nonprimitive::<String>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u8>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<u32>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_primitive::<i64>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .collect::<Vec<_>>();
                        let rhs_vec = find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .iter()
                        .filter_map(|s| {
                            s.map(|s| {
                                Subject::get_array_as_vec_nonprimitive::<String>(
                                    &s,
                                    rhs_values.get(index).unwrap(),
                                )
                                .unwrap_or_default()
                            })
                        })
                        .collect::<Vec<_>>();
                        let agg_values = lhs_vec
                            .into_iter()
                            .zip(rhs_vec.into_iter())
                            .map(|(l, r)| {
                                l.into_iter()
                                    .chain(r.into_iter())
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::Concat => match column_data_type {
                DataType::Utf8 => {
                    let lhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let rhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(
                            &lhs_table,
                            &batch_missing_vec,
                            rhs_values.get(index).unwrap(),
                        )?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .zip(rhs_vec.into_iter())
                        .map(|(l, r)| [l, r].join(""))
                        .collect::<Vec<_>>();
                    Arc::new(StringArray::from(agg_values))
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::And
            | DataColumnOperator::AndNot
            | DataColumnOperator::Or
            | DataColumnOperator::XOr
            | DataColumnOperator::LeftShift
            | DataColumnOperator::RightShift => match column_data_type {
                DataType::UInt8 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    column_binary_operator_arrow::<u8, UInt8Type>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                    )?
                }
                DataType::UInt32 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    column_binary_operator_arrow::<u32, UInt32Type>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                    )?
                }
                DataType::Int64 => {
                    let (rhs_column, rhs_arr) =
                        rhs_helper(rhs_values, &lhs_table, &batch_missing_vec, index)?;
                    if rhs_column.is_none() && rhs_arr.is_none() {
                        return Err(anyhow!(
                            "rhs column cannot be None for column operator {} with lhs column {column_name}.",
                            column_operators.get(index).unwrap()
                        ));
                    }
                    column_binary_operator_arrow::<i64, Int64Type>(
                        column_name,
                        &rhs_column.unwrap(),
                        column_operators.get(index).unwrap(),
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        &rhs_arr.unwrap(),
                    )?
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::Not => match column_data_type {
                DataType::UInt8 => column_unary_operator_arrow::<u8, UInt8Type>(
                    column_name,
                    column_operators.get(index).unwrap(),
                    &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                )?,
                DataType::UInt32 => column_unary_operator_arrow::<u32, UInt32Type>(
                    column_name,
                    column_operators.get(index).unwrap(),
                    &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                )?,
                DataType::Int64 => column_unary_operator_arrow::<i64, Int64Type>(
                    column_name,
                    column_operators.get(index).unwrap(),
                    &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                )?,
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::Len => match column_data_type {
                DataType::Utf8 => {
                    let lhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec
                        .into_iter()
                        .map(|s| s.len() as u32)
                        .collect::<Vec<_>>();
                    Arc::new(UInt32Array::from(agg_values))
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            }
            DataColumnOperator::BroadcastCount => {
                let num_rows = lhs_table.count_rows();
                let agg_vec = (0..num_rows).map(|v| v as u32).collect::<Vec<_>>();
                Arc::new(UInt32Array::from(agg_vec))
            }
            DataColumnOperator::BroadcastList => match column_data_type {
                DataType::UInt8 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u8>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::UInt32 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u32>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Int64 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<i64>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Float32 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<f32>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Float64 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<f64>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Utf8 => {
                    let lhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?;
                    let agg_values = lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<u8>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<u32>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<i64>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<f32>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<f64>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<f32, Float32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Float64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<f64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<f64, Float64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        let agg_values =
                            lhs_vec.iter().map(|_| lhs_vec.clone()).collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::BroadcastSet => match column_data_type {
                DataType::UInt8 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u8>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?
                    .into_iter()
                    .collect::<HashSet<_>>();
                    let agg_values = lhs_vec
                        .iter()
                        .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::UInt32 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<u32>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?
                    .into_iter()
                    .collect::<HashSet<_>>();
                    let agg_values = lhs_vec
                        .iter()
                        .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Int64 => {
                    let lhs_vec = Subject::get_array_as_vec_primitive::<i64>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?
                    .into_iter()
                    .collect::<HashSet<_>>();
                    let agg_values = lhs_vec
                        .iter()
                        .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::Utf8 => {
                    let lhs_vec = Subject::get_array_as_vec_nonprimitive::<String>(
                        &find_column(&lhs_table, &batch_missing_vec, column_name)?,
                        column_name,
                    )?
                    .into_iter()
                    .collect::<HashSet<_>>();
                    let agg_values = lhs_vec
                        .iter()
                        .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        agg_values,
                        column_data_type.clone(),
                    )
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::UInt32 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Int64 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                        .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    DataType::Utf8 => {
                        let lhs_vec = find_column(&lhs_table, &batch_missing_vec, column_name)?
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .filter_map(|s| {
                                s.map(|s| {
                                    Subject::get_array_as_vec_nonprimitive::<String>(
                                        &s,
                                        column_name,
                                    )
                                    .unwrap_or_default()
                                })
                            })
                            .flatten()
                            .collect::<HashSet<_>>();
                        let agg_values = lhs_vec
                            .iter()
                            .map(|_| lhs_vec.iter().cloned().collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            column_data_type.clone(),
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                            column_operators.get(index).unwrap()
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {column_data_type} for column operator {} and column {column_name}",
                        column_operators.get(index).unwrap()
                    ));
                }
            },
            DataColumnOperator::None
            | DataColumnOperator::Zeros
            | DataColumnOperator::Ones
            | DataColumnOperator::String
            | DataColumnOperator::Value => {
                find_column(&lhs_table, &batch_missing_vec, column_name)?
            }
        };

        // Try casting if possible
        let column_data_type = column_cast.data_type();
        let column_cast: ArrayRef = match cast_operators.get(index).unwrap() {
            DataCastOperator::Cast => {
                let to_type = cast_datatypes.get(index).unwrap();
                cast(&column_cast, to_type)?
            }
            DataCastOperator::BytesToString => {
                let to_type = cast_datatypes.get(index).unwrap();
                if to_type != &DataType::Utf8 {
                    return Err(anyhow!(
                        "Unsupported data type {to_type} for casting from Bytes to String for column {column_name}. The supported data type is Utf8."
                    ));
                }
                match column_data_type {
                    DataType::FixedSizeList(f, _) => match f.data_type() {
                        DataType::UInt8 => {
                            let lhs_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                            .unwrap_or_default()
                                    })
                                })
                                .collect::<Vec<_>>();
                            let cast_vec = lhs_vec
                                .into_iter()
                                .map(|v| String::from_utf8_lossy(&v).into_owned())
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(cast_vec))
                        }
                        DataType::UInt32 => {
                            let lhs_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                            .unwrap_or_default()
                                    })
                                })
                                .collect::<Vec<_>>();
                            let cast_vec = lhs_vec
                                .into_iter()
                                .map(|v| {
                                    let bytes = v.into_iter().map(|i| i as u8).collect::<Vec<_>>();
                                    String::from_utf8_lossy(&bytes).into_owned()
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(cast_vec))
                        }
                        DataType::Int64 => {
                            let lhs_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                            .unwrap_or_default()
                                    })
                                })
                                .collect::<Vec<_>>();
                            let cast_vec = lhs_vec
                                .into_iter()
                                .map(|v| {
                                    let bytes = v.into_iter().map(|i| i as u8).collect::<Vec<_>>();
                                    String::from_utf8_lossy(&bytes).into_owned()
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(cast_vec))
                        }
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {column_data_type} for casting from Bytes to String for column {column_name}. The supported data types are List-UInt8, List-UInt32, and List-Int64",
                            ));
                        }
                    },
                    DataType::List(f) => match f.data_type() {
                        DataType::UInt8 => {
                            let lhs_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        Subject::get_array_as_vec_primitive::<u8>(&s, column_name)
                                            .unwrap_or_default()
                                    })
                                })
                                .collect::<Vec<_>>();
                            let cast_vec = lhs_vec
                                .into_iter()
                                .map(|v| String::from_utf8_lossy(&v).into_owned())
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(cast_vec))
                        }
                        DataType::UInt32 => {
                            let lhs_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        Subject::get_array_as_vec_primitive::<u32>(&s, column_name)
                                            .unwrap_or_default()
                                    })
                                })
                                .collect::<Vec<_>>();
                            let cast_vec = lhs_vec
                                .into_iter()
                                .map(|v| {
                                    let bytes = v.into_iter().map(|i| i as u8).collect::<Vec<_>>();
                                    String::from_utf8_lossy(&bytes).into_owned()
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(cast_vec))
                        }
                        DataType::Int64 => {
                            let lhs_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        Subject::get_array_as_vec_primitive::<i64>(&s, column_name)
                                            .unwrap_or_default()
                                    })
                                })
                                .collect::<Vec<_>>();
                            let cast_vec = lhs_vec
                                .into_iter()
                                .map(|v| {
                                    let bytes = v.into_iter().map(|i| i as u8).collect::<Vec<_>>();
                                    String::from_utf8_lossy(&bytes).into_owned()
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(cast_vec))
                        }
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {column_data_type} for casting from Bytes to String for column {column_name}. The supported data types are List-UInt8, List-UInt32, and List-Int64",
                            ));
                        }
                    },
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for casting from Bytes to String for column {column_name}. The supported data types are List-UInt8, List-UInt32, and List-Int64",
                        ));
                    }
                }
            }
            DataCastOperator::Hash => {
                match (column_data_type, cast_datatypes.get(index).unwrap()) {
                    (DataType::Utf8, DataType::UInt32) => {
                        let lhs_vec = column_cast
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| hash_string::<u32>(s.unwrap_or_default()).unwrap_or_default())
                            .collect::<Vec<_>>();
                        Arc::new(UInt32Array::from(lhs_vec))
                    }
                    (DataType::Utf8, DataType::Int64) => {
                        let lhs_vec = column_cast
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| hash_string::<i64>(s.unwrap_or_default()).unwrap_or_default())
                            .collect::<Vec<_>>();
                        Arc::new(Int64Array::from(lhs_vec))
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for Hashing to {} for column {column_name}. The supported data types are from Utf8 to UInt32 and Int64",
                            cast_datatypes.get(index).unwrap()
                        ));
                    }
                }
            }
            DataCastOperator::None => column_cast,
        };

        // Inject into a string template
        let column_data_type = column_cast.data_type();
        let column_cast = if let Some(template) = cast_templates.get(index) {
            if template.is_empty() {
                column_cast
            } else {
                match column_data_type {
                    DataType::UInt8 => {
                        let template = SubjectScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<UInt8Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        Arc::new(StringArray::from(arr_vec))
                    }
                    DataType::UInt32 => {
                        let template = SubjectScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<UInt32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        Arc::new(StringArray::from(arr_vec))
                    }
                    DataType::Int64 => {
                        let template = SubjectScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        Arc::new(StringArray::from(arr_vec))
                    }
                    DataType::Float32 => {
                        let template = SubjectScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        Arc::new(StringArray::from(arr_vec))
                    }
                    DataType::Float64 => {
                        let template = SubjectScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        Arc::new(StringArray::from(arr_vec))
                    }
                    DataType::Utf8 => {
                        let template = SubjectScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        Arc::new(StringArray::from(arr_vec))
                    }
                    DataType::FixedSizeList(f, _) => match f.data_type() {
                        DataType::UInt8 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<u8>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::UInt32 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<u32>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Int64 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<i64>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Float32 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<f32>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Float64 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<f64>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Utf8 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec =
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s,
                                                column_name,
                                            )
                                            .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {column_data_type} for injecting into a String template for column {column_name}.",
                            ));
                        }
                    },
                    DataType::List(f) => match f.data_type() {
                        DataType::UInt8 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<u8>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::UInt32 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<u32>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Int64 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<i64>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Float32 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<f32>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Float64 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec = Subject::get_array_as_vec_primitive::<f64>(
                                            &s,
                                            column_name,
                                        )
                                        .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        DataType::Utf8 => {
                            let template = SubjectScript::new_from_template(template.to_string());
                            let arr_vec = column_cast
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| {
                                    s.map(|s| {
                                        let s_vec =
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s,
                                                column_name,
                                            )
                                            .unwrap_or_default();
                                        template
                                            .apply_template(
                                                &json!({column_name.to_string(): s_vec}),
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect::<Vec<_>>();
                            Arc::new(StringArray::from(arr_vec))
                        }
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {column_data_type} for injecting into a String template for column {column_name}.",
                            ));
                        }
                    },
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {column_data_type} for injecting into a String template for column {column_name}"
                        ));
                    }
                }
            }
        } else {
            column_cast
        };

        // Rename the columns
        if let Some(name) = as_columns.get(index) {
            if name.is_empty() {
                batch_vec.push((column_name, column_cast));
            } else {
                batch_vec.push((name, column_cast));
            }
        } else {
            batch_vec.push((column_name, column_cast));
        }
    }

    // Reorder the columns
    let batch_vec = reorder_batch_vec_columns(
        &batch_vec
            .into_iter()
            .map(|(&a, b)| (a, b))
            .collect::<Vec<_>>(),
        reorder_columns,
    );
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::device;

    use super::*;

    #[test]
    fn test_select() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "1"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "3"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // ------ String, UInt32, Cast, No operators ------
        let result = select(
            &["lhs_pk", "lhs_text", "lhs_metadata"],
            &[lhs_batch_1.clone(), lhs_batch_2.clone()],
            &["", "", ""],
            &["new_pk", "", "new_metadata"],
            &["new_pk", "lhs_text", "new_metadata"],
            &[
                DataColumnOperator::None,
                DataColumnOperator::None,
                DataColumnOperator::None,
            ],
            &[
                DataCastOperator::Cast,
                DataCastOperator::None,
                DataCastOperator::Cast,
            ],
            &[DataType::UInt32, DataType::Utf8, DataType::Float32],
            &["", "Into template {{ lhs_text }}", ""],
            &device,
        )?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("lhs_text");
        assert_eq!(
            lhs_text,
            vec![
                "Into template left",
                "Into template 1",
                "Into template left",
                "Into template 3"
            ]
        );
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("new_pk")?;
        assert_eq!(lhs_id, [0, 1, 2, 3]);
        let metadata = result_table.get_column_as_vec_primitive::<f32>("new_metadata")?;
        assert_eq!(metadata, [1., 2., 3., 4.]);

        // ------ String, UInt32, Cast, Operator, reorder ------
        let result = select(
            &[
                "lhs_pk",
                "lhs_text",
                "lhs_metadata",
                "new_text",
                "lhs_metadata",
            ],
            &[lhs_batch_1.clone(), lhs_batch_2.clone()],
            &["", "lhs_text", "new_pk", "lhs_text", "new_pk"],
            &[
                "new_pk",
                "new_text",
                "new_metadata",
                "newer_text",
                "new_metadata2",
            ],
            &["new_pk", "new_text", "newer_text", "new_metadata"],
            &[
                DataColumnOperator::None,
                DataColumnOperator::Concat,
                DataColumnOperator::Add,
                DataColumnOperator::List,
                DataColumnOperator::Add,
            ],
            &[
                DataCastOperator::Cast,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
            ],
            &[
                DataType::UInt32,
                DataType::Utf8,
                DataType::UInt32,
                DataType::Utf8,
                DataType::UInt32,
            ],
            &["", "Into template {{ lhs_text }}", "", ""],
            &device,
        )?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("new_text");
        assert_eq!(
            lhs_text,
            [
                "Into template leftleft",
                "Into template 11",
                "Into template leftleft",
                "Into template 33"
            ]
        );
        let lhs_text =
            result_table.get_column_as_vec_nested_nonprimitive::<String>("newer_text")?;
        assert_eq!(
            lhs_text,
            [
                ["Into template leftleft", "left"],
                ["Into template 11", "1"],
                ["Into template leftleft", "left"],
                ["Into template 33", "3"],
            ]
        );
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("new_pk")?;
        assert_eq!(lhs_id, [0, 1, 2, 3]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("new_metadata")?;
        assert_eq!(metadata, [1, 3, 5, 7]);
        let fields = result_table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        assert_eq!(fields, ["new_pk", "new_text", "newer_text", "new_metadata"]);

        // ------ String, UInt32, Float32, Missing column ------
        let result = select(
            &[
                "new_pk",
                "default_metadata",
                "broadcast_metadata",
                "lhs_pk",
                "lhs_metadata",
                "lhs_metadata",
            ],
            &[lhs_batch_1.clone(), lhs_batch_2.clone()],
            &["", "", "", "", "", ""],
            &[
                "new_pk1",
                "",
                "",
                "hash_pk",
                "min_metadata",
                "list_metadata",
            ],
            &[
                "new_pk1",
                "default_metadata",
                "broadcast_metadata",
                "hash_pk",
                "min_metadata",
                "list_metadata",
            ],
            &[
                DataColumnOperator::String,
                DataColumnOperator::Zeros,
                DataColumnOperator::Value,
                DataColumnOperator::None,
                DataColumnOperator::BroadcastMin,
                DataColumnOperator::BroadcastList,
            ],
            &[
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::None,
                DataCastOperator::Hash,
                DataCastOperator::None,
                DataCastOperator::None,
            ],
            &[
                DataType::Utf8,
                DataType::UInt32,
                DataType::Float32,
                DataType::UInt32,
                DataType::UInt32,
                DataType::UInt32,
            ],
            &["", "", "0.75", "", "", ""],
            &device,
        )?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("new_pk1");
        assert_eq!(lhs_text, ["", "", "", ""]);
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("default_metadata")?;
        assert_eq!(lhs_id, [0, 0, 0, 0]);
        let lhs_id = result_table.get_column_as_vec_primitive::<f32>("broadcast_metadata")?;
        assert_eq!(lhs_id, [0.75, 0.75, 0.75, 0.75]);
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("hash_pk")?;
        assert_eq!(lhs_id, [0, 1, 2, 3]);
        let column_names = result_table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        assert!(!column_names.contains(&"new_pk".to_string()));
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("min_metadata")?;
        assert_eq!(lhs_id, [1, 1, 1, 1]);
        let lhs_id = result_table.get_column_as_vec_nested_primitive::<u32>("list_metadata")?;
        assert_eq!(
            lhs_id,
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        );

        Ok(())
    }
}
