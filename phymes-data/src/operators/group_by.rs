use std::{collections::HashMap, fmt::Display, ops::Range, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayData, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, Float32Array, Float64Array,
        Int64Array, ListArray, ListBuilder, PrimitiveBuilder, RecordBatch, StringArray,
        StringBuilder, UInt8Array, UInt32Array,
    },
    buffer::Buffer,
    compute::kernels::partition::partition,
    datatypes::{
        ArrowNativeType, DataType, Field, Float32Type, Float64Type, Int64Type, Schema, UInt8Type,
        UInt32Type,
    },
};
use candle_core::{Device, Tensor, WithDType};
use num_traits::{Bounded, Num, NumCast};
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait};
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    ToolTrait, DataAggregatorOperator, DataConfig, DataOperatorTrait,
    operators::sort::sort,
};

/// Group the [RecordBatch] according to the `lhs_values` columns and aggregate using a specified aggregation operator over specified columns
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GroupBy {
    lhs_values: Vec<String>,
    agg_columns: Vec<String>,
    agg_operators: Vec<DataAggregatorOperator>,
}

impl MappableTrait for GroupBy {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for GroupBy {
    fn get_description(&self) -> String {
        "Group by user specified columns and aggregate user specified aggregation columns using the user specified aggregation operators.".to_string()
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
            "lhs_pk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table".to_string(),
                ),
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
        let function = Function {
            name: Self::get_static_name().to_string(),
            description: Some(self.get_description()),
            parameters: FunctionParameters {
                schema_type: JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
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

impl DataOperatorTrait for GroupBy {
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
        let agg_columns = self
            .agg_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let batches = group_by(
            &lhs_values,
            lhs_args,
            &agg_columns,
            &self.agg_operators,
            device,
        )?;
        Ok(batches)
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let agg_columns = config.agg_columns.clone().ok_or(anyhow!(
            "Missing `agg_columns` for `{}`.",
            Self::get_static_name()
        ))?;
        let agg_operators = config.agg_operators.clone().ok_or(anyhow!(
            "Missing `agg_operators` for `{}`.",
            Self::get_static_name()
        ))?;

        // Ensure that the array lengths for columns and operators match
        if agg_columns.len() != agg_operators.len() {
            return Err(anyhow!(
                "agg_columns length {} is not equal to the agg_operators length {}",
                agg_columns.len(),
                agg_operators.len()
            ));
        }

        Ok(GroupBy {
            lhs_values,
            agg_columns,
            agg_operators,
        })
    }
}

/// Partition a lexocographically sorted slice of [RecordBatch]es
fn partition_record_batches(lhs_values: &[&str], lhs_table: &Subject) -> Result<Vec<Range<usize>>> {
    let mut columns = Vec::new();
    for column_name in lhs_values.iter() {
        columns.push(lhs_table.get_column_as_array(column_name)?);
    }
    let ranges = partition(&columns)?.ranges();
    Ok(ranges)
}

/// Helper function to compute the aggregation operator for tensors
fn aggregator_operator_tensor(
    agg_column: &str,
    agg_operator: &DataAggregatorOperator,
    lhs_table: &Subject,
    tensor: &Tensor,
    range: &Range<usize>,
    device: &Device,
) -> Result<Tensor> {
    let gather_tensor = Tensor::arange(range.start as u8, range.end as u8, device)?;
    let agg_tensor = match agg_operator {
        DataAggregatorOperator::Sum => tensor
            .gather(&gather_tensor, candle_core::D::Minus1)?
            .sum(candle_core::D::Minus1)?,
        DataAggregatorOperator::Max => tensor
            .gather(&gather_tensor, candle_core::D::Minus1)?
            .max(candle_core::D::Minus1)?,
        DataAggregatorOperator::Min => tensor
            .gather(&gather_tensor, candle_core::D::Minus1)?
            .min(candle_core::D::Minus1)?,
        DataAggregatorOperator::Mean => tensor
            .gather(&gather_tensor, candle_core::D::Minus1)?
            .mean(candle_core::D::Minus1)?,
        DataAggregatorOperator::Var => tensor
            .gather(&gather_tensor, candle_core::D::Minus1)?
            .var(candle_core::D::Minus1)?,
        _ => {
            return Err(anyhow!(
                "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                lhs_table.get_column_data_type(agg_column)?
            ));
        }
    };
    Ok(agg_tensor)
}

/// Helper function to extract the aggregator column for primitive types
fn extract_aggregator_column_primitive<T>(
    group_column: &str,
    lhs_table: &Subject,
    ranges: &[Range<usize>],
) -> Vec<T>
where
    T: Num + Bounded + NumCast + Send + Sync + Clone + 'static,
{
    let array_vec = lhs_table
        .get_column_as_vec_primitive::<T>(group_column)
        .unwrap();
    let mut agg_vec = Vec::new();
    for range in ranges.iter() {
        let value = array_vec.get(range.start).unwrap();
        agg_vec.push(NumCast::from(value.clone()).unwrap());
    }
    agg_vec
}

/// Helper function to extract the aggregator column for primitive types
fn extract_aggregator_column_nonprimitive<T>(
    group_column: &str,
    lhs_table: &Subject,
    ranges: &[Range<usize>],
) -> Vec<T>
where
    T: From<String> + Clone + Display + 'static,
{
    let array_vec = lhs_table
        .get_column_as_vec_nonprimitive::<T>(group_column)
        .unwrap();
    let mut agg_vec = Vec::new();
    for range in ranges.iter() {
        let value = array_vec.get(range.start).unwrap();
        agg_vec.push(T::from(value.to_string()));
    }
    agg_vec
}

/// Helper function to extract the aggregator column for primitive types
fn extract_aggregator_column_nested_primitive<T>(
    group_column: &str,
    lhs_table: &Subject,
    ranges: &[Range<usize>],
) -> Vec<Vec<T>>
where
    T: Num + Bounded + NumCast + Send + Sync + Clone + 'static,
{
    let array_vec = lhs_table
        .get_column_as_vec_nested_primitive::<T>(group_column)
        .unwrap();
    let mut agg_vec = Vec::new();
    for range in ranges.iter() {
        let value = array_vec.get(range.start).unwrap();
        agg_vec.push(value.to_owned());
    }
    agg_vec
}

/// Helper function to build a fixed list primitive type
pub(crate) fn build_aggregator_column_fixed_size_list<T>(
    agg_vec: Vec<Vec<T>>,
    data_type: DataType,
) -> ArrayRef
where
    T: ArrowNativeType + 'static,
{
    let dim_0 = agg_vec.len();
    let dim_1 = agg_vec.first().unwrap().len();
    let list_values = agg_vec.into_iter().flatten().collect::<Vec<_>>();
    let value_data = ArrayData::builder(data_type.clone())
        .len(list_values.len())
        .add_buffer(Buffer::from_vec(list_values))
        .build()
        .unwrap();
    let list_data_type = DataType::FixedSizeList(
        Arc::new(Field::new_list_field(data_type, false)),
        dim_1 as i32,
    );
    let list_data = ArrayData::builder(list_data_type)
        .len(dim_0)
        .add_child_data(value_data)
        .build()
        .unwrap();
    Arc::new(FixedSizeListArray::from(list_data))
}

/// Helper function to build a list primitive type
pub fn build_aggregator_column_list_primitive<T, D>(
    agg_vec: Vec<Vec<T>>,
    data_type: DataType,
) -> ArrayRef
where
    T: ArrowNativeType + 'static,
    D: ArrowPrimitiveType<Native = T> + 'static,
{
    let value_builder = PrimitiveBuilder::<D>::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(data_type, false));
    for values in agg_vec {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    Arc::new(list_builder.finish())
}

/// Helper function to build a list nonprimitive type
pub fn build_aggregator_column_list_nonprimitive<T>(
    agg_vec: Vec<Vec<T>>,
    data_type: DataType,
) -> ArrayRef
where
    T: From<String> + 'static + std::convert::AsRef<str>,
{
    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(data_type, false));
    for v in agg_vec {
        for s in v {
            list_builder.values().append_value(s);
        }
        list_builder.append(true);
    }
    Arc::new(list_builder.finish())
}

/// Helper function to build the aggregation column for primitive types
///   using tensor operations (i.e., sum, min, max, mean, var)
fn build_aggregation_column_primitive_tensor<T>(
    agg_column: &str,
    agg_operator: &DataAggregatorOperator,
    lhs_table: &Subject,
    ranges: &[Range<usize>],
    device: &Device,
) -> Result<Vec<T>>
where
    T: Num + Bounded + NumCast + Send + Sync + WithDType + 'static,
{
    let array_vec = lhs_table.get_column_as_vec_primitive::<T>(agg_column)?;
    let tensor = Tensor::from_iter(array_vec, device)?;
    let mut agg_vec = Vec::new();
    for range in ranges.iter() {
        let agg_tensor = aggregator_operator_tensor(
            agg_column,
            agg_operator,
            lhs_table,
            &tensor,
            range,
            device,
        )?;
        agg_vec.push(agg_tensor.to_vec0::<T>()?);
    }
    Ok(agg_vec)
}

/// Helper function to build the aggregation column for the count operator
fn build_aggregation_column_count(ranges: &[Range<usize>]) -> ArrayRef {
    let agg_vec = ranges
        .iter()
        .map(|range| (range.end - range.start) as u32)
        .collect::<Vec<_>>();
    Arc::new(UInt32Array::from(agg_vec))
}

/// Helper function to extract out the aggregation ranges
fn extract_aggregation_ranges(
    agg_column: &str,
    lhs_table: &Subject,
    ranges: &[Range<usize>],
) -> Result<Vec<ArrayRef>> {
    let array_ref = lhs_table.get_column_as_array(agg_column)?;
    let mut agg_vec = Vec::new();
    for range in ranges.iter() {
        let gather_arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(
            range.start as u32..range.end as u32,
        ));
        let taken_arr = arrow::compute::take(&array_ref, &gather_arr, None)?;
        agg_vec.push(taken_arr);
    }
    Ok(agg_vec)
}

/// Helper function to create the new aggregation column name based on the original column name plus the aggregation operator
pub(crate) fn create_agg_column_name(
    agg_column: &str,
    agg_operator: &DataAggregatorOperator,
) -> String {
    format!("{agg_column}-{agg_operator}")
}

/// Group by specified columns and aggregate using a specified aggregation operator over specified columns
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the columns to group by
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `agg_columns` - Slice of Strings for the aggregation columns
/// * `agg_operators` - Slice of [DataAggregatorOperator]s specifying the aggregator operator to apply to each agg_column
/// * `device` - The compute device
#[instrument(skip(lhs_values, lhs_args, agg_columns, agg_operators, device))]
pub fn group_by(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    agg_columns: &[&str],
    agg_operators: &[DataAggregatorOperator],
    device: &Device,
) -> Result<RecordBatch> {
    // todo!(): need to account for the case of no `lhs_values`
    // Presort the lhs group by columns
    let mut lhs_sorted = RecordBatch::new_empty(Arc::new(Schema::empty()));
    for (iter, column_name) in lhs_values.iter().enumerate() {
        if iter > 0 {
            lhs_sorted = sort(column_name, &[lhs_sorted], true, device)?;
        } else {
            lhs_sorted = sort(column_name, lhs_args, true, device)?;
        }
    }

    // Wrap the lhs and rhs into an ArrowTable
    let lhs_table = Subject::get_builder()
        .with_name("group_by")
        .with_record_batches(vec![lhs_sorted])?
        .build()?;

    // Partition the group by columns
    let ranges = partition_record_batches(lhs_values, &lhs_table)?;
    let mut batch_vec = Vec::new();

    // Copy out the first partition for each group
    for group_column in lhs_values.iter() {
        let lhs_agg: ArrayRef = match lhs_table.get_column_data_type(group_column)? {
            DataType::UInt8 => {
                let agg_vec =
                    extract_aggregator_column_primitive::<u8>(group_column, &lhs_table, &ranges);
                Arc::new(UInt8Array::from(agg_vec))
            }
            DataType::UInt32 => {
                let agg_vec =
                    extract_aggregator_column_primitive::<u32>(group_column, &lhs_table, &ranges);
                Arc::new(UInt32Array::from(agg_vec))
            }
            DataType::Int64 => {
                let agg_vec =
                    extract_aggregator_column_primitive::<i64>(group_column, &lhs_table, &ranges);
                Arc::new(Int64Array::from(agg_vec))
            }
            DataType::Float32 => {
                let agg_vec =
                    extract_aggregator_column_primitive::<f32>(group_column, &lhs_table, &ranges);
                Arc::new(Float32Array::from(agg_vec))
            }
            DataType::Float64 => {
                let agg_vec =
                    extract_aggregator_column_primitive::<f64>(group_column, &lhs_table, &ranges);
                Arc::new(Float64Array::from(agg_vec))
            }
            DataType::Utf8 => {
                let agg_vec = extract_aggregator_column_nonprimitive::<String>(
                    group_column,
                    &lhs_table,
                    &ranges,
                );
                Arc::new(StringArray::from(agg_vec))
            }
            DataType::FixedSizeList(f, _) => match f.data_type() {
                DataType::UInt8 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<u8>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_fixed_size_list::<u8>(agg_vec, DataType::UInt8)
                }
                DataType::UInt32 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<u32>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_fixed_size_list::<u32>(agg_vec, DataType::UInt32)
                }
                DataType::Int64 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<i64>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_fixed_size_list::<i64>(agg_vec, DataType::Int64)
                }
                DataType::Float32 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<f32>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_fixed_size_list::<f32>(agg_vec, DataType::Float32)
                }
                DataType::Float64 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<f64>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_fixed_size_list::<f64>(agg_vec, DataType::Float64)
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type for column {}: {}",
                        group_column,
                        lhs_table.get_column_data_type(group_column)?
                    ));
                }
            },
            DataType::List(f) => match f.data_type() {
                DataType::UInt8 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<u8>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_vec,
                        DataType::UInt8,
                    )
                }
                DataType::UInt32 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<u32>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_vec,
                        DataType::UInt32,
                    )
                }
                DataType::Int64 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<i64>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_vec,
                        DataType::Int64,
                    )
                }
                DataType::Float32 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<f32>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        agg_vec,
                        DataType::Float32,
                    )
                }
                DataType::Float64 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<f64>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        agg_vec,
                        DataType::Float64,
                    )
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type for column {}: {}",
                        group_column,
                        lhs_table.get_column_data_type(group_column)?
                    ));
                }
            },
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    group_column,
                    lhs_table.get_column_data_type(group_column)?
                ));
            }
        };
        batch_vec.push((group_column.to_string(), lhs_agg));
    }

    // Apply the aggregation operators
    for (agg_column, agg_operator) in agg_columns.iter().zip(agg_operators.iter()) {
        let lhs_agg: ArrayRef = match agg_operator {
            DataAggregatorOperator::Count => build_aggregation_column_count(&ranges),
            DataAggregatorOperator::Concat => match lhs_table.get_column_data_type(agg_column)? {
                DataType::Utf8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                                .join("")
                        })
                        .collect::<Vec<_>>();
                    Arc::new(StringArray::from(agg_vecs))
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
            DataAggregatorOperator::ConcatSemicolonSeperator => match lhs_table
                .get_column_data_type(agg_column)?
            {
                DataType::Utf8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                                .join("; ")
                        })
                        .collect::<Vec<_>>();
                    Arc::new(StringArray::from(agg_vecs))
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
            DataAggregatorOperator::Max
            | DataAggregatorOperator::Mean
            | DataAggregatorOperator::Min
            | DataAggregatorOperator::Var
            | DataAggregatorOperator::Sum => match lhs_table.get_column_data_type(agg_column)? {
                DataType::UInt8 => {
                    let agg_vec = build_aggregation_column_primitive_tensor::<u8>(
                        agg_column,
                        agg_operator,
                        &lhs_table,
                        &ranges,
                        device,
                    )?;
                    Arc::new(UInt8Array::from(agg_vec))
                }
                DataType::UInt32 => {
                    let agg_vec = build_aggregation_column_primitive_tensor::<u32>(
                        agg_column,
                        agg_operator,
                        &lhs_table,
                        &ranges,
                        device,
                    )?;
                    Arc::new(UInt32Array::from(agg_vec))
                }
                DataType::Int64 => {
                    let agg_vec = build_aggregation_column_primitive_tensor::<i64>(
                        agg_column,
                        agg_operator,
                        &lhs_table,
                        &ranges,
                        device,
                    )?;
                    Arc::new(Int64Array::from(agg_vec))
                }
                DataType::Float32 => {
                    let agg_vec = build_aggregation_column_primitive_tensor::<f32>(
                        agg_column,
                        agg_operator,
                        &lhs_table,
                        &ranges,
                        device,
                    )?;
                    Arc::new(Float32Array::from(agg_vec))
                }
                DataType::Float64 => {
                    let agg_vec = build_aggregation_column_primitive_tensor::<f64>(
                        agg_column,
                        agg_operator,
                        &lhs_table,
                        &ranges,
                        device,
                    )?;
                    Arc::new(Float64Array::from(agg_vec))
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
            DataAggregatorOperator::First => match lhs_table.get_column_data_type(agg_column)? {
                DataType::UInt8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt8Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.first()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(UInt8Array::from(agg_values))
                }
                DataType::UInt32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.first()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(UInt32Array::from(agg_values))
                }
                DataType::Int64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.first()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(Int64Array::from(agg_values))
                }
                DataType::Float32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Float32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.first()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(Float32Array::from(agg_values))
                }
                DataType::Float64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Float64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.first()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(Float64Array::from(agg_values))
                }
                DataType::Utf8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| s.map(|s| s.to_owned()))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.first()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(StringArray::from(agg_values))
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<f32, Float32Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<f64, Float64Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.first()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
            DataAggregatorOperator::Last => match lhs_table.get_column_data_type(agg_column)? {
                DataType::UInt8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt8Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.last()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(UInt8Array::from(agg_values))
                }
                DataType::UInt32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.last()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(UInt32Array::from(agg_values))
                }
                DataType::Int64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.last()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(Int64Array::from(agg_values))
                }
                DataType::Float32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Float32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.last()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(Float32Array::from(agg_values))
                }
                DataType::Float64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Float64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.last()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(Float64Array::from(agg_values))
                }
                DataType::Utf8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| s.map(|s| s.to_owned()))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut agg_values = Vec::new();
                    for agg_vec in agg_vecs.into_iter() {
                        let agg_value = agg_vec.last()
                            .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                            .to_owned();
                        agg_values.push(agg_value);
                    }
                    Arc::new(StringArray::from(agg_values))
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_fixed_size_list(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<f32, Float32Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_primitive::<f64, Float64Type>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let mut agg_values = Vec::new();
                        for agg_vec in agg_vecs.into_iter() {
                            let agg_value = agg_vec.last()
                                .ok_or(anyhow!("Empty array for data type {} and aggregator operator {agg_operator} for column {agg_column}", lhs_table.get_column_data_type(agg_column)?))?
                                .to_owned();
                            agg_values.push(agg_value);
                        }
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_values,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
            DataAggregatorOperator::List => match lhs_table.get_column_data_type(agg_column)? {
                DataType::UInt8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt8Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::UInt32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::Int64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::Float32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Float32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::Float64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Float64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::Utf8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| s.map(|s| s.to_owned()))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<f32, Float32Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Float64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<f64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<f64, Float64Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
            DataAggregatorOperator::Set => match lhs_table.get_column_data_type(agg_column)? {
                DataType::UInt8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt8Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::UInt32 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<UInt32Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::Int64 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .iter()
                                .flatten()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::Utf8 => {
                    let agg_ranges = extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                    let agg_vecs = agg_ranges
                        .into_iter()
                        .map(|arr| {
                            arr.as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .iter()
                                .filter_map(|s| s.map(|s| s.to_owned()))
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        agg_vecs,
                        lhs_table.get_column_data_type(agg_column)?,
                    )
                }
                DataType::FixedSizeList(f, _) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<FixedSizeListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::UInt8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u8>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u8, UInt8Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::UInt32 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<u32>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<u32, UInt32Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Int64 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_primitive::<i64>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_primitive::<i64, Int64Type>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    DataType::Utf8 => {
                        let agg_ranges =
                            extract_aggregation_ranges(agg_column, &lhs_table, &ranges)?;
                        let agg_vecs = agg_ranges
                            .into_iter()
                            .map(|arr| {
                                arr.as_any()
                                    .downcast_ref::<ListArray>()
                                    .unwrap()
                                    .iter()
                                    .filter_map(|s| {
                                        s.map(|s| {
                                            Subject::get_array_as_vec_nonprimitive::<String>(
                                                &s, agg_column,
                                            )
                                            .unwrap_or_default()
                                        })
                                    })
                                    .collect::<HashSet<_>>()
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        build_aggregator_column_list_nonprimitive::<String>(
                            agg_vecs,
                            lhs_table.get_column_data_type(agg_column)?,
                        )
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} and aggregator operator {agg_operator} for column {agg_column}",
                        lhs_table.get_column_data_type(agg_column)?
                    ));
                }
            },
        };
        let columns_name = create_agg_column_name(agg_column, agg_operator);
        batch_vec.push((columns_name, lhs_agg));
    }

    // Create the output batch
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::device;

    use super::*;

    #[test]
    fn test_group_by() -> Result<()> {
        // ------ lhs_values = String ------
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
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
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Group the text
        let result = group_by(
            &["lhs_text"],
            &[lhs_batch_1, lhs_batch_2],
            &[
                "lhs_pk",
                "lhs_pk",
                "lhs_pk",
                "lhs_pk",
                "lhs_metadata",
                "lhs_metadata",
                "lhs_metadata",
                "lhs_metadata",
            ],
            &[
                DataAggregatorOperator::Concat,
                DataAggregatorOperator::Count,
                DataAggregatorOperator::List,
                DataAggregatorOperator::Set,
                DataAggregatorOperator::Sum,
                DataAggregatorOperator::Max,
                DataAggregatorOperator::List,
                DataAggregatorOperator::Set,
            ],
            &device,
        )?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("lhs_text");
        assert_eq!(lhs_text, ["left"]);
        let lhs_id = result_table.get_column_as_vec_str("lhs_pk-Concat");
        assert_eq!(lhs_id, ["0123"]);
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("lhs_pk-Count")?;
        assert_eq!(lhs_id, [4]);
        let lhs_id = result_table.get_column_as_vec_nested_nonprimitive::<String>("lhs_pk-List")?;
        assert_eq!(lhs_id, [["0", "1", "2", "3"]]);
        let lhs_id = result_table.get_column_as_vec_nested_nonprimitive::<String>("lhs_pk-Set")?;
        let mut lhs_id_sort = lhs_id.into_iter().flatten().collect::<Vec<_>>();
        lhs_id_sort.sort();
        assert_eq!(lhs_id_sort, ["0", "1", "2", "3"]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata-Sum")?;
        assert_eq!(metadata, [10]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata-Max")?;
        assert_eq!(metadata, [4]);
        let metadata =
            result_table.get_column_as_vec_nested_primitive::<u32>("lhs_metadata-List")?;
        assert_eq!(metadata, [[1, 2, 3, 4]]);
        let metadata =
            result_table.get_column_as_vec_nested_primitive::<u32>("lhs_metadata-Set")?;
        let mut metadata_sort = metadata.into_iter().flatten().collect::<Vec<_>>();
        metadata_sort.sort();
        assert_eq!(metadata_sort, [1, 2, 3, 4]);

        // ------ lhs_values = String, u32 ------
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
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
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;

        // Group the text
        let result = group_by(
            &["lhs_pk", "lhs_metadata"],
            &[lhs_batch_1, lhs_batch_2],
            &["lhs_text"],
            &[DataAggregatorOperator::Count],
            &device,
        )?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_id = result_table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_id, ["0", "1", "2", "3"]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata")?;
        assert_eq!(metadata, [1, 2, 3, 4]);
        let lhs_text = result_table.get_column_as_vec_primitive::<u32>("lhs_text-Count")?;
        assert_eq!(lhs_text, [1, 1, 1, 1]);

        Ok(())
    }
}
