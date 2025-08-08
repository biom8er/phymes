use std::{collections::HashMap, fmt::Display, ops::Range, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayData, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array, ListArray,
        RecordBatch, StringArray, UInt8Array, UInt32Array,
    },
    buffer::Buffer,
    compute::kernels::partition::partition,
    datatypes::{ArrowNativeType, DataType, Field, Schema},
};
use candle_core::{Device, Tensor, WithDType};
use num_traits::{Bounded, Num, NumCast};
use phymes_core::schemas::{chat_completion, types};
use phymes_core::{
    session::common_traits::{BuildableTrait, BuilderTrait},
    table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
};

use crate::{
    candle_data::data_config::DataAggregatorOperator,
    candle_operators::{
        data_operator::{DataOperatorTrait, make_error_record_batch},
        sort_column_and_indices::sort_column_and_indices,
    },
};

/// Sort the [RecordBatch] according to the `score` column and then apply the sorting order to the rest of the record batch columns
#[derive(Debug)]
pub struct GroupByAndAggregate {
    lhs_values: Vec<String>,
    agg_columns: Vec<String>,
    agg_operators: Vec<DataAggregatorOperator>,
}

impl DataOperatorTrait for GroupByAndAggregate {
    fn get_static_name() -> &'static str {
        "group-by-and-aggregate"
    }
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
        match group_by_and_aggregate(
            &lhs_values,
            lhs_args,
            &agg_columns,
            &self.agg_operators,
            device,
        ) {
            Ok(batch) => Ok(batch),
            Err(err) => Ok(make_error_record_batch(err.to_string().as_str())),
        }
    }
    fn new(
        _lhs_pk: &str,
        _lhs_fk: &str,
        lhs_values: &str,
        _rhs_pk: Option<&str>,
        _rhs_fk: Option<&str>,
        _rhs_values: Option<&str>,
        kwargs: Option<&str>,
    ) -> Self {
        // Attempt to parse the lhs_values
        let lhs_values: Vec<String> = serde_json::from_str(lhs_values).unwrap_or_default();

        // Attempt to parse the op_kwargs
        let ops_kwargs_default = "{\"agg_columns\": [], \"agg_operators\": []}";
        let ops_kwargs_str = kwargs.unwrap_or(ops_kwargs_default);
        let ops_kwargs: serde_json::Value = serde_json::from_str(ops_kwargs_str)
            .unwrap_or(serde_json::from_str(ops_kwargs_default).unwrap());
        let agg_columns = ops_kwargs
            .get("agg_columns")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>();
        let agg_operators = ops_kwargs
            .get("agg_operators")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| serde_json::from_value::<DataAggregatorOperator>(v.clone()).unwrap())
            .collect::<Vec<_>>();

        // Make the object
        GroupByAndAggregate {
            lhs_values,
            agg_columns,
            agg_operators,
        }
    }
    fn get_description() -> String {
        "Group by user specified columns and aggregate user specified aggregation columns using the user specified aggregation operators.".to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        // properties.insert(
        //     "rhs_name".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The name of the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        properties.insert(
            "lhs_pk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        // properties.insert(
        //     "rhs_pk".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The primary key column identifier for the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        // properties.insert(
        //     "lhs_fk".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The foriegn key column identifier for the left hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        // properties.insert(
        //     "rhs_fk".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The foriegn key column identifier for the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        properties.insert(
            "lhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The values column identifier for the left hand side table in the form of a JSON list of strings".to_string(),
                ),
                ..Default::default()
            }),
        );
        // properties.insert(
        //     "rhs_values".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The values column identifier for the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        let function = types::Function {
            name: Self::get_name(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
                ]),
            },
        };
        let tool = chat_completion::Tool {
            r#type: chat_completion::ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

/// Partition a lexocographically sorted slice of [RecordBatch]es
fn partition_record_batches(
    lhs_values: &[&str],
    lhs_table: &ArrowTable,
) -> Result<Vec<Range<usize>>> {
    let mut columns = Vec::new();
    for column_name in lhs_values.iter() {
        columns.push(lhs_table.get_column_as_array(column_name));
    }
    let ranges = partition(&columns)?.ranges();
    Ok(ranges)
}

/// Helper function to compute the aggregation operator for tensors
fn aggregator_operator_tensor(
    agg_column: &str,
    agg_operator: &DataAggregatorOperator,
    lhs_table: &ArrowTable,
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
                "Unsupported data type {} and aggregator operator {} for column {}",
                lhs_table.get_column_data_type(agg_column)?.to_string(),
                agg_operator.get_name(),
                agg_column,
            ));
        }
    };
    Ok(agg_tensor)
}

/// Helper function to extract the aggregator column for primitive types
fn extract_aggregator_column_primitive<T>(
    group_column: &str,
    lhs_table: &ArrowTable,
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
    lhs_table: &ArrowTable,
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
    lhs_table: &ArrowTable,
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
fn build_aggregator_column_fixed_size_list<T>(agg_vec: Vec<Vec<T>>, data_type: DataType) -> ArrayRef
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
fn build_aggregator_column_list<T>(agg_vec: Vec<Vec<T>>, data_type: DataType) -> ArrayRef
where
    T: ArrowNativeType + 'static,
{
    let dim_0 = agg_vec.len();
    let list_values = agg_vec.into_iter().flatten().collect::<Vec<_>>();
    let value_data = ArrayData::builder(data_type.clone())
        .len(list_values.len())
        .add_buffer(Buffer::from_vec(list_values))
        .build()
        .unwrap();
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(data_type, false)));
    let list_data = ArrayData::builder(list_data_type)
        .len(dim_0)
        .add_child_data(value_data)
        .build()
        .unwrap();
    Arc::new(ListArray::from(list_data))
}

/// Helper function to build the aggregation column
fn build_aggregation_column_primitive<T>(
    agg_column: &str,
    agg_operator: &DataAggregatorOperator,
    lhs_table: &ArrowTable,
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

/// Group by specified columns and aggregate using a specified aggregation operator over specified columns
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the columns to group by
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `agg_columns` - Slice of Strings for the aggregation columns
/// * `agg_operators` - Slice of [DataAggregator]s specifying the aggregator operator to apply to each agg_column
/// * `device` - The compute device
pub fn group_by_and_aggregate(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    agg_columns: &[&str],
    agg_operators: &[DataAggregatorOperator],
    device: &Device,
) -> Result<RecordBatch> {
    // Presort the lhs group by columns
    let mut lhs_sorted = RecordBatch::new_empty(Arc::new(Schema::empty()));
    for (iter, column_name) in lhs_values.iter().enumerate() {
        if iter > 0 {
            lhs_sorted = sort_column_and_indices(column_name, &[lhs_sorted], true, device)?;
        } else {
            lhs_sorted = sort_column_and_indices(column_name, lhs_args, true, device)?;
        }
    }

    // Wrap the lhs and rhs into an ArrowTable
    let lhs_table = ArrowTable::get_builder()
        .with_record_batches(vec![lhs_sorted])?
        .with_name("")
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
                        lhs_table.get_column_data_type(group_column)?.to_string()
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
                    build_aggregator_column_list::<u8>(agg_vec, DataType::UInt8)
                }
                DataType::UInt32 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<u32>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list::<u32>(agg_vec, DataType::UInt32)
                }
                DataType::Int64 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<i64>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list::<i64>(agg_vec, DataType::Int64)
                }
                DataType::Float32 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<f32>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list::<f32>(agg_vec, DataType::Float32)
                }
                DataType::Float64 => {
                    let agg_vec = extract_aggregator_column_nested_primitive::<f64>(
                        group_column,
                        &lhs_table,
                        &ranges,
                    );
                    build_aggregator_column_list::<f64>(agg_vec, DataType::Float64)
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type for column {}: {}",
                        group_column,
                        lhs_table.get_column_data_type(group_column)?.to_string()
                    ));
                }
            },
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    group_column,
                    lhs_table.get_column_data_type(group_column)?.to_string()
                ));
            }
        };
        batch_vec.push((group_column.to_string(), lhs_agg));
    }

    // Apply the aggregation operators
    for (agg_column, agg_operator) in agg_columns.iter().zip(agg_operators.iter()) {
        let lhs_agg: ArrayRef = match lhs_table.get_column_data_type(agg_column)? {
            DataType::UInt8 => {
                let agg_vec = build_aggregation_column_primitive::<u8>(
                    agg_column,
                    agg_operator,
                    &lhs_table,
                    &ranges,
                    device,
                )?;
                Arc::new(UInt8Array::from(agg_vec))
            }
            DataType::UInt32 => {
                let agg_vec = build_aggregation_column_primitive::<u32>(
                    agg_column,
                    agg_operator,
                    &lhs_table,
                    &ranges,
                    device,
                )?;
                Arc::new(UInt32Array::from(agg_vec))
            }
            DataType::Int64 => {
                let agg_vec = build_aggregation_column_primitive::<i64>(
                    agg_column,
                    agg_operator,
                    &lhs_table,
                    &ranges,
                    device,
                )?;
                Arc::new(Int64Array::from(agg_vec))
            }
            DataType::Float32 => {
                let agg_vec = build_aggregation_column_primitive::<f32>(
                    agg_column,
                    agg_operator,
                    &lhs_table,
                    &ranges,
                    device,
                )?;
                Arc::new(Float32Array::from(agg_vec))
            }
            DataType::Float64 => {
                let agg_vec = build_aggregation_column_primitive::<f64>(
                    agg_column,
                    agg_operator,
                    &lhs_table,
                    &ranges,
                    device,
                )?;
                Arc::new(Float64Array::from(agg_vec))
            }
            DataType::Utf8 => {
                let array_vec = lhs_table.get_column_as_vec_nonprimitive::<String>(agg_column)?;
                let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let gather_arr: ArrayRef = Arc::new(UInt8Array::from_iter_values(
                        range.start as u8..range.end as u8,
                    ));
                    let taken_arr = arrow::compute::take(&array_ref, &gather_arr, None)?;
                    let agg_value = match agg_operator {
                        DataAggregatorOperator::Count => format!("{}", taken_arr.len()),
                        DataAggregatorOperator::Concat => taken_arr
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                            .join(""),
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {} and aggregator operator {} for column {}",
                                lhs_table.get_column_data_type(agg_column)?.to_string(),
                                agg_operator.get_name(),
                                agg_column,
                            ));
                        }
                    };
                    agg_vec.push(agg_value);
                }
                Arc::new(StringArray::from(agg_vec))
            }
            DataType::FixedSizeList(f, _) => {
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let agg_value = match agg_operator {
                        DataAggregatorOperator::Count => vec![range.end - range.start],
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {} and aggregator operator {} for column {}",
                                lhs_table.get_column_data_type(agg_column)?.to_string(),
                                agg_operator.get_name(),
                                agg_column,
                            ));
                        }
                    };
                    agg_vec.push(agg_value);
                }
                match f.data_type() {
                    DataType::UInt8 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as u8).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<u8>(agg_vec, DataType::UInt8)
                    }
                    DataType::UInt32 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as u32).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<u32>(agg_vec, DataType::UInt32)
                    }
                    DataType::Int64 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as i64).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<i64>(agg_vec, DataType::Int64)
                    }
                    DataType::Float32 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as f32).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<f32>(agg_vec, DataType::Float32)
                    }
                    DataType::Float64 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as f64).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_fixed_size_list::<f64>(agg_vec, DataType::Float64)
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?.to_string()
                        ));
                    }
                }
            }
            DataType::List(f) => {
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let agg_value = match agg_operator {
                        DataAggregatorOperator::Count => vec![range.end - range.start],
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {} and aggregator operator {} for column {}",
                                lhs_table.get_column_data_type(agg_column)?.to_string(),
                                agg_operator.get_name(),
                                agg_column,
                            ));
                        }
                    };
                    agg_vec.push(agg_value);
                }
                match f.data_type() {
                    DataType::UInt8 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as u8).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list::<u8>(agg_vec, DataType::UInt8)
                    }
                    DataType::UInt32 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as u32).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list::<u32>(agg_vec, DataType::UInt32)
                    }
                    DataType::Int64 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as i64).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list::<i64>(agg_vec, DataType::Int64)
                    }
                    DataType::Float32 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as f32).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list::<f32>(agg_vec, DataType::Float32)
                    }
                    DataType::Float64 => {
                        let agg_vec = agg_vec
                            .into_iter()
                            .map(|v| v.into_iter().map(|v| v as f64).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        build_aggregator_column_list::<f64>(agg_vec, DataType::Float64)
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type for column {}: {}",
                            agg_column,
                            lhs_table.get_column_data_type(agg_column)?.to_string()
                        ));
                    }
                }
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    agg_column,
                    lhs_table.get_column_data_type(agg_column)?.to_string()
                ));
            }
        };
        let columns_name = format!("{agg_column}-{}", agg_operator.get_name());
        batch_vec.push((columns_name, lhs_agg));
    }

    // Create the output batch
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_group_by_and_aggregate() -> Result<()> {
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
        let result = group_by_and_aggregate(
            &["lhs_text"],
            &[lhs_batch_1, lhs_batch_2],
            &["lhs_pk", "lhs_pk", "lhs_metadata", "lhs_metadata"],
            &[
                DataAggregatorOperator::Concat,
                DataAggregatorOperator::Count,
                DataAggregatorOperator::Sum,
                DataAggregatorOperator::Max,
            ],
            &device,
        )?;
        let result_table = ArrowTable::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("lhs_text");
        assert_eq!(lhs_text, vec!["left"]);
        let lhs_id = result_table.get_column_as_vec_str("lhs_pk-Concat");
        assert_eq!(lhs_id, vec!["0123"]);
        let lhs_id = result_table.get_column_as_vec_str("lhs_pk-Count");
        assert_eq!(lhs_id, vec!["4"]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata-Sum")?;
        assert_eq!(metadata, vec![10]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata-Max")?;
        assert_eq!(metadata, vec![4]);

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
        let result = group_by_and_aggregate(
            &["lhs_pk", "lhs_metadata"],
            &[lhs_batch_1, lhs_batch_2],
            &["lhs_text"],
            &[DataAggregatorOperator::Count],
            &device,
        )?;
        let result_table = ArrowTable::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_id = result_table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_id, vec!["0", "1", "2", "3"]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata")?;
        assert_eq!(metadata, vec![1, 2, 3, 4]);
        let lhs_text = result_table.get_column_as_vec_str("lhs_text-Count");
        assert_eq!(lhs_text, vec!["1", "1", "1", "1"]);

        Ok(())
    }
}
