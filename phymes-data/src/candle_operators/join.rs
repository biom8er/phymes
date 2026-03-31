use arrow::{
    array::{
        Array, ArrayRef, Float32Array, Float64Array, Int64Array, StringArray, UInt8Array,
        UInt32Array,
    },
    compute::concat_batches,
    datatypes::{DataType, Float32Type, Float64Type, Int64Type, SchemaRef, UInt8Type, UInt32Type},
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, op::CmpOp};
use phymes_core::{
    BuildableTrait, BuilderTrait, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType,
    MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait, Tool, ToolType,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{
    DataJoinOperator, ToolTrait,
    candle_data::DataConfig,
    candle_operators::{
        data_operator::DataOperatorTrait,
        group_by::{
            build_aggregator_column_fixed_size_list, build_aggregator_column_list_nonprimitive,
            build_aggregator_column_list_primitive,
        },
        select::reorder_batch_vec_columns,
        sort::{sort, take_columns_by_indices},
    },
};

/// Inner join along the LHS foreign key and RHS PK of two [RecordBatch] ONLY the rows with matching values in common are returned
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Join {
    _lhs_pk: String,
    lhs_fk: String,
    _rhs_pk: String,
    rhs_fk: String,
    join_operator: DataJoinOperator,
}

impl MappableTrait for Join {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Join {
    fn get_description(&self) -> String {
        "Join two tables on their foreign keys".to_string()
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
            "rhs_name".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("The name of the right hand side table".to_string()),
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
            "rhs_pk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the right hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_fk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The foriegn key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_fk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The foriegn key column identifier for the right hand side table".to_string(),
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
                    "rhs_name".to_string(),
                    "lhs_fk".to_string(),
                    "rhs_fk".to_string(),
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

impl DataOperatorTrait for Join {
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_pk = config.lhs_pk.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;
        let lhs_fk = config.lhs_fk.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_fk` for `{}`.",
            Self::get_static_name()
        ))?;
        let rhs_pk = config.rhs_pk.clone().ok_or(anyhow!(
            "Missing `rhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;
        let rhs_fk = config.rhs_fk.to_owned().ok_or(anyhow!(
            "Missing `rhs_fk` for `{}`.",
            Self::get_static_name()
        ))?;
        let join_operator = config.join_operators.clone().ok_or(anyhow!(
            "Missing `join_operator` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(Join {
            _lhs_pk: lhs_pk,
            lhs_fk,
            _rhs_pk: rhs_pk,
            rhs_fk,
            join_operator,
        })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        join(
            &self.lhs_fk,
            lhs_args,
            &self.rhs_fk,
            rhs_args.ok_or(anyhow!(
                "Missing `rhs_args` for `{}`.",
                Self::get_static_name()
            ))?,
            &self.join_operator,
            device,
        )
    }
}

type JointInnerTensorResult = (Arc<dyn Array>, Tensor, Arc<dyn Array>, Tensor);

/// Helper method to compute the inner join using Tensors
fn join_inner_tensor(
    lhs_dim_0: usize,
    lhs_tensor: Tensor,
    rhs_dim_0: usize,
    rhs_tensor: Tensor,
    device: &Device,
) -> Result<JointInnerTensorResult> {
    let match_tensor = lhs_tensor
        .cmp(&rhs_tensor, CmpOp::Eq)?
        .to_dtype(DType::U32)?;

    // Convert the matches into indices
    let lhs_indices_tensor = (&Tensor::arange(1u32, (lhs_dim_0 + 1) as u32, device)?
        .reshape((lhs_dim_0, 1))?
        .broadcast_as((lhs_dim_0, rhs_dim_0))?
        * &match_tensor)?
        .flatten_all()?;
    let rhs_indices_tensor = (&Tensor::arange(1u32, (rhs_dim_0 + 1) as u32, device)?
        .reshape((1, rhs_dim_0))?
        .broadcast_as((lhs_dim_0, rhs_dim_0))?
        * &match_tensor)?
        .flatten_all()?;

    // Extract out the indices
    let lhs_indices = lhs_indices_tensor.to_vec1::<u32>()?;
    let lhs_indices = lhs_indices
        .into_iter()
        .filter_map(|v| if v >= 1 { Some(v - 1) } else { None })
        .collect::<Vec<_>>();
    let lhs_tensor = Tensor::from_iter(
        lhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
        device,
    )?;
    let lhs_arr: ArrayRef = Arc::new(UInt32Array::from(lhs_indices));
    let rhs_indices = rhs_indices_tensor.to_vec1::<u32>()?;
    let rhs_indices = rhs_indices
        .into_iter()
        .filter_map(|v| if v >= 1 { Some(v - 1) } else { None })
        .collect::<Vec<_>>();
    let rhs_tensor = Tensor::from_iter(
        rhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
        device,
    )?;
    let rhs_arr: ArrayRef = Arc::new(UInt32Array::from(rhs_indices));
    Ok((lhs_arr, lhs_tensor, rhs_arr, rhs_tensor))
}

type TakeColumnsByUnmatchedIndicesResult = (Vec<(String, Arc<dyn Array>)>, usize);

/// Take the columns according to the UNMATCHED indices over the specified columns
fn take_columns_by_unmatched_indices(
    column_names: &[String],
    table: &Subject,
    asort_arr: &ArrayRef,
    device: &Device,
) -> Result<TakeColumnsByUnmatchedIndicesResult> {
    let asort_vec = asort_arr
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .iter()
        .flatten()
        .collect::<Vec<u32>>();
    let unmatched_vec = (0..table.count_rows() as u32)
        .filter(|i| !asort_vec.contains(i))
        .collect::<Vec<_>>();
    let unmatched_tensor = Tensor::from_iter(
        unmatched_vec
            .iter()
            .map(|v| v.to_owned())
            .collect::<Vec<_>>(),
        device,
    )?;
    let n_unmatched = unmatched_vec.len();
    let unmatched_arr: ArrayRef = Arc::new(UInt32Array::from(unmatched_vec));
    let take = take_columns_by_indices(
        column_names,
        table,
        &unmatched_arr,
        &unmatched_tensor,
        device,
    )?;
    Ok((take, n_unmatched))
}

/// Build [RecordBatch] columns filled with defaults of a specified number of rows
fn build_default_columns(
    column_names: &[String],
    schema: &SchemaRef,
    n_rows: usize,
) -> Result<Vec<(String, Arc<dyn Array>)>> {
    let mut batch_vec = Vec::new();
    for column in column_names.iter() {
        let sorted_array: ArrayRef = match schema.field_with_name(column)?.data_type() {
            DataType::UInt8 => {
                let values_vec = (0..n_rows).map(|_| u8::default()).collect::<Vec<_>>();
                Arc::new(UInt8Array::from(values_vec))
            }
            DataType::UInt32 => {
                let values_vec = (0..n_rows).map(|_| u32::default()).collect::<Vec<_>>();
                Arc::new(UInt32Array::from(values_vec))
            }
            DataType::Int64 => {
                let values_vec = (0..n_rows).map(|_| i64::default()).collect::<Vec<_>>();
                Arc::new(Int64Array::from(values_vec))
            }
            DataType::Float32 => {
                let values_vec = (0..n_rows).map(|_| f32::default()).collect::<Vec<_>>();
                Arc::new(Float32Array::from(values_vec))
            }
            DataType::Float64 => {
                let values_vec = (0..n_rows).map(|_| f64::default()).collect::<Vec<_>>();
                Arc::new(Float64Array::from(values_vec))
            }
            DataType::Utf8 => {
                let values_vec = (0..n_rows).map(|_| String::new()).collect::<Vec<_>>();
                Arc::new(StringArray::from(values_vec))
            }
            DataType::FixedSizeList(f, s) => match f.data_type() {
                DataType::UInt8 => {
                    let values_vec = (0..n_rows)
                        .map(|_| (0..*s).map(|_| u8::default()).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<u8>(values_vec, DataType::UInt8)
                }
                DataType::UInt32 => {
                    let values_vec = (0..n_rows)
                        .map(|_| (0..*s).map(|_| u32::default()).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<u32>(values_vec, DataType::UInt32)
                }
                DataType::Int64 => {
                    let values_vec = (0..n_rows)
                        .map(|_| (0..*s).map(|_| i64::default()).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<i64>(values_vec, DataType::Int64)
                }
                DataType::Float32 => {
                    let values_vec = (0..n_rows)
                        .map(|_| (0..*s).map(|_| f32::default()).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<f32>(values_vec, DataType::Float32)
                }
                DataType::Float64 => {
                    let values_vec = (0..n_rows)
                        .map(|_| (0..*s).map(|_| f64::default()).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<f64>(values_vec, DataType::Float64)
                }
                DataType::Utf8 => {
                    let values_vec = (0..n_rows)
                        .map(|_| (0..*s).map(|_| String::new()).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(values_vec, DataType::Utf8)
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type for column {}: {}",
                        column,
                        schema.field_with_name(column)?.data_type()
                    ));
                }
            },
            DataType::List(f) => match f.data_type() {
                DataType::UInt8 => {
                    let values_vec = (0..n_rows).map(|_| vec![u8::default()]).collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        values_vec,
                        DataType::UInt8,
                    )
                }
                DataType::UInt32 => {
                    let values_vec = (0..n_rows)
                        .map(|_| vec![u32::default()])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        values_vec,
                        DataType::UInt32,
                    )
                }
                DataType::Int64 => {
                    let values_vec = (0..n_rows)
                        .map(|_| vec![i64::default()])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        values_vec,
                        DataType::Int64,
                    )
                }
                DataType::Float32 => {
                    let values_vec = (0..n_rows)
                        .map(|_| vec![f32::default()])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        values_vec,
                        DataType::Float32,
                    )
                }
                DataType::Float64 => {
                    let values_vec = (0..n_rows)
                        .map(|_| vec![f64::default()])
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        values_vec,
                        DataType::Float64,
                    )
                }
                DataType::Utf8 => {
                    let values_vec = (0..n_rows).map(|_| vec![String::new()]).collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(values_vec, DataType::Utf8)
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type for column {}: {}",
                        column,
                        schema.field_with_name(column)?.data_type()
                    ));
                }
            },
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    column,
                    schema.field_with_name(column)?.data_type()
                ));
            }
        };
        batch_vec.push((column.to_owned(), sorted_array));
    }
    Ok(batch_vec)
}

/**
Inner join along the LHS foreign key and RHS PK of two [RecordBatch]
  ONLY the rows with matching values in common are returned

# Arguments

* `lhs` - RecordBatch
* `lhs_fk` - Left hand side foreign key
* `rhs` - RecordBatch
* `rhs_fk` - Right hand side foreign key
* `join_operator` - The join operator to use
* `device` - The compute device

*/
#[instrument(skip(lhs_fk, lhs_args, rhs_fk, rhs_args, join_operator, device))]
pub fn join(
    lhs_fk: &str,
    lhs_args: &[RecordBatch],
    rhs_fk: &str,
    rhs_args: &[RecordBatch],
    join_operator: &DataJoinOperator,
    device: &Device,
) -> Result<RecordBatch> {
    // Presort the lhs and rhs according to PKs
    let lhs_sorted = sort(lhs_fk, lhs_args, true, device)?;
    let rhs_sorted = sort(rhs_fk, rhs_args, true, device)?;

    // Wrap the lhs and rhs into an ArrowTable
    let lhs_table = Subject::get_builder()
        .with_name("join_inner_lhs")
        .with_record_batches(vec![lhs_sorted])?
        .build()?;
    let rhs_table = Subject::get_builder()
        .with_name("join_inner_rhs")
        .with_record_batches(vec![rhs_sorted])?
        .build()?;

    // Join by the FKs
    assert_eq!(
        lhs_table.get_column_data_type(lhs_fk)?,
        rhs_table.get_column_data_type(rhs_fk)?,
        "LHS FK and RHS FK columns must be the same type."
    );
    // Find matches between FKs
    let (lhs_asort_arr, lhs_asort_tensor, rhs_asort_arr, rhs_asort_tensor) =
        match lhs_table.get_column_data_type(lhs_fk)? {
            DataType::UInt8 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<u8>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<u8>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::UInt32 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<u32>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<u32>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Int64 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<i64>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<i64>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Float32 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<f32>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<f32>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Float64 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<f64>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<f64>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Utf8 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_nonprimitive::<String>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_nonprimitive::<String>(rhs_fk)?;
                let mut lhs_indices = Vec::new();
                let mut rhs_indices = Vec::new();

                // Find matches between foreign keys
                for (li, lfk) in lhs_fk_vec.iter().enumerate() {
                    // check for the start of rfk_run
                    let mut rfk_found = false;
                    for (ri, rfk) in rhs_fk_vec.iter().enumerate() {
                        if lfk == rfk {
                            lhs_indices.push(li as u32);
                            rhs_indices.push(ri as u32);
                            rfk_found = true;
                        }

                        // take advantage of the presorting of fks to break early
                        if lfk != rfk && rfk_found {
                            break;
                        }
                    }
                }
                let lhs_tensor = Tensor::from_iter(
                    lhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
                    device,
                )?;
                let lhs_arr: ArrayRef = Arc::new(UInt32Array::from(lhs_indices));
                let rhs_tensor = Tensor::from_iter(
                    rhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
                    device,
                )?;
                let rhs_arr: ArrayRef = Arc::new(UInt32Array::from(rhs_indices));
                (lhs_arr, lhs_tensor, rhs_arr, rhs_tensor)
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    lhs_fk,
                    lhs_table.get_column_data_type(lhs_fk)?
                ));
            }
        };

    // Build the joined table
    let mut batch_vec = Vec::new();
    let lhs_columns: Vec<String> = lhs_table
        .get_schema()
        .fields()
        .iter()
        .map(|field| field.name().to_owned())
        .collect();
    batch_vec.extend(take_columns_by_indices(
        &lhs_columns,
        &lhs_table,
        &lhs_asort_arr,
        &lhs_asort_tensor,
        device,
    )?);

    // Skip the rhs_fk if it matches the lhs_fk
    let rhs_columns: Vec<String> = if lhs_fk == rhs_fk {
        rhs_table
            .get_schema()
            .fields()
            .iter()
            .filter_map(|field| {
                if field.name() == rhs_fk {
                    None
                } else {
                    Some(field.name().to_owned())
                }
            })
            .collect()
    } else {
        rhs_table
            .get_schema()
            .fields()
            .iter()
            .map(|field| field.name().to_owned())
            .collect()
    };

    // Build the inner join
    batch_vec.extend(take_columns_by_indices(
        &rhs_columns,
        &rhs_table,
        &rhs_asort_arr,
        &rhs_asort_tensor,
        device,
    )?);

    // Add the left, right, or both depending on the join operator
    let batch = match join_operator {
        DataJoinOperator::Inner => RecordBatch::try_from_iter(batch_vec)?,
        DataJoinOperator::LeftOuter => {
            // Build the unmatched LHS
            let mut lhs_batch_unmatched_vec = Vec::new();
            let (lhs_take, lhs_n_unmatched) = take_columns_by_unmatched_indices(
                &lhs_columns,
                &lhs_table,
                &lhs_asort_arr,
                device,
            )?;
            lhs_batch_unmatched_vec.extend(lhs_take);

            // Build the default RHS
            let rhs_batch_default_vec =
                build_default_columns(&rhs_columns, &rhs_table.get_schema(), lhs_n_unmatched)?;
            lhs_batch_unmatched_vec.extend(rhs_batch_default_vec);

            // Concatenate the unmatched and matched columns
            let lhs_batch_unmatched = RecordBatch::try_from_iter(lhs_batch_unmatched_vec)?;
            let batch_matched = RecordBatch::try_from_iter(batch_vec)?;
            let schema = batch_matched.schema().clone();
            concat_batches(&schema, &[batch_matched, lhs_batch_unmatched])?
        }
        DataJoinOperator::RightOuter => {
            let rhs_columns: Vec<String> = rhs_table
                .get_schema()
                .fields()
                .iter()
                .map(|field| field.name().to_owned())
                .collect();

            // Skip the lhs_fk if it matches the rhs_fk
            let lhs_columns: Vec<String> = if lhs_fk == rhs_fk {
                lhs_table
                    .get_schema()
                    .fields()
                    .iter()
                    .filter_map(|field| {
                        if field.name() == rhs_fk {
                            None
                        } else {
                            Some(field.name().to_owned())
                        }
                    })
                    .collect()
            } else {
                lhs_table
                    .get_schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().to_owned())
                    .collect()
            };

            // Build the unmatched RHS
            let (rhs_take, rhs_n_unmatched) = take_columns_by_unmatched_indices(
                &rhs_columns,
                &rhs_table,
                &rhs_asort_arr,
                device,
            )?;

            // Build the default LHS
            let mut rhs_batch_unmatched_vec = Vec::new();
            let lhs_batch_default_vec =
                build_default_columns(&lhs_columns, &lhs_table.get_schema(), rhs_n_unmatched)?;
            rhs_batch_unmatched_vec.extend(lhs_batch_default_vec);
            rhs_batch_unmatched_vec.extend(rhs_take);

            // Concatenate the unmatched and matched columns
            let batch_matched = RecordBatch::try_from_iter(batch_vec)?;
            let schema = batch_matched.schema().clone();
            let rhs_batch_unmatched_vec = reorder_batch_vec_columns(
                &rhs_batch_unmatched_vec
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect::<Vec<_>>(),
                &schema
                    .fields()
                    .into_iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>(),
            );
            let rhs_batch_unmatched = RecordBatch::try_from_iter(rhs_batch_unmatched_vec)?;
            concat_batches(&schema, &[batch_matched, rhs_batch_unmatched])?
        }
        DataJoinOperator::FullOuter => {
            // Build the unmatched LHS
            let mut lhs_batch_unmatched_vec = Vec::new();
            let (lhs_take, lhs_n_unmatched) = take_columns_by_unmatched_indices(
                &lhs_columns,
                &lhs_table,
                &lhs_asort_arr,
                device,
            )?;
            lhs_batch_unmatched_vec.extend(lhs_take);

            // Build the default RHS
            let rhs_batch_default_vec =
                build_default_columns(&rhs_columns, &rhs_table.get_schema(), lhs_n_unmatched)?;
            lhs_batch_unmatched_vec.extend(rhs_batch_default_vec);

            // Build the unmatched RHS
            let rhs_columns: Vec<String> = rhs_table
                .get_schema()
                .fields()
                .iter()
                .map(|field| field.name().to_owned())
                .collect();

            // Skip the lhs_fk if it matches the rhs_fk
            let lhs_columns: Vec<String> = if lhs_fk == rhs_fk {
                lhs_table
                    .get_schema()
                    .fields()
                    .iter()
                    .filter_map(|field| {
                        if field.name() == rhs_fk {
                            None
                        } else {
                            Some(field.name().to_owned())
                        }
                    })
                    .collect()
            } else {
                lhs_table
                    .get_schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().to_owned())
                    .collect()
            };
            let (rhs_take, rhs_n_unmatched) = take_columns_by_unmatched_indices(
                &rhs_columns,
                &rhs_table,
                &rhs_asort_arr,
                device,
            )?;

            // Build the default LHS
            let mut rhs_batch_unmatched_vec = Vec::new();
            let lhs_batch_default_vec =
                build_default_columns(&lhs_columns, &lhs_table.get_schema(), rhs_n_unmatched)?;
            rhs_batch_unmatched_vec.extend(lhs_batch_default_vec);
            rhs_batch_unmatched_vec.extend(rhs_take);

            // Concatenate the unmatched and matched columns
            let batch_matched = RecordBatch::try_from_iter(batch_vec)?;
            let schema = batch_matched.schema().clone();
            let lhs_batch_unmatched = RecordBatch::try_from_iter(lhs_batch_unmatched_vec)?;
            let rhs_batch_unmatched_vec = reorder_batch_vec_columns(
                &rhs_batch_unmatched_vec
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect::<Vec<_>>(),
                &schema
                    .fields()
                    .into_iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>(),
            );
            let rhs_batch_unmatched = RecordBatch::try_from_iter(rhs_batch_unmatched_vec)?;
            concat_batches(
                &schema,
                &[batch_matched, lhs_batch_unmatched, rhs_batch_unmatched],
            )?
        }
        _ => {
            return Err(anyhow!(
                "Join operator `{join_operator}` is not yet supported."
            ));
        }
    };
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::device;
    use arrow::array::{ArrayRef, StringArray, UInt8Array, UInt32Array};

    use super::*;

    #[test]
    fn test_join_inner() -> Result<()> {
        // ------ FK = String ------
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
        let rhs_ids_vec_1 = vec!["0", "2", "2"];
        let rhs_ids_array: ArrayRef = Arc::new(StringArray::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Chunk the documents
        let result = join(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &DataJoinOperator::Inner,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "2", "2"]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 3, 3]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "2", "2"]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 9, 10]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "right", "right"]);

        // ------ FK = u8 ------
        // Make the test record batches
        let lhs_ids_vec_1: Vec<u8> = vec![0, 1];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2: Vec<u8> = vec![2, 3];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let rhs_ids_vec_1: Vec<u8> = vec![0, 2, 2];
        let rhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Chunk the documents
        let result = join(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &DataJoinOperator::Inner,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 2, 2]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 3, 3]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 2, 2]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 9, 10]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "right", "right"]);

        Ok(())
    }

    #[test]
    fn test_join_left_outer() -> Result<()> {
        // ------ FK = String ------
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
        let rhs_ids_vec_1 = vec!["0", "2", "2"];
        let rhs_ids_array: ArrayRef = Arc::new(StringArray::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Chunk the documents
        let result = join(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &DataJoinOperator::LeftOuter,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "2", "2", "1", "3"]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 3, 3, 2, 4]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "2", "2", "", ""]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 9, 10, 0, 0]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "right", "right", "", ""]);

        // ------ FK = u8 ------
        // Make the test record batches
        let lhs_ids_vec_1: Vec<u8> = vec![0, 1];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2: Vec<u8> = vec![2, 3];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let rhs_ids_vec_1: Vec<u8> = vec![0, 2, 2];
        let rhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Chunk the documents
        let result = join(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &DataJoinOperator::LeftOuter,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 2, 2, 1, 3]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 3, 3, 2, 4]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 2, 2, 0, 0]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 9, 10, 0, 0]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "right", "right", "", ""]);

        Ok(())
    }

    #[test]
    fn test_join_right_outer() -> Result<()> {
        // ------ FK = String ------
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
        let rhs_ids_vec_1 = vec!["0", "2", "2"];
        let rhs_ids_array: ArrayRef = Arc::new(StringArray::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Chunk the documents
        let result = join(
            "rhs_pk",
            &[rhs_batch_1],
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            &DataJoinOperator::RightOuter,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "2", "2", "1", "3"]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 3, 3, 2, 4]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "2", "2", "", ""]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 9, 10, 0, 0]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "right", "right", "", ""]);

        // ------ FK = u8 ------
        // Make the test record batches
        let lhs_ids_vec_1: Vec<u8> = vec![0, 1];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2: Vec<u8> = vec![2, 3];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let rhs_ids_vec_1: Vec<u8> = vec![0, 2, 2];
        let rhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Chunk the documents
        let result = join(
            "rhs_pk",
            &[rhs_batch_1],
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            &DataJoinOperator::RightOuter,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 2, 2, 1, 3]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 3, 3, 2, 4]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 2, 2, 0, 0]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 9, 10, 0, 0]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "right", "right", "", ""]);

        Ok(())
    }

    #[test]
    fn test_join_full_outer() -> Result<()> {
        // ------ FK = String ------
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
        let rhs_ids_vec_1 = vec!["0", "4", "5"];
        let rhs_ids_array: ArrayRef = Arc::new(StringArray::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Chunk the documents
        let result = join(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &DataJoinOperator::FullOuter,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "1", "2", "3", "", ""]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 2, 3, 4, 0, 0]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left", "left", "", ""]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, ["0", "", "", "", "4", "5"]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 0, 0, 0, 9, 10]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "", "", "", "right", "right"]);

        // ------ FK = u8 ------
        // Make the test record batches
        let lhs_ids_vec_1: Vec<u8> = vec![0, 1];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2: Vec<u8> = vec![2, 3];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let rhs_ids_vec_1 = vec![0, 4, 5];
        let rhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Chunk the documents
        let result = join(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &DataJoinOperator::FullOuter,
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 1, 2, 3, 0, 0]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [1, 2, 3, 4, 0, 0]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["left", "left", "left", "left", "", ""]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, [0, 0, 0, 0, 4, 5]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, [8, 0, 0, 0, 9, 10]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, ["right", "", "", "", "right", "right"]);

        Ok(())
    }
}
