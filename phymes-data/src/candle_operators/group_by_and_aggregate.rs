use std::{ops::Range, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::{ArrayRef, RecordBatch, StringArray, UInt8Array}, compute::kernels::partition::partition, datatypes::{DataType, Schema}};
use candle_core::{Device, Tensor};
use phymes_core::{session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};
use tracing_subscriber::registry::Data;

use crate::{candle_data::data_config::DataAggregator, candle_operators::sort_scores_and_indices::sort_column_and_indices};


/// Partition a lexocographically sorted slice of [RecordBatch]es
fn partition_record_batches(lhs_values: &[&str], lhs_table: &ArrowTable) -> Result<Vec<Range<usize>>> {
    let mut columns = Vec::new();
    for column_name in lhs_values.iter() {
        columns.push(lhs_table.get_column_as_array(column_name));
    }
    let ranges = partition(&columns)?.ranges();
    Ok(ranges)
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
fn group_by_and_aggregate(lhs_values: &[&str], lhs_args: &[RecordBatch], agg_columns: &[&str], agg_operators: &[DataAggregator], device: &Device) -> Result<RecordBatch> {
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
    // batch_vec.extend(take_columns_by_indices(
    //     &lhs_values,
    //     &lhs_table,
    //     lhs_asort_arr,
    //     lhs_asort_tensor,
    //     device,
    // )?);

    // Apply the aggregation operators
    for (agg_column, agg_operator) in agg_columns.iter().zip(agg_operators.iter()) {
        let lhs_agg: ArrayRef = match lhs_table.get_column_data_type(agg_column)? {
            DataType::UInt8 => {
                let array_vec = lhs_table.get_column_as_vec_primitive::<u8>(agg_column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let gather_tensor = Tensor::arange(range.start as u8, range.end as u8, device)?;
                    let agg_tensor = match agg_operator {
                        DataAggregator::Sum => tensor.gather(&gather_tensor, candle_core::D::Minus1)?.sum(candle_core::D::Minus1)?,
                        DataAggregator::Max => tensor.gather(&gather_tensor, candle_core::D::Minus1)?.max(candle_core::D::Minus1)?,
                        DataAggregator::Min => tensor.gather(&gather_tensor, candle_core::D::Minus1)?.min(candle_core::D::Minus1)?,
                        DataAggregator::Mean => tensor.gather(&gather_tensor, candle_core::D::Minus1)?.mean(candle_core::D::Minus1)?,
                        DataAggregator::Var => tensor.gather(&gather_tensor, candle_core::D::Minus1)?.var(candle_core::D::Minus1)?,
                        _ => return Err(anyhow!(
                            "Unsupported data type {} and aggregator operator {} for column {}",
                            lhs_table.get_column_data_type(agg_column)?.to_string(),
                            agg_operator.get_name(),
                            agg_column,
                            
                        )),
                    };
                    agg_vec.push(agg_tensor.to_vec0::<u8>()?);
                }
                Arc::new(UInt8Array::from(agg_vec))
            }
            DataType::Utf8 => {
                let array_vec = lhs_table.get_column_as_vec_nonprimitive::<String>(agg_column).unwrap();
                let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let gather_arr: ArrayRef = Arc::new(UInt8Array::from_iter_values(range.start as u8..range.end as u8));
                    let taken_arr = arrow::compute::take(&array_ref, &gather_arr, None)?;
                    let agg_value = match agg_operator {
                        DataAggregator::Count => format!("{}", taken_arr.len()),
                        DataAggregator::Concat => taken_arr
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                            .join(""),
                        _ => return Err(anyhow!(
                            "Unsupported data type {} and aggregator operator {} for column {}",
                            lhs_table.get_column_data_type(agg_column)?.to_string(),
                            agg_operator.get_name(),
                            agg_column,
                            
                        )),
                    };
                    agg_vec.push(agg_value);
                }
                Arc::new(StringArray::from(agg_vec))
            }
            _ => return Err(anyhow!(
                "Unsupported data type for column {}: {}",
                agg_column,
                lhs_table.get_column_data_type(agg_column)?.to_string()
            )),
        };
        batch_vec.push((agg_column.to_string(), lhs_agg));
    }

    // Create the output batch
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}