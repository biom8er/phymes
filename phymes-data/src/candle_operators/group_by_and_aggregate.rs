use std::{ops::Range, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::{ArrayRef, RecordBatch, StringArray, UInt8Array, UInt32Array}, compute::kernels::partition::partition, datatypes::{DataType, Schema}};
use candle_core::{Device, Tensor};
use phymes_core::{session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};

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

/// Helper function to compute the aggregation operator for tensors
fn aggregator_operator_tensor(agg_column: &str, agg_operator: &DataAggregator, lhs_table: &ArrowTable, tensor: &Tensor, range: &Range<usize>, device: &Device) -> Result<Tensor> {
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
    Ok(agg_tensor)
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

    // Copy out the first partition for each group
    for group_column in lhs_values.iter() {
        let lhs_agg: ArrayRef = match lhs_table.get_column_data_type(group_column)? {
            DataType::UInt8 => {
                let array_vec = lhs_table.get_column_as_vec_primitive::<u8>(group_column).unwrap();
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    agg_vec.push(array_vec.get(range.start).unwrap().to_owned());
                }
                Arc::new(UInt8Array::from(agg_vec))
            }
            DataType::UInt32 => {
                let array_vec = lhs_table.get_column_as_vec_primitive::<u32>(group_column).unwrap();
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    agg_vec.push(array_vec.get(range.start).unwrap().to_owned());
                }
                Arc::new(UInt32Array::from(agg_vec))
            }
            // DM: repeat for all other types
            DataType::Utf8 => {
                let array_vec = lhs_table.get_column_as_vec_nonprimitive::<String>(group_column).unwrap();
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    agg_vec.push(array_vec.get(range.start).unwrap().to_owned());
                }
                Arc::new(StringArray::from(agg_vec))
            }
            _ => return Err(anyhow!(
                "Unsupported data type for column {}: {}",
                group_column,
                lhs_table.get_column_data_type(group_column)?.to_string()
            )),
        };
        batch_vec.push((group_column.to_string(), lhs_agg));
    }

    // Apply the aggregation operators
    for (agg_column, agg_operator) in agg_columns.iter().zip(agg_operators.iter()) {
        let lhs_agg: ArrayRef = match lhs_table.get_column_data_type(agg_column)? {
            DataType::UInt8 => {
                let array_vec = lhs_table.get_column_as_vec_primitive::<u8>(agg_column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let agg_tensor = aggregator_operator_tensor(&agg_column, agg_operator, &lhs_table, &tensor, range, device)?;
                    agg_vec.push(agg_tensor.to_vec0::<u8>()?);
                }
                Arc::new(UInt8Array::from(agg_vec))
            }
            DataType::UInt32 => {
                let array_vec = lhs_table.get_column_as_vec_primitive::<u32>(agg_column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let mut agg_vec = Vec::new();
                for range in ranges.iter() {
                    let agg_tensor = aggregator_operator_tensor(&agg_column, agg_operator, &lhs_table, &tensor, range, device)?;
                    agg_vec.push(agg_tensor.to_vec0::<u32>()?);
                }
                Arc::new(UInt32Array::from(agg_vec))
            }
            // DM: repeat for all other types
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
        let columns_name = format!("{}-{}", agg_column.to_string(), agg_operator.get_name());
        batch_vec.push((columns_name, lhs_agg));
    }

    // Create the output batch
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_ml::candle_assets::device::device;

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
            &[DataAggregator::Concat, DataAggregator::Count, DataAggregator::Sum, DataAggregator::Max], 
            &device)?;
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
            &[DataAggregator::Count], 
            &device)?;
        let result_table = ArrowTable::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_id = result_table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_id, vec!["0","1","2","3"]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata")?;
        assert_eq!(metadata, vec![1,2,3,4]);
        let lhs_text = result_table.get_column_as_vec_str("lhs_text-Count");
        assert_eq!(lhs_text, vec!["1","1","1","1"]);

        Ok(())
    }
}