use std::sync::Arc;

use arrow::{array::{ArrayRef, ListArray, PrimitiveArray, RecordBatch, StringArray, UInt32Array}, compute::{contains, ends_with, filter, ilike, in_list, in_list_utf8, like, nilike, nlike, regexp_is_match, starts_with}, datatypes::{DataType, UInt8Type}};
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor, WithDType};
use num_traits::{Bounded, Num, NumCast};
use phymes_core::{session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};
use tracing::instrument;

use crate::{candle_data::data_config::{DataComparatorOperator, DataComparatorPredicate}, candle_operators::{group_by_and_aggregate::build_aggregator_column_list, sort_column_and_indices::take_columns_by_indices}};

/// Helper function to compute the comparator operator for tensors
fn comparator_operator_tensor<T>(
    column_name: &str,
    index: usize,
    cmp_columns: &[&str],
    cmp_operator: &[DataComparatorOperator],
    lhs_table: &ArrowTable,
    device: &Device
) -> Result<Tensor> 
where
    T: Num + Bounded + NumCast + Send + Sync + Clone + WithDType + 'static,
{
    let values_vec = lhs_table.get_column_as_vec_primitive::<T>(column_name)?;
    let values_tensor = Tensor::from_iter(values_vec, device)?;
    let cmp_vec = lhs_table.get_column_as_vec_primitive::<T>(cmp_columns.get(index).unwrap())?;
    let cmp_tensor = Tensor::from_iter(cmp_vec, device)?;
    let tensor = match cmp_operator.get(index).unwrap() {
        DataComparatorOperator::Equals => values_tensor.eq(&cmp_tensor)?,
        DataComparatorOperator::NotEquals => values_tensor.ne(&cmp_tensor)?,
        DataComparatorOperator::LessThanOrEqualTo => values_tensor.le(&cmp_tensor)?,
        DataComparatorOperator::GreaterThanOrEqualTo => values_tensor.ge(&cmp_tensor)?,
        DataComparatorOperator::LessThan => values_tensor.lt(&cmp_tensor)?,
        DataComparatorOperator::GreaterThan => values_tensor.gt(&cmp_tensor)?,
        _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
            lhs_table.get_column_data_type(column_name)?.to_string(), 
            cmp_operator.get(index).unwrap().get_name())),
    };
    // Ensure all tensors are u32 for subsequent operations
    let tensor = tensor.to_dtype(candle_core::DType::U32)?;
    Ok(tensor)
}

/// Filter by specified columns using a specified comparator operator over specified columns
/// 
/// # Notes
/// * An SQL equivalent would be the following, e.g., where COL1 < COL2 AND ...
///   `lhs_values` = ["COL1", ...]
///   `cmp_columns` = ["COL2", ...]
///   `cmp_operator` = [DataComparatorOperator::LessThan, ...]
///   `cmp_predicate` = DataComparatorPredicate::All
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the columns to filter by
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `cmp_columns` - Slice of Strings for the comparator columns
/// * `cmp_operator` - Slice of [DataComparatorOperator]s specifying the comparator operator to apply to each cmp_column
/// * `cmp_predicate` - [DataComparatorPredicate]
/// * `device` - The compute device
#[instrument(skip(lhs_values, lhs_args, cmp_columns, cmp_operator, cmp_predicate, device))]
pub fn filter_columns_and_indices(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    cmp_columns: &[&str],
    cmp_operator: &[DataComparatorOperator],
    cmp_predicate: DataComparatorPredicate,
    device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs into an ArrowTable
    let lhs_table = ArrowTable::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?;

    // Apply the filter to each column based on type and comparator
    let mut predicate_tensors = Vec::new();
    for (index, column_name) in lhs_values.iter().enumerate() {
        let predicate_tensor = match lhs_table.get_column_data_type(column_name)? {
            DataType::UInt8 => comparator_operator_tensor::<u8>(&column_name, index, cmp_columns, cmp_operator, &lhs_table, device)?,
            DataType::UInt32 => comparator_operator_tensor::<u32>(&column_name, index, cmp_columns, cmp_operator, &lhs_table, device)?,
            DataType::Int64 => comparator_operator_tensor::<i64>(&column_name, index, cmp_columns, cmp_operator, &lhs_table, device)?,
            DataType::Float32 => comparator_operator_tensor::<f32>(&column_name, index, cmp_columns, cmp_operator, &lhs_table, device)?,
            DataType::Float64 => comparator_operator_tensor::<f64>(&column_name, index, cmp_columns, cmp_operator, &lhs_table, device)?,
            DataType::Utf8 => {
                // StringArray must be sorted on the CPU
                let values_arr = StringArray::from(lhs_table.get_column_as_vec_str(column_name));
                let cmp_arr = StringArray::from(lhs_table.get_column_as_vec_str(cmp_columns.get(index).unwrap()));
                let flags_array: Option<&StringArray> = None;
                let predicate_arr = match cmp_operator.get(index).unwrap() {
                    DataComparatorOperator::Like => like(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::NotLike => nlike(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::CaseInsensitiveLike => ilike(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::CaseInsensitiveNotLike => nilike(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::Contains => contains(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::EndsWith => ends_with(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::StartsWith => starts_with(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::RegExpIsMatch => regexp_is_match(&values_arr, &cmp_arr, flags_array)?,
                    _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                        lhs_table.get_column_data_type(column_name)?.to_string(), 
                        cmp_operator.get(index).unwrap().get_name())),
                };
                let predicate_vec = predicate_arr.into_iter().map(|s| s.unwrap_or_default() as u32).collect::<Vec<_>>();
                Tensor::from_iter(predicate_vec, device)?
            }
            DataType::FixedSizeList(f, _) | DataType::List(f) => match f.data_type() {
                DataType::UInt8 => {
                    let values_arr = build_aggregator_column_list::<u8>(
                        lhs_table.get_column_as_vec_nested_primitive::<u8>(column_name)?,
                        DataType::UInt8);
                    let values_arr = values_arr.as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap();
                    let cmp_arr: PrimitiveArray<UInt8Type> = PrimitiveArray::from_iter_values(lhs_table.get_column_as_vec_primitive::<u8>(cmp_columns.get(index).unwrap())?);
                    let predicate_arr = match cmp_operator.get(index).unwrap() {
                        DataComparatorOperator::InList => in_list(&cmp_arr, values_arr)?,
                        _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                            lhs_table.get_column_data_type(column_name)?.to_string(), 
                            cmp_operator.get(index).unwrap().get_name())),
                    };
                    let predicate_vec = predicate_arr.into_iter().map(|s| s.unwrap_or_default() as u32).collect::<Vec<_>>();
                    Tensor::from_iter(predicate_vec, device)?
                }
                // DM: and the rest...
                // DM: FixedSizeList and List for Utf8 is not yet supported
                // DataType::Utf8 => {
                //     let values_arr = build_aggregator_column_list::<String>(
                //         lhs_table.get_column_as_vec_nested_nonprimitive::<String>(column_name)?,
                //         DataType::Utf8);
                //     let cmp_arr = StringArray::from_iter_values(lhs_table.get_column_as_vec_str(cmp_columns.get(index).unwrap()));
                //     let predicate_arr = match cmp_operator.get(index).unwrap() {
                //         DataComparatorOperator::InListUtf8 => in_list_utf8(&cmp_arr, &values_arr)?,
                //         _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                //             lhs_table.get_column_data_type(column_name)?.to_string(), 
                //             cmp_operator.get(index).unwrap().get_name())),
                //     };
                //     let predicate_vec = predicate_arr.into_iter().map(|s| s.unwrap_or_default() as u8).collect::<Vec<_>>();
                //     Tensor::from_iter(predicate_vec, device)?
                // }
                _ => return Err(anyhow!("Unsupported data type {} for column {column_name}", lhs_table.get_column_data_type(column_name)?.to_string())),
            }
            _ => return Err(anyhow!("Unsupported data type {} for column {column_name}", lhs_table.get_column_data_type(column_name)?.to_string())),
        };
        predicate_tensors.push(predicate_tensor);
    }

    // Apply the comparison predicate across all filter columns and reduce to a single predicate
    let predicate_tensor = if predicate_tensors.len() < 2 {
        predicate_tensors.first().unwrap().detach()
    } else {
        match cmp_predicate {
            DataComparatorPredicate::All => {
                let mut predicate_tensor = (predicate_tensors.first().unwrap() * predicate_tensors.get(1).unwrap())?;
                for (index, tensor) in predicate_tensors.iter().enumerate() {
                    if index >= 2 {
                        predicate_tensor = (predicate_tensor * tensor)?;
                    }
                }
                predicate_tensor
            }
            DataComparatorPredicate::Any => {
                let mut predicate_tensor = (predicate_tensors.first().unwrap() + predicate_tensors.get(1).unwrap())?;
                for (index, tensor) in predicate_tensors.iter().enumerate() {
                    if index >= 2 {
                        predicate_tensor = (predicate_tensor + tensor)?;
                    }
                }
                // let zeros = predicate_tensor.zeros_like()?;
                // predicate_tensor = predicate_tensor.gt(&zeros)?;
                predicate_tensor
            }
        }
    };

    // Convert the predicate to a take vec based on it's indices (CPU,GPU with Candle)
    let indices_tensor = Tensor::arange(1 as u32, (predicate_tensor.dims1()? + 1) as u32, device)?;
    let zeros = indices_tensor.zeros_like()?;
    let (sorted, _asort) = predicate_tensor.where_cond(&indices_tensor, &zeros)?.sort_last_dim(true)?;
    let take_vec = sorted.to_vec1::<u32>()?
        .into_iter()
        .filter_map(|v| {
            if v > 0 {
                Some(v - 1)
            } else {
                None
            }
        }).collect::<Vec<u32>>();
    let take_tensor = Tensor::from_iter(take_vec.clone(), device)?;
    let take_arr: ArrayRef = Arc::new(UInt32Array::from_iter(take_vec));

    // // Convert the predicate to a take vec based on it's indices (CPU with Arrow)
    // let indices_vec = UInt32Array::from_iter(0 as u32..predicate_tensor.dims1()? as u32);
    // let predicate_arr = BooleanArray::from_iter(predicate_tensor.to_vec1::<bool>()?);
    // let take_arr = filter(&indices_vec, &predicate_arr)?;
    // let take_vec = take_arr.as_any()
    //     .downcast_ref::<UInt32Array>()
    //     .unwrap()
    //     .iter()
    //     .filter_map(|s| s.unwrap_or_default())
    //     .collect::<Vec<u32>>();
    // let take_tensor = Tensor::from_iter(values_vec, device)?;

    // Filter the rest of the columns according to the final filter
    let mut batch_vec = Vec::new();
    let columns: Vec<String> = lhs_table
        .get_schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    batch_vec.extend(take_columns_by_indices(
        &columns,
        &lhs_table,
        take_arr,
        take_tensor,
        device,
    )?);

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_filter_column_and_indices() -> Result<()> {
        // ------ String, UInt32, All ------
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
        let result = filter_columns_and_indices(
            &["lhs_pk", "lhs_metadata"],
            &[lhs_batch_1, lhs_batch_2],
            &["lhs_pk", "lhs_metadata"],
            &[
                DataComparatorOperator::Like,
                DataComparatorOperator::Equals,
            ],
            DataComparatorPredicate::All,
            &device,
        )?;
        let result_table = ArrowTable::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("lhs_text");
        assert_eq!(lhs_text, vec!["left","left","left","left"]);
        let lhs_id = result_table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_id, vec!["0", "1", "2", "3"]);
        let metadata = result_table.get_column_as_vec_primitive::<u32>("lhs_metadata")?;
        assert_eq!(metadata, vec![1, 2, 3, 4]);

        Ok(())
    }
}