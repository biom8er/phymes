use std::sync::Arc;

use arrow::{array::{ArrayRef, BooleanArray, GenericListArray, PrimitiveArray, RecordBatch, StringArray, UInt32Array, UInt8Array}, compute::{ilike, in_list, in_list_utf8, like, nilike, nlike, regexp_is_match, starts_with}, datatypes::DataType};
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use phymes_core::{session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};
use tracing::instrument;

use crate::candle_data::data_config::{DataComparatorOperator, DataComparatorPredicate};



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
    let predicate_tensors = Vec::new();
    for (index, column_name) in lhs_values.iter().enumerate() {
        let predicate_tensor = match lhs_table.get_column_data_type(column_name)? {
            DataType::UInt8 => {
                let values_vec = lhs_table.get_column_as_vec_primitive::<u8>(column_name)?;
                let values_tensor = Tensor::from_iter(values_vec, device)?;
                let cmp_vec = lhs_table.get_column_as_vec_primitive::<u8>(cmp_columns.get(index).unwrap())?;
                let cmp_tensor = Tensor::from_iter(cmp_vec, device)?;
                match cmp_operator.get(index).unwrap() {
                    DataComparatorOperator::Equals => values_tensor.eq(&cmp_tensor)?,
                    DataComparatorOperator::NotEquals => values_tensor.ne(&cmp_tensor)?,
                    DataComparatorOperator::LessThanOrEqualTo => values_tensor.le(&cmp_tensor)?,
                    DataComparatorOperator::GreaterThanOrEqualTo => values_tensor.ge(&cmp_tensor)?,
                    DataComparatorOperator::LessThan => values_tensor.lt(&cmp_tensor)?,
                    DataComparatorOperator::GreaterThan => values_tensor.gt(&cmp_tensor)?,
                    _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                        lhs_table.get_column_data_type(column_name)?.to_string(), 
                        cmp_operator.get(index).unwrap().get_name())),
                }
            }
            // DM: and the rest...
            DataType::Utf8 => {
                // StringArray must be sorted on the CPU
                let values_arr = StringArray::from(lhs_table.get_column_as_vec_str(column_name));
                let cmp_arr = StringArray::from(lhs_table.get_column_as_vec_str(cmp_columns.get(index).unwrap()));
                let predicate_arr = match cmp_operator.get(index).unwrap() {
                    DataComparatorOperator::Like => like(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::NotLike => nlike(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::CaseInsensitiveLike => ilike(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::CaseInsensitiveNotLike => nilike(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::Contains => contains(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::EndsWith => ends_with(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::StartsWith => starts_with(&values_arr, &cmp_arr)?,
                    DataComparatorOperator::RegExpIsMatch => regexp_is_match(&values_arr, &cmp_arr, None)?,
                    _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                        lhs_table.get_column_data_type(column_name)?.to_string(), 
                        cmp_operator.get(index).unwrap().get_name())),
                };
                let predicate_vec = predicate_arr.into_iter().filter_map(|s| s.unwrap_or_default()).collect::<Vec<_>>();
                Tensor::from_iter(predicate_vec, device)?
            }
            DataType::FixedSizeList(f, _) | DataType::List(f) => match f {
                DataType::UInt8 => {
                    let values_arr = GenericListArray::from_iter_primitive(lhs_table.get_column_as_vec_nested_primitive::<u8>(column_name)?);
                    let cmp_arr = PrimitiveArray::from_iter_values(lhs_table.get_column_as_vec_primitive::<u8>(cmp_columns.get(index).unwrap())?);
                    let predicate_arr = match cmp_operator.get(index).unwrap() {
                        DataComparatorOperator::InList => in_list(&cmp_arr, &values_arr)?,
                        _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                            lhs_table.get_column_data_type(column_name)?.to_string(), 
                            cmp_operator.get(index).unwrap().get_name())),
                    };
                    let predicate_vec = predicate_arr.into_iter().filter_map(|s| s.unwrap_or_default()).collect::<Vec<bool>>();
                    Tensor::from_iter(predicate_vec, device)?
                }
                // DM: and the rest...
                DataType::Utf8 => {
                    let values_arr = GenericListArray::from_iter(lhs_table.get_column_as_vec_nested_nonprimitive::<String>(column_name)?);
                    let cmp_arr = StringArray::from_iter_values(lhs_table.get_column_as_vec_str(cmp_columns.get(index).unwrap())?);
                    let predicate_arr = match cmp_operator.get(index).unwrap() {
                        DataComparatorOperator::InListUtf8 => in_list_utf8(&cmp_arr, &values_arr)?,
                        _ => return Err(anyhow!("Unsupported data type {} and comparator {} for column {column_name}", 
                            lhs_table.get_column_data_type(column_name)?.to_string(), 
                            cmp_operator.get(index).unwrap().get_name())),
                    };
                    let predicate_vec = predicate_arr.into_iter().filter_map(|s| s.unwrap_or_default()).collect::<Vec<bool>>();
                    Tensor::from_iter(predicate_vec, device)?
                }
                _ => return Err(anyhow!("Unsupported data type {} for column {column_name}", lhs_table.get_column_data_type(column_name)?.to_string())),
            }
            _ => return Err(anyhow!("Unsupported data type {} for column {column_name}", lhs_table.get_column_data_type(column_name)?.to_string())),
        };
        predicate_tensors.push(predicate_tensor);
    }

    // Apply the comparison predicate across all filter columns
    let (predicate_arr, predicate_tensor) = if predicate_tensors.len() < 2 {
        let predicate_arr = BooleanArray::from_iter(predicate_tensors.first().unwrap().to_vec1::<Vec<bool>()?);
        (predicate_arr, predicate_tensors.first().unwrap())
    } else {
        match cmp_predicate {
        DataComparatorPredicate::All {
            let mut predicate_tensor = (predicate_tensors.first().unwrap() * predicate_tensors.get(1).unwrap())?;
            for (index, tensor) in predicate_tensors.iter().enumerate() {
                if index >= 2 {
                    predicate_tensor = (predicate_tensor * predicate_tensors.get(index).unwrap())?;
                }
            }
            let predicate_arr = BooleanArray::from_iter(predicate_tensor.to_vec1::<Vec<bool>()?);
            (predicate_arr, predicate_tensors.first().unwrap())
        }
        DataComparatorPredicate::Any {
            let mut predicate_tensor = (predicate_tensors.first().unwrap() + predicate_tensors.get(1).unwrap())?;
            for (index, tensor) in predicate_tensors.iter().enumerate() {
                if index >= 2 {
                    predicate_tensor = (predicate_tensor + predicate_tensors.get(index).unwrap())?;
                }
            }
            let zeros = predicate_tensor.zeros_like(device)?;
            predicate_tensor = predicate_tensor.gt(&zeros)?;
            let predicate_arr = BooleanArray::from_iter(predicate_tensor.to_vec1::<Vec<bool>()?);
            (predicate_arr, predicate_tensors.first().unwrap())
        }
    }

    // Filter the rest of the columns according to the final filter

    // Insert the sorted column at the same position as in the schema
    let batch_vec = Vec::new();

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{ArrayData, FixedSizeListArray},
        buffer::Buffer,
        datatypes::Field,
    };
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_filter_column_and_indices() -> Result<()> {
        Ok
    }
}