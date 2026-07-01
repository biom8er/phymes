use arrow::{
    array::{
        Array, ArrayRef, Float32Array, Float64Array, Int64Array, StringArray, UInt8Array,
        UInt32Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{DataConfig, DataOperatorTrait, ToolTrait};

/// Sort the [RecordBatch] according to the `score` column and then apply the sorting order to the rest of the record batch columns
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Sort {
    lhs_values: String,
    asc: bool,
}

impl MappableTrait for Sort {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Sort {
    fn get_description(&self) -> String {
        "Sort the the list of computed scores in ascending order".to_string()
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
        properties.insert(
            "asc".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::Boolean),
                description: Some(
                    "true for sort order Ascending and false for sort order Descending".to_string(),
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
                    "asc".to_string(),
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

impl DataOperatorTrait for Sort {
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config
            .lhs_values
            .as_ref()
            .cloned()
            .ok_or(anyhow!(
                "Missing `lhs_values` for `{}`.",
                Self::get_static_name()
            ))?
            .first()
            .cloned()
            .ok_or(anyhow!(
                "`lhs_values` is empty for `{}`.",
                Self::get_static_name()
            ))?;
        let asc = config.asc.unwrap_or(true);

        Ok(Sort { lhs_values, asc })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        sort(&self.lhs_values, lhs_args, self.asc, device)
    }
}

/// Take the columns according to the indices over the specified columns
pub fn take_columns_by_indices(
    column_names: &[String],
    table: &Subject,
    asort_arr: &ArrayRef,
    asort_tensor: &Tensor,
    device: &Device,
) -> Result<Vec<(String, Arc<dyn Array>)>> {
    let mut batch_vec = Vec::new();
    for column in column_names.iter() {
        let sorted_array: ArrayRef = match table.get_column_data_type(column)? {
            DataType::UInt8 => {
                let array_vec = table.get_column_as_vec_primitive::<u8>(column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let sorted = tensor.gather(asort_tensor, candle_core::D::Minus1)?;
                let array_vec = sorted.to_vec1::<u8>()?;
                Arc::new(UInt8Array::from(array_vec))
            }
            DataType::UInt32 => {
                let array_vec = table.get_column_as_vec_primitive::<u32>(column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let sorted = tensor.gather(asort_tensor, candle_core::D::Minus1)?;
                let array_vec = sorted.to_vec1::<u32>()?;
                Arc::new(UInt32Array::from(array_vec))
            }
            DataType::Int64 => {
                let array_vec = table.get_column_as_vec_primitive::<i64>(column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let sorted = tensor.gather(asort_tensor, candle_core::D::Minus1)?;
                let array_vec = sorted.to_vec1::<i64>()?;
                Arc::new(Int64Array::from(array_vec))
            }
            DataType::Float32 => {
                let array_vec = table.get_column_as_vec_primitive::<f32>(column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let sorted = tensor.gather(asort_tensor, candle_core::D::Minus1)?;
                let array_vec = sorted.to_vec1::<f32>()?;
                Arc::new(Float32Array::from(array_vec))
            }
            DataType::Float64 => {
                let array_vec = table.get_column_as_vec_primitive::<f64>(column).unwrap();
                let tensor = Tensor::from_iter(array_vec, device)?;
                let sorted = tensor.gather(asort_tensor, candle_core::D::Minus1)?;
                let array_vec = sorted.to_vec1::<f64>()?;
                Arc::new(Float64Array::from(array_vec))
            }
            DataType::Utf8 => {
                // StringArray must be sorted on the CPU
                let array_ref: ArrayRef =
                    Arc::new(StringArray::from(table.get_column_as_vec_str(column)));
                arrow::compute::take(&array_ref, asort_arr, None)?
            }
            DataType::Boolean => {
                // DM: it maybe possible to move Boolean to the GPU in the future
                let array_ref: ArrayRef =
                    Arc::new(StringArray::from(table.get_column_as_vec_str(column)));
                arrow::compute::take(&array_ref, asort_arr, None)?
            }
            DataType::FixedSizeList(_f, _s) => {
                let array_ref: ArrayRef = table.get_column_as_array(column)?;
                arrow::compute::take(&array_ref, asort_arr, None)?
            }
            DataType::List(_f) => {
                let array_ref: ArrayRef = table.get_column_as_array(column)?;
                arrow::compute::take(&array_ref, asort_arr, None)?
            }
            // Note: Candle::Tensor library supports u8, u32, i64, bf16, f16, f32, f64
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    column,
                    table.get_column_data_type(column)?
                ));
            }
        };
        batch_vec.push((column.to_owned(), sorted_array));
    }
    Ok(batch_vec)
}

/**
Sort the [RecordBatch] according to the `score` column
  and then apply the sorting order to the rest of the record batch columns

# Arguments

* `lhs_values` - The name of the column to sort by, typically "score"
* `lhs` - RecordBatch with a column for `score`
* `asc` - true for ascending and false for descending
* `device` - The compute device

*/
#[instrument(skip(lhs_values, lhs_args, asc, device))]
pub fn sort(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    asc: bool,
    device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs into an ArrowTable
    let lhs_table = Subject::get_builder()
        .with_name("sort")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;

    // Extract out the column to sort by
    let (asort_arr, asort_tensor, lhs_sorted) = match lhs_table.get_column_data_type(lhs_values)? {
        DataType::UInt8 => {
            let array_vec = lhs_table
                .get_column_as_vec_primitive::<u8>(lhs_values)
                .unwrap();
            let tensor = Tensor::from_iter(array_vec, device)?;
            let (sorted, asort) = tensor.sort_last_dim(asc)?;
            let lhs_sorted: ArrayRef = Arc::new(UInt8Array::from(sorted.to_vec1::<u8>()?));
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(asort.to_vec1::<u32>()?));
            (asort_arr, asort, lhs_sorted)
        }
        DataType::UInt32 => {
            let array_vec = lhs_table
                .get_column_as_vec_primitive::<u32>(lhs_values)
                .unwrap();
            let tensor = Tensor::from_iter(array_vec, device)?;
            let (sorted, asort) = tensor.sort_last_dim(asc)?;
            let lhs_sorted: ArrayRef = Arc::new(UInt32Array::from(sorted.to_vec1::<u32>()?));
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(asort.to_vec1::<u32>()?));
            (asort_arr, asort, lhs_sorted)
        }
        DataType::Int64 => {
            let array_vec = lhs_table
                .get_column_as_vec_primitive::<i64>(lhs_values)
                .unwrap();
            let tensor = Tensor::from_iter(array_vec, device)?;
            let (sorted, asort) = tensor.sort_last_dim(asc)?;
            let lhs_sorted: ArrayRef = Arc::new(Int64Array::from(sorted.to_vec1::<i64>()?));
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(asort.to_vec1::<u32>()?));
            (asort_arr, asort, lhs_sorted)
        }
        DataType::Float32 => {
            let array_vec = lhs_table
                .get_column_as_vec_primitive::<f32>(lhs_values)
                .unwrap();
            let tensor = Tensor::from_iter(array_vec, device)?;
            let (sorted, asort) = tensor.sort_last_dim(asc)?;
            let lhs_sorted: ArrayRef = Arc::new(Float32Array::from(sorted.to_vec1::<f32>()?));
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(asort.to_vec1::<u32>()?));
            (asort_arr, asort, lhs_sorted)
        }
        DataType::Float64 => {
            let array_vec = lhs_table
                .get_column_as_vec_primitive::<f64>(lhs_values)
                .unwrap();
            let tensor = Tensor::from_iter(array_vec, device)?;
            let (sorted, asort) = tensor.sort_last_dim(asc)?;
            let lhs_sorted: ArrayRef = Arc::new(Float64Array::from(sorted.to_vec1::<f64>()?));
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(asort.to_vec1::<u32>()?));
            (asort_arr, asort, lhs_sorted)
        }
        DataType::Utf8 => {
            // StringArray must be sorted on the CPU
            let array_ref: ArrayRef = Arc::new(StringArray::from(
                lhs_table.get_column_as_vec_str(lhs_values),
            ));
            let sorted_indices = arrow::compute::sort_to_indices(
                &array_ref,
                Some(arrow::compute::SortOptions {
                    descending: !asc,
                    nulls_first: false,
                }),
                None,
            )?;
            let lhs_sorted = arrow::compute::take(&array_ref, &sorted_indices, None)?;
            let asort = sorted_indices
                .iter()
                .map(|v| v.unwrap_or_default())
                .collect::<Vec<u32>>();
            let asort_tensor = Tensor::new(asort, device)?;
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(sorted_indices));
            (asort_arr, asort_tensor, lhs_sorted)
        }
        DataType::FixedSizeList(_f, _s) => {
            let array_ref: ArrayRef = lhs_table.get_column_as_array(lhs_values)?;
            let sorted_indices = arrow::compute::sort_to_indices(
                &array_ref,
                Some(arrow::compute::SortOptions {
                    descending: !asc,
                    nulls_first: false,
                }),
                None,
            )?;
            let lhs_sorted = arrow::compute::take(&array_ref, &sorted_indices, None)?;
            let asort = sorted_indices
                .iter()
                .map(|v| v.unwrap_or_default())
                .collect::<Vec<u32>>();
            let asort_tensor = Tensor::new(asort, device)?;
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(sorted_indices));
            (asort_arr, asort_tensor, lhs_sorted)
        }
        DataType::List(_f) => {
            let array_ref: ArrayRef = lhs_table.get_column_as_array(lhs_values)?;
            let sorted_indices = arrow::compute::sort_to_indices(
                &array_ref,
                Some(arrow::compute::SortOptions {
                    descending: !asc,
                    nulls_first: false,
                }),
                None,
            )?;
            let lhs_sorted = arrow::compute::take(&array_ref, &sorted_indices, None)?;
            let asort = sorted_indices
                .iter()
                .map(|v| v.unwrap_or_default())
                .collect::<Vec<u32>>();
            let asort_tensor = Tensor::new(asort, device)?;
            let asort_arr: ArrayRef = Arc::new(UInt32Array::from(sorted_indices));
            (asort_arr, asort_tensor, lhs_sorted)
        }
        _ => {
            return Err(anyhow!(
                "Unsupported data type {} for column {lhs_values}",
                lhs_table.get_column_data_type(lhs_values)?
            ));
        }
    };

    // Sort the rest of the columns according to the sorted indices
    let mut batch_vec = Vec::new();
    let columns: Vec<String> = lhs_table
        .get_schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if field.name() != lhs_values {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    batch_vec.extend(take_columns_by_indices(
        &columns,
        &lhs_table,
        &asort_arr,
        &asort_tensor,
        device,
    )?);

    // Insert the sorted column at the same position as in the schema
    let index = lhs_table.get_schema().index_of(lhs_values).unwrap();
    batch_vec.insert(index, (lhs_values.to_string(), lhs_sorted));

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::device;
    use arrow::{
        array::{ArrayData, FixedSizeListArray},
        buffer::Buffer,
        datatypes::Field,
    };

    use super::*;

    #[test]
    fn test_sort() -> Result<()> {
        // Float32 test (original)
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_scores_vec_1: Vec<f32> = vec![1., 0.];
        let lhs_scores_array: ArrayRef = Arc::new(Float32Array::from(lhs_scores_vec_1));
        let batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array.clone()),
            ("score", lhs_scores_array),
            ("metadata", lhs_metadata_array.clone()),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array2: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array2: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_scores_vec_2: Vec<f32> = vec![3., 2.];
        let lhs_scores_array2: ArrayRef = Arc::new(Float32Array::from(lhs_scores_vec_2));
        let batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array2.clone()),
            ("score", lhs_scores_array2),
            ("metadata", lhs_metadata_array2.clone()),
        ])?;

        // Make the device
        let device = device(false)?;

        let result = sort("score", &[batch_1, batch_2], true, &device)?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, vec!["1", "0", "3", "2"]);
        let metadata = result
            .column_by_name("metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, vec![2, 1, 4, 3]);
        let scores = result
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(scores, vec![0., 1., 2., 3.]);

        // UInt8 test
        let ids = vec!["a", "b", "c", "d"];
        let ids_array: ArrayRef = Arc::new(StringArray::from(ids.clone()));
        let u8_vec: Vec<u8> = vec![10, 5, 20, 15];
        let u8_array: ArrayRef = Arc::new(UInt8Array::from(u8_vec.clone()));
        let batch = RecordBatch::try_from_iter(vec![
            ("id", ids_array.clone()),
            ("score", u8_array.clone()),
        ])?;
        let result = sort("score", &[batch], true, &device)?;
        let sorted_ids = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(sorted_ids, vec!["b", "a", "d", "c"]);
        let sorted_scores = result
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sorted_scores, vec![5, 10, 15, 20]);

        // Int64 test
        let ids = vec!["x", "y", "z"];
        let ids_array: ArrayRef = Arc::new(StringArray::from(ids.clone()));
        let i64_vec: Vec<i64> = vec![100, -50, 0];
        let i64_array: ArrayRef = Arc::new(Int64Array::from(i64_vec.clone()));
        let batch = RecordBatch::try_from_iter(vec![
            ("id", ids_array.clone()),
            ("score", i64_array.clone()),
        ])?;
        let result = sort("score", &[batch], true, &device)?;
        let sorted_ids = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(sorted_ids, vec!["y", "z", "x"]);
        let sorted_scores = result
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sorted_scores, vec![-50, 0, 100]);

        // Float64 test
        let ids = vec!["p", "q", "r"];
        let ids_array: ArrayRef = Arc::new(StringArray::from(ids.clone()));
        let f64_vec: Vec<f64> = vec![2.5, 1.1, 3.3];
        let f64_array: ArrayRef = Arc::new(Float64Array::from(f64_vec.clone()));
        let batch = RecordBatch::try_from_iter(vec![
            ("id", ids_array.clone()),
            ("score", f64_array.clone()),
        ])?;
        let result = sort("score", &[batch], true, &device)?;
        let sorted_ids = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(sorted_ids, vec!["q", "p", "r"]);
        let sorted_scores = result
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sorted_scores, vec![1.1, 2.5, 3.3]);

        // String test
        let ids = vec![1, 2, 3];
        let ids_array: ArrayRef = Arc::new(UInt32Array::from(ids.clone()));
        let str_vec = vec!["b", "a", "c"];
        let str_array: ArrayRef = Arc::new(StringArray::from(str_vec.clone()));
        let batch = RecordBatch::try_from_iter(vec![
            ("id", ids_array.clone()),
            ("score", str_array.clone()),
        ])?;
        let result = sort("score", &[batch], true, &device)?;
        let sorted_ids = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sorted_ids, vec![2, 1, 3]);
        let sorted_scores = result
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(sorted_scores, vec!["a", "b", "c"]);

        // Vec<Vec<u32>> test
        let ids = vec!["a", "b", "c"];
        let ids_array: ArrayRef = Arc::new(StringArray::from(ids.clone()));
        let list_values: Vec<u32> = vec![
            3, 4, // "a"
            1, 2, // "b"
            2, 3, // "c"
        ];
        let value_data = ArrayData::builder(DataType::UInt32)
            .len(6)
            .add_buffer(Buffer::from_vec(list_values))
            .build()
            .unwrap();
        let list_data_type =
            DataType::FixedSizeList(Arc::new(Field::new_list_field(DataType::UInt32, false)), 2);
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_child_data(value_data.clone())
            .build()
            .unwrap();
        let list_array: ArrayRef = Arc::new(FixedSizeListArray::from(list_data));
        let batch =
            RecordBatch::try_from_iter(vec![("id", ids_array.clone()), ("score", list_array)])?;
        let result = sort("score", &[batch], true, &device)?;
        let sorted_ids = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        // The order should be by the first element in each list: [1,2], [2,3], [3,4]
        assert_eq!(sorted_ids, vec!["b", "c", "a"]);

        Ok(())
    }
}
