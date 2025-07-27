use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray, UInt32Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;
use tracing::instrument;

/**
Sort the [RecordBatch] according to the `score` column
  and then apply the sorting order to the rest of the record batch columns

# Arguments

* `lhs` - RecordBatch with a column for `score`
* `asc` - true for ascending and false for descending
* `device` - The compute device

*/
#[instrument(skip(lhs, asc, device))]
pub fn sort_scores_and_indices(
    lhs: &[RecordBatch],
    asc: bool,
    device: &Device,
) -> Result<RecordBatch> {
    // Extract out the score
    let lhs_embeddings: Vec<f32> = lhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("score")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .map(|f| f.unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Create the lhs Tensors and sort
    let lhs_tensor = Tensor::from_iter(lhs_embeddings, device)?;
    let (sorted, asort) = lhs_tensor.sort_last_dim(asc)?;
    let sorted_vec: Vec<f32> = sorted.to_vec1::<f32>()?;
    let asort_vec: Vec<u32> = asort.to_vec1::<u32>()?;

    // Wrap the output into a record batch
    let mut batch_vec = Vec::new();
    let out_scores: ArrayRef = Arc::new(Float32Array::from(sorted_vec));
    batch_vec.push(("score", out_scores));

    // Sort the other columns...
    let sorted_indices: ArrayRef = Arc::new(UInt32Array::from(asort_vec));

    // ...Primitive columns can be done on the GPU
    // DM: repeat for all primitive types not just UInt32
    let columns: Vec<String> = lhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != "score") & (field.data_type() == &DataType::UInt32) {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_vec = lhs
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name(column)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::from_iter(array_vec, device)?;
        let sorted = tensor.gather(&asort, candle_core::D::Minus1)?;
        let array_vec = sorted.to_vec1::<u32>()?;
        let sorted_array: ArrayRef = Arc::new(UInt32Array::from(array_vec));
        batch_vec.push((column, sorted_array));
    }

    // ...StringArray columns must be done on the CPU
    let columns: Vec<String> = lhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != "score") & (field.data_type() == &DataType::Utf8) {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_vec = lhs
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name(column)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
        let sorted_array = arrow::compute::take(&array_ref, &sorted_indices, None)?;
        batch_vec.push((column, sorted_array));
    }

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::candle_ops::ops_processor::test_candle_ops_processor::make_embeddings_record_batch;

    use super::*;

    #[test]
    fn test_sort_scores_and_indices() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_scores_vec_1: Vec<f32> = vec![1., 0.];
        let lhs_scores_array: ArrayRef = Arc::new(Float32Array::from(lhs_scores_vec_1));
        let batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("score", lhs_scores_array),
            ("metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_scores_vec_2: Vec<f32> = vec![3., 2.];
        let lhs_scores_array: ArrayRef = Arc::new(Float32Array::from(lhs_scores_vec_2));
        let batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("score", lhs_scores_array),
            ("metadata", lhs_metadata_array),
        ])?;

        // Sort according to score
        let result = sort_scores_and_indices(&[batch_1, batch_2], true, &Device::Cpu)?;

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

        Ok(())
    }
}
