use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray},
    record_batch::RecordBatch,
};

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;
use tracing::instrument;

/**
Compute the relative similarity between two [RecordBatch]es
  where each [RecordBatch] represents a list of vector embeddings

# Arguments

* `lhs` - Query 2D Tensor
* `rhs` - Document chunk 2D Tensor
* `device` - The compute device

*/
#[instrument(skip(lhs, rhs, lhs_pk, lhs_values, rhs_pk, rhs_values, device))]
pub fn relative_similarity_scores(
    lhs: &[RecordBatch],
    rhs: &[RecordBatch],
    lhs_pk: &str,
    lhs_values: &str,
    rhs_pk: &str,
    rhs_values: &str,
    device: &Device,
) -> Result<RecordBatch> {
    // Extract out the lhs_id and the embeddings
    let lhs_embeddings = lhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(lhs_values)
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .iter()
                .map(|s| {
                    s.unwrap()
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .unwrap()
                        .iter()
                        .map(|f| f.unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let lhs_id = lhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(lhs_pk)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Extract out the rhs and the embeddings
    let rhs_embeddings = rhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(rhs_values)
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .iter()
                .map(|s| {
                    s.unwrap()
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .unwrap()
                        .iter()
                        .map(|f| f.unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let rhs_id = rhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(rhs_pk)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Create the lhs and rhs Tensors
    let lhs_dim_1 = lhs_embeddings.len();
    let lhs_dim_2 = lhs_embeddings.first().unwrap().len();
    let lhs_vec = lhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
    let lhs_tensor = Tensor::from_iter(lhs_vec, device)?.reshape((lhs_dim_1, lhs_dim_2))?;
    let rhs_dim_1 = rhs_embeddings.len();
    let rhs_dim_2 = rhs_embeddings.first().unwrap().len();
    let rhs_vec = rhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
    let rhs_tensor = Tensor::from_iter(rhs_vec, device)?.reshape((rhs_dim_1, rhs_dim_2))?;

    // Run the operation
    let result = relative_similarity_scores_tensor(&lhs_tensor, &rhs_tensor)?;
    let result_vec = result.to_vec2::<f32>()?;

    // Wrap the output into a record batch
    let mut out_lhs_id_vec = Vec::with_capacity(lhs_dim_1 * rhs_dim_2);
    let mut out_rhs_id_vec = Vec::with_capacity(lhs_dim_1 * rhs_dim_2);
    for lhs in lhs_id.iter() {
        for rhs in rhs_id.iter() {
            out_lhs_id_vec.push(lhs.to_string());
            out_rhs_id_vec.push(rhs.to_string());
        }
    }
    let out_scores_vec = result_vec.into_iter().flatten().collect::<Vec<_>>();
    let out_lhs_id: ArrayRef = Arc::new(StringArray::from(out_lhs_id_vec));
    let out_rhs_id: ArrayRef = Arc::new(StringArray::from(out_rhs_id_vec));
    let out_scores: ArrayRef = Arc::new(Float32Array::from(out_scores_vec));
    let batch = RecordBatch::try_from_iter(vec![
        (lhs_pk, out_lhs_id),
        (rhs_pk, out_rhs_id),
        ("score", out_scores),
    ])?;
    Ok(batch)
}

/**
Compute the relative similarity between two Tensors

# Arguments

* `lhs` - Query 2D Tensor
* `rhs` - Document chunk 2D Tensor
* `device` - The compute device

*/
#[instrument(skip(lhs, rhs))]
pub fn relative_similarity_scores_tensor(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let embd = Tensor::cat(&[&lhs, &rhs], 0)?;
    let norm = embd
        .broadcast_div(&embd.sqr()?.sum_keepdim(1)?.sqrt()?)?
        .contiguous()?;
    let scores = norm
        .narrow(0, 0, lhs.dims2()?.0)?
        .matmul(&norm.narrow(0, lhs.dims2()?.0, rhs.dims2()?.0)?.t()?)?;
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use crate::candle_ops::ops_processor::test_candle_ops_processor::make_embeddings_record_batch;

    use super::*;

    #[test]
    fn test_relative_similarity_scores_tensor() -> Result<()> {
        let lhs_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let rhs_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let scores_vec: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];
        let lhs = Tensor::from_iter(
            lhs_vec.into_iter().flatten().collect::<Vec<_>>(),
            &Device::Cpu,
        )?
        .reshape((3, 4))?;
        let rhs = Tensor::from_iter(
            rhs_vec.into_iter().flatten().collect::<Vec<_>>(),
            &Device::Cpu,
        )?
        .reshape((4, 4))?;
        let result = relative_similarity_scores_tensor(&lhs, &rhs)?;
        let result_vec = result
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(result_vec, scores_vec);

        Ok(())
    }

    #[test]
    fn test_relative_similarity_scores() -> Result<()> {
        // LHS and RHS record batches
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs = make_embeddings_record_batch("lhs_pk", lhs_ids_vec, lhs_embeddings_vec)?;
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs = make_embeddings_record_batch("rhs_pk", rhs_ids_vec, rhs_embeddings_vec)?;

        // Compute the relative similarity scores
        let result = relative_similarity_scores(
            &[lhs],
            &[rhs],
            "lhs_pk",
            "embedding",
            "rhs_pk",
            "embedding",
            &Device::Cpu,
        )?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        Ok(())
    }
}
