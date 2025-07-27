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
Inner join along the LHS foreign key and RHS PK of two [RecordBatch]
  ONLY the rows with matching values in common are returned

# Arguments

* `lhs` - RecordBatch
* `lhs_fk` - Left hand side foreign key
* `rhs` - RecordBatch
* `rhs_fk` - Right hand side foreign key
* `device` - The compute device

*/
#[instrument(skip(lhs, rhs, lhs_fk, rhs_fk, _device))]
pub fn join_inner(
    lhs: &[RecordBatch],
    lhs_fk: &str,
    rhs: &[RecordBatch],
    rhs_fk: &str,
    _device: &Device,
) -> Result<RecordBatch> {
    // Extract the foreign keys
    let lhs_fk_vec = lhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(lhs_fk)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rhs_fk_vec = rhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(rhs_fk)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Find matches between foreign keys
    let mut lhs_indices = Vec::new();
    let mut rhs_indices = Vec::new();
    let mut lhs_fk_matches_vec = Vec::new();
    let mut rhs_fk_matches_vec = Vec::new();
    for (li, lfk) in lhs_fk_vec.iter().enumerate() {
        for (ri, rfk) in rhs_fk_vec.iter().enumerate() {
            if lfk == rfk {
                lhs_indices.push(li);
                rhs_indices.push(ri);
                lhs_fk_matches_vec.push(lfk.to_owned());
                rhs_fk_matches_vec.push(rfk.to_owned());
            }
        }
    }

    // Build lhs columns
    let mut batch_vec = Vec::new();
    let array_ref: ArrayRef = Arc::new(StringArray::from(lhs_fk_matches_vec));
    batch_vec.push((lhs_fk, array_ref));
    let array_ref: ArrayRef = Arc::new(StringArray::from(rhs_fk_matches_vec));
    batch_vec.push((rhs_fk, array_ref));

    // ... starting with the lhs
    let columns: Vec<String> = lhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != lhs_fk) & (field.data_type() == &DataType::Utf8) {
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
        let array_vec = lhs_indices
            .iter()
            .map(|i| array_vec.get(*i).unwrap().to_owned())
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
        batch_vec.push((column, array_ref));
    }
    let columns: Vec<String> = lhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if field.data_type() == &DataType::UInt32 {
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
        let array_vec = lhs_indices
            .iter()
            .map(|i| array_vec.get(*i).unwrap().to_owned())
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(UInt32Array::from(array_vec));
        batch_vec.push((column, array_ref));
    }
    let columns: Vec<String> = lhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if field.data_type() == &DataType::Float32 {
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
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let array_vec = lhs_indices
            .iter()
            .map(|i| array_vec.get(*i).unwrap().to_owned())
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(Float32Array::from(array_vec));
        batch_vec.push((column, array_ref));
    }

    // ... and then the rhs
    let columns: Vec<String> = rhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != rhs_fk) & (field.data_type() == &DataType::Utf8) {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_vec = rhs
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
        let array_vec = rhs_indices
            .iter()
            .map(|i| array_vec.get(*i).unwrap().to_owned())
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
        batch_vec.push((column, array_ref));
    }
    let columns: Vec<String> = rhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if field.data_type() == &DataType::UInt32 {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_vec = rhs
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
        let array_vec = rhs_indices
            .iter()
            .map(|i| array_vec.get(*i).unwrap().to_owned())
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(UInt32Array::from(array_vec));
        batch_vec.push((column, array_ref));
    }
    let columns: Vec<String> = rhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if field.data_type() == &DataType::Float32 {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_vec = rhs
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name(column)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let array_vec = rhs_indices
            .iter()
            .map(|i| array_vec.get(*i).unwrap().to_owned())
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(Float32Array::from(array_vec));
        batch_vec.push((column, array_ref));
    }

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::candle_ops::ops_processor::test_candle_ops_processor::make_embeddings_record_batch;

    use super::*;

    #[test]
    fn test_join_inner() -> Result<()> {
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
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 9];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Chunk the documents
        let result = join_inner(
            &[lhs_batch_1, lhs_batch_2],
            "lhs_pk",
            &[rhs_batch_1],
            "rhs_pk",
            &Device::Cpu,
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
        assert_eq!(lhs_id, vec!["0", "2", "2"]);
        let metadata = result
            .column_by_name("lhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, vec![1, 3, 3]);
        let text = result
            .column_by_name("lhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, vec!["left", "left", "left"]);
        let lhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, vec!["0", "2", "2"]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, vec![8, 9, 9]);
        let text = result
            .column_by_name("rhs_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(text, vec!["right", "right", "right"]);

        Ok(())
    }
}
