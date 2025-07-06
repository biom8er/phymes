use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray, UInt32Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;

/**
Compute the relative similarity between two [RecordBatch]es
  where each [RecordBatch] represents a list of vector embeddings

# Arguments

* `lhs` - Query 2D Tensor
* `rhs` - Document chunk 2D Tensor
* `device` - The compute device

*/
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

/**
Sort the [RecordBatch] according to the `score` column
  and then apply the sorting order to the rest of the record batch columns

# Arguments

* `lhs` - RecordBatch with a column for `score`
* `asc` - true for ascending and false for descending
* `device` - The compute device

*/
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

/**
Chunk documents by splitting a StringArray column in a [RecordBatch]
  into multiple rows based on a defined criteria

# Notes
* inspired by <https://python.langchain.com/api_reference/_modules/langchain_text_splitters/character.html#RecursiveCharacterTextSplitter>
* inspired by <https://python.langchain.com/api_reference/_modules/langchain_text_splitters/character.html#CharacterTextSplitter>
* A column named `chunk_id` of type UInt32 is added to ensure uniqueness with lhs_pk and chunk_id

# Arguments

* `lhs` - `RecordBatch` with a column for 'score'
* `lhs_pk` - Left hand side primary key
* `lhs_value` - Left hand value key
* `chunk_size` - the length of the document chunks
* `chunk_overlap` - the length of overlap between document chunks
* `device` - The compute device

*/
pub fn chunk_documents(
    lhs: &[RecordBatch],
    lhs_pk: &str,
    lhs_values: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    _device: &Device,
) -> Result<RecordBatch> {
    // Extract out the document text
    let text = lhs
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(lhs_values)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        // Break the strings according to their size
        // DM: Implement a proper chunking function that is language specific...
        .map(|s| {
            let mut chunks = Vec::new();
            let mut doc = s.to_string();
            while doc.len() > chunk_size {
                let (s1, s2) = doc.split_at(chunk_size);
                chunks.push(s1.to_string());
                doc = [
                    s1[chunk_size - chunk_overlap..chunk_size].to_string(),
                    s2[..].to_string(),
                ]
                .join("");
            }
            chunks.push(doc);
            chunks
        })
        .collect::<Vec<_>>();

    // Wrap the output into a record batch
    let mut batch_vec = Vec::new();

    // Extract the rest of the columns according to type
    // Create new columns expanding when text vec size > 1
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
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .enumerate()
            .flat_map(|(index, s)| {
                let mut ar = Vec::new();
                (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                ar
            })
            .collect::<Vec<_>>();
        let sorted_array: ArrayRef = Arc::new(Float32Array::from(array_vec));
        batch_vec.push((column, sorted_array));
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
            .enumerate()
            .flat_map(|(index, s)| {
                let mut ar = Vec::new();
                (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                ar
            })
            .collect::<Vec<_>>();
        let sorted_array: ArrayRef = Arc::new(UInt32Array::from(array_vec));
        batch_vec.push((column, sorted_array));
    }
    let columns: Vec<String> = lhs
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != lhs_values)
                & (field.name() != "chunk_id")
                & (field.data_type() == &DataType::Utf8)
            {
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
            .enumerate()
            .flat_map(|(index, s)| {
                let mut ar = Vec::new();
                (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s.to_string()));
                ar
            })
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
        batch_vec.push((column, array_ref));
    }

    // Migrate the primary key to the chunk_id
    let array_vec = lhs
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
        .enumerate()
        .flat_map(|(index, s)| {
            let mut ar = Vec::new();
            (0..text.get(index).unwrap().len()).for_each(|i| {
                let s_new = format!("{s}_{i}");
                ar.push(s_new)
            });
            ar
        })
        .collect::<Vec<_>>();
    let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
    let chunk_id = "chunk_id".to_string();
    batch_vec.insert(0, (&chunk_id, array_ref));

    // flatten the text column
    let array_ref: ArrayRef = Arc::new(StringArray::from(
        text.into_iter().flatten().collect::<Vec<_>>(),
    ));
    let text_name = lhs_values.to_string();
    batch_vec.push((&text_name, array_ref));

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

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
            "embeddings",
            "rhs_pk",
            "embeddings",
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

    #[test]
    fn test_chunk_documents() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["01234597890123456789", "0123459789"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("text", lhs_text_array),
            ("metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["01234597890123456789", "0123459789"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("text", lhs_text_array),
            ("metadata", lhs_metadata_array),
        ])?;

        // Chunk the documents
        let result = chunk_documents(&[batch_1, batch_2], "lhs_pk", "text", 10, 2, &Device::Cpu)?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, vec!["0", "0", "0", "1", "2", "2", "2", "3"]);
        let metadata = result
            .column_by_name("metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, vec![1, 1, 1, 2, 3, 3, 3, 4]);
        let text = result
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            text,
            vec![
                "0123459789",
                "8901234567",
                "6789",
                "0123459789",
                "0123459789",
                "8901234567",
                "6789",
                "0123459789"
            ]
        );
        let chunk_id = result
            .column_by_name("chunk_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            chunk_id,
            vec!["0_0", "0_1", "0_2", "1_0", "2_0", "2_1", "2_2", "3_0"]
        );

        Ok(())
    }

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
