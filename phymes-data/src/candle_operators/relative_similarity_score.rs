use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int64Array, StringArray, UInt8Array, UInt32Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};
use phymes_core::{
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::table::{Table, TableBuilderTrait, TableTrait},
};

use crate::candle_operators::data_operator::make_error_record_batch;

use super::data_operator::DataOperatorTrait;
use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use phymes_core::schemas::{chat_completion, types};
use std::{collections::HashMap, sync::Arc};
use tracing::{event, instrument, Level};

/// Compute the relative similarity between two [RecordBatch]es where each [RecordBatch] represents a list of vector embeddings
#[derive(Debug)]
pub struct RelativeSimilarityScore {
    lhs_pk: String,
    _lhs_fk: String,
    lhs_values: String,
    _rhs_fk: String,
    rhs_pk: String,
    rhs_values: String,
}

impl MappableTrait for RelativeSimilarityScore {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for RelativeSimilarityScore {
    fn new(
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_values: &str,
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        rhs_values: Option<&str>,
        _kwargs: Option<&str>,
    ) -> Self {
        RelativeSimilarityScore {
            lhs_pk: lhs_pk.to_string(),
            _lhs_fk: lhs_fk.to_string(),
            lhs_values: lhs_values.to_string(),
            rhs_pk: rhs_pk.unwrap_or("rhs_pk").to_string(),
            _rhs_fk: rhs_fk.unwrap_or("rhs_fk").to_string(),
            rhs_values: rhs_values.unwrap_or("embedding").to_string(),
        }
    }
    fn get_description() -> String {
        "Compute the relative similarity score between two different lists of embedding vectors"
            .to_string()
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        match relative_similarity_score(
            &self.lhs_pk,
            &self.lhs_values,
            lhs_args,
            &self.rhs_pk,
            &self.rhs_values,
            rhs_args.unwrap_or(&[]),
            device,
        ) {
            Ok(batch) => Ok(batch),
            Err(err) => {
                event!(Level::ERROR, "{err}");
                Ok(make_error_record_batch(err.to_string().as_str()))
            },
        }
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_name".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The name of the right hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_pk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_pk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the right hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The values column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The values column identifier for the right hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_static_name().to_string(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
                    "rhs_name".to_string(),
                    "rhs_pk".to_string(),
                    "rhs_values".to_string(),
                ]),
            },
        };
        let tool = chat_completion::Tool {
            r#type: chat_completion::ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

/// Helper method to extract out the embeddings information from the LHS and RHS arguments
fn embeddings_to_tensor(
    values: &str,
    table: &Table,
    device: &Device,
) -> Result<(usize, usize, Tensor)> {
    match table.get_column_data_type(values)? {
        DataType::FixedSizeList(field, _) | DataType::List(field) => match field.data_type() {
            DataType::UInt8 => {
                let lhs_embeddings = table.get_column_as_vec_nested_primitive::<u8>(values)?;
                let lhs_dim_0 = lhs_embeddings.len();
                let lhs_dim_1 = lhs_embeddings.first().unwrap().len();
                let lhs_vec = lhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
                let lhs_tensor =
                    Tensor::from_iter(lhs_vec, device)?.reshape((lhs_dim_0, lhs_dim_1))?;
                Ok((lhs_dim_0, lhs_dim_1, lhs_tensor))
            }
            DataType::UInt32 => {
                let lhs_embeddings = table.get_column_as_vec_nested_primitive::<u32>(values)?;
                let lhs_dim_0 = lhs_embeddings.len();
                let lhs_dim_1 = lhs_embeddings.first().unwrap().len();
                let lhs_vec = lhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
                let lhs_tensor =
                    Tensor::from_iter(lhs_vec, device)?.reshape((lhs_dim_0, lhs_dim_1))?;
                Ok((lhs_dim_0, lhs_dim_1, lhs_tensor))
            }
            DataType::Int64 => {
                let lhs_embeddings = table.get_column_as_vec_nested_primitive::<i64>(values)?;
                let lhs_dim_0 = lhs_embeddings.len();
                let lhs_dim_1 = lhs_embeddings.first().unwrap().len();
                let lhs_vec = lhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
                let lhs_tensor =
                    Tensor::from_iter(lhs_vec, device)?.reshape((lhs_dim_0, lhs_dim_1))?;
                Ok((lhs_dim_0, lhs_dim_1, lhs_tensor))
            }
            // DataType::Float16 => {
            // }
            DataType::Float32 => {
                let lhs_embeddings = table.get_column_as_vec_nested_primitive::<f32>(values)?;
                let lhs_dim_0 = lhs_embeddings.len();
                let lhs_dim_1 = lhs_embeddings.first().unwrap().len();
                let lhs_vec = lhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
                let lhs_tensor =
                    Tensor::from_iter(lhs_vec, device)?.reshape((lhs_dim_0, lhs_dim_1))?;
                Ok((lhs_dim_0, lhs_dim_1, lhs_tensor))
            }
            DataType::Float64 => {
                let lhs_embeddings = table.get_column_as_vec_nested_primitive::<f64>(values)?;
                let lhs_dim_0 = lhs_embeddings.len();
                let lhs_dim_1 = lhs_embeddings.first().unwrap().len();
                let lhs_vec = lhs_embeddings.into_iter().flatten().collect::<Vec<_>>();
                let lhs_tensor =
                    Tensor::from_iter(lhs_vec, device)?.reshape((lhs_dim_0, lhs_dim_1))?;
                Ok((lhs_dim_0, lhs_dim_1, lhs_tensor))
            }
            _ => Err(anyhow!(
                "Unsupported data type for column {}: {}",
                values,
                field.data_type().to_string()
            )),
        },
        _ => Err(anyhow!(
            "Unsupported data type for column {}: {}",
            values,
            table.get_column_data_type(values)?.to_string()
        )),
    }
}

/// Helper method to calculate the relative similarity scores
fn tensor_to_scores(
    lhs_values: &str,
    lhs_table: &Table,
    lhs_tensor: Tensor,
    _rhs_values: &str,
    _rhs_table: &Table,
    rhs_tensor: Tensor,
) -> Result<ArrayRef> {
    let result = relative_similarity_scores_tensor(&lhs_tensor, &rhs_tensor)?.flatten_all()?;
    match lhs_table.get_column_data_type(lhs_values)? {
        DataType::FixedSizeList(field, _) | DataType::List(field) => match field.data_type() {
            DataType::UInt8 => {
                let result_vec = result.to_vec1::<u8>()?;
                Ok(Arc::new(UInt8Array::from(result_vec)))
            }
            DataType::UInt32 => {
                let result_vec = result.to_vec1::<u32>()?;
                Ok(Arc::new(UInt32Array::from(result_vec)))
            }
            DataType::Int64 => {
                let result_vec = result.to_vec1::<i64>()?;
                Ok(Arc::new(Int64Array::from(result_vec)))
            }
            // DataType::Float16 => {
            // }
            DataType::Float32 => {
                let result_vec = result.to_vec1::<f32>()?;
                Ok(Arc::new(Float32Array::from(result_vec)))
            }
            DataType::Float64 => {
                let result_vec = result.to_vec1::<f64>()?;
                Ok(Arc::new(Float64Array::from(result_vec)))
            }
            _ => Err(anyhow!(
                "Unsupported data type for column {}: {}",
                lhs_values,
                field.data_type().to_string()
            )),
        },
        _ => Err(anyhow!(
            "Unsupported data type for column {}: {}",
            lhs_values,
            lhs_table.get_column_data_type(lhs_values)?.to_string()
        )),
    }
}

/**
Compute the relative similarity between two [RecordBatch]es
  where each [RecordBatch] represents a list of vector embeddings

# Arguments

* `lhs_args` - Query 2D Tensor
* `lhs_args` - Document chunk 2D Tensor
* `device` - The compute device

*/
#[instrument(skip(lhs_pk, lhs_values, lhs_args, rhs_pk, rhs_values, rhs_args, device))]
fn relative_similarity_score(
    lhs_pk: &str,
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    rhs_pk: &str,
    rhs_values: &str,
    rhs_args: &[RecordBatch],
    device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs and rhs into an ArrowTable
    let lhs_table = Table::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?;
    let rhs_table = Table::get_builder()
        .with_record_batches(rhs_args.to_vec())?
        .with_name("")
        .build()?;

    // Compute the relative similarity score
    let (lhs_dim_0, _lhs_dim_1, lhs_tensor) = embeddings_to_tensor(lhs_values, &lhs_table, device)?;
    let (rhs_dim_0, _rhs_dim_1, rhs_tensor) = embeddings_to_tensor(rhs_values, &rhs_table, device)?;
    let out_scores = tensor_to_scores(
        lhs_values, &lhs_table, lhs_tensor, rhs_values, &rhs_table, rhs_tensor,
    )?;

    // Create the expanded LHS and RHS PKs
    let lhs_id: ArrayRef = match lhs_table.get_column_data_type(lhs_pk)? {
        // Broadcast along dim_0 by rhs_dim_0
        DataType::UInt8 => {
            let lhs_ids = lhs_table.get_column_as_vec_primitive::<u8>(lhs_pk)?;
            let lhs_tensor = Tensor::from_iter(lhs_ids, device)?
                .reshape((lhs_dim_0, 1))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let lhs_ids_vec = lhs_tensor.to_vec1::<u8>()?;
            Arc::new(UInt8Array::from(lhs_ids_vec))
        }
        DataType::UInt32 => {
            let lhs_ids = lhs_table.get_column_as_vec_primitive::<u32>(lhs_pk)?;
            let lhs_tensor = Tensor::from_iter(lhs_ids, device)?
                .reshape((lhs_dim_0, 1))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let lhs_ids_vec = lhs_tensor.to_vec1::<u32>()?;
            Arc::new(UInt32Array::from(lhs_ids_vec))
        }
        DataType::Int64 => {
            let lhs_ids = lhs_table.get_column_as_vec_primitive::<i64>(lhs_pk)?;
            let lhs_tensor = Tensor::from_iter(lhs_ids, device)?
                .reshape((lhs_dim_0, 1))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let lhs_ids_vec = lhs_tensor.to_vec1::<i64>()?;
            Arc::new(Int64Array::from(lhs_ids_vec))
        }
        DataType::Float32 => {
            let lhs_ids = lhs_table.get_column_as_vec_primitive::<f32>(lhs_pk)?;
            let lhs_tensor = Tensor::from_iter(lhs_ids, device)?
                .reshape((lhs_dim_0, 1))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let lhs_ids_vec = lhs_tensor.to_vec1::<f32>()?;
            Arc::new(Float32Array::from(lhs_ids_vec))
        }
        DataType::Float64 => {
            let lhs_ids = lhs_table.get_column_as_vec_primitive::<f64>(lhs_pk)?;
            let lhs_tensor = Tensor::from_iter(lhs_ids, device)?
                .reshape((lhs_dim_0, 1))?
                .broadcast_as((lhs_dim_0, lhs_dim_0))?
                .flatten_all()?;
            let lhs_ids_vec = lhs_tensor.to_vec1::<f64>()?;
            Arc::new(Float64Array::from(lhs_ids_vec))
        }
        DataType::Utf8 => {
            let lhs_ids = lhs_table.get_column_as_vec_nonprimitive::<String>(lhs_pk)?;
            let mut lhs_ids_vec = Vec::with_capacity(lhs_dim_0 * rhs_dim_0);
            for i in 0..lhs_dim_0 {
                for _j in 0..rhs_dim_0 {
                    lhs_ids_vec.push(lhs_ids.get(i).unwrap().to_owned());
                }
            }
            Arc::new(StringArray::from(lhs_ids_vec))
        }
        _ => {
            return Err(anyhow!(
                "Unsupported data type for column {}: {}",
                lhs_pk,
                lhs_table.get_column_data_type(lhs_pk)?.to_string()
            ));
        }
    };
    let rhs_id: ArrayRef = match rhs_table.get_column_data_type(rhs_pk)? {
        // Broadcast along dim_1 by lhs_dim_0
        DataType::UInt8 => {
            let rhs_ids = rhs_table.get_column_as_vec_primitive::<u8>(rhs_pk)?;
            let rhs_tensor = Tensor::from_iter(rhs_ids, device)?
                .reshape((1, rhs_dim_0))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let rhs_ids_vec = rhs_tensor.to_vec1::<u8>()?;
            Arc::new(UInt8Array::from(rhs_ids_vec))
        }
        DataType::UInt32 => {
            let rhs_ids = rhs_table.get_column_as_vec_primitive::<u32>(rhs_pk)?;
            let rhs_tensor = Tensor::from_iter(rhs_ids, device)?
                .reshape((1, rhs_dim_0))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let rhs_ids_vec = rhs_tensor.to_vec1::<u32>()?;
            Arc::new(UInt32Array::from(rhs_ids_vec))
        }
        DataType::Int64 => {
            let rhs_ids = rhs_table.get_column_as_vec_primitive::<i64>(rhs_pk)?;
            let rhs_tensor = Tensor::from_iter(rhs_ids, device)?
                .reshape((1, rhs_dim_0))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let rhs_ids_vec = rhs_tensor.to_vec1::<i64>()?;
            Arc::new(Int64Array::from(rhs_ids_vec))
        }
        DataType::Float32 => {
            let rhs_ids = rhs_table.get_column_as_vec_primitive::<f32>(rhs_pk)?;
            let rhs_tensor = Tensor::from_iter(rhs_ids, device)?
                .reshape((1, rhs_dim_0))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let rhs_ids_vec = rhs_tensor.to_vec1::<f32>()?;
            Arc::new(Float32Array::from(rhs_ids_vec))
        }
        DataType::Float64 => {
            let rhs_ids = rhs_table.get_column_as_vec_primitive::<f64>(rhs_pk)?;
            let rhs_tensor = Tensor::from_iter(rhs_ids, device)?
                .reshape((1, rhs_dim_0))?
                .broadcast_as((lhs_dim_0, rhs_dim_0))?
                .flatten_all()?;
            let rhs_ids_vec = rhs_tensor.to_vec1::<f64>()?;
            Arc::new(Float64Array::from(rhs_ids_vec))
        }
        DataType::Utf8 => {
            let rhs_ids = rhs_table.get_column_as_vec_nonprimitive::<String>(rhs_pk)?;
            let mut rhs_ids_vec = Vec::with_capacity(lhs_dim_0 * rhs_dim_0);
            for _i in 0..lhs_dim_0 {
                for j in 0..rhs_dim_0 {
                    rhs_ids_vec.push(rhs_ids.get(j).unwrap().to_owned());
                }
            }
            Arc::new(StringArray::from(rhs_ids_vec))
        }
        // DM: assuming that the PKs are not nested types...
        _ => {
            return Err(anyhow!(
                "Unsupported data type for column {}: {}",
                rhs_pk,
                rhs_table.get_column_data_type(rhs_pk)?.to_string()
            ));
        }
    };

    // Create the output batch
    let batch = RecordBatch::try_from_iter(vec![
        (lhs_pk, lhs_id),
        (rhs_pk, rhs_id),
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
    use phymes_core::session::common_traits::device;

    use crate::candle_data::data_processor::test_candle_ops_processor::{
        make_embeddings_record_batch_str_f32, make_embeddings_record_batch_u32_f32,
    };

    use super::*;

    #[test]
    fn test_relative_similarity_scores_tensor() -> Result<()> {
        let device = device(false)?;
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
        let lhs = Tensor::from_iter(lhs_vec.into_iter().flatten().collect::<Vec<_>>(), &device)?
            .reshape((3, 4))?;
        let rhs = Tensor::from_iter(rhs_vec.into_iter().flatten().collect::<Vec<_>>(), &device)?
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
        // ------ PK = String ------
        // LHS and RHS record batches
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs = make_embeddings_record_batch_str_f32("lhs_pk", lhs_ids_vec, lhs_embeddings_vec)?;
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs = make_embeddings_record_batch_str_f32("rhs_pk", rhs_ids_vec, rhs_embeddings_vec)?;

        // Make the device
        let device = device(false)?;

        // Compute the relative similarity scores
        let result = relative_similarity_score(
            "lhs_pk",
            "embedding",
            &[lhs],
            "rhs_pk",
            "embedding",
            &[rhs],
            &device,
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

        // ------ PK = u32 ------
        // LHS and RHS record batches
        let lhs_ids_vec: Vec<u32> = vec![1, 2, 3];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs = make_embeddings_record_batch_u32_f32("lhs_pk", lhs_ids_vec, lhs_embeddings_vec)?;
        let rhs_ids_vec: Vec<u32> = vec![1, 2, 3, 4];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs = make_embeddings_record_batch_u32_f32("rhs_pk", rhs_ids_vec, rhs_embeddings_vec)?;

        // Compute the relative similarity scores
        let result = relative_similarity_score(
            "lhs_pk",
            "embedding",
            &[lhs],
            "rhs_pk",
            "embedding",
            &[rhs],
            &device,
        )?;

        // Expected values
        let lhs_ids_test = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let rhs_ids_test = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .column_by_name("rhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
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
