use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray}, datatypes::{DataType, Field, Schema, SchemaRef}, record_batch::RecordBatch
};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;
use super::data_operator::DataOperatorTrait;
use phymes_ai::openai_asset::{chat_completion, types};

/// Compute the relative similarity between two [RecordBatch]es where each [RecordBatch] represents a list of vector embeddings
#[derive(Debug)]
pub struct RelativeSimilarityScores {
    lhs_pk: String,
    lhs_fk: String,
    lhs_values: String,
    rhs_fk: String,
    rhs_pk: String,
    rhs_values: String,
}

impl DataOperatorTrait for RelativeSimilarityScores {
    fn get_name() -> String {
        "relative_similarity_scores".to_string()
    }
    fn new( 
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_values: &str,
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        rhs_values: Option<&str>,
        _kwargs: Option<&str>) -> Self {
        RelativeSimilarityScores {
            lhs_pk: lhs_pk.to_string(),
            lhs_fk: lhs_fk.to_string(),
            lhs_values: lhs_values.to_string(),
            rhs_pk: rhs_pk.unwrap_or("rhs_pk").to_string(),
            rhs_fk: rhs_fk.unwrap_or("rhs_fk").to_string(),
            rhs_values: rhs_values.unwrap_or("embedding").to_string(),
        }
    }
    fn get_description() -> String {
        "Compute the relative similarity scores between two record batches".to_string()
    }
    fn forward(&self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device
    ) -> Result<RecordBatch> {
        relative_similarity_scores(
            &self.lhs_pk,
            &self.lhs_values,
            lhs_args,
            &self.rhs_pk,
            &self.rhs_values,
            rhs_args.unwrap_or(&[]),
            device,
        )
    }
    fn get_schema_lhs_input(&self,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {        
        let lhs_pk = Field::new(self.lhs_pk.clone(), DataType::Utf8, false);
        let lhs_fk = Field::new(self.lhs_fk.clone(), DataType::Utf8, false);
        let embed_size = list_size.unwrap_or(2);
        let list_data_type = DataType::FixedSizeList(
            Arc::new(Field::new_list_field(DataType::Float32, false)),
            embed_size.try_into().unwrap(),
        );
        assert_eq!(self.lhs_values, "embedding");
        let lhs_value = Field::new(self.lhs_values.clone(), list_data_type, false);
        let mut fields = vec![lhs_pk, lhs_fk, lhs_value];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn get_schema_rhs_input(&self,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        let rhs_pk = Field::new(self.rhs_pk.clone(), DataType::Utf8, false);
        let rhs_fk = Field::new(self.rhs_fk.clone(), DataType::Utf8, false);
        let embed_size = list_size.unwrap_or(2);
        let list_data_type = DataType::FixedSizeList(
            Arc::new(Field::new_list_field(DataType::Float32, false)),
            embed_size.try_into().unwrap(),
        );
        assert_eq!(self.rhs_values, "embedding");
        let rhs_values = Field::new(self.rhs_values.clone(), list_data_type, false);
        let mut fields = vec![rhs_pk, rhs_fk, rhs_values];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn get_schema_output(&self,
        _list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {        
        let lhs_pk = Field::new(self.lhs_pk.clone(), DataType::Utf8, false);
        let rhs_pk = Field::new(self.rhs_pk.clone(), DataType::Utf8, false);
        let score = Field::new("score", DataType::Float32, false);
        let mut fields = vec![lhs_pk, rhs_pk, score];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn check_schema_lhs_input(&self,
        other: SchemaRef,
    ) -> Result<Option<bool>> {
        if other.column_with_name(&self.lhs_pk).is_none() {
            return Err(anyhow!(
                "LHS input is missing column for lhs_pk {}.",
                self.lhs_pk
            ));
        }
        if other.column_with_name("embedding").is_none() {
            return Err(anyhow!("LHS input is missing column for embedding."));
        }
        Ok(Some(true))}
    fn check_schema_rhs_input(&self,
        other: SchemaRef,
    ) -> Result<Option<bool>> {
        if other.column_with_name(&self.rhs_pk).is_none() {
            return Err(anyhow!(
                "RHS input is missing column for rhs_pk {}.",
                self.rhs_pk
            ));
        }
        if other.column_with_name("embedding").is_none() {
            return Err(anyhow!("RHS input is missing column for embedding."));
        }
        Ok(Some(true))
    }
    fn check_schema_output(&self,
        other: SchemaRef,
    ) -> Result<Option<bool>> {        
        if other.column_with_name(&self.lhs_pk).is_none() {
            return Err(anyhow!("LHS output is missing column for lhs_pk."));
        }
        if other.column_with_name(&self.rhs_pk).is_none() {
            return Err(anyhow!("RHS output is missing column for rhs_pk."));
        }
        if other.column_with_name("embedding").is_none() {
            return Err(anyhow!("Output is missing column for embedding."));
        }
        Ok(Some(true))
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
        // properties.insert(
        //     "rhs_name".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The name of the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        properties.insert(
            "lhs_pk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table"
                        .to_string(),
                ),
                ..Default::default()
            }),
        );
        // properties.insert(
        //     "rhs_pk".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The primary key column identifier for the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        // properties.insert(
        //     "lhs_fk".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The foriegn key column identifier for the left hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        // properties.insert(
        //     "rhs_fk".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The foriegn key column identifier for the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
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
        // properties.insert(
        //     "rhs_values".to_string(),
        //     Box::new(types::JSONSchemaDefine {
        //         schema_type: Some(types::JSONSchemaType::String),
        //         description: Some("The values column identifier for the right hand side table".to_string()),
        //         ..Default::default()
        //     }),
        // );
        let function = types::Function {
            name: Self::get_name(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
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

/**
Compute the relative similarity between two [RecordBatch]es
  where each [RecordBatch] represents a list of vector embeddings

# Arguments

* `lhs` - Query 2D Tensor
* `rhs` - Document chunk 2D Tensor
* `device` - The compute device

*/
#[instrument(skip(lhs_pk, lhs_values, lhs_args, rhs_pk, rhs_values, rhs_args, device))]
fn relative_similarity_scores(
    lhs_pk: &str,
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    rhs_pk: &str,
    rhs_values: &str,
    rhs_args: &[RecordBatch],
    device: &Device,
) -> Result<RecordBatch> {
    // Extract out the lhs_id and the embeddings
    let lhs_embeddings = lhs_args
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

    let lhs_id = lhs_args
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
    let rhs_embeddings = rhs_args
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

    let rhs_id = rhs_args
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
    use crate::candle_data::data_processor::test_candle_ops_processor::make_embeddings_record_batch;

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
            "lhs_pk",
            "embedding",
            &[lhs],
            "rhs_pk",
            "embedding",
            &[rhs],
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
