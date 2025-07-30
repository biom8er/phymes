use arrow::{
    array::{ArrayRef, Float32Array, StringArray, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use phymes_ml::openai_asset::{chat_completion, types};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::candle_operators::data_operator::DataOperatorTrait;

/// Sort the [RecordBatch] according to the `score` column and then apply the sorting order to the rest of the record batch columns
#[derive(Debug)]
pub struct SortScoresAndIndices {
    lhs_values: String,
    asc: bool,
}

impl DataOperatorTrait for SortScoresAndIndices {
    fn get_static_name() -> &'static str {
        "sort-scores-and-indices"
    }
    fn new(
        _lhs_pk: &str,
        _lhs_fk: &str,
        lhs_value: &str,
        _rhs_pk: Option<&str>,
        _rhs_fk: Option<&str>,
        _rhs_value: Option<&str>,
        kwargs: Option<&str>
    ) -> Self {
        // Attempt to parse the op_kwargs
        let ops_kwargs_default = "{\"asc\": false}";
        let ops_kwargs_str = kwargs.unwrap_or(ops_kwargs_default);
        let ops_kwargs: serde_json::Value = serde_json::from_str(ops_kwargs_str)
            .unwrap_or(serde_json::from_str(ops_kwargs_default).unwrap());
        SortScoresAndIndices {
            lhs_values: lhs_value.to_string(),
            asc: ops_kwargs
                .get("asc")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        }
    }
    fn forward(&self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device
    ) -> Result<RecordBatch> {
        assert_eq!(self.lhs_values, "score", "The score column must be named 'score'");
        sort_scores_and_indices(lhs_args, self.asc, device)
    }
    fn get_schema_lhs_input(&self,
        _list_size: Option<usize>,
        other: Option<Vec<arrow::datatypes::Field>>,
    ) -> Option<arrow::datatypes::SchemaRef> {        
        assert_eq!(self.lhs_values, "score");
        let lhs_value = Field::new(self.lhs_values.clone(), DataType::Float32, false);
        let mut fields = vec![lhs_value];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))        
    }
    fn get_schema_rhs_input(&self,
        _list_size: Option<usize>,
        _other: Option<Vec<arrow::datatypes::Field>>,
    ) -> Option<arrow::datatypes::SchemaRef> {
        None
    }
    fn get_schema_output(&self,
        _list_size: Option<usize>,
        other: Option<Vec<arrow::datatypes::Field>>,
    ) -> Option<arrow::datatypes::SchemaRef> { 
        let score = Field::new("score", DataType::Float32, false);
        let mut fields = vec![score];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn check_schema_lhs_input(&self,
        other: arrow::datatypes::SchemaRef,
    ) -> Result<Option<bool>> {
        if other.column_with_name("score").is_none() {
            return Err(anyhow!("LHS input is missing column for score."));
        }
        Ok(Some(true))
    }
    fn check_schema_rhs_input(&self,
        _other: arrow::datatypes::SchemaRef,
    ) -> Result<Option<bool>> {
        Ok(None)
    }
    fn check_schema_output(&self,
        other: arrow::datatypes::SchemaRef,
    ) -> Result<Option<bool>> {        
        if other.column_with_name("score").is_none() {
            return Err(anyhow!("LHS output is missing column for score."));
        }
        Ok(Some(true))
    }
    fn get_description() -> String {
        "Sort the the list of computed scores in ascending order".to_string()
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
