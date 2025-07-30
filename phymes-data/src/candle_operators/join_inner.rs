use arrow::{
    array::{ArrayRef, Float32Array, StringArray, UInt32Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use candle_core::Device;
use phymes_ml::openai_asset::{chat_completion, types};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::candle_operators::data_operator::DataOperatorTrait;

/// Inner join along the LHS foreign key and RHS PK of two [RecordBatch] ONLY the rows with matching values in common are returned
#[derive(Debug)]
pub struct JoinInner {
    lhs_pk: String,
    lhs_fk: String,
    rhs_pk: String,
    rhs_fk: String,
}

impl DataOperatorTrait for JoinInner {
    fn get_static_name() -> &'static str {
        "join-inner"
    }
    fn new(
        lhs_pk: &str,
        lhs_fk: &str,
        _lhs_value: &str,
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        _rhs_value: Option<&str>,
        _kwargs: Option<&str>,
    ) -> Self {
        JoinInner {
            lhs_pk: lhs_pk.to_string(),
            lhs_fk: lhs_fk.to_string(),
            rhs_pk: rhs_pk.unwrap_or("rhs_pk").to_string(),
            rhs_fk: rhs_fk.unwrap_or("rhs_fk").to_string(),
        }
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        join_inner(
            &self.lhs_fk,
            lhs_args,
            &self.rhs_fk,
            rhs_args.unwrap(),
            device,
        )
    }
    fn get_schema_lhs_input(
        &self,
        _list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        let lhs_pk = Field::new(self.lhs_pk.clone(), DataType::Utf8, false);
        let lhs_fk = Field::new(self.lhs_fk.clone(), DataType::Utf8, false);
        let mut fields = vec![lhs_pk, lhs_fk];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn get_schema_rhs_input(
        &self,
        _list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        let rhs_pk = Field::new(self.rhs_pk.clone(), DataType::Utf8, false);
        let rhs_fk = Field::new(self.rhs_fk.clone(), DataType::Utf8, false);
        let mut fields = vec![rhs_pk, rhs_fk];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn get_schema_output(
        &self,
        _list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        let lhs_fk = Field::new(self.lhs_fk.clone(), DataType::Utf8, false);
        let rhs_fk = Field::new(self.rhs_fk.clone(), DataType::Utf8, false);
        let mut fields = vec![lhs_fk, rhs_fk];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn check_schema_lhs_input(&self, other: SchemaRef) -> Result<Option<bool>> {
        if other.column_with_name(&self.lhs_fk).is_none() {
            return Err(anyhow!(
                "LHS input is missing column for lhs_fk {}.",
                self.lhs_fk
            ));
        }
        Ok(Some(true))
    }
    fn check_schema_rhs_input(&self, other: SchemaRef) -> Result<Option<bool>> {
        if other.column_with_name(&self.rhs_fk).is_none() {
            return Err(anyhow!(
                "RHS input is missing column for rhs_fk {}.",
                self.rhs_fk
            ));
        }
        Ok(Some(true))
    }
    fn check_schema_output(&self, other: SchemaRef) -> Result<Option<bool>> {
        if other.column_with_name(&self.lhs_fk).is_none() {
            return Err(anyhow!("LHS output is missing column for lhs_fk."));
        }
        if other.column_with_name(&self.rhs_fk).is_none() {
            return Err(anyhow!("RHS output is missing column for rhs_fk."));
        }
        Ok(Some(true))
    }
    fn get_description() -> String {
        "Join two tables on their foreign keys".to_string()
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
                    "The primary key column identifier for the left hand side table".to_string(),
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
Inner join along the LHS foreign key and RHS PK of two [RecordBatch]
  ONLY the rows with matching values in common are returned

# Arguments

* `lhs` - RecordBatch
* `lhs_fk` - Left hand side foreign key
* `rhs` - RecordBatch
* `rhs_fk` - Right hand side foreign key
* `device` - The compute device

*/
#[instrument(skip(lhs_fk, lhs_args, rhs_fk, rhs_args, _device))]
pub fn join_inner(
    lhs_fk: &str,
    lhs_args: &[RecordBatch],
    rhs_fk: &str,
    rhs_args: &[RecordBatch],
    _device: &Device,
) -> Result<RecordBatch> {
    // Extract the foreign keys
    let lhs_fk_vec = lhs_args
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
    let rhs_fk_vec = rhs_args
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
    let columns: Vec<String> = lhs_args
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
        let array_vec = lhs_args
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
    let columns: Vec<String> = lhs_args
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
        let array_vec = lhs_args
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
    let columns: Vec<String> = lhs_args
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
        let array_vec = lhs_args
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
    let columns: Vec<String> = rhs_args
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
        let array_vec = rhs_args
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
    let columns: Vec<String> = rhs_args
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
        let array_vec = rhs_args
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
    let columns: Vec<String> = rhs_args
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
        let array_vec = rhs_args
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
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
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
