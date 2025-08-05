use arrow::{
    array::{Array, ArrayRef, UInt8Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, op::CmpOp};
use phymes_core::{
    session::common_traits::{BuildableTrait, BuilderTrait},
    table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
};
use phymes_core::schemas::{chat_completion, types};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::candle_operators::{
    data_operator::DataOperatorTrait,
    sort_column_and_indices::{sort_column_and_indices, take_columns_by_indices},
};

/// Inner join along the LHS foreign key and RHS PK of two [RecordBatch] ONLY the rows with matching values in common are returned
#[derive(Debug)]
pub struct JoinInner {
    _lhs_pk: String,
    lhs_fk: String,
    _rhs_pk: String,
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
            _lhs_pk: lhs_pk.to_string(),
            lhs_fk: lhs_fk.to_string(),
            _rhs_pk: rhs_pk.unwrap_or("rhs_pk").to_string(),
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

type JointInnerTensorResult = (Arc<dyn Array>, Tensor, Arc<dyn Array>, Tensor);

/// Helper method to compute the inner join using Tensors
fn join_inner_tensor(
    lhs_dim_0: usize,
    lhs_tensor: Tensor,
    rhs_dim_0: usize,
    rhs_tensor: Tensor,
    device: &Device,
) -> Result<JointInnerTensorResult> {
    let match_tensor = lhs_tensor.cmp(&rhs_tensor, CmpOp::Eq)?;

    // Convert the matches into indices
    let lhs_indices_tensor = (&Tensor::arange(1u8, (lhs_dim_0 + 1) as u8, device)?
        .reshape((lhs_dim_0, 1))?
        .broadcast_as((lhs_dim_0, rhs_dim_0))?
        * &match_tensor)?
        .flatten_all()?;
    let rhs_indices_tensor = (&Tensor::arange(1u8, (rhs_dim_0 + 1) as u8, device)?
        .reshape((1, rhs_dim_0))?
        .broadcast_as((lhs_dim_0, rhs_dim_0))?
        * &match_tensor)?
        .flatten_all()?;

    // Extract out the indices
    let lhs_indices = lhs_indices_tensor.to_vec1::<u8>()?;
    let lhs_indices = lhs_indices
        .into_iter()
        .filter_map(|v| if v >= 1 { Some(v - 1) } else { None })
        .collect::<Vec<_>>();
    let lhs_tensor = Tensor::from_iter(
        lhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
        device,
    )?;
    let lhs_arr: ArrayRef = Arc::new(UInt8Array::from(lhs_indices));
    let rhs_indices = rhs_indices_tensor.to_vec1::<u8>()?;
    let rhs_indices = rhs_indices
        .into_iter()
        .filter_map(|v| if v >= 1 { Some(v - 1) } else { None })
        .collect::<Vec<_>>();
    let rhs_tensor = Tensor::from_iter(
        rhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
        device,
    )?;
    let rhs_arr: ArrayRef = Arc::new(UInt8Array::from(rhs_indices));
    Ok((lhs_arr, lhs_tensor, rhs_arr, rhs_tensor))
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
#[instrument(skip(lhs_fk, lhs_args, rhs_fk, rhs_args, device))]
pub fn join_inner(
    lhs_fk: &str,
    lhs_args: &[RecordBatch],
    rhs_fk: &str,
    rhs_args: &[RecordBatch],
    device: &Device,
) -> Result<RecordBatch> {
    // Presort the lhs and rhs according to PKs
    let lhs_sorted = sort_column_and_indices(lhs_fk, lhs_args, true, device)?;
    let rhs_sorted = sort_column_and_indices(rhs_fk, rhs_args, true, device)?;

    // Wrap the lhs and rhs into an ArrowTable
    let lhs_table = ArrowTable::get_builder()
        .with_record_batches(vec![lhs_sorted])?
        .with_name("")
        .build()?;
    let rhs_table = ArrowTable::get_builder()
        .with_record_batches(vec![rhs_sorted])?
        .with_name("")
        .build()?;

    // Join by the FKs
    assert_eq!(
        lhs_table.get_column_data_type(lhs_fk)?,
        rhs_table.get_column_data_type(rhs_fk)?,
        "LHS FK and RHS FK columns must be the same type."
    );
    let (lhs_asort_arr, lhs_asort_tensor, rhs_asort_arr, rhs_asort_tensor) =
        match lhs_table.get_column_data_type(lhs_fk)? {
            DataType::UInt8 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<u8>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<u8>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::UInt32 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<u32>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<u32>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Int64 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<i64>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<i64>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Float32 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<f32>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<f32>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Float64 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_primitive::<f64>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_primitive::<f64>(rhs_fk)?;
                let lhs_dim_0 = lhs_fk_vec.len();
                let rhs_dim_0 = rhs_fk_vec.len();

                // Broadcast along dims 0 and 1 to find the matching FKs
                let lhs_tensor = Tensor::from_iter(lhs_fk_vec, device)?
                    .reshape((lhs_dim_0, 1))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                let rhs_tensor = Tensor::from_iter(rhs_fk_vec, device)?
                    .reshape((1, rhs_dim_0))?
                    .broadcast_as((lhs_dim_0, rhs_dim_0))?;
                join_inner_tensor(lhs_dim_0, lhs_tensor, rhs_dim_0, rhs_tensor, device)?
            }
            DataType::Utf8 => {
                let lhs_fk_vec = lhs_table.get_column_as_vec_nonprimitive::<String>(lhs_fk)?;
                let rhs_fk_vec = rhs_table.get_column_as_vec_nonprimitive::<String>(rhs_fk)?;
                let mut lhs_indices = Vec::new();
                let mut rhs_indices = Vec::new();

                // Find matches between foreign keys
                for (li, lfk) in lhs_fk_vec.iter().enumerate() {
                    for (ri, rfk) in rhs_fk_vec.iter().enumerate() {
                        if lfk == rfk {
                            lhs_indices.push(li as u8);
                            rhs_indices.push(ri as u8);
                        }
                    }
                }
                let lhs_tensor = Tensor::from_iter(
                    lhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
                    device,
                )?;
                let lhs_arr: ArrayRef = Arc::new(UInt8Array::from(lhs_indices));
                let rhs_tensor = Tensor::from_iter(
                    rhs_indices.iter().map(|v| v.to_owned()).collect::<Vec<_>>(),
                    device,
                )?;
                let rhs_arr: ArrayRef = Arc::new(UInt8Array::from(rhs_indices));
                (lhs_arr, lhs_tensor, rhs_arr, rhs_tensor)
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    lhs_fk,
                    lhs_table.get_column_data_type(lhs_fk)?.to_string()
                ));
            }
        };

    // Build the joined table
    let mut batch_vec = Vec::new();
    let lhs_columns: Vec<String> = lhs_table
        .get_schema()
        .fields()
        .iter()
        .map(|field| field.name().to_owned())
        .collect();
    batch_vec.extend(take_columns_by_indices(
        &lhs_columns,
        &lhs_table,
        lhs_asort_arr,
        lhs_asort_tensor,
        device,
    )?);
    let rhs_columns: Vec<String> = rhs_table
        .get_schema()
        .fields()
        .iter()
        .map(|field| field.name().to_owned())
        .collect();
    batch_vec.extend(take_columns_by_indices(
        &rhs_columns,
        &rhs_table,
        rhs_asort_arr,
        rhs_asort_tensor,
        device,
    )?);
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray, UInt8Array, UInt32Array};
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_join_inner() -> Result<()> {
        // ------ FK = String ------
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
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["right", "right", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("rhs_pk", rhs_ids_array),
            ("rhs_text", rhs_text_array),
            ("rhs_metadata", rhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Chunk the documents
        let result = join_inner(
            "lhs_pk",
            &[lhs_batch_1, lhs_batch_2],
            "rhs_pk",
            &[rhs_batch_1],
            &device,
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
        assert_eq!(metadata, vec![8, 9, 10]);
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

        // ------ FK = u8 ------
        // Make the test record batches
        let lhs_ids_vec_1: Vec<u8> = vec![0, 1];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2: Vec<u8> = vec![2, 3];
        let lhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let rhs_ids_vec_1: Vec<u8> = vec![0, 2, 2];
        let rhs_ids_array: ArrayRef = Arc::new(UInt8Array::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![8, 9, 10];
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
            &device,
        )?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, vec![0, 2, 2]);
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
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, vec![0, 2, 2]);
        let metadata = result
            .column_by_name("rhs_metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, vec![8, 9, 10]);
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
