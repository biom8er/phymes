use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use candle_core::{Device, Tensor};
use phymes_core::{
    BuildableTrait, BuilderTrait, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType,
    MappableTrait, Table, TableBuilderTrait, TableTrait, Tool, ToolType,
};
use tracing::instrument;

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Compute the normalized start and end times in a [RecordBatch]
#[derive(Debug, Default)]
pub struct NormalizeTime {
    lhs_values: Vec<String>,
}

impl MappableTrait for NormalizeTime {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for NormalizeTime {
    fn get_description(&self) -> String {
        "Compute the normalized start and end times and duration between start and end times."
            .to_string()
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
            "op_kwargs".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "DataCastOperator and DataType with optional column renaming and template injection in the form of a JSON object".to_string(),
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
                    "lhs_values".to_string(),
                    "op_kwargs".to_string(),
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

impl DataOperatorTrait for NormalizeTime {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        let lhs_values = self
            .lhs_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        normalize_time(&lhs_values, lhs_args, device)
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!("Missing `lhs_values` for `{}`.", Self::get_static_name()))?;
        Ok(NormalizeTime { lhs_values })
    }
}

/// Compute the normalized start and end times in a [RecordBatch]
///
/// # Notes
///
/// * The existence of a `start_timestamp` and `end_timestamp` columns of types UInt64
/// * New columns for `start_time-Norm`, `end_time-Norm`, and `duration` will be created
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings specifying the `start_timestamp` and `end_timestamp` columns
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
#[instrument(skip(lhs_args, device))]
pub fn normalize_time(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    device: &Device,
) -> Result<RecordBatch> {
    if lhs_values.len() != 2 {
        return Err(anyhow!(
            "Two lhs_values columns for `start_timestamp` and `end_timestamp` need to be provided. lhs_values {lhs_values:?} were provided."
        ));
    }
    // Wrap the lhs into an ArrowTable
    let lhs_table = Table::get_builder()
        .with_name("normalize_time")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;

    // Determine the minimum start time
    let start_time_vec = lhs_table
        .get_column_as_vec_primitive::<i64>(lhs_values.first().unwrap())?
        .into_iter()
        .collect::<Vec<_>>();
    let start_time_tensor = Tensor::from_iter(start_time_vec, device)?;
    let min_tensor = start_time_tensor
        .min_all()?
        .broadcast_as(start_time_tensor.shape())?;

    // Normalize the start and time
    let start_time_norm_tensor = start_time_tensor.sub(&min_tensor)?;
    let end_time_vec = lhs_table
        .get_column_as_vec_primitive::<i64>(lhs_values.get(1).unwrap())?
        .into_iter()
        .collect::<Vec<_>>();
    let end_time_tensor = Tensor::from_iter(end_time_vec, device)?;
    let end_time_norm_tensor = end_time_tensor.sub(&min_tensor)?;

    // Compute the duration
    let duration_tensor = end_time_tensor.sub(&start_time_tensor)?;

    // Convert tensors to Arrays
    let start_time_norm_vec = start_time_norm_tensor.to_vec1::<i64>()?;
    let start_time_norm_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(start_time_norm_vec));
    let end_time_norm_vec = end_time_norm_tensor.to_vec1::<i64>()?;
    let end_time_norm_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(end_time_norm_vec));
    let duration_vec = duration_tensor.to_vec1::<i64>()?;
    let duration_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(duration_vec));

    // add the start_time_norm and end_time_norm columns to the table
    let start_col_name = format!("{}-normalized", lhs_values.first().unwrap());
    let end_col_name = format!("{}-normalized", lhs_values.get(1).unwrap());
    let mut batch_vec = vec![
        (start_col_name.as_str(), start_time_norm_arr),
        (end_col_name.as_str(), end_time_norm_arr),
        ("duration", duration_arr),
    ];
    let schema = lhs_table.get_schema();
    for field in schema.fields().iter() {
        batch_vec.push((field.name(), lhs_table.get_column_as_array(field.name())));
    }
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Array, StringArray};
    use phymes_core::device;

    use super::*;

    #[test]
    fn test_normalize_time() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_start_vec_1: Vec<i64> = vec![5, 10];
        let lhs_start_array: ArrayRef = Arc::new(Int64Array::from(lhs_start_vec_1));
        let lhs_end_vec_1: Vec<i64> = vec![10, 20];
        let lhs_end_array: ArrayRef = Arc::new(Int64Array::from(lhs_end_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("start_timestamp", lhs_start_array),
            ("end_timestamp", lhs_end_array),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_start_vec_2: Vec<i64> = vec![20, 30];
        let lhs_start_array: ArrayRef = Arc::new(Int64Array::from(lhs_start_vec_2));
        let lhs_end_vec_1: Vec<i64> = vec![30, 100];
        let lhs_end_array: ArrayRef = Arc::new(Int64Array::from(lhs_end_vec_1));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("start_timestamp", lhs_start_array),
            ("end_timestamp", lhs_end_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Test
        let result = normalize_time(
            &["start_timestamp", "end_timestamp"],
            &[lhs_batch_1, lhs_batch_2],
            &device,
        )?;
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let start_time_norm =
            result_table.get_column_as_vec_primitive::<i64>("start_timestamp-normalized")?;
        assert_eq!(start_time_norm, [0, 5, 15, 25]);
        let end_time_norm =
            result_table.get_column_as_vec_primitive::<i64>("end_timestamp-normalized")?;
        assert_eq!(end_time_norm, [5, 15, 25, 95]);
        let end_time_norm = result_table.get_column_as_vec_primitive::<i64>("duration")?;
        assert_eq!(end_time_norm, [5, 10, 10, 70]);

        Ok(())
    }
}
