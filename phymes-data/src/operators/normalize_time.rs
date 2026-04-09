use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch},
    datatypes::Schema,
};
use candle_core::{Device, Tensor};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{DataConfig, DataOperatorTrait, ToolTrait, operators::sort};

/// Compute the normalized start and end times in a [RecordBatch] and remove any gaps in time
#[derive(Debug, Default, Serialize, Deserialize)]
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
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(NormalizeTime { lhs_values })
    }
}

/// Compute the normalized time and duration
fn normalize_time_tensor(
    lhs_values: &[&str],
    lhs_table: &Subject,
    device: &Device,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    // Determine the minimum start time
    let start_time_vec = lhs_table
        .get_column_as_vec_primitive::<i64>(lhs_values.first().unwrap())?
        .into_iter()
        .collect::<Vec<_>>();
    let start_time_tensor = Tensor::from_iter(start_time_vec, device)?;
    let min_tensor = start_time_tensor
        .min_all()?
        .broadcast_as(start_time_tensor.shape())?;

    // Normalize the start and end time
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
    let end_time_norm_vec = end_time_norm_tensor.to_vec1::<i64>()?;
    let duration_vec = duration_tensor.to_vec1::<i64>()?;

    Ok((start_time_norm_vec, end_time_norm_vec, duration_vec))
}

/// Compute the normalized start and end times in a [RecordBatch] and remove any gaps in time
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

    // Pre-sort by start then end time
    let mut lhs_sorted = RecordBatch::new_empty(Arc::new(Schema::empty()));
    for (iter, column_name) in lhs_values.iter().enumerate() {
        if iter > 0 {
            lhs_sorted = sort(column_name, &[lhs_sorted], true, device)?;
        } else {
            lhs_sorted = sort(column_name, lhs_args, true, device)?;
        }
    }

    // Check for a single interval which is trivially non-overlapping
    let count = lhs_sorted.num_rows();
    if count <= 1 {
        return Ok(lhs_sorted);
    }

    // Wrap the lhs into an ArrowTable and extract out the start and end times
    let lhs_table = Subject::get_builder()
        .with_name("normalize_time")
        .with_record_batches(vec![lhs_sorted])?
        .build()?;

    // Normalize to a 0 start time
    let (start_time_norm_vec, end_time_norm_vec, duration_vec) =
        normalize_time_tensor(lhs_values, &lhs_table, device)?;

    // Find and remove gaps in time
    let mut gap_cum: i128 = 0;
    let mut last_end: i128 = 0;
    let mut start_time = Vec::new();
    let mut end_time = Vec::new();
    for (i, (s, e)) in start_time_norm_vec
        .into_iter()
        .zip(end_time_norm_vec.into_iter())
        .enumerate()
    {
        if i > 0 {
            let gap = s as i128 - last_end;
            if gap > 0 {
                // println!("gap: {gap}; gap_cum: {gap_cum}");
                gap_cum += gap;
            }
            start_time.push(s - gap_cum as i64);
            end_time.push(e - gap_cum as i64);
            if e as i128 > last_end {
                last_end = e as i128;
            }
        } else {
            start_time.push(s);
            end_time.push(e);
            last_end = e as i128;
        }
    }

    // Convert to Arrays
    let start_time_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(start_time));
    let end_time_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(end_time));
    let duration_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(duration_vec));

    // add the start_time_norm and end_time_norm columns to the table
    let start_col_name = format!("{}-normalized", lhs_values.first().unwrap());
    let end_col_name = format!("{}-normalized", lhs_values.get(1).unwrap());
    let mut batch_vec = vec![
        (start_col_name.as_str(), start_time_arr),
        (end_col_name.as_str(), end_time_arr),
        ("duration", duration_arr),
    ];
    let schema = lhs_table.get_schema();
    for field in schema.fields().iter() {
        batch_vec.push((field.name(), lhs_table.get_column_as_array(field.name())?));
    }
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::device;
    use arrow::array::{Int64Array, StringArray};

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
        let result_table = Subject::get_builder()
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

    #[test]
    fn test_trim_time_gaps() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_start_vec_1: Vec<i64> = vec![5, 8];
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
        let lhs_start_vec_2: Vec<i64> = vec![25, 50];
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
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let start_time_norm =
            result_table.get_column_as_vec_primitive::<i64>("start_timestamp-normalized")?;
        assert_eq!(start_time_norm, [0, 3, 15, 20]);
        let end_time_norm =
            result_table.get_column_as_vec_primitive::<i64>("end_timestamp-normalized")?;
        assert_eq!(end_time_norm, [5, 15, 20, 70]);

        Ok(())
    }
}
