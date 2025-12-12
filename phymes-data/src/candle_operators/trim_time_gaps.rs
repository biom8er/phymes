use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{array::{ArrayRef, Int64Array, RecordBatch}, datatypes::Schema};
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType,
    MappableTrait, Table, TableBuilderTrait, TableTrait, Tool, ToolType,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::{DataOperatorTrait, sort}};

/// Remove gaps in time in a [RecordBatch]
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TrimTimeGaps {
    lhs_values: Vec<String>,
}

impl MappableTrait for TrimTimeGaps {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for TrimTimeGaps {
    fn get_description(&self) -> String {
        "remove gaps in time."
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

impl DataOperatorTrait for TrimTimeGaps {
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
        trim_time_gaps(&lhs_values, lhs_args, device)
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(TrimTimeGaps { lhs_values })
    }
}

/// Remove gaps in time in a [RecordBatch]
///
/// # Notes
///
/// * The existence of a `start_timestamp` and `end_timestamp` columns of types UInt64
/// * No new columns will be created but the order of the columns will be changed
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings specifying the `start_timestamp` and `end_timestamp` columns
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
#[instrument(skip(lhs_args, device))]
pub fn trim_time_gaps(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    device: &Device,
) -> Result<RecordBatch> {
    if lhs_values.len() != 2 {
        return Err(anyhow!(
            "Two lhs_values columns for `start_timestamp` and `end_timestamp` need to be provided. lhs_values {lhs_values:?} were provided."
        ));
    }
    // Pre-sort by end and start time
    let mut lhs_sorted = RecordBatch::new_empty(Arc::new(Schema::empty()));
    for (iter, column_name) in lhs_values.iter().enumerate() {
        if iter > 0 {
            lhs_sorted = sort(column_name, &[lhs_sorted], true, device)?;
        } else {
            lhs_sorted = sort(column_name, lhs_args, true, device)?;
        }
    }

    // Wrap the lhs into an ArrowTable and extract out the start and end times
    let lhs_table = Table::get_builder()
        .with_name("trim_time_gaps")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let start_time_vec = lhs_table
        .get_column_as_vec_primitive::<i64>(lhs_values.first().unwrap())?
        .into_iter()
        .collect::<Vec<_>>();
    let end_time_vec = lhs_table
        .get_column_as_vec_primitive::<i64>(lhs_values.get(1).unwrap())?
        .into_iter()
        .collect::<Vec<_>>();

    // Find and remove the gaps in time
    let mut gap_cum: i64 = 0;
    let mut start_time = Vec::new();
    let mut end_time = Vec::new();
    for (i, (s, e)) in start_time_vec.iter().zip(end_time_vec.iter()).enumerate() {
        if i > 0 {
            let gap = s - end_time_vec.get(i-1).unwrap();
            if gap > 0 {
                gap_cum += gap;
                start_time.push(s-gap_cum);
                end_time.push(e-gap_cum);
            } else {
                start_time.push(*s);
                end_time.push(*e);
            }
        } else {
            start_time.push(*s);
            end_time.push(*e);
        }       
    }

    // Convert tensors to Arrays
    let start_time_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(start_time));
    let end_time_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(end_time));

    // Replace the start_time and end_time columns in the table
    let mut batch_vec = vec![
        (lhs_values.first().unwrap().to_string(), start_time_arr),
        (lhs_values.get(1).unwrap().to_string(), end_time_arr),
    ];
    let schema = lhs_table.get_schema();
    for field in schema.fields().iter() {
        if !lhs_values.contains(&field.name().as_str()) {
            batch_vec.push((field.name().to_string(), lhs_table.get_column_as_array(field.name())?));
        }        
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
        let result = trim_time_gaps(
            &["start_timestamp", "end_timestamp"],
            &[lhs_batch_1, lhs_batch_2],
            &device,
        )?;
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let start_time_norm =
            result_table.get_column_as_vec_primitive::<i64>("start_timestamp")?;
        assert_eq!(start_time_norm, [5, 8, 20, 25]);
        let end_time_norm =
            result_table.get_column_as_vec_primitive::<i64>("end_timestamp")?;
        assert_eq!(end_time_norm, [10, 20, 25, 75]);

        Ok(())
    }
}
