use arrow::{
    array::{ArrayRef, RecordBatch, UInt32Array},
    datatypes::{DataType, Field, Schema},
};

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{
    DataAggregatorOperator, DataConfig, DataOperatorTrait, ToolTrait,
    operators::{
        group_by::{create_agg_column_name, group_by},
        sort::take_columns_by_indices,
    },
};

/// Inner join along the LHS foreign key and RHS PK of two [RecordBatch] ONLY the rows with matching values in common are returned
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Pivot {
    lhs_values: Vec<String>,
    agg_columns: Vec<String>,
    agg_operators: Vec<DataAggregatorOperator>,
    default_values: Vec<String>,
    pvt_columns: Vec<String>,
}

impl MappableTrait for Pivot {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Pivot {
    fn get_description(&self) -> String {
        "Pivot on selected columns".to_string()
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
            "rhs_name".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("The name of the right hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_pk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_pk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the right hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_fk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The foriegn key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_fk".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "The foriegn key column identifier for the right hand side table".to_string(),
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
                    "rhs_name".to_string(),
                    "lhs_fk".to_string(),
                    "rhs_fk".to_string(),
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

impl DataOperatorTrait for Pivot {
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let agg_columns = config.agg_columns.clone().ok_or(anyhow!(
            "Missing `agg_columns` for `{}`.",
            Self::get_static_name()
        ))?;
        let agg_operators = config.agg_operators.clone().ok_or(anyhow!(
            "Missing `agg_operators` for `{}`.",
            Self::get_static_name()
        ))?;
        let default_values = config.default_values.clone().ok_or(anyhow!(
            "Missing `default_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let pvt_columns = config.pvt_columns.clone().ok_or(anyhow!(
            "Missing `pvt_columns` for `{}`.",
            Self::get_static_name()
        ))?;

        Ok(Pivot {
            lhs_values,
            agg_columns,
            agg_operators,
            default_values,
            pvt_columns,
        })
    }
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
        let agg_columns = self
            .agg_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let default_values = self
            .default_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let pvt_columns = self
            .pvt_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        pivot(
            &lhs_values,
            lhs_args,
            &agg_columns,
            &self.agg_operators,
            &default_values,
            &pvt_columns,
            device,
        )
    }
}

/// CPU only pivot operation to account for missing values
fn pivot_missing_values(
    pvt_columns: &[&str],
    default_values: &[&str],
    new_agg_columns: &[String],
    pvt_columns_table: &Subject,
    pvt_rows_table: &Subject,
    pvt_values_table: &Subject,
) -> Result<RecordBatch> {
    // Initialize the pivot table schema
    let mut pvt_fields = pvt_rows_table
        .get_schema()
        .fields
        .into_iter()
        .map(|f| Field::new(f.name(), f.data_type().to_owned(), false))
        .collect::<Vec<_>>();

    // Build the pivot table rows
    let mut rows_vec = Vec::new();
    let pvt_rows_json = pvt_rows_table.to_json_object()?;
    let pvt_columns_json = pvt_columns_table.to_json_object()?;
    let pvt_values_json = pvt_values_table.to_json_object()?;
    for row in pvt_rows_json.iter() {
        // Make the new row
        let mut map = row.clone();
        for column in pvt_columns_json.iter() {
            // Prepare the new column name
            let column_name_vec = pvt_columns
                .iter()
                .map(|&key| column.get(key).unwrap().as_str().unwrap().to_string())
                .collect::<Vec<_>>();
            let column_name = column_name_vec.join("-");

            let mut found = false;
            for item in pvt_values_json.iter() {
                // Check for row matches
                let mut rows_match = true;
                for (k, v) in row.iter() {
                    if item.get(k).unwrap() != v {
                        rows_match = false;
                        break;
                    }
                }
                // Check for column matches
                let mut columns_match = true;
                for (k, v) in column.iter() {
                    if item.get(k).unwrap() != v {
                        columns_match = false;
                        break;
                    }
                }
                if rows_match && columns_match {
                    // Add new columns to the row
                    for agg_column_name in new_agg_columns.iter() {
                        let new_column_name = format!("{column_name}-{agg_column_name}");
                        map.insert(
                            new_column_name.clone(),
                            item.get(agg_column_name).unwrap().to_owned(),
                        );
                        let field = Field::new(
                            new_column_name,
                            pvt_values_table
                                .get_schema()
                                .field_with_name(agg_column_name)
                                .unwrap()
                                .data_type()
                                .to_owned(),
                            false,
                        );
                        if !pvt_fields.contains(&field) {
                            pvt_fields.push(field);
                        }
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                // Add new columns to the row with default values
                for (agg_column_name, agg_default_value) in
                    new_agg_columns.iter().zip(default_values.iter())
                {
                    let new_column_name = format!("{column_name}-{agg_column_name}");
                    let data_type = pvt_values_table
                        .get_schema()
                        .field_with_name(agg_column_name)
                        .unwrap()
                        .data_type()
                        .to_owned();
                    match data_type {
                        DataType::UInt8 => {
                            let default_value = agg_default_value.parse::<u8>()?;
                            map.insert(
                                new_column_name.clone(),
                                serde_json::Value::from(default_value),
                            );
                        }
                        DataType::UInt32 => {
                            let default_value = agg_default_value.parse::<u32>()?;
                            map.insert(
                                new_column_name.clone(),
                                serde_json::Value::from(default_value),
                            );
                        }
                        DataType::Int64 => {
                            let default_value = agg_default_value.parse::<i64>()?;
                            map.insert(
                                new_column_name.clone(),
                                serde_json::Value::from(default_value),
                            );
                        }
                        DataType::Float32 => {
                            let default_value = agg_default_value.parse::<f32>()?;
                            map.insert(
                                new_column_name.clone(),
                                serde_json::Value::from(default_value),
                            );
                        }
                        DataType::Float64 => {
                            let default_value = agg_default_value.parse::<f64>()?;
                            map.insert(
                                new_column_name.clone(),
                                serde_json::Value::from(default_value),
                            );
                        }
                        DataType::Utf8 => {
                            map.insert(
                                new_column_name.clone(),
                                serde_json::Value::from(agg_default_value.to_string()),
                            );
                        }
                        _ => {
                            return Err(anyhow!(
                                "Unsupported data type {data_type} for default value {agg_default_value} and column {agg_column_name}."
                            ));
                        }
                    }
                    let field = Field::new(new_column_name, data_type, false);
                    if !pvt_fields.contains(&field) {
                        pvt_fields.push(field);
                    }
                }
            }
        }
        rows_vec.push(serde_json::Value::from(map));
    }

    // Build the pivot table
    let table = Subject::get_builder()
        .with_name("pivot_missing_values")
        .with_schema(Arc::new(Schema::new(pvt_fields)))
        .with_json_values(&rows_vec)?
        .build()?;
    Ok(table.get_record_batches_own().pop().unwrap())
}

/// Hardware accelerated version when there are no missing values
fn pivot_values(
    lhs_values: &[&str],
    pvt_columns: &[&str],
    new_agg_columns: &[String],
    pvt_columns_table: &Subject,
    pvt_rows_table: &Subject,
    pvt_values_table: &Subject,
    device: &Device,
) -> Result<RecordBatch> {
    // Build the pivot table columns
    let mut batch_vec = Vec::new();
    for (i, obj) in pvt_columns_table.to_json_object()?.iter().enumerate() {
        let start: u32 = i as u32 * pvt_rows_table.count_rows() as u32;
        let end: u32 = start + pvt_rows_table.count_rows() as u32;
        let asort_arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(
            (start..end).collect::<Vec<u32>>(),
        ));
        let asort_tensor = Tensor::arange(start, end, device)?;

        // Take each lhs_values column (only once)
        if i == 0 {
            let taken = take_columns_by_indices(
                &lhs_values.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                pvt_values_table,
                &asort_arr,
                &asort_tensor,
                device,
            )?;
            batch_vec.extend(taken);
        }

        // Prepare the new column name
        let column_name_vec = pvt_columns
            .iter()
            .map(|&key| obj.get(key).unwrap().as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        let column_name = column_name_vec.join("-");

        // take each agg_column based on the pvt_columns_group
        let taken = take_columns_by_indices(
            new_agg_columns,
            pvt_values_table,
            &asort_arr,
            &asort_tensor,
            device,
        )?;
        for (name, arr) in taken {
            batch_vec.push((format!("{column_name}-{name}"), arr));
        }
    }
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

/// Create a spreadsheet-style pivot table as a [RecordBatch].
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the rows to group by
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `agg_columns` - Slice of Strings for the columns to aggregate and fill the pivot table columns
/// * `agg_operators` - Slice of [DataAggregatorOperator]s specifying the aggregator operator to apply to each lhs_value column
/// * `default_values` - Slice of Strings representing the default value when missing values are encountered
/// * `pvt_columns` - Slice of Strings for the columns to group by
/// * `device` - The compute device
#[instrument(skip(
    lhs_values,
    lhs_args,
    agg_columns,
    agg_operators,
    default_values,
    pvt_columns,
    device
))]
pub fn pivot(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    agg_columns: &[&str],
    agg_operators: &[DataAggregatorOperator],
    default_values: &[&str],
    pvt_columns: &[&str],
    device: &Device,
) -> Result<RecordBatch> {
    // Group and aggregate by the lhs_values and pvt_columns
    // Note that the pvt_columns are last so that the group partition ranges can be used directly to extract out the columns for the pivot table
    let pvt_values: &[&str] = &lhs_values
        .iter()
        .chain(pvt_columns)
        .copied()
        .collect::<Vec<&str>>();
    let pvt_values_group = group_by(pvt_values, lhs_args, agg_columns, agg_operators, device)?;
    let pvt_values_table = Subject::get_builder()
        .with_name("pivot")
        .with_record_batches(vec![pvt_values_group])?
        .build()?;

    // Make the values column names
    let new_agg_columns = agg_columns
        .iter()
        .zip(agg_operators.iter())
        .map(|(agg_col, agg_op)| create_agg_column_name(agg_col, agg_op))
        .collect::<Vec<_>>();

    // Extract out just the values and pvt columns for grouping
    let mut pvt_columns_vec = Vec::new();
    let mut pvt_values_vec = Vec::new();
    for column_name in pvt_columns {
        let arr = pvt_values_table.get_column_as_array(column_name)?;
        pvt_columns_vec.push((column_name, arr));
    }
    for column_name in lhs_values {
        let arr = pvt_values_table.get_column_as_array(column_name)?;
        pvt_values_vec.push((column_name, arr));
    }

    // Group the columns and the rows
    let pvt_columns_batch = RecordBatch::try_from_iter(pvt_columns_vec)?;
    let pvt_values_batches = RecordBatch::try_from_iter(pvt_values_vec)?;
    let pvt_columns_group = group_by(pvt_columns, &[pvt_columns_batch], &[], &[], device)?;
    let pvt_rows_group = group_by(lhs_values, &[pvt_values_batches], &[], &[], device)?;

    // Wrap the all grouped batches into tables
    let pvt_columns_table = Subject::get_builder()
        .with_name("pvt_columns_table")
        .with_record_batches(vec![pvt_columns_group])?
        .build()?;
    let pvt_rows_table = Subject::get_builder()
        .with_name("pvt_rows_table")
        .with_record_batches(vec![pvt_rows_group])?
        .build()?;

    // Check that there are no missing values
    if pvt_columns_table.count_rows() * pvt_rows_table.count_rows() == pvt_values_table.count_rows()
    {
        pivot_values(
            lhs_values,
            pvt_columns,
            &new_agg_columns,
            &pvt_columns_table,
            &pvt_rows_table,
            &pvt_values_table,
            device,
        )
    } else {
        pivot_missing_values(
            pvt_columns,
            default_values,
            &new_agg_columns,
            &pvt_columns_table,
            &pvt_rows_table,
            &pvt_values_table,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::device;
    use arrow::array::{ArrayRef, StringArray, UInt32Array};

    use super::*;

    #[test]
    fn test_pivot_without_missing_values() -> Result<()> {
        // Make the test record batches
        let lhs_a_vec_1 = vec!["foo", "foo", "foo", "foo", "foo"];
        let lhs_a_array: ArrayRef = Arc::new(StringArray::from(lhs_a_vec_1));
        let lhs_b_vec_1 = vec!["one", "one", "one", "two", "two"];
        let lhs_b_array: ArrayRef = Arc::new(StringArray::from(lhs_b_vec_1));
        let lhs_c_vec_1 = vec!["small", "large", "large", "small", "small"];
        let lhs_c_array: ArrayRef = Arc::new(StringArray::from(lhs_c_vec_1));
        let lhs_d_vec_1: Vec<u32> = vec![1, 2, 2, 3, 3];
        let lhs_d_array: ArrayRef = Arc::new(UInt32Array::from(lhs_d_vec_1));
        let lhs_e_vec_1: Vec<u32> = vec![2, 4, 5, 5, 6];
        let lhs_e_array: ArrayRef = Arc::new(UInt32Array::from(lhs_e_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("a", lhs_a_array),
            ("b", lhs_b_array),
            ("c", lhs_c_array),
            ("d", lhs_d_array),
            ("e", lhs_e_array),
        ])?;
        let lhs_a_vec_1 = vec!["bar", "bar", "bar", "bar", "foo"];
        let lhs_a_array: ArrayRef = Arc::new(StringArray::from(lhs_a_vec_1));
        let lhs_b_vec_1 = vec!["one", "one", "two", "two", "two"];
        let lhs_b_array: ArrayRef = Arc::new(StringArray::from(lhs_b_vec_1));
        let lhs_c_vec_1 = vec!["large", "small", "small", "large", "large"];
        let lhs_c_array: ArrayRef = Arc::new(StringArray::from(lhs_c_vec_1));
        let lhs_d_vec_1: Vec<u32> = vec![4, 5, 6, 7, 0];
        let lhs_d_array: ArrayRef = Arc::new(UInt32Array::from(lhs_d_vec_1));
        let lhs_e_vec_1: Vec<u32> = vec![6, 8, 9, 9, 0];
        let lhs_e_array: ArrayRef = Arc::new(UInt32Array::from(lhs_e_vec_1));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("a", lhs_a_array),
            ("b", lhs_b_array),
            ("c", lhs_c_array),
            ("d", lhs_d_array),
            ("e", lhs_e_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Make the pivot table
        let result = pivot(
            &["a", "b"],
            &[lhs_batch_1, lhs_batch_2],
            &["d"],
            &[DataAggregatorOperator::Sum],
            &["0"],
            &["c"],
            &device,
        )?;

        let lhs_a = result
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_a, vec!["bar", "foo", "bar", "foo"]);
        let lhs_b = result
            .column_by_name("b")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_b, vec!["one", "one", "two", "two"]);
        let lhs_large_d = result
            .column_by_name("large-d-Sum")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs_large_d, vec![4, 4, 7, 0]);
        let lhs_small_d = result
            .column_by_name("small-d-Sum")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs_small_d, vec![5, 1, 6, 6]);

        Ok(())
    }

    #[test]
    fn test_pivot_with_missing_values() -> Result<()> {
        // Make the test record batches
        let lhs_a_vec_1 = vec!["foo", "foo", "foo", "foo", "foo"];
        let lhs_a_array: ArrayRef = Arc::new(StringArray::from(lhs_a_vec_1));
        let lhs_b_vec_1 = vec!["one", "one", "one", "two", "two"];
        let lhs_b_array: ArrayRef = Arc::new(StringArray::from(lhs_b_vec_1));
        let lhs_c_vec_1 = vec!["small", "large", "large", "small", "small"];
        let lhs_c_array: ArrayRef = Arc::new(StringArray::from(lhs_c_vec_1));
        let lhs_d_vec_1: Vec<u32> = vec![1, 2, 2, 3, 3];
        let lhs_d_array: ArrayRef = Arc::new(UInt32Array::from(lhs_d_vec_1));
        let lhs_e_vec_1: Vec<u32> = vec![2, 4, 5, 5, 6];
        let lhs_e_array: ArrayRef = Arc::new(UInt32Array::from(lhs_e_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("a", lhs_a_array),
            ("b", lhs_b_array),
            ("c", lhs_c_array),
            ("d", lhs_d_array),
            ("e", lhs_e_array),
        ])?;
        let lhs_a_vec_1 = vec!["bar", "bar", "bar", "bar"];
        let lhs_a_array: ArrayRef = Arc::new(StringArray::from(lhs_a_vec_1));
        let lhs_b_vec_1 = vec!["one", "one", "two", "two"];
        let lhs_b_array: ArrayRef = Arc::new(StringArray::from(lhs_b_vec_1));
        let lhs_c_vec_1 = vec!["large", "small", "small", "large"];
        let lhs_c_array: ArrayRef = Arc::new(StringArray::from(lhs_c_vec_1));
        let lhs_d_vec_1: Vec<u32> = vec![4, 5, 6, 7];
        let lhs_d_array: ArrayRef = Arc::new(UInt32Array::from(lhs_d_vec_1));
        let lhs_e_vec_1: Vec<u32> = vec![6, 8, 9, 9];
        let lhs_e_array: ArrayRef = Arc::new(UInt32Array::from(lhs_e_vec_1));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("a", lhs_a_array),
            ("b", lhs_b_array),
            ("c", lhs_c_array),
            ("d", lhs_d_array),
            ("e", lhs_e_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Make the pivot table
        let result = pivot(
            &["a", "b"],
            &[lhs_batch_1, lhs_batch_2],
            &["d"],
            &[DataAggregatorOperator::Sum],
            &["0"],
            &["c"],
            &device,
        )?;

        let lhs_a = result
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_a, vec!["bar", "foo", "bar", "foo"]);
        let lhs_b = result
            .column_by_name("b")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_b, vec!["one", "one", "two", "two"]);
        let lhs_large_d = result
            .column_by_name("large-d-Sum")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs_large_d, vec![4, 4, 7, 0]);
        let lhs_small_d = result
            .column_by_name("small-d-Sum")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs_small_d, vec![5, 1, 6, 6]);

        Ok(())
    }
}
