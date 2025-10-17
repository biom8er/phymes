use arrow::array::{ArrayRef, RecordBatch, UInt32Array};

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use phymes_core::{
    schemas::{chat_completion, types},
    session::common_traits::MappableTrait,
};
use phymes_core::{
    session::common_traits::{BuildableTrait, BuilderTrait},
    table::table_trait::{Table, TableBuilderTrait, TableTrait},
};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{candle_data::data_config::{DataAggregatorOperator, DataConfig}, candle_operators::{
    data_operator::DataOperatorTrait, group_by_and_aggregate::{create_agg_column_name, group_by_and_aggregate}, sort_column_and_indices::take_columns_by_indices
}};

/// Inner join along the LHS foreign key and RHS PK of two [RecordBatch] ONLY the rows with matching values in common are returned
#[derive(Debug)]
pub struct Pivot {
    lhs_values: Vec<String>,
    agg_columns: Vec<String>,
    agg_operators: Vec<DataAggregatorOperator>,
    pvt_columns: Vec<String>,
}

impl MappableTrait for Pivot {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for Pivot {
    fn new(config: &DataConfig) -> Self {
        let lhs_values = config.lhs_values.to_owned();
        let agg_columns = config.agg_columns.clone().unwrap_or(Vec::new());
        let agg_operators = config.agg_operators.clone().unwrap_or(Vec::new());
        let pvt_columns = config.pvt_columns.clone().unwrap_or(Vec::new());
        
        Pivot {
            lhs_values,
            agg_columns,
            agg_operators,
            pvt_columns
        }
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
        let pvt_columns = self
            .pvt_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        pivot(&lhs_values, &lhs_args, &agg_columns, 
            &self.agg_operators,
            &pvt_columns, device)
    }
    fn get_description() -> String {
        "Pivot on selected columns".to_string()
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
            "lhs_fk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The foriegn key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "rhs_fk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The foriegn key column identifier for the right hand side table".to_string(),
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
                    "rhs_name".to_string(),
                    "lhs_fk".to_string(),
                    "rhs_fk".to_string(),
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

/// Group by specified columns and aggregate using a specified aggregation operator over specified columns
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the rows to group by
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `agg_columns` - Slice of Strings for the columns to aggregate and fill the pivot table columns
/// * `agg_operators` - Slice of [DataAggregatorOperator]s specifying the aggregator operator to apply to each lhs_value column
/// * `pvt_columns` - Slice of Strings for the columns to group by
/// * `fill_value` - Value to replace missing values with (in the resulting pivot table, after aggregation)
/// * `device` - The compute device
#[instrument(skip(lhs_values,lhs_args,agg_columns,agg_operators,pvt_columns,device))]
pub fn pivot(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    agg_columns: &[&str],
    agg_operators: &[DataAggregatorOperator],
    pvt_columns: &[&str],
    device: &Device,
) -> Result<RecordBatch> {
    // Group and aggregate by the lhs_values and pvt_columns
    // Note that the pvt_columns are last so that the gorup partition ranges can be used directly to extract out the columns for the pivot table
    let pvt_values: &[&str] = &lhs_values.iter().chain(pvt_columns).map(|&s| s).collect::<Vec<&str>>();
    let pvt_values_group = group_by_and_aggregate(pvt_values, lhs_args, agg_columns, agg_operators, device)?;
    let pvt_values_table = Table::get_builder()
        .with_record_batches(vec![pvt_values_group])?
        .with_name("")
        .build()?;

    // Make the values column names
    let new_agg_columns = agg_columns.iter().zip(agg_operators.iter())
        .map(|(agg_col, agg_op)| create_agg_column_name(agg_col, agg_op))
        .collect::<Vec<_>>();

    // Extract out just the values and pvt columns for grouping
    let mut pvt_columns_vec = Vec::new();
    let mut pvt_values_vec = Vec::new();
    for column_name in pvt_columns {
        let arr = pvt_values_table.get_column_as_array(column_name);
        pvt_columns_vec.push((column_name, arr));
    }
    for column_name in lhs_values {
        let arr = pvt_values_table.get_column_as_array(column_name);
        pvt_values_vec.push((column_name, arr));
    }

    // Group the columns and the rows
    let pvt_columns_batch = RecordBatch::try_from_iter(pvt_columns_vec)?;
    let pvt_values_batches = RecordBatch::try_from_iter(pvt_values_vec)?;
    let pvt_columns_group = group_by_and_aggregate(pvt_columns, &[pvt_columns_batch], &[], &[], device)?;
    let pvt_rows_group = group_by_and_aggregate(lhs_values, &[pvt_values_batches], &[], &[], device)?;

    // Check that there are no missing values
    if pvt_columns_group.num_rows() * pvt_rows_group.num_rows() != pvt_values_table.count_rows() {
        return Err(anyhow!("Cannot make the pivot table because there are missing values: pvt_columns {}, pvt_rows {}, and pvt_values {}.",
            pvt_columns_group.num_rows(), pvt_rows_group.num_rows(), pvt_values_table.count_rows()));
    }

    // Wrap the all grouped batches into tables
    let pvt_columns_tab = Table::get_builder()
        .with_record_batches(vec![pvt_columns_group])?
        .with_name("")
        .build()?;
    let pvt_rows_tab = Table::get_builder()
        .with_record_batches(vec![pvt_rows_group])?
        .with_name("")
        .build()?;

    // Build the pivot table columns
    let mut batch_vec = Vec::new();
    for (i, obj) in pvt_columns_tab.to_json_object()?.iter().enumerate() {
        let start: u32 = i as u32 * pvt_rows_tab.count_rows() as u32;
        let end: u32 = start + pvt_rows_tab.count_rows() as u32;
        let asort_arr: ArrayRef = Arc::new(UInt32Array::from_iter_values((start..end).collect::<Vec<u32>>()));
        let asort_tensor = Tensor::arange(start, end, device)?;

        // Take each lhs_values column (only once)
        if i == 0 {
            let taken = take_columns_by_indices(
                &lhs_values.iter().map(|s| s.to_string()).collect::<Vec<_>>(), 
                &pvt_values_table, &asort_arr, &asort_tensor, device)?;
            batch_vec.extend(taken);
        }

        // take each agg_column based on the pvt_columns_group
        let taken = take_columns_by_indices(&new_agg_columns, &pvt_values_table, &asort_arr, &asort_tensor, device)?;
        for (name, arr) in taken {
            let mut column_name_vec = pvt_columns.iter()
                .map(|&key| obj.get(key).unwrap().as_str().unwrap().to_string())
                .collect::<Vec<_>>();
            column_name_vec.push(name);
            let column_name = column_name_vec.join("-");
            batch_vec.push((column_name, arr));
        }        
    }
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray, UInt32Array};
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_pivot() -> Result<()> {
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