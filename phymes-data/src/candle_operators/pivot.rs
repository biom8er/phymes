use arrow::{
    array::{Array, ArrayRef, UInt8Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, op::CmpOp};
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
        
        Pivot {
            lhs_values,
            agg_columns,
            agg_operators,
        }
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
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
        todo!()
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

    // Build the pivot table columns by take each agg_column based on the pvt_columns_group
    let mut batch_vec = Vec::new();
    for i in 0..pvt_columns_tab.count_rows() {
        let start = i*pvt_rows_tab.count_rows();
        let end = start + pvt_rows_tab.count_rows() + 1;
        // TODO: make asort arr/tensor
        let taken = take_columns_by_indices(&new_agg_columns, &pvt_values_table, asort_arr, asort_tensor, device)?;
        for (name, arr) in taken {
            // TODO: remake the name to include the pvt_columns
            batch_vec.push((name, arr))
        }
        
    }
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
        let lhs_c_vec_1 = vec!["large", "small", "small","large"];
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

        // Chunk the documents
        let result = pivot(
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

        Ok(())
    }
}
