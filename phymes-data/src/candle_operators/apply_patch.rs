use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::{DataType, Field, Schema}};
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, Function, FunctionParameters, JSONSchemaDefine,
    JSONSchemaType, MappableTrait, Table, TableBuilderTrait, TableTrait, Tool,
    ToolType, create_bytes_record_batch, create_mermaid_content_template_batch,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::instrument;

use crate::{
    DataColumnOperator, DataComparatorOperator, DataComparatorPredicate, DataJoinOperator, PatchOperator, ToolTrait, apply_patch_auto, candle_data::DataConfig, candle_operators::{DataOperatorTrait, group_by::build_aggregator_column_list_nonprimitive, join::join, select::select}, filter
};

/// Inject a table into a string template
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ApplyPatch {
    lhs_values: String,
    rhs_values: Vec<String>,
    lhs_pk: String,
    rhs_pk: String,
}

impl MappableTrait for ApplyPatch {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for ApplyPatch {
    fn get_description(&self) -> String {
        "Inject a table into a string template.".to_string()
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
            "op_kwargs".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "template, table_expression, and input_template in the form of a JSON object"
                        .to_string(),
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
                required: Some(vec!["lhs_name".to_string(), "op_kwargs".to_string()]),
            },
        };
        let tool = Tool {
            r#type: ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

impl DataOperatorTrait for ApplyPatch {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        // Check for empty rhs_args and change to None
        let rhs_args = rhs_args.ok_or(anyhow!("Missing `rhs_args` for `ApplyPatch` Operator."))?;
        apply_patch(
            lhs_args,
            rhs_args,
            &self.lhs_values,
            &self.rhs_values.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &self.lhs_pk,
            &self.rhs_pk,
            device,
        )
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config
            .lhs_values
            .as_ref()
            .cloned()
            .ok_or(anyhow!(
                "Missing `lhs_values` for `{}`.",
                Self::get_static_name()
            ))?
            .first()
            .cloned()
            .ok_or(anyhow!(
                "`lhs_values` is empty for `{}`.",
                Self::get_static_name()
            ))?;
        let rhs_values = config
            .rhs_values
            .as_ref()
            .cloned()
            .ok_or(anyhow!(
                "Missing `rhs_values` for `{}`.",
                Self::get_static_name()
            ))?;
        let lhs_pk = config.lhs_pk.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;
        let rhs_pk = config.rhs_pk.clone().ok_or(anyhow!(
            "Missing `rhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;

        if rhs_values.len() != 2 {
            return Err(anyhow!("rhs_values is less than 2. The first element should be the name of the `diff` column and the second element should be the name of the `operator` column."))
        }

        // Make the object
        Ok(ApplyPatch {
            lhs_values,
            rhs_values,
            lhs_pk,
            rhs_pk
        })
    }
}

/// Apply patches to [RecordBatch]es
///
/// # Notes
/// 
/// - Each [RecordBatch] is treated as a seperate "workspace" where each row
///   is the equivalent of a "file", and a patch is applied per "file" analogous to `git`
/// - `Create` operations will generate a new row
/// - `Delete` operations will remove a row
/// - `Update` operations will update the row in place
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es that the patches will be applied to
/// * `rhs_args` - Slice of [RecordBatch]es that contain the patches
/// * `lhs_values` - The name of the column to apply the patches to (MUST be of type Utf8)
/// * `rhs_values` - Slice of [String] where the first element is the name of the column that contains the patch 
///   (MUST be of type Utf8 in either V4A Diff or Universal Diff formats), and the second element is the name of the column that
///   containers the operator
/// * `lhs_pk` - The name of the column with the full path of the "file" or another unique identifier to match the patch on
/// * `rhs_pk` - The name of the column with the full path of the "file" or another unique identifier to match the patch on
/// * `device` - The compute device
#[instrument(skip(lhs_args, rhs_args, lhs_values, rhs_values, lhs_pk, rhs_pk, device))]
pub fn apply_patch(
    lhs_args: &[RecordBatch],
    rhs_args: &[RecordBatch],
    lhs_values: &str,
    rhs_values: &[&str],
    lhs_pk: &str,
    rhs_pk: &str,
    device: &Device,
) -> Result<RecordBatch> {
    // Extract out the RHS values
    let diff_column = rhs_values.first().map_or("diff", |v| v);
    let operator_column = rhs_values.get(1).map_or("operator", |v| v);

    // Filter RHS for `Delete`
    let rhs_delete = select(&["delete"], rhs_args, &[], &[], &[], &[DataColumnOperator::Value], &[], &[DataType::Utf8], &[PatchOperator::Delete.to_string().as_str()], device)?;
    let rhs_delete = filter(&[operator_column], &[rhs_delete], &["delete"], &[DataComparatorOperator::Like], &DataComparatorPredicate::All, device)?;
    // let rhs_delete = select(
    //     &rhs_args.first().unwrap().schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(), 
    //     &[rhs_delete], &[], &[], &[], &[], &[], &[], &[], device)?;    

    // Apply `Delete`
    let rhs_table = Table::get_builder()
        .with_name("apply_patch rhs_args")
        .with_record_batches(vec![rhs_delete])?
        .build()?;
    let lhs_delete = match rhs_table.get_column_data_type(rhs_pk)? {
        // DM: and the other primitive types...
        DataType::Utf8 => {
            let values_vec = rhs_table.get_column_as_vec_nonprimitive::<String>(rhs_pk)?;
            let lhs_deleted: Result<Vec<RecordBatch>> = lhs_args.iter()
                .map(|batch| {
                    // New colum
                    let agg_vec = (0..batch.num_rows()).map(|_| values_vec.clone()).collect::<Vec<_>>();
                    let new_arr = build_aggregator_column_list_nonprimitive::<String>(agg_vec, DataType::Utf8);

                    // New schema               
                    let new_fields = batch.schema().fields().iter().cloned()
                        .chain([Arc::new(Field::new("delete", DataType::Utf8, false))])
                        .collect::<Vec<_>>();
                    let new_schema = Arc::new(Schema::new(new_fields));

                    // New batches
                    let mut new_columns = batch.columns().to_vec();
                    new_columns.push(new_arr);
                    let new_batch = RecordBatch::try_new(new_schema.clone(), new_columns)?;
                    Ok(new_batch)
                })
                .collect();
            lhs_deleted?
        }
        // DM: and the other nested types...
        _ => {
            return Err(anyhow!(
                "Unsupported data type {} for column {rhs_pk}",
                rhs_table.get_column_data_type(rhs_pk)?
            ));
        }
    };
    let lhs_delete = filter(&["delete"], &lhs_delete, &[lhs_pk], &[DataComparatorOperator::NotInListUtf8], &DataComparatorPredicate::All, device)?;
    let lhs_delete = select(
        &lhs_args.first().unwrap().schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(), 
        &[lhs_delete], &[], &[], &[], &[], &[], &[], &[], device)?;

    // Filter RHS for `Update`
    let rhs_update = select(&["update"], rhs_args, &[], &[], &[], &[DataColumnOperator::Value], &[], &[DataType::Utf8], &[PatchOperator::Update.to_string().as_str()], device)?;
    let rhs_update = filter(&[operator_column], &[rhs_update], &["update"], &[DataComparatorOperator::Like], &DataComparatorPredicate::All, device)?;
    let rhs_update = select(
        &rhs_args.first().unwrap().schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(), 
        &[rhs_update], &[], &[], &[], &[], &[], &[], &[], device)?;
    
    // Join LHS and filtered RHS on lhs_pk and rhs_pk
    let lhs_update = join(lhs_pk, &[lhs_delete], rhs_pk, &[rhs_update], &DataJoinOperator::LeftOuter, device)?;

    // Apply `Update`
    let lhs_update_table = Table::get_builder()
        .with_name("apply_patch lhs_update")
        .with_record_batches(vec![lhs_update])?
        .build()?;
    let original = lhs_update_table.get_column_as_vec_str(lhs_values);
    let patches = lhs_update_table.get_column_as_vec_str(diff_column);
    let modified: Result<Vec<String>> = original.into_iter()
        .zip(patches.into_iter())
        .map(|(o, p)| apply_patch_auto(o, p, false))
        .collect();
    let mut lhs_updated_batch_vec = Vec::new();

    // Apply `Create`
    let rhs_create = select(&["create"], rhs_args, &[], &[], &[], &[DataColumnOperator::Value], &[], &[DataType::Utf8], &[PatchOperator::Create.to_string().as_str()], device)?;
    let rhs_create = filter(&[operator_column], &[rhs_create], &["create"], &[DataComparatorOperator::Like], &DataComparatorPredicate::All, device)?;
    
    let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_core::test_table::make_test_table_chat;

    use crate::{device, template::test_minimal_html};

    use super::*;

    #[test]
    fn test_apply_patch_() -> Result<()> {

        Ok(())
    }
}
