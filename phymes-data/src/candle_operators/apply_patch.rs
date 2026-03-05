use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array},
    datatypes::{DataType, Field, Int64Type, Schema, UInt32Type},
};
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType,
    MappableTrait, Table, TableBuilderTrait, TableTrait, Tool, ToolType,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    DataCastOperator, DataColumnOperator, DataComparatorOperator, DataComparatorPredicate,
    DataJoinOperator, PatchOperator, ToolTrait, apply_patch_auto,
    candle_data::DataConfig,
    candle_operators::{
        DataOperatorTrait,
        group_by::{
            build_aggregator_column_list_nonprimitive, build_aggregator_column_list_primitive,
        },
        join::join,
        select::select,
    },
    filter,
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
            &self
                .rhs_values
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
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
        let rhs_values = config.rhs_values.as_ref().cloned().ok_or(anyhow!(
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
            return Err(anyhow!(
                "rhs_values is less than 2. The first element should be the name of the `diff` column and the second element should be the name of the `operator` column."
            ));
        }

        // Make the object
        Ok(ApplyPatch {
            lhs_values,
            rhs_values,
            lhs_pk,
            rhs_pk,
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
    // Extract out LHS and RHS values that will be re-used
    let diff_column = rhs_values.first().map_or("diff", |v| v);
    let operator_column = rhs_values.get(1).map_or("operator", |v| v);
    let rhs_columns = rhs_args
        .first()
        .ok_or(anyhow!("Missing rhs_args in apply_patch."))?
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect::<Vec<_>>();
    let lhs_columns = lhs_args
        .first()
        .ok_or(anyhow!("Missing lhs_args in apply_patch."))?
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect::<Vec<_>>();

    // Filter RHS for `Delete`
    let rhs_delete = {
        let select_lhs_cols = rhs_columns
            .iter()
            .map(|f| f.as_str())
            .chain(["delete"])
            .collect::<Vec<_>>();
        let select_rhs_cols = select_lhs_cols.iter().map(|_| "").collect::<Vec<_>>();
        let select_column_operators = rhs_columns
            .iter()
            .map(|_| DataColumnOperator::None)
            .chain([DataColumnOperator::Value])
            .collect::<Vec<_>>();
        let select_cast_operators = rhs_columns
            .iter()
            .map(|_| DataCastOperator::None)
            .chain([DataCastOperator::None])
            .collect::<Vec<_>>();
        let select_cast_datatypes = rhs_columns
            .iter()
            .map(|_| DataType::Utf8)
            .chain([DataType::Utf8])
            .collect::<Vec<_>>();
        let patch_operator = PatchOperator::Delete.to_string();
        let select_cast_templates = rhs_columns
            .iter()
            .map(|_| "")
            .chain([patch_operator.as_str()])
            .collect::<Vec<_>>();
        let rhs_delete = select(
            &select_lhs_cols,
            rhs_args,
            &select_rhs_cols,
            &select_lhs_cols,
            &select_lhs_cols,
            &select_column_operators,
            &select_cast_operators,
            &select_cast_datatypes,
            &select_cast_templates,
            device,
        )?;
        filter(
            &[operator_column],
            &[rhs_delete],
            &["delete"],
            &[DataComparatorOperator::Like],
            &DataComparatorPredicate::All,
            device,
        )?
    };

    // Apply `Delete`
    let rhs_table = Table::get_builder()
        .with_name("apply_patch rhs_args")
        .with_record_batches(vec![rhs_delete])?
        .build()?;
    let lhs_deleted = match rhs_table.get_column_data_type(rhs_pk)? {
        DataType::UInt32 => {
            let values_vec = rhs_table.get_column_as_vec_primitive::<u32>(rhs_pk)?;
            let lhs_deleted: Result<Vec<RecordBatch>> = lhs_args
                .iter()
                .map(|batch| {
                    // New colum
                    let agg_vec = (0..batch.num_rows())
                        .map(|_| values_vec.clone())
                        .collect::<Vec<_>>();
                    let new_arr = build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        agg_vec,
                        DataType::UInt32,
                    );

                    // New schema
                    let list_data_type =
                        DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)));
                    let new_fields = batch
                        .schema()
                        .fields()
                        .iter()
                        .cloned()
                        .chain([Arc::new(Field::new("delete", list_data_type, false))])
                        .collect::<Vec<_>>();
                    let new_schema = Arc::new(Schema::new(new_fields));

                    // New batches
                    let mut new_columns = batch.columns().to_vec();
                    new_columns.push(new_arr);
                    let new_batch = RecordBatch::try_new(new_schema.clone(), new_columns)?;
                    Ok(new_batch)
                })
                .collect();
            filter(
                &["delete"],
                &lhs_deleted?,
                &[lhs_pk],
                &[DataComparatorOperator::NotInList],
                &DataComparatorPredicate::All,
                device,
            )?
        }
        DataType::Int64 => {
            let values_vec = rhs_table.get_column_as_vec_primitive::<i64>(rhs_pk)?;
            let lhs_deleted: Result<Vec<RecordBatch>> = lhs_args
                .iter()
                .map(|batch| {
                    // New colum
                    let agg_vec = (0..batch.num_rows())
                        .map(|_| values_vec.clone())
                        .collect::<Vec<_>>();
                    let new_arr = build_aggregator_column_list_primitive::<i64, Int64Type>(
                        agg_vec,
                        DataType::Int64,
                    );

                    // New schema
                    let list_data_type =
                        DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false)));
                    let new_fields = batch
                        .schema()
                        .fields()
                        .iter()
                        .cloned()
                        .chain([Arc::new(Field::new("delete", list_data_type, false))])
                        .collect::<Vec<_>>();
                    let new_schema = Arc::new(Schema::new(new_fields));

                    // New batches
                    let mut new_columns = batch.columns().to_vec();
                    new_columns.push(new_arr);
                    let new_batch = RecordBatch::try_new(new_schema.clone(), new_columns)?;
                    Ok(new_batch)
                })
                .collect();
            filter(
                &["delete"],
                &lhs_deleted?,
                &[lhs_pk],
                &[DataComparatorOperator::NotInList],
                &DataComparatorPredicate::All,
                device,
            )?
        }
        DataType::Utf8 => {
            let values_vec = rhs_table.get_column_as_vec_nonprimitive::<String>(rhs_pk)?;
            let lhs_deleted: Result<Vec<RecordBatch>> = lhs_args
                .iter()
                .map(|batch| {
                    // New colum
                    let agg_vec = (0..batch.num_rows())
                        .map(|_| values_vec.clone())
                        .collect::<Vec<_>>();
                    let new_arr = build_aggregator_column_list_nonprimitive::<String>(
                        agg_vec,
                        DataType::Utf8,
                    );

                    // New schema
                    let list_data_type =
                        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
                    let new_fields = batch
                        .schema()
                        .fields()
                        .iter()
                        .cloned()
                        .chain([Arc::new(Field::new("delete", list_data_type, false))])
                        .collect::<Vec<_>>();
                    let new_schema = Arc::new(Schema::new(new_fields));

                    // New batches
                    let mut new_columns = batch.columns().to_vec();
                    new_columns.push(new_arr);
                    let new_batch = RecordBatch::try_new(new_schema.clone(), new_columns)?;
                    Ok(new_batch)
                })
                .collect();
            filter(
                &["delete"],
                &lhs_deleted?,
                &[lhs_pk],
                &[DataComparatorOperator::NotInListUtf8],
                &DataComparatorPredicate::All,
                device,
            )?
        }
        // DM: and the other nested types...
        _ => {
            return Err(anyhow!(
                "Unsupported data type {} for column {rhs_pk}",
                rhs_table.get_column_data_type(rhs_pk)?
            ));
        }
    };
    let lhs_delete = {
        let select_lhs_cols = lhs_columns.iter().map(|f| f.as_str()).collect::<Vec<_>>();
        let select_rhs_cols = select_lhs_cols.iter().map(|_| "").collect::<Vec<_>>();
        let select_column_operators = lhs_columns
            .iter()
            .map(|_| DataColumnOperator::None)
            .collect::<Vec<_>>();
        let select_cast_operators = lhs_columns
            .iter()
            .map(|_| DataCastOperator::None)
            .collect::<Vec<_>>();
        let select_cast_datatypes = lhs_columns
            .iter()
            .map(|_| DataType::Utf8)
            .collect::<Vec<_>>();
        let select_cast_templates = lhs_columns.iter().map(|_| "").collect::<Vec<_>>();
        select(
            &select_lhs_cols,
            &[lhs_deleted],
            &select_rhs_cols,
            &select_lhs_cols,
            &select_lhs_cols,
            &select_column_operators,
            &select_cast_operators,
            &select_cast_datatypes,
            &select_cast_templates,
            device,
        )?
    };

    // Filter RHS for `Update`
    let rhs_update = {
        let select_lhs_cols = rhs_columns
            .iter()
            .map(|f| f.as_str())
            .chain(["update"])
            .collect::<Vec<_>>();
        let select_rhs_cols = select_lhs_cols.iter().map(|_| "").collect::<Vec<_>>();
        let select_column_operators = rhs_columns
            .iter()
            .map(|_| DataColumnOperator::None)
            .chain([DataColumnOperator::Value])
            .collect::<Vec<_>>();
        let select_cast_operators = rhs_columns
            .iter()
            .map(|_| DataCastOperator::None)
            .chain([DataCastOperator::None])
            .collect::<Vec<_>>();
        let select_cast_datatypes = rhs_columns
            .iter()
            .map(|_| DataType::Utf8)
            .chain([DataType::Utf8])
            .collect::<Vec<_>>();
        let patch_operator = PatchOperator::Update.to_string();
        let select_cast_templates = rhs_columns
            .iter()
            .map(|_| "")
            .chain([patch_operator.as_str()])
            .collect::<Vec<_>>();
        let rhs_update = select(
            &select_lhs_cols,
            rhs_args,
            &select_rhs_cols,
            &select_lhs_cols,
            &select_lhs_cols,
            &select_column_operators,
            &select_cast_operators,
            &select_cast_datatypes,
            &select_cast_templates,
            device,
        )?;
        let rhs_update = filter(
            &[operator_column],
            &[rhs_update],
            &["update"],
            &[DataComparatorOperator::Like],
            &DataComparatorPredicate::All,
            device,
        )?;
        let select_lhs_cols = rhs_columns.iter().map(|f| f.as_str()).collect::<Vec<_>>();
        let select_rhs_cols = select_lhs_cols.iter().map(|_| "").collect::<Vec<_>>();
        let select_column_operators = rhs_columns
            .iter()
            .map(|_| DataColumnOperator::None)
            .collect::<Vec<_>>();
        let select_cast_operators = rhs_columns
            .iter()
            .map(|_| DataCastOperator::None)
            .collect::<Vec<_>>();
        let select_cast_datatypes = rhs_columns
            .iter()
            .map(|_| DataType::Utf8)
            .collect::<Vec<_>>();
        let select_cast_templates = rhs_columns.iter().map(|_| "").collect::<Vec<_>>();
        select(
            &select_lhs_cols,
            &[rhs_update],
            &select_rhs_cols,
            &select_lhs_cols,
            &select_lhs_cols,
            &select_column_operators,
            &select_cast_operators,
            &select_cast_datatypes,
            &select_cast_templates,
            device,
        )?
    };

    // Join LHS and filtered RHS on lhs_pk and rhs_pk
    let lhs_update = if rhs_update.num_rows() > 0 {
        let lhs_update = join(
            lhs_pk,
            &[lhs_delete],
            rhs_pk,
            &[rhs_update],
            &DataJoinOperator::LeftOuter,
            device,
        )?;

        // Apply `Update`
        let lhs_update_table = Table::get_builder()
            .with_name("apply_patch lhs_update")
            .with_record_batches(vec![lhs_update])?
            .build()?;
        let original = lhs_update_table.get_column_as_vec_str(lhs_values);
        let patches = lhs_update_table.get_column_as_vec_str(diff_column);
        let modified: Result<Vec<String>> = original
            .into_iter()
            .zip(patches.into_iter())
            .map(|(o, p)| {
                if p.is_empty() {
                    Ok(o.to_string())
                } else {
                    apply_patch_auto(o, p, false)
                }
            })
            .collect();
        let modified_arr: ArrayRef = Arc::new(StringArray::from(modified?));
        let mut lhs_updated_batch_vec = Vec::new();
        for field in lhs_args.first().unwrap().schema().fields() {
            if field.name() == lhs_values {
                lhs_updated_batch_vec.push((field.name().to_string(), modified_arr.clone()))
            } else {
                lhs_updated_batch_vec.push((
                    field.name().to_string(),
                    lhs_update_table.get_column_as_array(field.name())?,
                ))
            }
        }
        RecordBatch::try_from_iter(lhs_updated_batch_vec)?
    } else {
        lhs_delete
    };

    // Filter RHS for `Create`
    let rhs_create = {
        let select_lhs_cols = rhs_columns
            .iter()
            .map(|f| f.as_str())
            .chain(["create"])
            .collect::<Vec<_>>();
        let select_rhs_cols = select_lhs_cols.iter().map(|_| "").collect::<Vec<_>>();
        let select_column_operators = rhs_columns
            .iter()
            .map(|_| DataColumnOperator::None)
            .chain([DataColumnOperator::Value])
            .collect::<Vec<_>>();
        let select_cast_operators = rhs_columns
            .iter()
            .map(|_| DataCastOperator::None)
            .chain([DataCastOperator::None])
            .collect::<Vec<_>>();
        let select_cast_datatypes = rhs_columns
            .iter()
            .map(|_| DataType::Utf8)
            .chain([DataType::Utf8])
            .collect::<Vec<_>>();
        let patch_operator = PatchOperator::Create.to_string();
        let select_cast_templates = rhs_columns
            .iter()
            .map(|_| "")
            .chain([patch_operator.as_str()])
            .collect::<Vec<_>>();
        let rhs_create = select(
            &select_lhs_cols,
            rhs_args,
            &select_rhs_cols,
            &select_lhs_cols,
            &select_lhs_cols,
            &select_column_operators,
            &select_cast_operators,
            &select_cast_datatypes,
            &select_cast_templates,
            device,
        )?;
        filter(
            &[operator_column],
            &[rhs_create],
            &["create"],
            &[DataComparatorOperator::Like],
            &DataComparatorPredicate::All,
            device,
        )?
    };

    // Join LHS and filtered RHS on lhs_pk and rhs_pk
    let lhs_create = if rhs_create.num_rows() > 0 {
        let lhs_create = join(
            lhs_pk,
            &[lhs_update],
            rhs_pk,
            &[rhs_create],
            &DataJoinOperator::FullOuter,
            device,
        )?;

        // Apply `Create`
        let lhs_create_table = Table::get_builder()
            .with_name("apply_patch lhs_create")
            .with_record_batches(vec![lhs_create])?
            .build()?;
        let original = lhs_create_table.get_column_as_vec_str(lhs_values);
        let patches = lhs_create_table.get_column_as_vec_str(diff_column);
        let (modified_arr, pks_arr) = match lhs_create_table.get_column_data_type(lhs_pk)? {
            DataType::UInt32 => {
                let lhs_pks = lhs_create_table.get_column_as_vec_primitive::<u32>(lhs_pk)?;
                let rhs_pks = lhs_create_table.get_column_as_vec_primitive::<u32>(rhs_pk)?;
                let modified_pks: Result<(Vec<String>, Vec<u32>)> = original
                    .into_iter()
                    .zip(patches.into_iter())
                    .zip(lhs_pks.into_iter())
                    .zip(rhs_pks.into_iter())
                    .map(|(((o, p), lhs_pk), rhs_pk)| {
                        if p.is_empty() {
                            Ok((o.to_string(), lhs_pk))
                        } else {
                            let modified = apply_patch_auto(o, p, true)?;
                            Ok((modified, rhs_pk))
                        }
                    })
                    .collect();
                let (modified, pks) = modified_pks?;
                let modified_arr: ArrayRef = Arc::new(StringArray::from(modified));
                let pks_arr: ArrayRef = Arc::new(UInt32Array::from(pks));
                (modified_arr, pks_arr)
            }
            DataType::Int64 => {
                let lhs_pks = lhs_create_table.get_column_as_vec_primitive::<i64>(lhs_pk)?;
                let rhs_pks = lhs_create_table.get_column_as_vec_primitive::<i64>(rhs_pk)?;
                let modified_pks: Result<(Vec<String>, Vec<i64>)> = original
                    .into_iter()
                    .zip(patches.into_iter())
                    .zip(lhs_pks.into_iter())
                    .zip(rhs_pks.into_iter())
                    .map(|(((o, p), lhs_pk), rhs_pk)| {
                        if p.is_empty() {
                            Ok((o.to_string(), lhs_pk))
                        } else {
                            let modified = apply_patch_auto(o, p, true)?;
                            Ok((modified, rhs_pk))
                        }
                    })
                    .collect();
                let (modified, pks) = modified_pks?;
                let modified_arr: ArrayRef = Arc::new(StringArray::from(modified));
                let pks_arr: ArrayRef = Arc::new(Int64Array::from(pks));
                (modified_arr, pks_arr)
            }
            DataType::Utf8 => {
                let lhs_pks = lhs_create_table.get_column_as_vec_nonprimitive::<String>(lhs_pk)?;
                let rhs_pks = lhs_create_table.get_column_as_vec_nonprimitive::<String>(rhs_pk)?;
                let modified_pks: Result<(Vec<String>, Vec<String>)> = original
                    .into_iter()
                    .zip(patches.into_iter())
                    .zip(lhs_pks.into_iter())
                    .zip(rhs_pks.into_iter())
                    .map(|(((o, p), lhs_pk), rhs_pk)| {
                        if p.is_empty() {
                            Ok((o.to_string(), lhs_pk))
                        } else {
                            let modified = apply_patch_auto(o, p, true)?;
                            Ok((modified, rhs_pk))
                        }
                    })
                    .collect();
                let (modified, pks) = modified_pks?;
                let modified_arr: ArrayRef = Arc::new(StringArray::from(modified));
                let pks_arr: ArrayRef = Arc::new(StringArray::from(pks));
                (modified_arr, pks_arr)
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported data type {} for column {lhs_pk}",
                    lhs_create_table.get_column_data_type(lhs_pk)?
                ));
            }
        };
        let mut lhs_created_batch_vec = Vec::new();
        for field in lhs_args.first().unwrap().schema().fields() {
            if field.name() == lhs_values {
                lhs_created_batch_vec.push((field.name().to_string(), modified_arr.clone()))
            } else if field.name() == lhs_pk {
                lhs_created_batch_vec.push((field.name().to_string(), pks_arr.clone()))
            } else {
                lhs_created_batch_vec.push((
                    field.name().to_string(),
                    lhs_create_table.get_column_as_array(field.name())?,
                ))
            }
        }
        RecordBatch::try_from_iter(lhs_created_batch_vec)?
    } else {
        lhs_update
    };

    Ok(lhs_create)
}

#[cfg(test)]
mod tests {
    use crate::{PatchOperator, device};

    use super::*;

    #[test]
    fn test_apply_patch_all() -> Result<()> {
        // Create the mock repository
        let repo_pks = vec![0, 1, 2, 3, 4];
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
            r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
            "pub mod extra;",
            r#"mod todo;
pub use todo::Todo"#,
            "pub struct Todo {}",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_pks: ArrayRef = Arc::new(UInt32Array::from(repo_pks));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let repo_batch = RecordBatch::try_from_iter(vec![
            ("repo_pk", repo_pks),
            ("repo_path", repo_paths),
            ("code", code),
        ])?;

        // Create the mock patches
        let patch_pks = vec![1, 3, 5];
        let patch_paths = [
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/other.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let operations = vec![
            PatchOperator::Delete.to_string(),
            PatchOperator::Update.to_string(),
            PatchOperator::Create.to_string(),
        ];
        let patches = [
            "",
            "@@ pub mod extra;\n+pub mod other;\n",
            "+pub struct Other {}\n*** End Patch",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let patch_pks: ArrayRef = Arc::new(UInt32Array::from(patch_pks));
        let patch_paths: ArrayRef = Arc::new(StringArray::from(patch_paths));
        let operations: ArrayRef = Arc::new(StringArray::from(operations));
        let patches: ArrayRef = Arc::new(StringArray::from(patches));
        let patch_batch = RecordBatch::try_from_iter(vec![
            ("patch_pk", patch_pks),
            ("patch_path", patch_paths),
            ("operation", operations),
            ("patch", patches),
        ])?;

        // Make the device
        let device = device(false)?;

        // --- PK = String ---
        // Patch the repository
        let result = apply_patch(
            std::slice::from_ref(&repo_batch),
            std::slice::from_ref(&patch_batch),
            "code",
            &["patch", "operation"],
            "repo_path",
            "patch_path",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [0, 3, 4, 2, 0]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/other.rs"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub mod extra;",
                "pub struct Other {}"
            ]
        );

        // --- PK = UInt32 ---
        // Patch the repository
        let result = apply_patch(
            &[repo_batch],
            &[patch_batch],
            "code",
            &["patch", "operation"],
            "repo_pk",
            "patch_pk",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [0, 2, 3, 4, 5]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                ""
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod extra;",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub struct Other {}"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_apply_patch_missing_delete() -> Result<()> {
        // Create the mock repository
        let repo_pks = vec![0, 1, 2, 3, 4];
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
            r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
            "pub mod extra;",
            r#"mod todo;
pub use todo::Todo"#,
            "pub struct Todo {}",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_pks: ArrayRef = Arc::new(UInt32Array::from(repo_pks));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let repo_batch = RecordBatch::try_from_iter(vec![
            ("repo_pk", repo_pks),
            ("repo_path", repo_paths),
            ("code", code),
        ])?;

        // Create the mock patches
        let patch_pks = vec![3, 5];
        let patch_paths = [
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/other.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let operations = vec![
            PatchOperator::Update.to_string(),
            PatchOperator::Create.to_string(),
        ];
        let patches = [
            "@@ pub mod extra;\n+pub mod other;\n",
            "+pub struct Other {}\n*** End Patch",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let patch_pks: ArrayRef = Arc::new(UInt32Array::from(patch_pks));
        let patch_paths: ArrayRef = Arc::new(StringArray::from(patch_paths));
        let operations: ArrayRef = Arc::new(StringArray::from(operations));
        let patches: ArrayRef = Arc::new(StringArray::from(patches));
        let patch_batch = RecordBatch::try_from_iter(vec![
            ("patch_pk", patch_pks),
            ("patch_path", patch_paths),
            ("operation", operations),
            ("patch", patches),
        ])?;

        // Make the device
        let device = device(false)?;

        // --- PK = String ---
        // Patch the repository
        let result = apply_patch(
            std::slice::from_ref(&repo_batch),
            std::slice::from_ref(&patch_batch),
            "code",
            &["patch", "operation"],
            "repo_path",
            "patch_path",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [0, 3, 4, 2, 1, 0]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/other.rs"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub mod extra;",
                "use anyhow::Result;\nfn main() -> Result<()> {\n    Ok(())\n}",
                "pub struct Other {}"
            ]
        );

        // --- PK = UInt32 ---
        // Patch the repository
        let result = apply_patch(
            &[repo_batch],
            &[patch_batch],
            "code",
            &["patch", "operation"],
            "repo_pk",
            "patch_pk",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [0, 1, 2, 3, 4, 5]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                ""
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "use anyhow::Result;\nfn main() -> Result<()> {\n    Ok(())\n}",
                "pub mod extra;",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub struct Other {}"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_apply_patch_missing_update() -> Result<()> {
        // Create the mock repository
        let repo_pks = vec![0, 1, 2, 3, 4];
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
            r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
            "pub mod extra;",
            r#"mod todo;
pub use todo::Todo"#,
            "pub struct Todo {}",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_pks: ArrayRef = Arc::new(UInt32Array::from(repo_pks));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let repo_batch = RecordBatch::try_from_iter(vec![
            ("repo_pk", repo_pks),
            ("repo_path", repo_paths),
            ("code", code),
        ])?;

        // Create the mock patches
        let patch_pks = vec![1, 5];
        let patch_paths = [
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/extras/other.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let operations = vec![
            PatchOperator::Delete.to_string(),
            PatchOperator::Create.to_string(),
        ];
        let patches = ["", "+pub struct Other {}\n*** End Patch"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let patch_pks: ArrayRef = Arc::new(UInt32Array::from(patch_pks));
        let patch_paths: ArrayRef = Arc::new(StringArray::from(patch_paths));
        let operations: ArrayRef = Arc::new(StringArray::from(operations));
        let patches: ArrayRef = Arc::new(StringArray::from(patches));
        let patch_batch = RecordBatch::try_from_iter(vec![
            ("patch_pk", patch_pks),
            ("patch_path", patch_paths),
            ("operation", operations),
            ("patch", patches),
        ])?;

        // Make the device
        let device = device(false)?;

        // --- PK = String ---
        // Patch the repository
        let result = apply_patch(
            std::slice::from_ref(&repo_batch),
            std::slice::from_ref(&patch_batch),
            "code",
            &["patch", "operation"],
            "repo_path",
            "patch_path",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [0, 3, 4, 2, 0]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/other.rs"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "mod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub mod extra;",
                "pub struct Other {}"
            ]
        );

        // --- PK = UInt32 ---
        // Patch the repository
        let result = apply_patch(
            &[repo_batch],
            &[patch_batch],
            "code",
            &["patch", "operation"],
            "repo_pk",
            "patch_pk",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [0, 2, 3, 4, 5]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                ""
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod extra;",
                "mod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub struct Other {}"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_apply_patch_missing_create() -> Result<()> {
        // Create the mock repository
        let repo_pks = vec![0, 1, 2, 3, 4];
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
            r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
            "pub mod extra;",
            r#"mod todo;
pub use todo::Todo"#,
            "pub struct Todo {}",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_pks: ArrayRef = Arc::new(UInt32Array::from(repo_pks));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let repo_batch = RecordBatch::try_from_iter(vec![
            ("repo_pk", repo_pks),
            ("repo_path", repo_paths),
            ("code", code),
        ])?;

        // Create the mock patches
        let patch_pks = vec![1, 3];
        let patch_paths = [
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/extras/mod.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let operations = vec![
            PatchOperator::Delete.to_string(),
            PatchOperator::Update.to_string(),
        ];
        let patches = ["", "@@ pub mod extra;\n+pub mod other;\n"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let patch_pks: ArrayRef = Arc::new(UInt32Array::from(patch_pks));
        let patch_paths: ArrayRef = Arc::new(StringArray::from(patch_paths));
        let operations: ArrayRef = Arc::new(StringArray::from(operations));
        let patches: ArrayRef = Arc::new(StringArray::from(patches));
        let patch_batch = RecordBatch::try_from_iter(vec![
            ("patch_pk", patch_pks),
            ("patch_path", patch_paths),
            ("operation", operations),
            ("patch", patches),
        ])?;

        // Make the device
        let device = device(false)?;

        // --- PK = String ---
        // Patch the repository
        let result = apply_patch(
            std::slice::from_ref(&repo_batch),
            std::slice::from_ref(&patch_batch),
            "code",
            &["patch", "operation"],
            "repo_path",
            "patch_path",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [3, 0, 4, 2]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub struct Todo {}",
                "pub mod extra;"
            ]
        );

        // --- PK = UInt32 ---
        // Patch the repository
        let result = apply_patch(
            &[repo_batch],
            &[patch_batch],
            "code",
            &["patch", "operation"],
            "repo_pk",
            "patch_pk",
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_name("test_apply_patch")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [3, 0, 2, 4]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/todo.rs"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("code")?;
        assert_eq!(
            test,
            [
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod extra;",
                "pub struct Todo {}"
            ]
        );

        Ok(())
    }
}
