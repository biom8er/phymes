use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray, UInt8Array,
        UInt32Array,
    },
    compute::cast,
    datatypes::DataType,
};
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType,
    MappableTrait, Table, TableBuilderTrait, TableScript, TableTrait, Tool, ToolType,
    from_str_to_data_type,
};
use serde_json::json;
use tracing::instrument;

use crate::{
    candle_data::{DataCastOperator, DataConfig},
    candle_operators::DataOperatorTrait,
};

/// Select and cast the [RecordBatch]es based on the [DataCastOperator] and [DataType] with optional column renaming and template injection
#[derive(Debug)]
pub struct SelectAndCast {
    lhs_values: Vec<String>,
    as_columns: Vec<String>,
    cast_operators: Vec<DataCastOperator>,
    cast_datatypes: Vec<DataType>,
    cast_templates: Vec<String>,
}

impl MappableTrait for SelectAndCast {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for SelectAndCast {
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
        let as_columns = self
            .as_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let cast_templates = self
            .cast_templates
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        select_and_cast(
            &lhs_values,
            lhs_args,
            &as_columns,
            &self.cast_operators,
            &self.cast_datatypes,
            &cast_templates,
            device,
        )
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!("Missing `lhs_values` for `{}`.", Self::get_static_name()))?;
        let as_columns = config.as_columns.clone().ok_or(anyhow!("Missing `as_columns` for `{}`.", Self::get_static_name()))?;
        let cast_operators = config.cast_operators.clone().ok_or(anyhow!("Missing `cast_operators` for `{}`.", Self::get_static_name()))?;
        let cast_datatypes = config
            .cast_datatypes
            .clone()
            .ok_or(anyhow!("Missing `cast_datatypes` for `{}`.", Self::get_static_name()))?
            .iter()
            .map(|s| from_str_to_data_type(s).unwrap())
            .collect::<Vec<_>>();
        let cast_templates = config.cast_templates.clone().ok_or(anyhow!("Missing `cast_templates` for `{}`.", Self::get_static_name()))?;
        
        // Ensure that the array lengths match
        if lhs_values.len() != as_columns.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the as_columns length {}",
                lhs_values.len(),
                as_columns.len()
            ));
        } else if lhs_values.len() != cast_operators.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the cast_operators length {}",
                lhs_values.len(),
                cast_operators.len()
            ));
        } else if lhs_values.len() != cast_datatypes.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the cast_datatypes length {}",
                lhs_values.len(),
                cast_datatypes.len()
            ));
        } else if lhs_values.len() != cast_templates.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the with_templates length {}",
                lhs_values.len(),
                cast_templates.len()
            ));
        }

        Ok(SelectAndCast {
            lhs_values,
            as_columns,
            cast_operators,
            cast_datatypes,
            cast_templates,
        })
    }
    fn get_description() -> String {
        "Cast specified columns using a specified cast operator and cast data type with optional column renaming and template injection."
            .to_string()
    }
    fn get_json_tool_schema() -> String {
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
            description: Some(Self::get_description()),
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

/// Cast specified columns using a specified cast operator and cast data type with optional column renaming and template injection
///
/// # Notes
/// * An SQL equivalent would be the following at the row level, e.g., SELECT CAST('100' AS INTEGER);
/// * An SQL equivalent would be the following at the table level, e.g., SELECT COUNT(COL1) AS count...
///   `lhs_values` = ["COL1", ...]
///   `as_columns` = ["count", ...]
///   `cast_operators` = [DataCastOperator::Cast, ...]
///   `cast_datatypes` = [DataType::UInt32, ...]
///   `cast_templates` = ["", ...] ignored since this is not a cast to DataType::Utf8
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the columns to cast
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `as_columns` - Slice of [String]s for the columns to rename to
/// * `cast_operators` - Slice of [DataCastOperator]s specifying the cast operator to apply to each lhs_values
/// * `cast_datatypes` - Slice of [DataType]s specifying the data type to cast each lhs_values to
/// * `cast_templates` - Slice of [String]s specifying the template to use when casting each lhs_value to a [String] representation
///   where the template is a simple minijinja template with a single expression for the column
///   e.g., "Hello {{ COL1 }}"
/// * `device` - The compute device
#[instrument(skip(
    lhs_values,
    lhs_args,
    as_columns,
    cast_operators,
    cast_datatypes,
    cast_templates,
    _device
))]
pub fn select_and_cast(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    as_columns: &[&str],
    cast_operators: &[DataCastOperator],
    cast_datatypes: &[DataType],
    cast_templates: &[&str],
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs into an ArrowTable
    let lhs_table = Table::get_builder()
        .with_name("select_and_cast")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;

    // Apply the cast and optional column renaming and template injection based on the lhs_values
    let mut batch_vec = Vec::new();
    for (index, column_name) in lhs_values.iter().enumerate() {
        // Try casting if possible
        let (column_cast, column_data_type) = match cast_operators.get(index).unwrap() {
            DataCastOperator::Cast => {
                let to_type = cast_datatypes.get(index).unwrap();
                let arr = cast(&lhs_table.get_column_as_array(column_name), to_type)?;
                (arr, to_type.to_owned())
            }
            DataCastOperator::None => (
                lhs_table.get_column_as_array(column_name),
                lhs_table.get_column_data_type(column_name)?,
            ),
        };

        // Inject into a string template
        let column_cast = if let Some(template) = cast_templates.get(index) {
            if template.is_empty() {
                column_cast
            } else {
                match column_data_type {
                    DataType::UInt8 => {
                        let template = TableScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<UInt8Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(arr_vec));
                        arr
                    }
                    DataType::UInt32 => {
                        let template = TableScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<UInt32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(arr_vec));
                        arr
                    }
                    DataType::Int64 => {
                        let template = TableScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(arr_vec));
                        arr
                    }
                    DataType::Float32 => {
                        let template = TableScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(arr_vec));
                        arr
                    }
                    DataType::Float64 => {
                        let template = TableScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(arr_vec));
                        arr
                    }
                    DataType::Utf8 => {
                        let template = TableScript::new_from_template(template.to_string());
                        let arr_vec = column_cast
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                template
                                    .apply_template(
                                        &json!({column_name.to_string(): s.unwrap_or_default()}),
                                    )
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let arr: ArrayRef = Arc::new(StringArray::from(arr_vec));
                        arr
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported data type {} for injecting into a String template for column {column_name}",
                            lhs_table.get_column_data_type(column_name)?
                        ));
                    }
                }
            }
        } else {
            column_cast
        };

        // Rename the columns
        if let Some(name) = as_columns.get(index) {
            if name.is_empty() {
                batch_vec.push((column_name, column_cast));
            } else {
                batch_vec.push((name, column_cast));
            }
        } else {
            batch_vec.push((column_name, column_cast));
        }
    }

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_core::device;

    use super::*;

    #[test]
    fn test_select_and_cast() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "1"];
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
        let lhs_text_vec_2 = vec!["left", "3"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // ------ String, UInt32, All ------
        // Group the text
        let result = select_and_cast(
            &["lhs_pk", "lhs_text", "lhs_metadata"],
            &[lhs_batch_1.clone(), lhs_batch_2.clone()],
            &["new_pk", "", "new_metadata"],
            &[
                DataCastOperator::Cast,
                DataCastOperator::None,
                DataCastOperator::Cast,
            ],
            &[DataType::UInt32, DataType::Utf8, DataType::Float32],
            &["", "Into template {{ lhs_text }}", ""],
            &device,
        )?;
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("lhs_text");
        assert_eq!(
            lhs_text,
            vec![
                "Into template left",
                "Into template 1",
                "Into template left",
                "Into template 3"
            ]
        );
        let lhs_id = result_table.get_column_as_vec_primitive::<u32>("new_pk")?;
        assert_eq!(lhs_id, vec![0, 1, 2, 3]);
        let metadata = result_table.get_column_as_vec_primitive::<f32>("new_metadata")?;
        assert_eq!(metadata, vec![1., 2., 3., 4.]);

        Ok(())
    }
}
