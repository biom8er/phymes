use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray, UInt8Array,
        UInt32Array,
    },
    compute::cast,
    datatypes::{DataType, Float32Type, Float64Type, Int64Type, UInt8Type, UInt32Type},
};

use anyhow::{Result, anyhow};
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use phymes_diagnostics::HashSet;
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{
    DataConfig, DataOperatorTrait, ToolTrait,
    operators::group_by::{
        build_aggregator_column_fixed_size_list, build_aggregator_column_list_nonprimitive,
        build_aggregator_column_list_primitive,
    },
};

/// Unpivot (melt) a [RecordBatch] from wide to long format
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Melt {
    lhs_values: Vec<String>,
    pvt_columns: Vec<String>,
}

impl MappableTrait for Melt {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Melt {
    fn get_description(&self) -> String {
        "Unpivot (melt) from wide to long format".to_string()
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

impl DataOperatorTrait for Melt {
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let pvt_columns = config.pvt_columns.clone().ok_or(anyhow!(
            "Missing `pvt_columns` for `{}`.",
            Self::get_static_name()
        ))?;

        Ok(Melt {
            lhs_values,
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
        let pvt_columns = self
            .pvt_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        melt(&lhs_values, lhs_args, &pvt_columns, device)
    }
}

fn expand_column_inner<T>(n_rows: usize, values_vec: &[T]) -> Vec<T>
where
    T: Clone + 'static,
{
    values_vec
        .iter()
        .flat_map(|v| (0..n_rows).map(|_| v.clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}
fn expand_column_outer<T>(n_rows: usize, values_vec: &[T]) -> Vec<T>
where
    T: Clone + 'static,
{
    (0..n_rows)
        .flat_map(|_| values_vec.to_vec())
        .collect::<Vec<_>>()
}

/// Unpivot (melt) a [RecordBatch] from wide to long format
///
/// # Notes
/// * This function is useful to massage a [RecordBatch] into a format where one or more columns are identifier variables (lhs_values),
///   while all other columns, considered measured variables (pvt_columns), are unpivoted to the row axis, leaving just two non-identifier columns, `variable` and `value`.
/// * If all [DataType]s in the `pvt_columns` are not the same, they values will be cast to [String]s and a third column for `data_type` will be added
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the columns to use as the identifier variables
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `pvt_columns` - Slice of Strings for the columns to unpivot. If not specified, uses all columns that are not set as lhs_values
/// * `device` - The compute device
#[instrument(skip(lhs_values, lhs_args, pvt_columns, _device))]
pub fn melt(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    pvt_columns: &[&str],
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs_args into a table
    let lhs_table = Subject::get_builder()
        .with_name("melt")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;

    // Determine what columns to unpivot
    let variable_columns = if pvt_columns.is_empty() {
        lhs_table
            .get_schema()
            .fields()
            .iter()
            .filter_map(|f| {
                if lhs_values.contains(&f.name().as_str()) {
                    None
                } else {
                    Some(f.name().to_string())
                }
            })
            .collect::<Vec<_>>()
    } else {
        pvt_columns
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    };

    // Begin creating the RecordBatch
    let mut batch_vec = Vec::new();
    for column_name in lhs_values {
        let arr: ArrayRef = match lhs_table.get_column_data_type(column_name)? {
            DataType::UInt8 => {
                let col_vec = lhs_table.get_column_as_vec_primitive::<u8>(column_name)?;
                let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                Arc::new(UInt8Array::from(expanded_vec))
            }
            DataType::UInt32 => {
                let col_vec = lhs_table.get_column_as_vec_primitive::<u32>(column_name)?;
                let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                Arc::new(UInt32Array::from(expanded_vec))
            }
            DataType::Int64 => {
                let col_vec = lhs_table.get_column_as_vec_primitive::<i64>(column_name)?;
                let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                Arc::new(Int64Array::from(expanded_vec))
            }
            DataType::Float32 => {
                let col_vec = lhs_table.get_column_as_vec_primitive::<f32>(column_name)?;
                let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                Arc::new(Float32Array::from(expanded_vec))
            }
            DataType::Float64 => {
                let col_vec = lhs_table.get_column_as_vec_primitive::<f64>(column_name)?;
                let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                Arc::new(Float64Array::from(expanded_vec))
            }
            DataType::Utf8 => {
                let col_vec = lhs_table.get_column_as_vec_nonprimitive::<String>(column_name)?;
                let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                Arc::new(StringArray::from(expanded_vec))
            }
            DataType::FixedSizeList(f, _) => match f.data_type() {
                DataType::UInt8 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<u8>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_fixed_size_list::<u8>(expanded_vec, DataType::UInt8)
                }
                DataType::UInt32 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<u32>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_fixed_size_list::<u32>(expanded_vec, DataType::UInt32)
                }
                DataType::Int64 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<i64>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_fixed_size_list::<i64>(expanded_vec, DataType::Int64)
                }
                DataType::Float32 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<f32>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_fixed_size_list::<f32>(expanded_vec, DataType::Float32)
                }
                DataType::Float64 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<f64>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_fixed_size_list::<f64>(expanded_vec, DataType::Float64)
                }
                // DM: Note the conversion from fixedSizeList to List
                DataType::Utf8 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_nonprimitive::<String>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_nonprimitive::<String>(
                        expanded_vec,
                        DataType::Utf8,
                    )
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} for the identifier column {column_name} for Melt Operator. The supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, and FixedList and List versions.",
                        lhs_table.get_column_data_type(column_name)?
                    ));
                }
            },
            DataType::List(f) => match f.data_type() {
                DataType::UInt8 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<u8>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        expanded_vec,
                        DataType::UInt8,
                    )
                }
                DataType::UInt32 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<u32>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        expanded_vec,
                        DataType::UInt32,
                    )
                }
                DataType::Int64 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<i64>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        expanded_vec,
                        DataType::Int64,
                    )
                }
                DataType::Float32 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<f32>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        expanded_vec,
                        DataType::Float32,
                    )
                }
                DataType::Float64 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_primitive::<f64>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        expanded_vec,
                        DataType::Float64,
                    )
                }
                DataType::Utf8 => {
                    let col_vec =
                        lhs_table.get_column_as_vec_nested_nonprimitive::<String>(column_name)?;
                    let expanded_vec = expand_column_outer(variable_columns.len(), &col_vec);
                    build_aggregator_column_list_nonprimitive::<String>(
                        expanded_vec,
                        DataType::Utf8,
                    )
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} for the identifier column {column_name} for Melt Operator. The supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, and FixedList and List versions.",
                        lhs_table.get_column_data_type(column_name)?
                    ));
                }
            },
            _ => {
                return Err(anyhow!(
                    "Unsupported data type {} for the identifier column {column_name} for Melt Operator. The supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, and FixedList and List versions.",
                    lhs_table.get_column_data_type(column_name)?
                ));
            }
        };
        batch_vec.push((column_name, arr));
    }

    // Create the variables column
    let n_rows = lhs_table.count_rows();
    let variable_vec = expand_column_inner(n_rows, &variable_columns);
    batch_vec.push((&"variable", Arc::new(StringArray::from(variable_vec))));

    // Check if all columns have the same data type
    let data_types = variable_columns
        .iter()
        .map(|name| lhs_table.get_column_data_type(name).unwrap())
        .collect::<Vec<_>>();
    let data_types_set = data_types.iter().collect::<HashSet<_>>();
    if data_types_set.len() > 1 {
        // Cast values to strings
        let mut values_vec = Vec::new();
        for column_name in variable_columns.iter() {
            let arr = cast(
                &lhs_table.get_column_as_array(column_name)?,
                &DataType::Utf8,
            )?;
            let values = Subject::get_array_as_vec_nonprimitive::<String>(&arr, column_name)?;
            values_vec.extend(values);
        }
        batch_vec.push((&"value", Arc::new(StringArray::from(values_vec))));

        // Create the data type column
        let data_types_vec = expand_column_inner(n_rows, &data_types)
            .into_iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>();
        batch_vec.push((&"data_type", Arc::new(StringArray::from(data_types_vec))));
    } else {
        // Create the values column
        let arr: ArrayRef = match data_types.first().unwrap() {
            DataType::UInt8 => {
                let expanded_vec = variable_columns
                    .iter()
                    .flat_map(|name| lhs_table.get_column_as_vec_primitive::<u8>(name).unwrap())
                    .collect::<Vec<_>>();
                Arc::new(UInt8Array::from(expanded_vec))
            }
            DataType::UInt32 => {
                let expanded_vec = variable_columns
                    .iter()
                    .flat_map(|name| lhs_table.get_column_as_vec_primitive::<u32>(name).unwrap())
                    .collect::<Vec<_>>();
                Arc::new(UInt32Array::from(expanded_vec))
            }
            DataType::Int64 => {
                let expanded_vec = variable_columns
                    .iter()
                    .flat_map(|name| lhs_table.get_column_as_vec_primitive::<i64>(name).unwrap())
                    .collect::<Vec<_>>();
                Arc::new(Int64Array::from(expanded_vec))
            }
            DataType::Float32 => {
                let expanded_vec = variable_columns
                    .iter()
                    .flat_map(|name| lhs_table.get_column_as_vec_primitive::<f32>(name).unwrap())
                    .collect::<Vec<_>>();
                Arc::new(Float32Array::from(expanded_vec))
            }
            DataType::Float64 => {
                let expanded_vec = variable_columns
                    .iter()
                    .flat_map(|name| lhs_table.get_column_as_vec_primitive::<f64>(name).unwrap())
                    .collect::<Vec<_>>();
                Arc::new(Float64Array::from(expanded_vec))
            }
            DataType::Utf8 => {
                let expanded_vec = variable_columns
                    .iter()
                    .flat_map(|name| {
                        lhs_table
                            .get_column_as_vec_nonprimitive::<String>(name)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Arc::new(StringArray::from(expanded_vec))
            }
            DataType::FixedSizeList(f, _) => match f.data_type() {
                DataType::UInt8 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<u8>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<u8>(expanded_vec, DataType::UInt8)
                }
                DataType::UInt32 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<u32>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<u32>(expanded_vec, DataType::UInt32)
                }
                DataType::Int64 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<i64>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<i64>(expanded_vec, DataType::Int64)
                }
                DataType::Float32 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<f32>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<f32>(expanded_vec, DataType::Float32)
                }
                DataType::Float64 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<f64>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_fixed_size_list::<f64>(expanded_vec, DataType::Float64)
                }
                // DM: Note the conversion from fixedSizeList to List
                DataType::Utf8 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_nonprimitive::<String>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        expanded_vec,
                        DataType::Utf8,
                    )
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} for the values column {variable_columns:?} for the Melt Operator. The supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, and FixedList and List versions.",
                        data_types.first().unwrap()
                    ));
                }
            },
            DataType::List(f) => match f.data_type() {
                DataType::UInt8 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<u8>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u8, UInt8Type>(
                        expanded_vec,
                        DataType::UInt8,
                    )
                }
                DataType::UInt32 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<u32>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<u32, UInt32Type>(
                        expanded_vec,
                        DataType::UInt32,
                    )
                }
                DataType::Int64 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<i64>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<i64, Int64Type>(
                        expanded_vec,
                        DataType::Int64,
                    )
                }
                DataType::Float32 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<f32>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f32, Float32Type>(
                        expanded_vec,
                        DataType::Float32,
                    )
                }
                DataType::Float64 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_primitive::<f64>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_primitive::<f64, Float64Type>(
                        expanded_vec,
                        DataType::Float64,
                    )
                }
                DataType::Utf8 => {
                    let expanded_vec = variable_columns
                        .iter()
                        .flat_map(|name| {
                            lhs_table
                                .get_column_as_vec_nested_nonprimitive::<String>(name)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    build_aggregator_column_list_nonprimitive::<String>(
                        expanded_vec,
                        DataType::Utf8,
                    )
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported data type {} for the values column {variable_columns:?} for the Melt Operator. The supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, and FixedList and List versions.",
                        data_types.first().unwrap()
                    ));
                }
            },
            _ => {
                return Err(anyhow!(
                    "Unsupported data type {} for the values column {variable_columns:?} for the Melt Operator. The supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, and FixedList and List versions.",
                    data_types.first().unwrap()
                ));
            }
        };
        batch_vec.push((&"value", arr));
    }

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use crate::device;
    use arrow::array::{ArrayRef, StringArray, UInt32Array};

    use super::*;

    #[test]
    fn test_melt() -> Result<()> {
        // Make the test record batches
        let lhs_a_vec_1 = vec!["a", "b", "c"];
        let lhs_a_array: ArrayRef = Arc::new(StringArray::from(lhs_a_vec_1));
        let lhs_b_vec_1: Vec<u32> = vec![1, 3, 5];
        let lhs_b_array: ArrayRef = Arc::new(UInt32Array::from(lhs_b_vec_1));
        let lhs_c_vec_1: Vec<u32> = vec![2, 4, 6];
        let lhs_c_array: ArrayRef = Arc::new(UInt32Array::from(lhs_c_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("A", lhs_a_array),
            ("B", lhs_b_array),
            ("C", lhs_c_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Make the pivot table
        let result = melt(&["A"], std::slice::from_ref(&lhs_batch_1), &["B"], &device)?;

        let lhs_a = result
            .column_by_name("A")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_a, vec!["a", "b", "c"]);
        let lhs = result
            .column_by_name("variable")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec!["B", "B", "B"]);
        let lhs = result
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec![1, 3, 5]);

        // Make the pivot table
        let result = melt(
            &["A"],
            std::slice::from_ref(&lhs_batch_1),
            &["B", "C"],
            &device,
        )?;

        let lhs_a = result
            .column_by_name("A")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_a, vec!["a", "b", "c", "a", "b", "c"]);
        let lhs = result
            .column_by_name("variable")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec!["B", "B", "B", "C", "C", "C"]);
        let lhs = result
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec![1, 3, 5, 2, 4, 6]);

        // Make the pivot table
        let result = melt(
            &["A", "B"],
            std::slice::from_ref(&lhs_batch_1),
            &["C"],
            &device,
        )?;

        let lhs_a = result
            .column_by_name("A")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_a, vec!["a", "b", "c"]);
        let lhs = result
            .column_by_name("B")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec![1, 3, 5]);
        let lhs = result
            .column_by_name("variable")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec!["C", "C", "C"]);
        let lhs = result
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec![2, 4, 6]);

        // Make the pivot table
        let result = melt(&["B"], &[lhs_batch_1], &["A", "C"], &device)?;

        let lhs = result
            .column_by_name("B")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec![1, 3, 5, 1, 3, 5]);
        let lhs = result
            .column_by_name("variable")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec!["A", "A", "A", "C", "C", "C"]);
        let lhs = result
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs, vec!["a", "b", "c", "2", "4", "6"]);
        let lhs = result
            .column_by_name("data_type")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            lhs,
            vec!["Utf8", "Utf8", "Utf8", "UInt32", "UInt32", "UInt32"]
        );

        Ok(())
    }
}
