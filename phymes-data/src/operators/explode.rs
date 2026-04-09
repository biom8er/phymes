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
use phymes_subject::{
    BuildableTrait, BuilderTrait, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType,
    MappableTrait, Table, TableBuilderTrait, TableTrait, Tool, ToolType,
};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{
    ToolTrait,
    candle_data::DataConfig,
    candle_operators::{
        data_operator::DataOperatorTrait,
        group_by_and_aggregate::{
            build_aggregator_column_fixed_size_list, build_aggregator_column_list_nonprimitive,
            build_aggregator_column_list_primitive,
        },
    },
};

/// Transform each element of a list-like to a row
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Explode {
    lhs_values: Vec<String>,
}

impl MappableTrait for Explode {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Explode {
    fn get_description(&self) -> String {
        "Transform each element of a list-like to a row".to_string()
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

impl DataOperatorTrait for Explode {
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;

        Ok(Explode {
            lhs_values,
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
        explode(&lhs_values, lhs_args, device)
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

/// Transform each element of a list-like to a row
///
/// # Notes
/// * Only nested columns (i.e., List and FixedSizeList) can be exploded; Non-nested columns will through an error
/// * FixedSizeList columns with the same size will be exploded simultaneous (i.e., for N FixedSizeList columns of size 6, 6 additional rows will be added).
/// * FixedSizeList without the same size and List columns with or without the same size will be exploded combinatorially (i.e., for 2 List columns of size 6, 36 additional rows will be added).
/// * See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html
///
/// # Arguments
///
/// * `lhs_values` - Slice of Strings for the columns to explode
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
#[instrument(skip(lhs_values, lhs_args, _device))]
pub fn explode(
    lhs_values: &[&str],
    lhs_args: &[RecordBatch],
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs_args into a table
    let lhs_table = Table::get_builder()
        .with_name("explode")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;

    // Determine if simultaneous or combinatorial explosion should be applied
    let lhs_types_vec = lhs_table
        .get_schema()
        .fields()
        .iter()
        .filter_map(|f| {
            if lhs_values.contains(&f.name().as_str()) {
                None
            } else {
                Some(f.data_type().to_owned())
            }
        })
        .collect::<Vec<_>>();
    let lhs_types_set = lhs_types_vec.iter().collect::<HashSet<_>>();
    if lhs_types_set.len() > 1 {
        todo!()
    } else {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray, UInt32Array};
    use crate::device;

    use super::*;

    #[test]
    fn test_explode() -> Result<()> {
        todo!()
    }
}
