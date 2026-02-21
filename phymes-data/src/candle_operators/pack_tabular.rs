use std::collections::HashMap;

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, CsvFormat, DataFormat, Function, FunctionParameters,
    JSONSchemaDefine, JSONSchemaType, MappableTrait, Table, TableBuilderTrait, TableTrait, Tool,
    ToolType, create_blob_batch, create_chat_record_batch,
};
use phymes_diagnostics::create_timestamp_micros;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Extract tabular data in either CSV or JSON format from Bytes
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PackTabular {
    format: DataFormat,
}

impl MappableTrait for PackTabular {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for PackTabular {
    fn get_description(&self) -> String {
        "Pack tabular data in either CSV or JSON format from Bytes".to_string()
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
                description: Some("DataSummaryFormat object as a String".to_string()),
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

impl DataOperatorTrait for PackTabular {
    fn new(config: &DataConfig) -> Result<Self>
    where
        Self: Sized,
    {
        let format = config.format.clone().ok_or(anyhow!(
            "Missing `format` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(PackTabular { format })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        pack_tabular(lhs_args, &self.format)
    }
}

/// Helper function to convert a [Table] into the desired output [DataFormat]
///
/// # Arguments
/// `table` - the [Table] containing the data
/// `format` - the desired output [DataFormat]
/// `content` - Optional string to include JUST the contents of column data `content`
///   which is needed for some tool calling and visualization generation methods
pub fn table_and_data_format_to_record_batch(
    table: &Table,
    format: &DataFormat,
    content: Option<&str>,
) -> Result<RecordBatch> {
    match format {
        DataFormat::None => {
            // Extract out the content
            let content = if let Some(content) = content {
                match table.get_column_as_vec_string(content) {
                    Ok(column) => column.join(""),
                    Err(_err) => String::new(),
                }
            } else {
                serde_json::to_string(&table.to_json_object()?)?
            };
            // Wrap into a record batch
            create_chat_record_batch(
                vec!["tool".to_string()], // DM: Change when upgrading to Qwen 3 "function"
                vec![content],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Csv(csv_format) => {
            // Convert to CSV and wrap into a blob batch
            let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::CsvDefault => {
            // Convert to CSV and wrap into a blob batch
            let csv_format = CsvFormat {
                ..Default::default()
            };
            let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Bytes => {
            // Convert to bytes directly
            let bytes = table.to_bytes()?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes.to_vec()],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Json(_) | DataFormat::JsonDefault | DataFormat::JsonSchema => {
            // Convert to JSON
            let bytes = table.to_json()?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Html | DataFormat::Txt => {
            // Extract out the values column and concatenate into a single String to form the document
            let bytes = table
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Pdf | DataFormat::Ipc | DataFormat::Xml | DataFormat::Owl => {
            Err(anyhow!("{format} format is not yet supported."))
        }
    }
}

/// Pack tabular data in either CSV or JSON format from Bytes
#[instrument(skip(lhs_args))]
pub fn pack_tabular(lhs_args: &[RecordBatch], format: &DataFormat) -> Result<RecordBatch> {
    // Pack the values
    let args_table = Table::get_builder()
        .with_name("pack_tabular")
        .with_record_batches(lhs_args.to_vec())?
        .build()?
        .concat_record_batches()?;

    // Convert to the desired format
    let batch = table_and_data_format_to_record_batch(&args_table, &format, None)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::candle_data::test_candle_ops_processor::make_embeddings_record_batch_str_f32;
    use arrow::array::{ArrayRef, StringArray};

    use super::*;

    #[tokio::test]
    async fn test_pack_tabular_message_format() -> Result<()> {
        // Create the input
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch =
            make_embeddings_record_batch_str_f32("lhs_pk", lhs_ids_vec, lhs_embeddings_vec)?;

        // Pack the tabular data
        let result = pack_tabular(&[lhs_batch], &DataFormat::None)?;

        // Wrap the results in a table
        let partitions = Table::get_builder()
            .with_name("pack_tabular")
            .with_record_batches(vec![result])?
            .build()?
            .concat_record_batches()?;

        // Check the results
        assert_eq!(partitions.count_rows(), 1);
        // DM: change after upgrading to Qwen 3 series
        // assert_eq!(partitions.get_column_as_vec_str("role"), ["function"]);
        assert_eq!(partitions.get_column_as_vec_str("role"), ["tool"]);
        assert_eq!(
            partitions.get_column_as_vec_str("content"),
            [
                "[{\"embedding\":[1.0,1.0,1.0,1.0],\"lhs_pk\":\"1\"},{\"embedding\":[0.0,1.0,0.0,1.0],\"lhs_pk\":\"2\"},{\"embedding\":[0.0,0.0,0.0,1.0],\"lhs_pk\":\"3\"}]"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_pack_tabular_blob_formats() -> Result<()> {
        // Create the input
        let lhs_ids_vec = vec!["1", "2", "3"];
        let ids_ar: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec));
        let lhs_batch = RecordBatch::try_from_iter(vec![("lhs_pk", ids_ar)])?;

        // Pack the tabular data
        let result = pack_tabular(&[lhs_batch], &DataFormat::CsvDefault)?;

        // Wrap the results in a table
        let partitions = Table::get_builder()
            .with_name("pack_tabular")
            .with_record_batches(vec![result])?
            .build()?
            .concat_record_batches()?;

        // Check the results
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(
            partitions.get_column_as_vec_str("filename"),
            ["pack_tabular"]
        );
        assert_eq!(partitions.get_column_as_vec_str("extension"), ["csv"]);
        assert_eq!(partitions.get_column_as_vec_str("metadata"), ["assistant"]);
        let contents_vec = partitions.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let mut contents_str = Vec::new();
        for contents in contents_vec.into_iter() {
            contents_str.push(String::from_utf8(contents)?);
        }
        let contents_join = contents_str.join("");
        assert_eq!(contents_join, "lhs_pk\n1\n2\n3\n");

        Ok(())
    }
}
