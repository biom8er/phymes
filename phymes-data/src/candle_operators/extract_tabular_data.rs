use std::collections::HashMap;

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    schemas::{chat_completion, types},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
};
use tracing::{Level, event, instrument};

use crate::{candle_data::summary_config::DataSummaryFormat, candle_operators::data_operator::{make_error_record_batch, DataOperatorTrait}};

/// Extract tabular data in either CSV or JSON format from Bytes
#[derive(Debug)]
pub struct ExtractTabularData {
    lhs_values: String,
    format: DataSummaryFormat,
}

impl MappableTrait for ExtractTabularData {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for ExtractTabularData {
    fn new(
        _lhs_pk: &str,
        _lhs_fk: &str,
        lhs_values: &str,
        _rhs_pk: Option<&str>,
        _rhs_fk: Option<&str>,
        _rhs_values: Option<&str>,
        kwargs: Option<&str>,
    ) -> Self
    where
        Self: Sized,
    {
        let format = match kwargs {
            Some(kw) => match serde_json::from_str(kw) {
                Ok(format) => format,
                Err(err) => {
                    event!(Level::ERROR, "Failed to parse ExtractTabularData kwargs: {err}, using default.");
                    DataSummaryFormat::default()
                }
            },
            None => {
                event!(Level::ERROR, "No ExtractTabularData kwargs were provided, using default.");
                DataSummaryFormat::default()
            }
        };
        ExtractTabularData {
            lhs_values: lhs_values.to_string(),
            format
        }
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        match extract_tabular_data(&self.lhs_values, lhs_args, &self.format) {
            Ok(batch) => Ok(batch),
            Err(err) => Ok(make_error_record_batch(err.to_string().as_str())),
        }
    }
    fn get_description() -> String {
        "Extract tabular data in either CSV or JSON format from Bytes".to_string()
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
            "lhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The values column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "op_kwargs".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "DataSummaryFormat object as a String".to_string(),
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
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
                    "op_kwargs".to_string(),
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

/// Extract tabular data in either CSV or JSON format from Bytes
#[instrument(skip(lhs_values, lhs_args))]
pub fn extract_tabular_data(lhs_values: &str, lhs_args: &[RecordBatch], format: &DataSummaryFormat) -> Result<RecordBatch> {
    let args_table = ArrowTable::get_builder()
        .with_name("")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table.get_column_as_vec_nested_primitive::<u8>(lhs_values)?;
    let table = match format {
        DataSummaryFormat::Csv(csv_format) => {
            ArrowTable::get_builder()
            .with_name("attachment")
            .with_csv(values_vec.last().unwrap(), csv_format.delimiter, csv_format.header, csv_format.batch_size)?
            .build()?
        }
        DataSummaryFormat::Json(json_format) => {
            ArrowTable::get_builder()
            .with_name("attachment")
            .with_json(values_vec.last().unwrap(), json_format.batch_size)?
            .build()?
        }
        _ => return Err(anyhow!("Unsupported format {:?} for extract_tabular_data operator.", format)),        
    };

    let batch = table.get_record_batches_own().remove(0);
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float32Array, StringArray};
    use phymes_core::{
        schemas::available_subjects::create_blob_batch, session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}
    };

    use crate::candle_data::summary_config::CsvFormat;

    use super::*;

    
    pub fn make_scores_table() -> Result<ArrowTable> {
        let lhs_ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let scores: ArrayRef = Arc::new(Float32Array::from(vec![3.0, 2.0, 1.0]));
        let batch = RecordBatch::try_from_iter(vec![("lhs_pk", lhs_ids), ("score", scores)])?;
        ArrowTable::get_builder()
            .with_name("scores")
            .with_record_batches(vec![batch])?
            .build()
    }

    #[test]
    fn test_extract_tabular_data_csv_format() {
        let csv_format = CsvFormat { ..Default::default() };

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data.to_csv(csv_format.delimiter, csv_format.header).unwrap();
        let csv_batch = create_blob_batch(vec!["attachment".to_string()], vec!["csv".to_string()], vec![bytes], vec!["".to_string()]).unwrap();

        // Extract the tabular data
        let extracted = extract_tabular_data(
            "bytes",
            &vec![csv_batch],
            &DataSummaryFormat::Csv(csv_format),
        ).unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = ArrowTable::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let lhs_pk = table.get_column_as_vec_str("lhs_pk");
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(lhs_pk, vec!["a", "b", "c"]);
        assert_eq!(score, vec![3.0, 2.0, 1.0]);
    }
}
