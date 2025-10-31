use std::collections::HashMap;

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, CsvFormat, DataFormat, Function, FunctionParameters,
    JSONSchemaDefine, JSONSchemaType, JsonFormat, MappableTrait, Table, TableBuilder,
    TableBuilderTrait, TableTrait, Tool, ToolType,
};
use tracing::instrument;

use crate::{candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Extract tabular data in either CSV or JSON format from Bytes
#[derive(Debug)]
pub struct ExtractTabularData {
    lhs_values: String,
    format: DataFormat,
}

impl MappableTrait for ExtractTabularData {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for ExtractTabularData {
    fn new(config: &DataConfig) -> Self
    where
        Self: Sized,
    {
        let lhs_values = config
            .lhs_values
            .as_ref()
            .cloned()
            .unwrap_or_default()
            .first()
            .cloned()
            .unwrap_or_default();
        let format = config.format.unwrap_or_default();
        ExtractTabularData { lhs_values, format }
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        extract_tabular_data(&self.lhs_values, lhs_args, &self.format)
    }
    fn get_description() -> String {
        "Extract tabular data in either CSV or JSON format from Bytes".to_string()
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
                description: Some("DataSummaryFormat object as a String".to_string()),
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

/// Extract tabular data in either CSV or JSON format from Bytes
#[instrument(skip(lhs_values, lhs_args))]
pub fn extract_tabular_data(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
) -> Result<RecordBatch> {
    let args_table = Table::get_builder()
        .with_name("extract_tabular_data")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table.get_column_as_vec_nested_primitive::<u8>(lhs_values)?;
    let table = match format {
        DataFormat::Csv(csv_format) => Table::get_builder()
            .with_name("attachment")
            .with_csv(
                values_vec.last().unwrap(),
                csv_format.delimiter,
                csv_format.header,
                csv_format.batch_size,
            )?
            .build()?,
        DataFormat::CsvDefault => {
            let csv_format = CsvFormat::default();
            Table::get_builder()
                .with_name("attachment")
                .with_csv(
                    values_vec.last().unwrap(),
                    csv_format.delimiter,
                    csv_format.header,
                    csv_format.batch_size,
                )?
                .build()?
        }
        DataFormat::Json(json_format) => Table::get_builder()
            .with_name("attachment")
            .with_json(values_vec.last().unwrap(), json_format.batch_size)?
            .build()?,
        DataFormat::JsonDefault => {
            let json_format = JsonFormat::default();
            Table::get_builder()
                .with_name("attachment")
                .with_json(values_vec.last().unwrap(), json_format.batch_size)?
                .build()?
        }
        DataFormat::Ipc => TableBuilder::new_from_ipc_stream(values_vec.last().unwrap())?
            .with_name("attachment")
            .build()?,
        _ => {
            return Err(anyhow!(
                "Unsupported format {format:?} for extract_tabular_data operator."
            ));
        }
    };

    let batch = table.get_record_batches_own().remove(0);
    Ok(batch)
}

pub mod test_extract_tabular_data {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float32Array, StringArray};
    use phymes_core::{BuildableTrait, BuilderTrait, Table, TableBuilderTrait};

    pub fn make_scores_table() -> Result<Table> {
        let lhs_ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let scores: ArrayRef = Arc::new(Float32Array::from(vec![3.0, 2.0, 1.0]));
        let batch = RecordBatch::try_from_iter(vec![("lhs_pk", lhs_ids), ("score", scores)])?;
        Table::get_builder()
            .with_name("scores")
            .with_record_batches(vec![batch])?
            .build()
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{
        BuildableTrait, BuilderTrait, CsvFormat, DataFormat, JsonFormat, Table, TableBuilderTrait,
        TableTrait, create_blob_batch,
    };
    use phymes_diagnostics::create_timestamp_micros;

    use crate::candle_operators::extract_tabular_data::test_extract_tabular_data::make_scores_table;

    use super::*;

    #[test]
    fn test_extract_tabular_data_csv_format() {
        let csv_format = CsvFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data
            .to_csv(csv_format.delimiter, csv_format.header)
            .unwrap();
        let csv_batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["csv".to_string()],
            vec![bytes],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted =
            extract_tabular_data("bytes", &[csv_batch], &DataFormat::Csv(csv_format)).unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Table::get_builder()
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

    #[test]
    fn test_extract_tabular_data_json_format() {
        let json_format = JsonFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data.to_json().unwrap();
        let json_batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![bytes],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted =
            extract_tabular_data("bytes", &[json_batch], &DataFormat::Json(json_format)).unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Table::get_builder()
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
