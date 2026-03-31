use std::{
    collections::HashMap,
    io::{Cursor, Read},
};

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use flate2::read::{DeflateDecoder, GzDecoder, ZlibDecoder};
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, CsvFormat, DataEncoding, DataFormat, Function,
    FunctionParameters, JSONSchemaDefine, JSONSchemaType, JsonFormat, JsonSchemaTrait,
    MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait, Tool, ToolType,
    open_alex,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Extract tabular data in either CSV or JSON format from Bytes
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExtractTabular {
    lhs_values: String,
    encoding: DataEncoding,
    format: DataFormat,
    schema: AvailableSubjects,
}

impl MappableTrait for ExtractTabular {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for ExtractTabular {
    fn get_description(&self) -> String {
        "Extract tabular data in either CSV or JSON format from Bytes".to_string()
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

impl DataOperatorTrait for ExtractTabular {
    fn new(config: &DataConfig) -> Result<Self>
    where
        Self: Sized,
    {
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
        let encoding = config.encoding.clone().ok_or(anyhow!(
            "Missing `encoding` for `{}`.",
            Self::get_static_name()
        ))?;
        let format = config.format.clone().ok_or(anyhow!(
            "Missing `format` for `{}`.",
            Self::get_static_name()
        ))?;
        let schema = config.schema.ok_or(anyhow!(
            "Missing `schema` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(ExtractTabular {
            lhs_values,
            encoding,
            format,
            schema,
        })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        extract_tabular(
            &self.lhs_values,
            lhs_args,
            &self.encoding,
            &self.format,
            &self.schema,
        )
    }
}

/// Extract tabular data in either CSV or JSON format from Bytes
#[instrument(skip(lhs_values, lhs_args))]
pub fn extract_tabular(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    encoding: &DataEncoding,
    format: &DataFormat,
    schema: &AvailableSubjects,
) -> Result<RecordBatch> {
    // Extract out the values
    let args_table = Subject::get_builder()
        .with_name("extract_tabular")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table
        .get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // Deflate the values depending upon the specified encoding
    let values_vec = match encoding {
        DataEncoding::Deflate => {
            let cursor = Cursor::new(values_vec);
            let mut decoder = DeflateDecoder::new(cursor);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out)?;
            out
        }
        DataEncoding::Zlib => {
            let cursor = Cursor::new(values_vec);
            let mut decoder = ZlibDecoder::new(cursor);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out)?;
            out
        }
        DataEncoding::Gz => {
            let cursor = Cursor::new(values_vec);
            let mut decoder = GzDecoder::new(cursor);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out)?;
            out
        }
        DataEncoding::None => values_vec,
    };

    // Parse the values depending upon the specified format
    let table = match format {
        DataFormat::Csv(csv_format) => Subject::get_builder()
            .with_name("attachment")
            .with_csv(
                &values_vec,
                csv_format.delimiter,
                csv_format.header,
                csv_format.batch_size,
            )?
            .build()?,
        DataFormat::CsvDefault => {
            let csv_format = CsvFormat::default();
            Subject::get_builder()
                .with_name("attachment")
                .with_csv(
                    &values_vec,
                    csv_format.delimiter,
                    csv_format.header,
                    csv_format.batch_size,
                )?
                .build()?
        }
        DataFormat::Json(json_format) => Subject::get_builder()
            .with_name("attachment")
            .with_json(&values_vec, json_format.batch_size)?
            .build()?,
        DataFormat::JsonDefault => {
            let json_format = JsonFormat::default();
            Subject::get_builder()
                .with_name("attachment")
                .with_json(&values_vec, json_format.batch_size)?
                .build()?
        }
        DataFormat::JsonSchema => match schema {
            AvailableSubjects::OpenAlexResponseWorks => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseWorks>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseWorks")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(_err) => {
                        match open_alex::OpenAlexResponseWorks::from_jsonl(&values_vec) {
                            Ok(open_alex_response) => {
                                let batch = open_alex_response.to_record_batch("extract_tabular")?;
                                Subject::get_builder()
                                    .with_name("OpenAlexResponseWorks")
                                    .with_record_batches(vec![batch])?
                                    .build()?
                            }
                            Err(err) => {
                                return Err(anyhow!(
                                    "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                                ));
                            }
                        }
                    }
                }
            }
            AvailableSubjects::OpenAlexResponseAuthors => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseAuthors>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseAuthors")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            AvailableSubjects::OpenAlexResponseInstitutions => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseInstitution>(&values_vec)
                {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseInstitutions")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            AvailableSubjects::OpenAlexResponseTopics => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseTopic>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseTopics")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            AvailableSubjects::OpenAlexResponseAwards => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseAward>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseAwards")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            AvailableSubjects::OpenAlexResponseFunders => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseFunder>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseFunders")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            AvailableSubjects::OpenAlexResponsePublishers => {
                match serde_json::from_slice::<open_alex::OpenAlexResponsePublisher>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponsePublishers")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            AvailableSubjects::OpenAlexResponseSources => {
                match serde_json::from_slice::<open_alex::OpenAlexResponseSource>(&values_vec) {
                    Ok(open_alex_response) => {
                        let batch = open_alex_response.to_record_batch("extract_tabular")?;
                        Subject::get_builder()
                            .with_name("OpenAlexResponseSources")
                            .with_record_batches(vec![batch])?
                            .build()?
                    }
                    Err(err) => {
                        return Err(anyhow!(
                            "Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."
                        ));
                    }
                }
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported format `{format}` and schema `{schema}` for extract_tabular operator."
                ));
            }
        },
        DataFormat::Ipc => SubjectBuilder::new_from_ipc_stream(&values_vec)?
            .with_name("attachment")
            .build()?,
        _ => {
            return Err(anyhow!(
                "Unsupported format {format:?} for extract_tabular operator."
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
    use phymes_core::{BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait};

    pub fn make_scores_table() -> Result<Subject> {
        let lhs_ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let scores: ArrayRef = Arc::new(Float32Array::from(vec![3.0, 2.0, 1.0]));
        let batch = RecordBatch::try_from_iter(vec![("lhs_pk", lhs_ids), ("score", scores)])?;
        Subject::get_builder()
            .with_name("scores")
            .with_record_batches(vec![batch])?
            .build()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use bytes::Bytes;
    use flate2::{
        Compression,
        write::{DeflateEncoder, GzEncoder, ZlibEncoder},
    };
    use phymes_core::{
        BuildableTrait, BuilderTrait, CsvFormat, DataFormat, JsonFormat, Subject,
        SubjectBuilderTrait, SubjectTrait, create_attachments_batch,
    };
    use phymes_diagnostics::create_timestamp_micros;

    use crate::candle_operators::extract_tabular::test_extract_tabular_data::make_scores_table;

    use super::*;

    #[test]
    fn test_extract_tabular_csv_format_none_encoding() {
        let csv_format = CsvFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data
            .to_csv(csv_format.delimiter, csv_format.header)
            .unwrap();
        let csv_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["csv".to_string()],
            vec![bytes],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[csv_batch],
            &DataEncoding::None,
            &DataFormat::Csv(csv_format),
            &AvailableSubjects::Empty,
        )
        .unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let lhs_pk = table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_csv_format_deflate_encoding() {
        let csv_format = CsvFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data
            .to_csv(csv_format.delimiter, csv_format.header)
            .unwrap();
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&bytes).unwrap();
        let bytes_encoded = encoder.finish().unwrap();
        let csv_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["csv.deflate".to_string()],
            vec![bytes_encoded],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[csv_batch],
            &DataEncoding::Deflate,
            &DataFormat::Csv(csv_format),
            &AvailableSubjects::Empty,
        )
        .unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let lhs_pk = table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_csv_format_zlib_encoding() {
        let csv_format = CsvFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data
            .to_csv(csv_format.delimiter, csv_format.header)
            .unwrap();
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&bytes).unwrap();
        let bytes_encoded = encoder.finish().unwrap();
        let csv_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["csv.zz".to_string()],
            vec![bytes_encoded],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[csv_batch],
            &DataEncoding::Zlib,
            &DataFormat::Csv(csv_format),
            &AvailableSubjects::Empty,
        )
        .unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let lhs_pk = table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_csv_format_gz_encoding() {
        let csv_format = CsvFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data
            .to_csv(csv_format.delimiter, csv_format.header)
            .unwrap();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&bytes).unwrap();
        let bytes_encoded = encoder.finish().unwrap();
        let csv_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["csv.gz".to_string()],
            vec![bytes_encoded],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[csv_batch],
            &DataEncoding::Gz,
            &DataFormat::Csv(csv_format),
            &AvailableSubjects::Empty,
        )
        .unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let lhs_pk = table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_json_format_none_encoding() {
        let json_format = JsonFormat::default();

        // Make the tabular data
        let tabular_data = make_scores_table().unwrap();
        let bytes = tabular_data.to_json().unwrap();
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![bytes],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::Json(json_format),
            &AvailableSubjects::Empty,
        )
        .unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 2);
        assert_eq!(extracted.num_rows(), 3);

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let lhs_pk = table.get_column_as_vec_str("lhs_pk");
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_schema_format_none_encoding() -> Result<()> {
        // OpenAlexResponseWorks
        // Make the tabular data
        let open_alex_response_str = "{\"meta\":{\"count\":11164054,\"db_response_time_ms\":25,\"page\":1,\"per_page\":1,\"groups_count\":null},\"results\":[{\"id\":\"https://openalex.org/W3038568908\",\"doi\":\"https://doi.org/10.1585/pfr.15.2402039\",\"title\":\"Radiation Resistant Camera System for Monitoring Deuterium Plasma Discharges in the Large Helical Device\",\"display_name\":\"Radiation Resistant Camera System for Monitoring Deuterium Plasma Discharges in the Large Helical Device\",\"publication_year\":2020,\"publication_date\":\"2020-06-08\",\"ids\":{\"openalex\":\"https://openalex.org/W3038568908\",\"doi\":\"https://doi.org/10.1585/pfr.15.2402039\",\"mag\":\"3038568908\"},\"language\":\"en\",\"primary_location\":{\"id\":\"doi:10.1585/pfr.15.2402039\",\"is_oa\":true,\"landing_page_url\":\"https://doi.org/10.1585/pfr.15.2402039\",\"pdf_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"source\":{\"id\":\"https://openalex.org/S46033839\",\"display_name\":\"Plasma and Fusion Research\",\"issn_l\":\"1880-6821\",\"issn\":[\"1880-6821\"],\"is_oa\":true,\"is_in_doaj\":false,\"is_core\":true,\"host_organization\":\"https://openalex.org/P4328135220\",\"host_organization_name\":\"Japan Society of Plasma Science and Nuclear Fusion Research\",\"host_organization_lineage\":[\"https://openalex.org/P4328135220\"],\"host_organization_lineage_names\":[\"Japan Society of Plasma Science and Nuclear Fusion Research\"],\"type\":\"journal\"},\"license\":null,\"license_id\":null,\"version\":\"publishedVersion\",\"is_accepted\":true,\"is_published\":true,\"raw_source_name\":\"Plasma and Fusion Research\",\"raw_type\":\"journal-article\"},\"type\":\"article\",\"indexed_in\":[\"crossref\"],\"open_access\":{\"is_oa\":true,\"oa_status\":\"diamond\",\"oa_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"any_repository_has_fulltext\":false},\"authorships\":[{\"author_position\":\"first\",\"author\":{\"id\":\"https://openalex.org/A5039600762\",\"display_name\":\"M. Shoji\",\"orcid\":\"https://orcid.org/0000-0003-0655-7347\"},\"institutions\":[{\"id\":\"https://openalex.org/I199525922\",\"display_name\":\"National Institutes of Natural Sciences\",\"ror\":\"https://ror.org/055n47h92\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\"]},{\"id\":\"https://openalex.org/I4210108322\",\"display_name\":\"National Institute for Fusion Science\",\"ror\":\"https://ror.org/01t3wyv61\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\",\"https://openalex.org/I4210108322\"]}],\"countries\":[\"JP\"],\"is_corresponding\":true,\"raw_author_name\":\"Mamoru SHOJI\",\"raw_affiliation_strings\":[\"National Institute for Fusion Science, National Institutes of Natural Sciences\"],\"affiliations\":[{\"raw_affiliation_string\":\"National Institute for Fusion Science, National Institutes of Natural Sciences\",\"institution_ids\":[\"https://openalex.org/I4210108322\",\"https://openalex.org/I199525922\"]}]},{\"author_position\":\"last\",\"author\":{\"id\":null,\"display_name\":\"LHD Experiment Group\",\"orcid\":null},\"institutions\":[{\"id\":\"https://openalex.org/I199525922\",\"display_name\":\"National Institutes of Natural Sciences\",\"ror\":\"https://ror.org/055n47h92\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\"]},{\"id\":\"https://openalex.org/I4210108322\",\"display_name\":\"National Institute for Fusion Science\",\"ror\":\"https://ror.org/01t3wyv61\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\",\"https://openalex.org/I4210108322\"]}],\"countries\":[\"JP\"],\"is_corresponding\":false,\"raw_author_name\":\"LHD Experiment Group\",\"raw_affiliation_strings\":[\"National Institute for Fusion Science, National Institutes of Natural Sciences\"],\"affiliations\":[{\"raw_affiliation_string\":\"National Institute for Fusion Science, National Institutes of Natural Sciences\",\"institution_ids\":[\"https://openalex.org/I4210108322\",\"https://openalex.org/I199525922\"]}]}],\"institutions\":[],\"countries_distinct_count\":1,\"institutions_distinct_count\":2,\"corresponding_author_ids\":[\"https://openalex.org/A5039600762\"],\"corresponding_institution_ids\":[\"https://openalex.org/I199525922\",\"https://openalex.org/I4210108322\"],\"apc_list\":null,\"apc_paid\":null,\"fwci\":0.40325236,\"has_fulltext\":true,\"cited_by_count\":801216,\"citation_normalized_percentile\":{\"value\":0.86901083,\"is_in_top_1_percent\":false,\"is_in_top_10_percent\":false},\"cited_by_percentile_year\":{\"min\":89,\"max\":100},\"biblio\":{\"volume\":\"15\",\"issue\":\"0\",\"first_page\":\"2402039\",\"last_page\":\"2402039\"},\"is_retracted\":false,\"is_paratext\":false,\"is_xpac\":false,\"primary_topic\":{\"id\":\"https://openalex.org/T10346\",\"display_name\":\"Magnetic confinement fusion research\",\"score\":0.9991000294685364,\"subfield\":{\"id\":\"https://openalex.org/subfields/3106\",\"display_name\":\"Nuclear and High Energy Physics\"},\"field\":{\"id\":\"https://openalex.org/fields/31\",\"display_name\":\"Physics and Astronomy\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}},\"topics\":[{\"id\":\"https://openalex.org/T10346\",\"display_name\":\"Magnetic confinement fusion research\",\"score\":0.9991000294685364,\"subfield\":{\"id\":\"https://openalex.org/subfields/3106\",\"display_name\":\"Nuclear and High Energy Physics\"},\"field\":{\"id\":\"https://openalex.org/fields/31\",\"display_name\":\"Physics and Astronomy\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}},{\"id\":\"https://openalex.org/T11949\",\"display_name\":\"Nuclear Physics and Applications\",\"score\":0.9987999796867371,\"subfield\":{\"id\":\"https://openalex.org/subfields/3108\",\"display_name\":\"Radiation\"},\"field\":{\"id\":\"https://openalex.org/fields/31\",\"display_name\":\"Physics and Astronomy\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}},{\"id\":\"https://openalex.org/T10592\",\"display_name\":\"Fusion materials and technologies\",\"score\":0.998199999332428,\"subfield\":{\"id\":\"https://openalex.org/subfields/2505\",\"display_name\":\"Materials Chemistry\"},\"field\":{\"id\":\"https://openalex.org/fields/25\",\"display_name\":\"Materials Science\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}}],\"keywords\":[{\"id\":\"https://openalex.org/keywords/radiation\",\"display_name\":\"Radiation\",\"score\":0.7057818174362183},{\"id\":\"https://openalex.org/keywords/plasma\",\"display_name\":\"Plasma\",\"score\":0.5598242878913879},{\"id\":\"https://openalex.org/keywords/materials-science\",\"display_name\":\"Materials science\",\"score\":0.5517664551734924},{\"id\":\"https://openalex.org/keywords/optics\",\"display_name\":\"Optics\",\"score\":0.5239154100418091},{\"id\":\"https://openalex.org/keywords/shield\",\"display_name\":\"Shield\",\"score\":0.5098416209220886},{\"id\":\"https://openalex.org/keywords/neutron\",\"display_name\":\"Neutron\",\"score\":0.4559711515903473},{\"id\":\"https://openalex.org/keywords/nuclear-engineering\",\"display_name\":\"Nuclear engineering\",\"score\":0.3836207985877991},{\"id\":\"https://openalex.org/keywords/physics\",\"display_name\":\"Physics\",\"score\":0.32291728258132935},{\"id\":\"https://openalex.org/keywords/nuclear-physics\",\"display_name\":\"Nuclear physics\",\"score\":0.13794386386871338},{\"id\":\"https://openalex.org/keywords/geology\",\"display_name\":\"Geology\",\"score\":0.05549171566963196}],\"concepts\":[{\"id\":\"https://openalex.org/C153385146\",\"wikidata\":\"https://www.wikidata.org/wiki/Q18335\",\"display_name\":\"Radiation\",\"level\":2,\"score\":0.7057818174362183},{\"id\":\"https://openalex.org/C82706917\",\"wikidata\":\"https://www.wikidata.org/wiki/Q10251\",\"display_name\":\"Plasma\",\"level\":2,\"score\":0.5598242878913879},{\"id\":\"https://openalex.org/C192562407\",\"wikidata\":\"https://www.wikidata.org/wiki/Q228736\",\"display_name\":\"Materials science\",\"level\":0,\"score\":0.5517664551734924},{\"id\":\"https://openalex.org/C120665830\",\"wikidata\":\"https://www.wikidata.org/wiki/Q14620\",\"display_name\":\"Optics\",\"level\":1,\"score\":0.5239154100418091},{\"id\":\"https://openalex.org/C138081364\",\"wikidata\":\"https://www.wikidata.org/wiki/Q852013\",\"display_name\":\"Shield\",\"level\":2,\"score\":0.5098416209220886},{\"id\":\"https://openalex.org/C152568617\",\"wikidata\":\"https://www.wikidata.org/wiki/Q2348\",\"display_name\":\"Neutron\",\"level\":2,\"score\":0.4559711515903473},{\"id\":\"https://openalex.org/C116915560\",\"wikidata\":\"https://www.wikidata.org/wiki/Q83504\",\"display_name\":\"Nuclear engineering\",\"level\":1,\"score\":0.3836207985877991},{\"id\":\"https://openalex.org/C121332964\",\"wikidata\":\"https://www.wikidata.org/wiki/Q413\",\"display_name\":\"Physics\",\"level\":0,\"score\":0.32291728258132935},{\"id\":\"https://openalex.org/C185544564\",\"wikidata\":\"https://www.wikidata.org/wiki/Q81197\",\"display_name\":\"Nuclear physics\",\"level\":1,\"score\":0.13794386386871338},{\"id\":\"https://openalex.org/C127313418\",\"wikidata\":\"https://www.wikidata.org/wiki/Q1069\",\"display_name\":\"Geology\",\"level\":0,\"score\":0.05549171566963196},{\"id\":\"https://openalex.org/C5900021\",\"wikidata\":\"https://www.wikidata.org/wiki/Q163082\",\"display_name\":\"Petrology\",\"level\":1,\"score\":0.0},{\"id\":\"https://openalex.org/C127413603\",\"wikidata\":\"https://www.wikidata.org/wiki/Q11023\",\"display_name\":\"Engineering\",\"level\":0,\"score\":0.0}],\"mesh\":[],\"locations_count\":1,\"locations\":[{\"id\":\"doi:10.1585/pfr.15.2402039\",\"is_oa\":true,\"landing_page_url\":\"https://doi.org/10.1585/pfr.15.2402039\",\"pdf_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"source\":{\"id\":\"https://openalex.org/S46033839\",\"display_name\":\"Plasma and Fusion Research\",\"issn_l\":\"1880-6821\",\"issn\":[\"1880-6821\"],\"is_oa\":true,\"is_in_doaj\":false,\"is_core\":true,\"host_organization\":\"https://openalex.org/P4328135220\",\"host_organization_name\":\"Japan Society of Plasma Science and Nuclear Fusion Research\",\"host_organization_lineage\":[\"https://openalex.org/P4328135220\"],\"host_organization_lineage_names\":[\"Japan Society of Plasma Science and Nuclear Fusion Research\"],\"type\":\"journal\"},\"license\":null,\"license_id\":null,\"version\":\"publishedVersion\",\"is_accepted\":true,\"is_published\":true,\"raw_source_name\":\"Plasma and Fusion Research\",\"raw_type\":\"journal-article\"}],\"best_oa_location\":{\"id\":\"doi:10.1585/pfr.15.2402039\",\"is_oa\":true,\"landing_page_url\":\"https://doi.org/10.1585/pfr.15.2402039\",\"pdf_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"source\":{\"id\":\"https://openalex.org/S46033839\",\"display_name\":\"Plasma and Fusion Research\",\"issn_l\":\"1880-6821\",\"issn\":[\"1880-6821\"],\"is_oa\":true,\"is_in_doaj\":false,\"is_core\":true,\"host_organization\":\"https://openalex.org/P4328135220\",\"host_organization_name\":\"Japan Society of Plasma Science and Nuclear Fusion Research\",\"host_organization_lineage\":[\"https://openalex.org/P4328135220\"],\"host_organization_lineage_names\":[\"Japan Society of Plasma Science and Nuclear Fusion Research\"],\"type\":\"journal\"},\"license\":null,\"license_id\":null,\"version\":\"publishedVersion\",\"is_accepted\":true,\"is_published\":true,\"raw_source_name\":\"Plasma and Fusion Research\",\"raw_type\":\"journal-article\"},\"sustainable_development_goals\":[{\"score\":0.8799999952316284,\"display_name\":\"Affordable and clean energy\",\"id\":\"https://metadata.un.org/sdg/7\"}],\"awards\":[],\"funders\":[],\"has_content\":{\"grobid_xml\":true,\"pdf\":true},\"content_urls\":{\"pdf\":\"https://content.openalex.org/works/W3038568908.pdf\",\"grobid_xml\":\"https://content.openalex.org/works/W3038568908.grobid-xml\"},\"referenced_works_count\":8,\"referenced_works\":[\"https://openalex.org/W2069091362\",\"https://openalex.org/W2151240562\",\"https://openalex.org/W2527753843\",\"https://openalex.org/W2590699823\",\"https://openalex.org/W2783171299\",\"https://openalex.org/W2806477398\",\"https://openalex.org/W2922014310\",\"https://openalex.org/W2945236265\"],\"related_works\":[\"https://openalex.org/W2606430476\",\"https://openalex.org/W2069389872\",\"https://openalex.org/W2024680443\",\"https://openalex.org/W1992734408\",\"https://openalex.org/W2909752308\",\"https://openalex.org/W2074503354\",\"https://openalex.org/W2353473218\",\"https://openalex.org/W2060642378\",\"https://openalex.org/W2094345694\",\"https://openalex.org/W2889162861\"],\"abstract_inverted_index\":{\"Radiation\":[0],\"resistant\":[1,196],\"camera\":[2],\"system\":[3,18],\"was\":[4,98],\"constructed\":[5],\"for\":[6],\"monitoring\":[7],\"deuterium\":[8],\"plasma\":[9,44],\"discharges\":[10],\"in\":[11,42,52,69,83,118],\"the\":[12,43,47,62,88,91,94,105,108,112,115,119,124,129,132,139,142,145,151,159,163,174,178,187,191,194],\"Large\":[13],\"Helical\":[14],\"Device\":[15],\"(LHD).\":[16],\"This\":[17,181],\"has\":[19,134],\"contributed\":[20],\"to\":[21,32,123,162,186],\"safe\":[22],\"operation\":[23],\"during\":[24],\"two\":[25],\"experimental\":[26],\"campaigns\":[27],\"without\":[28],\"serious\":[29],\"problems\":[30],\"due\":[31],\"radiation\":[33,95,109,143,160,195],\"(neutrons\":[34],\"and\":[35,111],\"gamma-rays).\":[36],\"The\":[37,64],\"cameras\":[38,65,133],\"steadily\":[39],\"functioned\":[40],\"even\":[41],\"discharge\":[45],\"with\":[46,78,158],\"maximum\":[48],\"neutron\":[49],\"emission\":[50],\"rate\":[51],\"FY\":[53],\"2017,\":[54],\"though\":[55],\"some\":[56,169],\"bright\":[57,154,170],\"specks\":[58,155,171],\"temporarily\":[59],\"appeared\":[60],\"on\":[61,144,177],\"images.\":[63],\"have\":[66],\"been\":[67,135],\"installed\":[68],\"shield\":[70,92,120],\"boxes\":[71,76],\"which\":[72,103,165],\"consist\":[73],\"of\":[74,90,107,114,128,131,138,141,153,190,193],\"lead\":[75],\"covered\":[77],\"10%\":[79],\"borated\":[80],\"polyethylene\":[81],\"blocks\":[82],\"all\":[84],\"directions.\":[85],\"For\":[86],\"optimizing\":[87],\"design\":[89],\"box,\":[93],\"flux\":[96,110,161],\"distribution\":[97],\"calculated\":[99],\"by\":[100,173],\"MCNP-6\":[101],\"code,\":[102],\"reveals\":[104],\"reduction\":[106],\"change\":[113],\"energy\":[116],\"spectra\":[117],\"box.\":[121],\"Thanks\":[122],\"optimization,\":[125],\"significant\":[126],\"extension\":[127,189],\"lifetime\":[130,192],\"realized.\":[136],\"Investigation\":[137],\"influence\":[140],\"CCD\":[146],\"image\":[147,179],\"sensor\":[148],\"shows\":[149],\"that\":[150,168],\"number\":[152],\"generally\":[156],\"increases\":[157],\"camera,\":[164],\"also\":[166,183],\"indicates\":[167],\"disappear\":[172],\"self-annealing\":[175],\"process\":[176],\"sensor.\":[180],\"phenomenon\":[182],\"highly\":[184],\"contributes\":[185],\"further\":[188],\"cameras.\":[197]},\"counts_by_year\":[{\"year\":2026,\"cited_by_count\":1},{\"year\":2025,\"cited_by_count\":801210},{\"year\":2024,\"cited_by_count\":2},{\"year\":2022,\"cited_by_count\":2},{\"year\":2021,\"cited_by_count\":1}],\"updated_date\":\"2025-11-06T03:46:38.306776\",\"created_date\":\"2025-10-10T00:00:00\"}],\"group_by\":[]}\n";
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseWorks,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "WorkTable",
                "WorkAuthorshipTable",
                "WorkLocationTable",
                "WorkOpenAccessTable",
                "WorkBiblioTable",
                "WorkCitationPercentileTable",
                "WorkCitedByPercentileYearTable",
                "WorkCountsByYearTable",
                "WorkConceptTable",
                "WorkTopicTable",
                "WorkKeywordTable",
                "WorkSdgTagTable",
                "WorkCorrespondingAuthorTable",
                "WorkCorrespondingInstitutionTable",
                "WorkIndexedInTable",
                "WorkIdsTable",
                "WorkReferencedWorksTable",
                "WorkRelatedWorksTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let subjects = table.get_column_as_vec_str("subject");
        assert_eq!(
            subjects,
            [
                "WorkTable",
                "WorkAuthorshipTable",
                "WorkLocationTable",
                "WorkOpenAccessTable",
                "WorkBiblioTable",
                "WorkCitationPercentileTable",
                "WorkCitedByPercentileYearTable",
                "WorkCountsByYearTable",
                "WorkConceptTable",
                "WorkTopicTable",
                "WorkKeywordTable",
                "WorkSdgTagTable",
                "WorkCorrespondingAuthorTable",
                "WorkCorrespondingInstitutionTable",
                "WorkIndexedInTable",
                "WorkIdsTable",
                "WorkReferencedWorksTable",
                "WorkRelatedWorksTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(
            test,
            [
                "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc",
                "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"
            ]
        );
        let test_tables: Result<Vec<Subject>> = table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .map(|b| {
                SubjectBuilder::new_from_ipc_stream(&b)?
                    .with_name("extracted_bytes")
                    .build()
            })
            .collect();
        let test = test_tables?
            .into_iter()
            .zip(subjects)
            .map(|(t, s)| (s.to_string(), t))
            .collect::<HashMap<_, _>>();
        assert_eq!(test.get("WorkTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkAuthorshipTable").unwrap().count_rows(), 2);
        assert_eq!(test.get("WorkLocationTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkOpenAccessTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkBiblioTable").unwrap().count_rows(), 1);
        assert_eq!(
            test.get("WorkCitationPercentileTable")
                .unwrap()
                .count_rows(),
            1
        );
        assert_eq!(
            test.get("WorkCitedByPercentileYearTable")
                .unwrap()
                .count_rows(),
            1
        );
        assert_eq!(test.get("WorkCountsByYearTable").unwrap().count_rows(), 5);
        assert_eq!(test.get("WorkConceptTable").unwrap().count_rows(), 12);
        assert_eq!(test.get("WorkTopicTable").unwrap().count_rows(), 3);
        assert_eq!(test.get("WorkKeywordTable").unwrap().count_rows(), 10);
        assert!(!test.contains_key("WorkMeshTagTable"));
        assert_eq!(test.get("WorkSdgTagTable").unwrap().count_rows(), 1);
        assert_eq!(
            test.get("WorkCorrespondingAuthorTable")
                .unwrap()
                .count_rows(),
            1
        );
        assert_eq!(
            test.get("WorkCorrespondingInstitutionTable")
                .unwrap()
                .count_rows(),
            2
        );
        assert_eq!(test.get("WorkIndexedInTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkIdsTable").unwrap().count_rows(), 1);
        assert_eq!(
            test.get("WorkReferencedWorksTable").unwrap().count_rows(),
            8
        );
        assert_eq!(test.get("WorkRelatedWorksTable").unwrap().count_rows(), 10);

        // OpenAlexResponseAuthors
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":8329749,"db_response_time_ms":88,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/A5043579160","orcid":null,"display_name":"T. Tokuzawa","display_name_alternatives":["K. Tanaka","LHD Experiment Group","T Tokuzawa","T. Tokuzawa","TOKUZAWA, Tokihiko","Tokihiko TOKUZAWA","Tokihiko Tokuzawa","Tokihiko Tokuzawa Tokihiko Tokuzawa","Tokuzawa Tokihiko","Tokuzawa, T."],"works_count":671592,"cited_by_count":6540,"summary_stats":{"2yr_mean_citedness":9.238536019859872e-05,"h_index":40,"i10_index":184},"ids":{"openalex":"https://openalex.org/A5043579160","orcid":"https://orcid.org/0000-0001-5473-2109"},"affiliations":[{"institution":{"id":"https://openalex.org/I1289243028","ror":"https://ror.org/01qz5mb56","display_name":"Oak Ridge National Laboratory","country_code":"US","type":"facility","lineage":["https://openalex.org/I1289243028","https://openalex.org/I1330989302","https://openalex.org/I39565521","https://openalex.org/I4210159294"]},"years":[2008]},{"institution":{"id":"https://openalex.org/I135598925","ror":"https://ror.org/00p4k0j84","display_name":"Kyushu University","country_code":"JP","type":"education","lineage":["https://openalex.org/I135598925"]},"years":[2026,2025,2024,2023,2022,2014,2013]},{"institution":{"id":"https://openalex.org/I146399215","ror":"https://ror.org/02956yf07","display_name":"University of Tsukuba","country_code":"JP","type":"education","lineage":["https://openalex.org/I146399215"]},"years":[1999,1998,1997,1996,1995,1994,1993]},{"institution":{"id":"https://openalex.org/I199525922","ror":"https://ror.org/055n47h92","display_name":"National Institutes of Natural Sciences","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922"]},"years":[2026,2025,2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2012,2011,2010,2009,2008,2006,2005,2004,2003,2002,2001,2000]},{"institution":{"id":"https://openalex.org/I200475212","ror":"https://ror.org/0516ah480","display_name":"The Graduate University for Advanced Studies, SOKENDAI","country_code":"JP","type":"education","lineage":["https://openalex.org/I200475212"]},"years":[2026,2025,2024,2023,2022,2021,2020,2019,2018,2016,2015,2010,2008,2007,2006,2005,2003,2000]},{"institution":{"id":"https://openalex.org/I22299242","ror":"https://ror.org/02kpeqv85","display_name":"Kyoto University","country_code":"JP","type":"education","lineage":["https://openalex.org/I22299242"]},"years":[2005]},{"institution":{"id":"https://openalex.org/I2799567181","ror":"https://ror.org/03vn1ts68","display_name":"Princeton Plasma Physics Laboratory","country_code":"US","type":"facility","lineage":["https://openalex.org/I1330989302","https://openalex.org/I20089843","https://openalex.org/I2799567181","https://openalex.org/I39565521"]},"years":[2014]},{"institution":{"id":"https://openalex.org/I4200000001","ror":"https://ror.org/02nr0ka47","display_name":"OpenAlex","country_code":"CA","type":"nonprofit","lineage":["https://openalex.org/I4200000001"]},"years":[2025,2024,2023,2021,2018,2016,2015,2014,2013,2012,2006,2005,2004,2003,2002,1999]},{"institution":{"id":"https://openalex.org/I4210108322","ror":"https://ror.org/01t3wyv61","display_name":"National Institute for Fusion Science","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922","https://openalex.org/I4210108322"]},"years":[2026,2025,2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]},{"institution":{"id":"https://openalex.org/I4210110163","ror":"https://ror.org/01yk36x23","display_name":"Nippon Soken (Japan)","country_code":"JP","type":"company","lineage":["https://openalex.org/I4210110163"]},"years":[2020]},{"institution":{"id":"https://openalex.org/I4210125919","ror":"https://ror.org/02vtgg877","display_name":"Fusion (United States)","country_code":"US","type":"company","lineage":["https://openalex.org/I4210125919"]},"years":[2025,2006]},{"institution":{"id":"https://openalex.org/I4210149442","ror":"https://ror.org/05rwjyj14","display_name":"Fusion Academy","country_code":"US","type":"education","lineage":["https://openalex.org/I4210149442"]},"years":[2025,2021,2006]},{"institution":{"id":"https://openalex.org/I4210158445","ror":"https://ror.org/004tze884","display_name":"Institute of Natural Science","country_code":"KP","type":"education","lineage":["https://openalex.org/I4210158445"]},"years":[2024]},{"institution":{"id":"https://openalex.org/I4843557","ror":"https://ror.org/03e5eem51","display_name":"Budker Institute of Nuclear Physics","country_code":"RU","type":"facility","lineage":["https://openalex.org/I1313323035","https://openalex.org/I1313323035","https://openalex.org/I4210096333","https://openalex.org/I4210127387","https://openalex.org/I4843557"]},"years":[2010]},{"institution":{"id":"https://openalex.org/I50357001","ror":"https://ror.org/03ths8210","display_name":"Universidad Carlos III de Madrid","country_code":"ES","type":"education","lineage":["https://openalex.org/I50357001"]},"years":[2008]},{"institution":{"id":"https://openalex.org/I60134161","ror":"https://ror.org/04chrp450","display_name":"Nagoya University","country_code":"JP","type":"education","lineage":["https://openalex.org/I60134161"]},"years":[2022,2014,2005,2000]},{"institution":{"id":"https://openalex.org/I74801974","ror":"https://ror.org/057zh3y96","display_name":"The University of Tokyo","country_code":"JP","type":"education","lineage":["https://openalex.org/I74801974"]},"years":[2022,2012]}],"last_known_institutions":[{"id":"https://openalex.org/I135598925","ror":"https://ror.org/00p4k0j84","display_name":"Kyushu University","country_code":"JP","type":"education","lineage":["https://openalex.org/I135598925"]},{"id":"https://openalex.org/I199525922","ror":"https://ror.org/055n47h92","display_name":"National Institutes of Natural Sciences","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922"]},{"id":"https://openalex.org/I200475212","ror":"https://ror.org/0516ah480","display_name":"The Graduate University for Advanced Studies, SOKENDAI","country_code":"JP","type":"education","lineage":["https://openalex.org/I200475212"]},{"id":"https://openalex.org/I4210108322","ror":"https://ror.org/01t3wyv61","display_name":"National Institute for Fusion Science","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922","https://openalex.org/I4210108322"]}],"topics":[{"id":"https://openalex.org/T10346","display_name":"Magnetic confinement fusion research","count":379,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10159","display_name":"Ionosphere and magnetosphere dynamics","count":176,"score":0.9998999834060669,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10781","display_name":"Plasma Diagnostics and Applications","count":102,"score":0.9998000264167786,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11367","display_name":"Particle accelerators and beam dynamics","count":102,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10592","display_name":"Fusion materials and technologies","count":96,"score":0.9997000098228455,"subfield":{"id":"https://openalex.org/subfields/2505","display_name":"Materials Chemistry"},"field":{"id":"https://openalex.org/fields/25","display_name":"Materials Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}}],"topic_share":[{"id":"https://openalex.org/T10346","display_name":"Magnetic confinement fusion research","value":0.0016848,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10592","display_name":"Fusion materials and technologies","value":0.0007482,"subfield":{"id":"https://openalex.org/subfields/2505","display_name":"Materials Chemistry"},"field":{"id":"https://openalex.org/fields/25","display_name":"Materials Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10384","display_name":"Laser-Plasma Interactions and Diagnostics","value":0.0005514,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10781","display_name":"Plasma Diagnostics and Applications","value":0.000435,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11367","display_name":"Particle accelerators and beam dynamics","value":0.0003862,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}}],"x_concepts":[{"id":"121332964","wikidata":"https://www.wikidata.org/wiki/Q413","display_name":"Physics","score":0.9342754483222961},{"id":"142757262","wikidata":"https://www.wikidata.org/wiki/Q76436","display_name":"Doppler effect","score":0.8120940923690796},{"id":"192562407","wikidata":"https://www.wikidata.org/wiki/Q228736","display_name":"Materials science","score":0.7516918182373047},{"id":"2779843651","wikidata":"https://www.wikidata.org/wiki/Q7390335","display_name":"SIGNAL (programming language)","score":0.6705514192581177},{"id":"165838908","wikidata":"https://www.wikidata.org/wiki/Q736777","display_name":"Calibration","score":0.6469423770904541}],"counts_by_year":[{"year":1993,"works_count":1,"oa_works_count":0,"cited_by_count":33},{"year":1994,"works_count":1,"oa_works_count":0,"cited_by_count":9},{"year":1995,"works_count":4,"oa_works_count":0,"cited_by_count":38},{"year":1996,"works_count":1,"oa_works_count":0,"cited_by_count":1},{"year":1997,"works_count":9,"oa_works_count":0,"cited_by_count":76},{"year":1998,"works_count":5,"oa_works_count":0,"cited_by_count":19},{"year":1999,"works_count":18,"oa_works_count":3,"cited_by_count":682},{"year":2000,"works_count":15,"oa_works_count":5,"cited_by_count":165},{"year":2001,"works_count":36,"oa_works_count":8,"cited_by_count":933},{"year":2002,"works_count":19,"oa_works_count":4,"cited_by_count":362},{"year":2003,"works_count":30,"oa_works_count":9,"cited_by_count":539},{"year":2004,"works_count":22,"oa_works_count":5,"cited_by_count":374},{"year":2005,"works_count":21,"oa_works_count":6,"cited_by_count":484},{"year":2006,"works_count":34,"oa_works_count":14,"cited_by_count":342},{"year":2007,"works_count":16,"oa_works_count":3,"cited_by_count":231},{"year":2008,"works_count":34,"oa_works_count":16,"cited_by_count":401},{"year":2009,"works_count":6,"oa_works_count":1,"cited_by_count":78},{"year":2010,"works_count":25,"oa_works_count":6,"cited_by_count":344},{"year":2011,"works_count":21,"oa_works_count":1,"cited_by_count":99},{"year":2012,"works_count":16,"oa_works_count":5,"cited_by_count":156},{"year":2013,"works_count":19,"oa_works_count":8,"cited_by_count":167},{"year":2014,"works_count":14,"oa_works_count":5,"cited_by_count":115},{"year":2015,"works_count":21,"oa_works_count":6,"cited_by_count":83},{"year":2016,"works_count":31,"oa_works_count":4,"cited_by_count":69},{"year":2017,"works_count":13,"oa_works_count":6,"cited_by_count":228},{"year":2018,"works_count":12,"oa_works_count":7,"cited_by_count":143},{"year":2019,"works_count":9,"oa_works_count":1,"cited_by_count":120},{"year":2020,"works_count":9,"oa_works_count":4,"cited_by_count":35},{"year":2021,"works_count":8,"oa_works_count":6,"cited_by_count":57},{"year":2022,"works_count":12,"oa_works_count":9,"cited_by_count":83},{"year":2023,"works_count":4,"oa_works_count":3,"cited_by_count":12},{"year":2024,"works_count":13,"oa_works_count":12,"cited_by_count":60},{"year":2025,"works_count":11,"oa_works_count":9,"cited_by_count":2},{"year":2026,"works_count":671078,"oa_works_count":671078,"cited_by_count":0}],"longest_name":"Tokihiko Tokuzawa Tokihiko Tokuzawa","parsed_longest_name":{"first":"tokihiko","middle":"tokuzawa tokihiko","last":"tokuzawa","suffix":"","nickname":""},"block_key":"t tokuzawa","works_api_url":"https://api.openalex.org/works?filter=author.id:A5043579160","updated_date":"2026-02-16T12:19:24","created_date":"2016-06-24T00:00:00"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseAuthors,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "AuthorTable",
                "AuthorDisplayNameAlternativesTable",
                "AuthorAffiliationTable",
                "AuthorLastKnownInstitutionsTable",
                "AuthorIdsTable",
                "AuthorSummaryStatsTable",
                "AuthorCountsByYearTable",
                "AuthorConceptTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            [
                "AuthorTable",
                "AuthorDisplayNameAlternativesTable",
                "AuthorAffiliationTable",
                "AuthorLastKnownInstitutionsTable",
                "AuthorIdsTable",
                "AuthorSummaryStatsTable",
                "AuthorCountsByYearTable",
                "AuthorConceptTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(
            test,
            ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]
        );

        // OpenAlexResponseInstitutions
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":31340,"db_response_time_ms":4,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/I27837315","ror":"https://ror.org/00jmfr291","display_name":"University of Michigan","country_code":"US","type":"education","type_id":"https://openalex.org/institution-types/education","lineage":["https://openalex.org/I27837315"],"homepage_url":"https://www.umich.edu","image_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/University%20of%20Michigan%20logo.svg","image_thumbnail_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/University%20of%20Michigan%20logo.svg&width=300","display_name_acronyms":["UM"],"display_name_alternatives":["UMich","University of Michigan","University of Michigan\u2013Ann Arbor","Universit\u00e9 du Michigan"],"repositories":[{"id":"https://openalex.org/S4306400393","display_name":"Deep Blue (University of Michigan)","host_organization":"https://openalex.org/I27837315","host_organization_name":"University of Michigan","host_organization_lineage":["https://openalex.org/I27837315"]},{"id":"https://openalex.org/S4306400708","display_name":"CINECA IRIS Institutional Research Information System (IRIS Istituto Nazionale di Ricerca Metrologica)","host_organization":"https://openalex.org/I27837315","host_organization_name":"University of Michigan","host_organization_lineage":["https://openalex.org/I27837315"]}],"works_count":941687,"cited_by_count":59852336,"summary_stats":{"2yr_mean_citedness":3.1827793123154584,"h_index":2006,"i10_index":619796},"ids":{"openalex":"https://openalex.org/I27837315","ror":"https://ror.org/00jmfr291","grid":"grid.214458.e","wikipedia":"http://en.wikipedia.org/wiki/University_of_Michigan","wikidata":null},"geo":{"city":"Ann Arbor","geonames_city_id":"4984247","region":"Michigan","country_code":"US","country":"United States","latitude":42.27756,"longitude":-83.74088},"international":{},"associated_institutions":[{"id":"https://openalex.org/I2801799315","ror":"https://ror.org/034npj057","display_name":"Hurley Medical Center","country_code":"US","type":"healthcare","relationship":"related"},{"id":"https://openalex.org/I4210092198","ror":"https://ror.org/01c3xc117","display_name":"University of Michigan\u2013Flint","country_code":"US","type":"education","relationship":"related"},{"id":"https://openalex.org/I4210104572","ror":"https://ror.org/015tnsz82","display_name":"Michigan Sea Grant","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4210114445","ror":"https://ror.org/01zcpa714","display_name":"Michigan Medicine","country_code":"US","type":"healthcare","relationship":"related"},{"id":"https://openalex.org/I4210130704","ror":"https://ror.org/035wtm547","display_name":"University of Michigan\u2013Dearborn","country_code":"US","type":"education","relationship":"related"},{"id":"https://openalex.org/I4210163254","ror":"https://ror.org/057mgcy61","display_name":"Michigan Space Grant Consortium","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4387153780","ror":"https://ror.org/02q7mkh03","display_name":"Inter-university Consortium for Political and Social Research","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4387153798","ror":"https://ror.org/00rx1p510","display_name":"University of Michigan Press","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4387154935","ror":"https://ror.org/04pk7zz41","display_name":"Arctic Long Term Ecological Research","country_code":"US","type":"facility","relationship":"related"},{"id":"https://openalex.org/I4387930282","ror":"https://ror.org/02hhndj92","display_name":"University of Michigan Biological Station","country_code":"US","type":"facility","relationship":"child"},{"id":"https://openalex.org/I4402554065","ror":"https://ror.org/04xgzv028","display_name":"Cooperative Institute for Great Lakes Research","country_code":"US","type":"facility","relationship":"child"},{"id":"https://openalex.org/I4404532909","ror":"https://ror.org/02bkkgm47","display_name":"Center for Complex Particle Systems","country_code":"US","type":"education","relationship":"child"},{"id":"https://openalex.org/I4405258370","ror":"https://ror.org/00hv7q333","display_name":"Zettawatt-Equivalent Ultrashort pulse laser System","country_code":"US","type":"facility","relationship":"child"},{"id":"https://openalex.org/I4405258921","ror":"https://ror.org/01df9hd73","display_name":"Institute for Social Research","country_code":"US","type":"other","relationship":"child"}],"counts_by_year":[{"year":2027,"works_count":4,"oa_works_count":4,"cited_by_count":0},{"year":2026,"works_count":2487,"oa_works_count":1589,"cited_by_count":71},{"year":2025,"works_count":22243,"oa_works_count":14901,"cited_by_count":27126},{"year":2024,"works_count":23505,"oa_works_count":16530,"cited_by_count":110287},{"year":2023,"works_count":23035,"oa_works_count":16404,"cited_by_count":226656},{"year":2022,"works_count":456362,"oa_works_count":448843,"cited_by_count":333830},{"year":2021,"works_count":22297,"oa_works_count":15504,"cited_by_count":484898},{"year":2020,"works_count":25893,"oa_works_count":18305,"cited_by_count":707421},{"year":2019,"works_count":22929,"oa_works_count":15352,"cited_by_count":672563},{"year":2018,"works_count":20675,"oa_works_count":12227,"cited_by_count":719217},{"year":2017,"works_count":18315,"oa_works_count":9827,"cited_by_count":774510},{"year":2016,"works_count":16843,"oa_works_count":9158,"cited_by_count":801111},{"year":2015,"works_count":15713,"oa_works_count":8216,"cited_by_count":879665},{"year":2014,"works_count":15322,"oa_works_count":7746,"cited_by_count":767144},{"year":2013,"works_count":14809,"oa_works_count":7378,"cited_by_count":788449},{"year":2012,"works_count":14549,"oa_works_count":7305,"cited_by_count":810578},{"year":2011,"works_count":13207,"oa_works_count":5956,"cited_by_count":787692},{"year":2010,"works_count":12627,"oa_works_count":5548,"cited_by_count":844939}],"roles":[{"role":"funder","id":"https://openalex.org/F4320309652","works_count":32442},{"role":"institution","id":"https://openalex.org/I27837315","works_count":941687},{"role":"publisher","id":"https://openalex.org/P4310316579","works_count":1272}],"topics":[{"id":"https://openalex.org/T10048","display_name":"Particle physics theoretical and experimental studies","count":6010,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10325","display_name":"Astro and Planetary Science","count":4389,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11949","display_name":"Nuclear Physics and Applications","count":4109,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3108","display_name":"Radiation"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10039","display_name":"Stellar, planetary, and galactic studies","count":3994,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10235","display_name":"Health disparities and outcomes","count":3930,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3306","display_name":"Health"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10391","display_name":"Healthcare Policy and Management","count":3616,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10527","display_name":"High-Energy Particle Collisions Research","count":3600,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10804","display_name":"Health Systems, Economic Evaluations, Quality of Life","count":3410,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10224","display_name":"Quantum Chromodynamics and Particle Interactions","count":3330,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12646","display_name":"Inorganic Fluorides and Related Compounds","count":3260,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1604","display_name":"Inorganic Chemistry"},"field":{"id":"https://openalex.org/fields/16","display_name":"Chemistry"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10522","display_name":"Medical Imaging Techniques and Applications","count":3241,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2741","display_name":"Radiology, Nuclear Medicine and Imaging"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10251","display_name":"Solar and Space Plasma Dynamics","count":3076,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10026","display_name":"Galaxies: Formation, Evolution, Phenomena","count":3066,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10182","display_name":"Child and Adolescent Psychosocial and Emotional Development","count":3007,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3203","display_name":"Clinical Psychology"},"field":{"id":"https://openalex.org/fields/32","display_name":"Psychology"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10159","display_name":"Ionosphere and magnetosphere dynamics","count":2996,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10477","display_name":"Astrophysics and Star Formation Studies","count":2721,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10077","display_name":"Neuroscience and Neuropharmacology Research","count":2703,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2804","display_name":"Cellular and Molecular Neuroscience"},"field":{"id":"https://openalex.org/fields/28","display_name":"Neuroscience"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T10269","display_name":"Epigenetics and DNA Methylation","count":2665,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T12564","display_name":"Sensor Technology and Measurement Systems","count":2596,"score":0.9994000196456909,"subfield":{"id":"https://openalex.org/subfields/1705","display_name":"Computer Networks and Communications"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11930","display_name":"Cardiac, Anesthesia and Surgical Outcomes","count":2552,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2705","display_name":"Cardiology and Cardiovascular Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T12917","display_name":"Astronomy and Astrophysical Research","count":2517,"score":0.9998999834060669,"subfield":{"id":"https://openalex.org/subfields/3105","display_name":"Instrumentation"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10015","display_name":"Genomics and Phylogenetic Studies","count":2512,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T10543","display_name":"Prostate Cancer Treatment and Research","count":2469,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2740","display_name":"Pulmonary and Respiratory Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10645","display_name":"Cardiac Arrest and Resuscitation","count":2447,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2711","display_name":"Emergency Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10521","display_name":"RNA and protein synthesis mechanisms","count":2422,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}}],"topic_share":[{"id":"https://openalex.org/T14440","display_name":"Quality of Life Measurement","value":0.1774779,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T14320","display_name":"Optics and Image Analysis","value":0.039857,"subfield":{"id":"https://openalex.org/subfields/1404","display_name":"Management Information Systems"},"field":{"id":"https://openalex.org/fields/14","display_name":"Business, Management and Accounting"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T12281","display_name":"Animal testing and alternatives","value":0.0361171,"subfield":{"id":"https://openalex.org/subfields/3404","display_name":"Small Animals"},"field":{"id":"https://openalex.org/fields/34","display_name":"Veterinary"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10652","display_name":"Racial and Ethnic Identity Research","value":0.0281192,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T11170","display_name":"Biomimetic flight and propulsion mechanisms","value":0.0269944,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11539","display_name":"Survey Methodology and Nonresponse","value":0.0260274,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T12646","display_name":"Inorganic Fluorides and Related Compounds","value":0.0243669,"subfield":{"id":"https://openalex.org/subfields/1604","display_name":"Inorganic Chemistry"},"field":{"id":"https://openalex.org/fields/16","display_name":"Chemistry"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10843","display_name":"Diversity and Career in Medicine","value":0.0241965,"subfield":{"id":"https://openalex.org/subfields/3318","display_name":"Gender Studies"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T13736","display_name":"Steroid Chemistry and Biochemistry","value":0.0239324,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T11890","display_name":"Scientific Measurement and Uncertainty Evaluation","value":0.0238833,"subfield":{"id":"https://openalex.org/subfields/1804","display_name":"Statistics, Probability and Uncertainty"},"field":{"id":"https://openalex.org/fields/18","display_name":"Decision Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10875","display_name":"Pesticide Exposure and Toxicity","value":0.0237792,"subfield":{"id":"https://openalex.org/subfields/1110","display_name":"Plant Science"},"field":{"id":"https://openalex.org/fields/11","display_name":"Agricultural and Biological Sciences"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T11033","display_name":"Contact Dermatitis and Allergies","value":0.0231777,"subfield":{"id":"https://openalex.org/subfields/2708","display_name":"Dermatology"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T12564","display_name":"Sensor Technology and Measurement Systems","value":0.0228602,"subfield":{"id":"https://openalex.org/subfields/1705","display_name":"Computer Networks and Communications"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13075","display_name":"Legal Systems and Judicial Processes","value":0.0208929,"subfield":{"id":"https://openalex.org/subfields/3320","display_name":"Political Science and International Relations"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T13648","display_name":"Cerebrovascular and genetic disorders","value":0.0203678,"subfield":{"id":"https://openalex.org/subfields/2728","display_name":"Neurology"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10261","display_name":"Genetic Associations and Epidemiology","value":0.0198635,"subfield":{"id":"https://openalex.org/subfields/1311","display_name":"Genetics"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T13138","display_name":"Legal and Constitutional Studies","value":0.0195024,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10235","display_name":"Health disparities and outcomes","value":0.0190722,"subfield":{"id":"https://openalex.org/subfields/3306","display_name":"Health"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10026","display_name":"Galaxies: Formation, Evolution, Phenomena","value":0.0185358,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10391","display_name":"Healthcare Policy and Management","value":0.0184369,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T13856","display_name":"Advanced Power Generation Technologies","value":0.0182906,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10845","display_name":"Advanced Causal Inference Techniques","value":0.0181425,"subfield":{"id":"https://openalex.org/subfields/2613","display_name":"Statistics and Probability"},"field":{"id":"https://openalex.org/fields/26","display_name":"Mathematics"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12174","display_name":"Hospital Admissions and Outcomes","value":0.0180397,"subfield":{"id":"https://openalex.org/subfields/2711","display_name":"Emergency Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T14288","display_name":"Law, Rights, and Freedoms","value":0.0177982,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10543","display_name":"Prostate Cancer Treatment and Research","value":0.0173817,"subfield":{"id":"https://openalex.org/subfields/2740","display_name":"Pulmonary and Respiratory Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}}],"is_super_system":false,"works_api_url":"https://api.openalex.org/works?filter=institutions.id:I27837315","updated_date":"2026-02-17T06:01:00","created_date":"2016-06-24T00:00:00"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseInstitutions,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "InstitutionTable",
                "InstitutionDisplayNameAcronymsTable",
                "InstitutionDisplayNameAlternativesTable",
                "InstitutionGeoTable",
                "InstitutionIdsTable",
                "InstitutionAssociatedInstitutionTable",
                "InstitutionRepositoryTable",
                "InstitutionRoleTable",
                "InstitutionInternationalNamesTable",
                "InstitutionSummaryStatsTable",
                "InstitutionCountsByYearTable",
                "InstitutionLineageTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            [
                "InstitutionTable",
                "InstitutionDisplayNameAcronymsTable",
                "InstitutionDisplayNameAlternativesTable",
                "InstitutionGeoTable",
                "InstitutionIdsTable",
                "InstitutionAssociatedInstitutionTable",
                "InstitutionRepositoryTable",
                "InstitutionRoleTable",
                "InstitutionInternationalNamesTable",
                "InstitutionSummaryStatsTable",
                "InstitutionCountsByYearTable",
                "InstitutionLineageTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(
            test,
            [
                "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"
            ]
        );

        // OpenAlexResponseTopics
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":8,"db_response_time_ms":3,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/T11636","display_name":"Artificial Intelligence in Healthcare and Education","description":"This cluster of papers explores the intersection of artificial intelligence and medicine, focusing on applications in healthcare, medical imaging, clinical decision support, and the ethical challenges associated with AI implementation. It delves into topics such as machine learning, big data, precision medicine, and the potential impact of AI on health equity.","keywords":["Artificial Intelligence","Machine Learning","Healthcare","Medical Imaging","Clinical Decision Support","Ethical Challenges","Big Data","Precision Medicine","Radiology","Health Equity"],"ids":{"openalex":"https://openalex.org/T11636","wikipedia":"https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare"},"subfield":{"id":"https://openalex.org/subfields/2718","display_name":"Health Informatics"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"},"siblings":[],"relevance_score":9815.777,"works_count":105814,"cited_by_count":734445,"works_api_url":"https://api.openalex.org/works?filter=topics.id:T11636","updated_date":"2026-02-17T03:01:20","created_date":"2024-01-23T15:27:11"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseTopics,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "TopicTable",
                "TopicDomainTable",
                "TopicFieldTable",
                "TopicSubfieldTable",
                "TopicIdsTable",
                "TopicKeywordTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            [
                "TopicTable",
                "TopicDomainTable",
                "TopicFieldTable",
                "TopicSubfieldTable",
                "TopicIdsTable",
                "TopicKeywordTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);

        // OpenAlexResponseAwards
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":812094,"db_response_time_ms":89,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/G367708065","display_name":null,"description":null,"funder_award_id":"ACI-1548562","funder":{"id":"https://openalex.org/F4320306076","display_name":"National Science Foundation","doi":"10.13039/100000001"},"funded_outputs":["https://openalex.org/W4229450796","https://openalex.org/W4386332515","https://openalex.org/W2973716325","https://openalex.org/W4280503678","https://openalex.org/W4225427013","https://openalex.org/W4353032408","https://openalex.org/W3040495109","https://openalex.org/W2944375456","https://openalex.org/W4309666031","https://openalex.org/W4382680958","https://openalex.org/W3111929606","https://openalex.org/W2973017822","https://openalex.org/W3090889650","https://openalex.org/W2807017198","https://openalex.org/W3163667393","https://openalex.org/W3193948519","https://openalex.org/W4413938370","https://openalex.org/W2889884874","https://openalex.org/W4306683079","https://openalex.org/W3202649610","https://openalex.org/W2757355312","https://openalex.org/W4284897357","https://openalex.org/W3016474637","https://openalex.org/W4323784182","https://openalex.org/W3186381305","https://openalex.org/W3009185795","https://openalex.org/W4321749159","https://openalex.org/W2808891067","https://openalex.org/W3080795007","https://openalex.org/W2927335729","https://openalex.org/W4312094587","https://openalex.org/W3211615441","https://openalex.org/W3180066382","https://openalex.org/W4380324181","https://openalex.org/W3012279662","https://openalex.org/W4381481438","https://openalex.org/W3212058632","https://openalex.org/W2959589769","https://openalex.org/W3016109661","https://openalex.org/W2995750797","https://openalex.org/W2958180913","https://openalex.org/W2783955689","https://openalex.org/W2808310977","https://openalex.org/W3041099733","https://openalex.org/W3108100525","https://openalex.org/W4221161170","https://openalex.org/W2910451094","https://openalex.org/W2967331743","https://openalex.org/W4283031114","https://openalex.org/W4404746809","https://openalex.org/W4367048807","https://openalex.org/W3049026313","https://openalex.org/W3156392632","https://openalex.org/W4290772317","https://openalex.org/W4405834039","https://openalex.org/W2914190032","https://openalex.org/W2763406770","https://openalex.org/W2966369432","https://openalex.org/W3037833063","https://openalex.org/W4317463145","https://openalex.org/W2921706278","https://openalex.org/W4396934224","https://openalex.org/W4245301932","https://openalex.org/W3009078026","https://openalex.org/W3027352832","https://openalex.org/W3081774741","https://openalex.org/W3094863819","https://openalex.org/W2996598492","https://openalex.org/W2888358433","https://openalex.org/W4323655240","https://openalex.org/W4366442058","https://openalex.org/W2998573338","https://openalex.org/W2793239507","https://openalex.org/W4231415642","https://openalex.org/W3155811562","https://openalex.org/W4311587056","https://openalex.org/W3215839012","https://openalex.org/W2621330091","https://openalex.org/W3014017236","https://openalex.org/W3190572573","https://openalex.org/W2792407267","https://openalex.org/W3043660595","https://openalex.org/W2775304381","https://openalex.org/W3206267578","https://openalex.org/W2981571987","https://openalex.org/W4210272447","https://openalex.org/W4220704665","https://openalex.org/W2739346506","https://openalex.org/W4293194105","https://openalex.org/W4224979504","https://openalex.org/W3217665205","https://openalex.org/W2999310956","https://openalex.org/W2792251869","https://openalex.org/W2896477922","https://openalex.org/W4312206542","https://openalex.org/W4280523132","https://openalex.org/W2943187574","https://openalex.org/W4388671707","https://openalex.org/W4376956016","https://openalex.org/W4294344767"],"funded_outputs_count":1229,"amount":null,"currency":null,"funding_type":null,"funder_scheme":null,"start_date":null,"end_date":null,"start_year":null,"end_year":null,"landing_page_url":null,"doi":null,"provenance":"crossref_work.grants","lead_investigator":null,"co_lead_investigator":null,"investigators":null,"works_api_url":"https://api.openalex.org/works?filter=awards.id:G367708065","updated_date":"2026-02-10T15:33:03","created_date":"2026-01-20T07:04:32"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseAwards,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            ["AwardTable", "AwardFunderTable", "AwardFundedOutputsTable"]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            ["extract_tabular", "extract_tabular", "extract_tabular"]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            ["AwardTable", "AwardFunderTable", "AwardFundedOutputsTable"]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc"]);

        // OpenAlexResponseFunders
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":11178,"db_response_time_ms":7,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/F4320306076","display_name":"National Science Foundation","alternate_titles":["U.S. National Science Foundation","USA NSF","USNSF","NSF","US National Science Foundation","US NSF"],"country_code":"US","description":"United States government agency","homepage_url":"https://www.nsf.gov","image_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/NSF logo.png","image_thumbnail_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/NSF logo.png&width=300","awards_count":812094,"works_count":1483062,"cited_by_count":67229732,"summary_stats":{"2yr_mean_citedness":5.0423750233524185,"h_index":1612,"i10_index":876511},"ids":{"openalex":"https://openalex.org/F4320306076","ror":"https://ror.org/021nxhr62","wikidata":"https://www.wikidata.org/entity/Q304878","crossref":"100000001","doi":"10.13039/100000001"},"counts_by_year":[{"year":2604,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2029,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2026,"works_count":21,"oa_works_count":21,"cited_by_count":17},{"year":2025,"works_count":35573,"oa_works_count":26998,"cited_by_count":69651},{"year":2024,"works_count":55401,"oa_works_count":40081,"cited_by_count":389173},{"year":2023,"works_count":73608,"oa_works_count":56924,"cited_by_count":832747},{"year":2022,"works_count":70667,"oa_works_count":54502,"cited_by_count":1185018},{"year":2021,"works_count":75155,"oa_works_count":58933,"cited_by_count":1759054},{"year":2020,"works_count":74603,"oa_works_count":59843,"cited_by_count":2337598},{"year":2019,"works_count":73517,"oa_works_count":50558,"cited_by_count":2863451},{"year":2018,"works_count":78896,"oa_works_count":47546,"cited_by_count":3452727},{"year":2017,"works_count":76336,"oa_works_count":43598,"cited_by_count":3512939},{"year":2016,"works_count":75071,"oa_works_count":42766,"cited_by_count":3621167},{"year":2015,"works_count":68733,"oa_works_count":37782,"cited_by_count":3422307},{"year":2014,"works_count":63750,"oa_works_count":32548,"cited_by_count":3134829},{"year":2013,"works_count":54489,"oa_works_count":27499,"cited_by_count":2639033},{"year":2012,"works_count":50950,"oa_works_count":24960,"cited_by_count":2688011},{"year":2011,"works_count":47361,"oa_works_count":21774,"cited_by_count":2622788},{"year":2010,"works_count":39086,"oa_works_count":18098,"cited_by_count":2348734},{"year":2009,"works_count":35609,"oa_works_count":16546,"cited_by_count":2235251},{"year":2008,"works_count":33669,"oa_works_count":15116,"cited_by_count":2191220},{"year":2007,"works_count":31175,"oa_works_count":13411,"cited_by_count":2074237},{"year":2006,"works_count":30259,"oa_works_count":11657,"cited_by_count":1994887},{"year":2005,"works_count":27482,"oa_works_count":10274,"cited_by_count":1921585},{"year":2004,"works_count":24097,"oa_works_count":9163,"cited_by_count":1810144},{"year":2003,"works_count":21735,"oa_works_count":7732,"cited_by_count":1728214},{"year":2002,"works_count":20951,"oa_works_count":6786,"cited_by_count":1528195},{"year":2001,"works_count":15891,"oa_works_count":5989,"cited_by_count":1271228},{"year":2000,"works_count":15589,"oa_works_count":5724,"cited_by_count":1233339},{"year":1999,"works_count":14290,"oa_works_count":5245,"cited_by_count":1093380},{"year":1998,"works_count":13405,"oa_works_count":5119,"cited_by_count":964337},{"year":1997,"works_count":12267,"oa_works_count":4674,"cited_by_count":798291},{"year":1996,"works_count":11535,"oa_works_count":4156,"cited_by_count":725080},{"year":1995,"works_count":10133,"oa_works_count":3272,"cited_by_count":590613},{"year":1994,"works_count":9882,"oa_works_count":3094,"cited_by_count":553601},{"year":1993,"works_count":9136,"oa_works_count":2752,"cited_by_count":530105},{"year":1992,"works_count":9058,"oa_works_count":2664,"cited_by_count":543050},{"year":1991,"works_count":8710,"oa_works_count":2586,"cited_by_count":477473},{"year":1990,"works_count":8381,"oa_works_count":2617,"cited_by_count":488273},{"year":1989,"works_count":8002,"oa_works_count":2358,"cited_by_count":463569},{"year":1988,"works_count":7438,"oa_works_count":2300,"cited_by_count":412153},{"year":1987,"works_count":6666,"oa_works_count":2050,"cited_by_count":437093},{"year":1986,"works_count":6283,"oa_works_count":1913,"cited_by_count":366537},{"year":1985,"works_count":5687,"oa_works_count":1767,"cited_by_count":325486},{"year":1984,"works_count":5296,"oa_works_count":1646,"cited_by_count":312923},{"year":1983,"works_count":5192,"oa_works_count":1618,"cited_by_count":321181},{"year":1982,"works_count":4773,"oa_works_count":1515,"cited_by_count":267671},{"year":1981,"works_count":4678,"oa_works_count":1467,"cited_by_count":266349},{"year":1980,"works_count":4435,"oa_works_count":1483,"cited_by_count":228113},{"year":1979,"works_count":4083,"oa_works_count":1414,"cited_by_count":230276},{"year":1978,"works_count":3933,"oa_works_count":1382,"cited_by_count":211910},{"year":1977,"works_count":3533,"oa_works_count":1273,"cited_by_count":250381},{"year":1976,"works_count":3263,"oa_works_count":1195,"cited_by_count":142922},{"year":1975,"works_count":3164,"oa_works_count":1162,"cited_by_count":149342},{"year":1974,"works_count":3203,"oa_works_count":1308,"cited_by_count":138248},{"year":1973,"works_count":3013,"oa_works_count":1174,"cited_by_count":124274},{"year":1972,"works_count":3045,"oa_works_count":1167,"cited_by_count":138055},{"year":1971,"works_count":2560,"oa_works_count":1068,"cited_by_count":121699},{"year":1970,"works_count":2311,"oa_works_count":904,"cited_by_count":86217},{"year":1969,"works_count":2088,"oa_works_count":749,"cited_by_count":90140},{"year":1968,"works_count":1817,"oa_works_count":645,"cited_by_count":68929},{"year":1967,"works_count":1618,"oa_works_count":569,"cited_by_count":71593},{"year":1966,"works_count":1483,"oa_works_count":581,"cited_by_count":63281},{"year":1965,"works_count":1380,"oa_works_count":556,"cited_by_count":61723},{"year":1964,"works_count":1243,"oa_works_count":444,"cited_by_count":50899},{"year":1963,"works_count":1013,"oa_works_count":343,"cited_by_count":50458},{"year":1962,"works_count":790,"oa_works_count":313,"cited_by_count":34067},{"year":1961,"works_count":547,"oa_works_count":197,"cited_by_count":28104},{"year":1960,"works_count":452,"oa_works_count":187,"cited_by_count":23406},{"year":1959,"works_count":318,"oa_works_count":138,"cited_by_count":12729},{"year":1958,"works_count":215,"oa_works_count":90,"cited_by_count":8448},{"year":1957,"works_count":199,"oa_works_count":73,"cited_by_count":6615},{"year":1956,"works_count":124,"oa_works_count":40,"cited_by_count":4299},{"year":1955,"works_count":93,"oa_works_count":26,"cited_by_count":6006},{"year":1954,"works_count":47,"oa_works_count":20,"cited_by_count":1635},{"year":1953,"works_count":25,"oa_works_count":15,"cited_by_count":255},{"year":1952,"works_count":8,"oa_works_count":6,"cited_by_count":0},{"year":1951,"works_count":2,"oa_works_count":1,"cited_by_count":133},{"year":1950,"works_count":3,"oa_works_count":3,"cited_by_count":1},{"year":1948,"works_count":2,"oa_works_count":2,"cited_by_count":0},{"year":1947,"works_count":1,"oa_works_count":0,"cited_by_count":258},{"year":1946,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1945,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1944,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1942,"works_count":1,"oa_works_count":0,"cited_by_count":31},{"year":1941,"works_count":1,"oa_works_count":1,"cited_by_count":1},{"year":1939,"works_count":4,"oa_works_count":4,"cited_by_count":9},{"year":1938,"works_count":1,"oa_works_count":0,"cited_by_count":3},{"year":1933,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1932,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1931,"works_count":2,"oa_works_count":1,"cited_by_count":2},{"year":1930,"works_count":3,"oa_works_count":0,"cited_by_count":1},{"year":1925,"works_count":2,"oa_works_count":2,"cited_by_count":0},{"year":1923,"works_count":2,"oa_works_count":0,"cited_by_count":62},{"year":1920,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":1918,"works_count":1,"oa_works_count":0,"cited_by_count":7},{"year":1916,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1912,"works_count":3,"oa_works_count":1,"cited_by_count":42},{"year":1911,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1909,"works_count":1,"oa_works_count":0,"cited_by_count":12},{"year":1907,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1905,"works_count":2,"oa_works_count":2,"cited_by_count":39},{"year":1904,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":1902,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1901,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1900,"works_count":7,"oa_works_count":5,"cited_by_count":93},{"year":1899,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1898,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1895,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1890,"works_count":2,"oa_works_count":2,"cited_by_count":53},{"year":1885,"works_count":1,"oa_works_count":1,"cited_by_count":1},{"year":1880,"works_count":4,"oa_works_count":1,"cited_by_count":261},{"year":1879,"works_count":2,"oa_works_count":0,"cited_by_count":3},{"year":1870,"works_count":3,"oa_works_count":0,"cited_by_count":9},{"year":1868,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1856,"works_count":1,"oa_works_count":0,"cited_by_count":0}],"roles":[{"role":"funder","id":"https://openalex.org/F4320306076","works_count":1483062},{"role":"institution","id":"https://openalex.org/I1311060795","works_count":11573},{"role":"publisher","id":"https://openalex.org/P4365365866","works_count":1634}],"updated_date":"2026-02-17T04:02:36","created_date":"2023-02-13T20:32:25"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseFunders,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "FunderTable",
                "FunderAlternativeTitlesTable",
                "FunderIdsTable",
                "FunderRoleTable",
                "FunderCountsByYearTable",
                "FunderSummaryStatsTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            [
                "FunderTable",
                "FunderAlternativeTitlesTable",
                "FunderIdsTable",
                "FunderRoleTable",
                "FunderCountsByYearTable",
                "FunderSummaryStatsTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);

        // OpenAlexResponsePublishers
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":1,"db_response_time_ms":2,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/P4310320990","display_name":"Elsevier BV","alternate_titles":["Elsevier","elsevier.com","Elsevier Science","Uitg. Elsevier","\u0627\u0644\u0633\u0641\u06cc\u0631","\u0627\u0644\u0633\u0648\u06cc\u0631","\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0627\u0644\u0632\u0648\u06cc\u0631","\u0644\u0648\u062f\u0648\u06cc\u06a9 \u0627\u0644\u0633\u0641\u06cc\u0631","\u7231\u601d\u552f\u5c14"],"hierarchy_level":0,"parent_publisher":null,"lineage":["https://openalex.org/P4310320990"],"relevance_score":284962.6,"country_codes":["NL"],"homepage_url":"https://www.relx.com","image_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/Elsevier.svg","image_thumbnail_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/Elsevier.svg&width=300","works_count":23655674,"cited_by_count":625228497,"summary_stats":{"2yr_mean_citedness":3.4622258009937292,"h_index":2835,"i10_index":10151843},"ids":{"openalex":"https://openalex.org/P4310320990","ror":"https://ror.org/02scfj030","wikidata":"https://www.wikidata.org/entity/Q746413"},"counts_by_year":[{"year":2026,"works_count":132361,"cited_by_count":2934},{"year":2025,"works_count":1105481,"cited_by_count":1568829},{"year":2024,"works_count":1019915,"cited_by_count":6245105},{"year":2023,"works_count":905670,"cited_by_count":10010090},{"year":2022,"works_count":854619,"cited_by_count":13580974},{"year":2021,"works_count":846284,"cited_by_count":17740053},{"year":2020,"works_count":845947,"cited_by_count":22691156},{"year":2019,"works_count":777403,"cited_by_count":20660655},{"year":2018,"works_count":754286,"cited_by_count":21294221},{"year":2017,"works_count":779143,"cited_by_count":21583100},{"year":2016,"works_count":739657,"cited_by_count":21215879},{"year":2015,"works_count":686108,"cited_by_count":20660202},{"year":2014,"works_count":656717,"cited_by_count":20247736}],"roles":[{"role":"funder","id":"https://openalex.org/F4320308305","works_count":133},{"role":"institution","id":"https://openalex.org/I1318003438","works_count":2611},{"role":"institution","id":"https://openalex.org/I4210160603","works_count":658},{"role":"publisher","id":"https://openalex.org/P4310320990","works_count":23655674}],"sources_api_url":"https://api.openalex.org/sources?data-version=2&filter=host_organization.id:P4310320990","updated_date":"2026-02-17T15:56:39","created_date":"2023-01-01T00:00:00"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponsePublishers,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "PublisherTable",
                "PublisherAlternativeTitlesTable",
                "PublisherCountryCodeTable",
                "PublisherLineageTable",
                "PublisherIdsTable",
                "PublisherRoleTable",
                "PublisherCountsByYearTable",
                "PublisherSummaryStatsTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            [
                "PublisherTable",
                "PublisherAlternativeTitlesTable",
                "PublisherCountryCodeTable",
                "PublisherLineageTable",
                "PublisherIdsTable",
                "PublisherRoleTable",
                "PublisherCountsByYearTable",
                "PublisherSummaryStatsTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(
            test,
            ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]
        );

        // OpenAlexResponseSources
        println!("OpenAlexResponseSources");
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":163192,"db_response_time_ms":38,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/S4210203682","issn_l":"0366-4457","issn":["0366-4457"],"display_name":"Bulletin of Miscellaneous Information (Royal Gardens Kew)","host_organization":null,"host_organization_name":"JSTOR","host_organization_lineage":[null],"works_count":7180292,"oa_works_count":90593,"cited_by_count":546522,"summary_stats":{"2yr_mean_citedness":0.026968122578372666,"h_index":248,"i10_index":8763},"is_oa":false,"is_in_doaj":false,"is_in_doaj_since_year":null,"is_high_oa_rate":false,"is_high_oa_rate_since_year":null,"is_in_scielo":false,"is_ojs":false,"is_core":false,"oa_flip_year":null,"first_publication_year":0,"last_publication_year":2026,"ids":{"openalex":"https://openalex.org/S4210203682","issn_l":"0366-4457","issn":["0366-4457"],"mag":"4210203682","wikidata":"https://www.wikidata.org/entity/Q5735532"},"homepage_url":"http://www.archive.org/details/mobot31753002257050","apc_prices":[],"apc_usd":null,"country_code":"GB","societies":[],"alternate_titles":["Bulletin of miscellaneous Information, Kew"],"type":"journal","topics":[{"id":"https://openalex.org/T10346","display_name":"Magnetic confinement fusion research","count":954698,"score":0.9980999827384949,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11367","display_name":"Particle accelerators and beam dynamics","count":467294,"score":0.9815000295639038,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10592","display_name":"Fusion materials and technologies","count":270780,"score":0.998199999332428,"subfield":{"id":"https://openalex.org/subfields/2505","display_name":"Materials Chemistry"},"field":{"id":"https://openalex.org/fields/25","display_name":"Materials Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11808","display_name":"Superconducting Materials and Applications","count":206045,"score":0.9955999851226807,"subfield":{"id":"https://openalex.org/subfields/2204","display_name":"Biomedical Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13370","display_name":"Diverse Scientific and Economic Studies","count":205573,"score":0.6348000168800354,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10978","display_name":"Prenatal Screening and Diagnostics","count":178156,"score":0.9941999912261963,"subfield":{"id":"https://openalex.org/subfields/2735","display_name":"Pediatrics, Perinatology and Child Health"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10781","display_name":"Plasma Diagnostics and Applications","count":173658,"score":0.984499990940094,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12692","display_name":"Magnetic Field Sensors Techniques","count":173481,"score":0.9957000017166138,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11993","display_name":"Atomic and Subatomic Physics Research","count":141827,"score":0.9990000128746033,"subfield":{"id":"https://openalex.org/subfields/3107","display_name":"Atomic and Molecular Physics, and Optics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12552","display_name":"Fetal and Pediatric Neurological Disorders","count":101815,"score":0.9976000189781189,"subfield":{"id":"https://openalex.org/subfields/2735","display_name":"Pediatrics, Perinatology and Child Health"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T11986","display_name":"Scientific Computing and Data Management","count":99901,"score":0.9908000230789185,"subfield":{"id":"https://openalex.org/subfields/1802","display_name":"Information Systems and Management"},"field":{"id":"https://openalex.org/fields/18","display_name":"Decision Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T11044","display_name":"Particle Detector Development and Performance","count":95009,"score":0.9958000183105469,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10159","display_name":"Ionosphere and magnetosphere dynamics","count":88053,"score":0.9994999766349792,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11841","display_name":"Nuclear Structure and Function","count":86806,"score":0.9991000294685364,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T10251","display_name":"Solar and Space Plasma Dynamics","count":80114,"score":0.9987000226974487,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11603","display_name":"Dust and Plasma Wave Phenomena","count":79902,"score":0.9994000196456909,"subfield":{"id":"https://openalex.org/subfields/3107","display_name":"Atomic and Molecular Physics, and Optics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13398","display_name":"Data Analysis with R","count":75357,"score":0.9018999934196472,"subfield":{"id":"https://openalex.org/subfields/1702","display_name":"Artificial Intelligence"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12859","display_name":"Cell Image Analysis Techniques","count":73105,"score":0.9970999956130981,"subfield":{"id":"https://openalex.org/subfields/1304","display_name":"Biophysics"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T12760","display_name":"Laser Design and Applications","count":72307,"score":0.9976999759674072,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13650","display_name":"Computational Physics and Python Applications","count":70311,"score":0.9363999962806702,"subfield":{"id":"https://openalex.org/subfields/1702","display_name":"Artificial Intelligence"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10559","display_name":"Particle Accelerators and Free-Electron Lasers","count":68660,"score":0.9896000027656555,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10384","display_name":"Laser-Plasma Interactions and Diagnostics","count":63449,"score":0.9980999827384949,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12917","display_name":"Astronomy and Astrophysical Research","count":62173,"score":0.9937000274658203,"subfield":{"id":"https://openalex.org/subfields/3105","display_name":"Instrumentation"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11175","display_name":"Gyrotron and Vacuum Electronics Research","count":61856,"score":0.996999979019165,"subfield":{"id":"https://openalex.org/subfields/3107","display_name":"Atomic and Molecular Physics, and Optics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12019","display_name":"Calibration and Measurement Techniques","count":60325,"score":0.9930999875068665,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}}],"topic_share":[{"id":"https://openalex.org/T13436","display_name":"Space Technology and Applications","value":0.1691124,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12692","display_name":"Magnetic Field Sensors Techniques","value":0.1584834,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12762","display_name":"Crystallography and Radiation Phenomena","value":0.1544982,"subfield":{"id":"https://openalex.org/subfields/3104","display_name":"Condensed Matter Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13418","display_name":"Photocathodes and Microchannel Plates","value":0.1281034,"subfield":{"id":"https://openalex.org/subfields/2204","display_name":"Biomedical Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12019","display_name":"Calibration and Measurement Techniques","value":0.1204871,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12300","display_name":"Advanced Electrical Measurement Techniques","value":0.1131355,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13592","display_name":"Advanced Scientific Techniques and Applications","value":0.1062319,"subfield":{"id":"https://openalex.org/subfields/2308","display_name":"Management, Monitoring, Policy and Law"},"field":{"id":"https://openalex.org/fields/23","display_name":"Environmental Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11841","display_name":"Nuclear Structure and Function","value":0.105413,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T11993","display_name":"Atomic and Subatomic Physics Research","value":0.099937,"subfield":{"id":"https://openalex.org/subfields/3107","display_name":"Atomic and Molecular Physics, and Optics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13500","display_name":"Probability and Statistical Research","value":0.098291,"subfield":{"id":"https://openalex.org/subfields/2613","display_name":"Statistics and Probability"},"field":{"id":"https://openalex.org/fields/26","display_name":"Mathematics"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10346","display_name":"Magnetic confinement fusion research","value":0.0963214,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11290","display_name":"Preterm Birth and Chorioamnionitis","value":0.0954082,"subfield":{"id":"https://openalex.org/subfields/2713","display_name":"Epidemiology"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10592","display_name":"Fusion materials and technologies","value":0.0953928,"subfield":{"id":"https://openalex.org/subfields/2505","display_name":"Materials Chemistry"},"field":{"id":"https://openalex.org/fields/25","display_name":"Materials Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10624","display_name":"Silicon and Solar Cell Technologies","value":0.0952839,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11005","display_name":"Radiation Effects in Electronics","value":0.094685,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11367","display_name":"Particle accelerators and beam dynamics","value":0.094576,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T14372","display_name":"Environmental and Biological Research in Conflict Zones","value":0.0931541,"subfield":{"id":"https://openalex.org/subfields/2308","display_name":"Management, Monitoring, Policy and Law"},"field":{"id":"https://openalex.org/fields/23","display_name":"Environmental Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11808","display_name":"Superconducting Materials and Applications","value":0.0924136,"subfield":{"id":"https://openalex.org/subfields/2204","display_name":"Biomedical Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T14163","display_name":"Astronomical Observations and Instrumentation","value":0.0877892,"subfield":{"id":"https://openalex.org/subfields/2206","display_name":"Computational Mechanics"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13332","display_name":"Historical Studies on Reproduction, Gender, Health, and Societal Changes","value":0.0868176,"subfield":{"id":"https://openalex.org/subfields/1202","display_name":"History"},"field":{"id":"https://openalex.org/fields/12","display_name":"Arts and Humanities"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T12930","display_name":"Biomedical and Chemical Research","value":0.0864934,"subfield":{"id":"https://openalex.org/subfields/2746","display_name":"Surgery"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T11603","display_name":"Dust and Plasma Wave Phenomena","value":0.0853494,"subfield":{"id":"https://openalex.org/subfields/3107","display_name":"Atomic and Molecular Physics, and Optics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13898","display_name":"Diverse Interdisciplinary Research Studies","value":0.0849673,"subfield":{"id":"https://openalex.org/subfields/1702","display_name":"Artificial Intelligence"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12254","display_name":"Machine Learning in Bioinformatics","value":0.0847067,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T14455","display_name":"Technology and Education Systems","value":0.0844656,"subfield":{"id":"https://openalex.org/subfields/1705","display_name":"Computer Networks and Communications"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}}],"counts_by_year":[{"year":9999,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":9771,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":9651,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":9498,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":9011,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":9005,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8915,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8867,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8858,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8746,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8718,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8710,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8706,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8553,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8539,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8461,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8430,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8333,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":8248,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8157,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8145,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":8059,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":7945,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":7915,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":7777,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":7641,"works_count":10,"oa_works_count":0,"cited_by_count":0},{"year":7515,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":6409,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5998,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5681,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5669,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5668,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5667,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5661,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5660,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":5600,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5589,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5574,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":4666,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":4215,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":4008,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3320,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3101,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3023,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3011,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3010,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3003,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":3000,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2996,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2924,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2913,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2910,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2858,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2703,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":2556,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":2555,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":2554,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2553,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":2552,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2501,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2453,"works_count":1,"oa_works_count":0,"cited_by_count":2},{"year":2436,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2371,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2333,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2222,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":2211,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2208,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2207,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2206,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2110,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2101,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2100,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2099,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2077,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2070,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":2066,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":2063,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":2058,"works_count":6,"oa_works_count":2,"cited_by_count":0},{"year":2057,"works_count":5,"oa_works_count":1,"cited_by_count":0},{"year":2045,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2040,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2037,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":2036,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2034,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2033,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":2031,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2030,"works_count":6,"oa_works_count":4,"cited_by_count":0},{"year":2029,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2028,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":2026,"works_count":7,"oa_works_count":2,"cited_by_count":0},{"year":2025,"works_count":250,"oa_works_count":6,"cited_by_count":1},{"year":2024,"works_count":45037,"oa_works_count":4855,"cited_by_count":1222},{"year":2023,"works_count":73598,"oa_works_count":7029,"cited_by_count":708},{"year":2022,"works_count":85402,"oa_works_count":612,"cited_by_count":197},{"year":2021,"works_count":84402,"oa_works_count":1038,"cited_by_count":2053},{"year":2020,"works_count":83063,"oa_works_count":1602,"cited_by_count":1044},{"year":2019,"works_count":62144,"oa_works_count":1749,"cited_by_count":2180},{"year":2018,"works_count":47355,"oa_works_count":1431,"cited_by_count":6045},{"year":2017,"works_count":59035,"oa_works_count":644,"cited_by_count":3396},{"year":2016,"works_count":62119,"oa_works_count":493,"cited_by_count":3074},{"year":2015,"works_count":70322,"oa_works_count":325,"cited_by_count":2424},{"year":2014,"works_count":89003,"oa_works_count":272,"cited_by_count":4411},{"year":2013,"works_count":68659,"oa_works_count":365,"cited_by_count":4435},{"year":2012,"works_count":62272,"oa_works_count":432,"cited_by_count":8616},{"year":2011,"works_count":83424,"oa_works_count":491,"cited_by_count":13117},{"year":2010,"works_count":47190,"oa_works_count":1606,"cited_by_count":30174},{"year":2009,"works_count":35891,"oa_works_count":1274,"cited_by_count":25907},{"year":2008,"works_count":32775,"oa_works_count":933,"cited_by_count":16530},{"year":2007,"works_count":38566,"oa_works_count":952,"cited_by_count":17026},{"year":2006,"works_count":37732,"oa_works_count":502,"cited_by_count":5181},{"year":2005,"works_count":26245,"oa_works_count":978,"cited_by_count":5289},{"year":2004,"works_count":21130,"oa_works_count":533,"cited_by_count":3864},{"year":2003,"works_count":18428,"oa_works_count":516,"cited_by_count":3336},{"year":2002,"works_count":17515,"oa_works_count":498,"cited_by_count":3804},{"year":2001,"works_count":17167,"oa_works_count":436,"cited_by_count":3417},{"year":2000,"works_count":15834,"oa_works_count":205,"cited_by_count":4287},{"year":1999,"works_count":12161,"oa_works_count":109,"cited_by_count":4118},{"year":1998,"works_count":9506,"oa_works_count":127,"cited_by_count":4044},{"year":1997,"works_count":9604,"oa_works_count":96,"cited_by_count":3056},{"year":1996,"works_count":10135,"oa_works_count":125,"cited_by_count":2953},{"year":1995,"works_count":10257,"oa_works_count":110,"cited_by_count":5237},{"year":1994,"works_count":10435,"oa_works_count":185,"cited_by_count":4700},{"year":1993,"works_count":10561,"oa_works_count":158,"cited_by_count":6574},{"year":1992,"works_count":12910,"oa_works_count":162,"cited_by_count":7437},{"year":1991,"works_count":11174,"oa_works_count":167,"cited_by_count":3488},{"year":1990,"works_count":9745,"oa_works_count":131,"cited_by_count":7504},{"year":1989,"works_count":8862,"oa_works_count":182,"cited_by_count":5571},{"year":1988,"works_count":9729,"oa_works_count":124,"cited_by_count":5294},{"year":1987,"works_count":8155,"oa_works_count":110,"cited_by_count":4412},{"year":1986,"works_count":7506,"oa_works_count":127,"cited_by_count":4726},{"year":1985,"works_count":9027,"oa_works_count":108,"cited_by_count":5394},{"year":1984,"works_count":8139,"oa_works_count":128,"cited_by_count":5339},{"year":1983,"works_count":7288,"oa_works_count":100,"cited_by_count":5018},{"year":1982,"works_count":8170,"oa_works_count":77,"cited_by_count":4054},{"year":1981,"works_count":9125,"oa_works_count":114,"cited_by_count":4444},{"year":1980,"works_count":9052,"oa_works_count":87,"cited_by_count":6033},{"year":1979,"works_count":6512,"oa_works_count":105,"cited_by_count":4323},{"year":1978,"works_count":7378,"oa_works_count":87,"cited_by_count":4613},{"year":1977,"works_count":6487,"oa_works_count":85,"cited_by_count":5203},{"year":1976,"works_count":10555,"oa_works_count":96,"cited_by_count":14055},{"year":1975,"works_count":8963,"oa_works_count":127,"cited_by_count":4723},{"year":1974,"works_count":8780,"oa_works_count":111,"cited_by_count":6926},{"year":1973,"works_count":8785,"oa_works_count":175,"cited_by_count":3802},{"year":1972,"works_count":6661,"oa_works_count":76,"cited_by_count":8075},{"year":1971,"works_count":5719,"oa_works_count":276,"cited_by_count":5198},{"year":1970,"works_count":4746,"oa_works_count":122,"cited_by_count":6623},{"year":1969,"works_count":4132,"oa_works_count":441,"cited_by_count":10825},{"year":1968,"works_count":3920,"oa_works_count":266,"cited_by_count":3840},{"year":1967,"works_count":4196,"oa_works_count":230,"cited_by_count":3717},{"year":1966,"works_count":3521,"oa_works_count":418,"cited_by_count":2787},{"year":1965,"works_count":3459,"oa_works_count":167,"cited_by_count":2797},{"year":1964,"works_count":2827,"oa_works_count":63,"cited_by_count":3248},{"year":1963,"works_count":3167,"oa_works_count":71,"cited_by_count":4449},{"year":1962,"works_count":2956,"oa_works_count":67,"cited_by_count":2781},{"year":1961,"works_count":3035,"oa_works_count":75,"cited_by_count":2239},{"year":1960,"works_count":3661,"oa_works_count":68,"cited_by_count":1163},{"year":1959,"works_count":2895,"oa_works_count":101,"cited_by_count":3199},{"year":1958,"works_count":3557,"oa_works_count":60,"cited_by_count":2735},{"year":1957,"works_count":5322,"oa_works_count":73,"cited_by_count":1285},{"year":1956,"works_count":6366,"oa_works_count":75,"cited_by_count":2124},{"year":1955,"works_count":7671,"oa_works_count":65,"cited_by_count":2988},{"year":1954,"works_count":8668,"oa_works_count":45,"cited_by_count":824},{"year":1953,"works_count":11807,"oa_works_count":40,"cited_by_count":2986},{"year":1952,"works_count":13662,"oa_works_count":59,"cited_by_count":518},{"year":1951,"works_count":10390,"oa_works_count":46,"cited_by_count":4526},{"year":1950,"works_count":12248,"oa_works_count":67,"cited_by_count":921},{"year":1949,"works_count":10475,"oa_works_count":51,"cited_by_count":1344},{"year":1948,"works_count":8774,"oa_works_count":66,"cited_by_count":1327},{"year":1947,"works_count":13038,"oa_works_count":161,"cited_by_count":2119},{"year":1946,"works_count":10815,"oa_works_count":177,"cited_by_count":900},{"year":1945,"works_count":9941,"oa_works_count":670,"cited_by_count":372},{"year":1944,"works_count":8709,"oa_works_count":1082,"cited_by_count":381},{"year":1943,"works_count":5760,"oa_works_count":816,"cited_by_count":533},{"year":1942,"works_count":7275,"oa_works_count":295,"cited_by_count":649},{"year":1941,"works_count":9907,"oa_works_count":136,"cited_by_count":1890},{"year":1940,"works_count":9952,"oa_works_count":185,"cited_by_count":767},{"year":1939,"works_count":9596,"oa_works_count":69,"cited_by_count":1507},{"year":1938,"works_count":8218,"oa_works_count":64,"cited_by_count":650},{"year":1937,"works_count":7596,"oa_works_count":51,"cited_by_count":1531},{"year":1936,"works_count":6677,"oa_works_count":52,"cited_by_count":1167},{"year":1935,"works_count":6262,"oa_works_count":44,"cited_by_count":514},{"year":1934,"works_count":6904,"oa_works_count":61,"cited_by_count":480},{"year":1933,"works_count":4898,"oa_works_count":53,"cited_by_count":1004},{"year":1932,"works_count":4739,"oa_works_count":63,"cited_by_count":675},{"year":1931,"works_count":4935,"oa_works_count":64,"cited_by_count":531},{"year":1930,"works_count":7019,"oa_works_count":54,"cited_by_count":934},{"year":1929,"works_count":8817,"oa_works_count":56,"cited_by_count":793},{"year":1928,"works_count":9817,"oa_works_count":71,"cited_by_count":4495},{"year":1927,"works_count":9479,"oa_works_count":73,"cited_by_count":3141},{"year":1926,"works_count":8465,"oa_works_count":67,"cited_by_count":1132},{"year":1925,"works_count":7516,"oa_works_count":70,"cited_by_count":546},{"year":1924,"works_count":6605,"oa_works_count":69,"cited_by_count":848},{"year":1923,"works_count":8043,"oa_works_count":83,"cited_by_count":926},{"year":1922,"works_count":12260,"oa_works_count":488,"cited_by_count":4453},{"year":1921,"works_count":10052,"oa_works_count":433,"cited_by_count":1982},{"year":1920,"works_count":11044,"oa_works_count":499,"cited_by_count":2274},{"year":1919,"works_count":9095,"oa_works_count":427,"cited_by_count":1851},{"year":1918,"works_count":7861,"oa_works_count":429,"cited_by_count":1618},{"year":1917,"works_count":8918,"oa_works_count":474,"cited_by_count":2211},{"year":1916,"works_count":8953,"oa_works_count":464,"cited_by_count":1604},{"year":1915,"works_count":8182,"oa_works_count":476,"cited_by_count":2143},{"year":1914,"works_count":8482,"oa_works_count":502,"cited_by_count":1790},{"year":1913,"works_count":8264,"oa_works_count":503,"cited_by_count":1920},{"year":1912,"works_count":7792,"oa_works_count":502,"cited_by_count":2386},{"year":1911,"works_count":7491,"oa_works_count":474,"cited_by_count":1952},{"year":1910,"works_count":7958,"oa_works_count":484,"cited_by_count":1659},{"year":1909,"works_count":6983,"oa_works_count":521,"cited_by_count":2921},{"year":1908,"works_count":7587,"oa_works_count":474,"cited_by_count":2133},{"year":1907,"works_count":7199,"oa_works_count":467,"cited_by_count":1570},{"year":1906,"works_count":6437,"oa_works_count":453,"cited_by_count":2392},{"year":1905,"works_count":7300,"oa_works_count":425,"cited_by_count":1588},{"year":1904,"works_count":6310,"oa_works_count":406,"cited_by_count":2513},{"year":1903,"works_count":6238,"oa_works_count":404,"cited_by_count":1881},{"year":1902,"works_count":6129,"oa_works_count":375,"cited_by_count":2025},{"year":1901,"works_count":6212,"oa_works_count":368,"cited_by_count":1677},{"year":1900,"works_count":7037,"oa_works_count":352,"cited_by_count":2060},{"year":1899,"works_count":6591,"oa_works_count":396,"cited_by_count":1485},{"year":1898,"works_count":5798,"oa_works_count":340,"cited_by_count":1828},{"year":1897,"works_count":5605,"oa_works_count":315,"cited_by_count":1833},{"year":1896,"works_count":5899,"oa_works_count":344,"cited_by_count":1938},{"year":1895,"works_count":5719,"oa_works_count":338,"cited_by_count":3994},{"year":1894,"works_count":5147,"oa_works_count":300,"cited_by_count":1990},{"year":1893,"works_count":5693,"oa_works_count":347,"cited_by_count":1292},{"year":1892,"works_count":5587,"oa_works_count":321,"cited_by_count":1616},{"year":1891,"works_count":5327,"oa_works_count":307,"cited_by_count":1485},{"year":1890,"works_count":5310,"oa_works_count":338,"cited_by_count":1249},{"year":1889,"works_count":5083,"oa_works_count":304,"cited_by_count":1487},{"year":1888,"works_count":4848,"oa_works_count":309,"cited_by_count":1282},{"year":1887,"works_count":4821,"oa_works_count":272,"cited_by_count":1574},{"year":1886,"works_count":4824,"oa_works_count":238,"cited_by_count":897},{"year":1885,"works_count":4877,"oa_works_count":251,"cited_by_count":2261},{"year":1884,"works_count":4957,"oa_works_count":259,"cited_by_count":620},{"year":1883,"works_count":4677,"oa_works_count":221,"cited_by_count":675},{"year":1882,"works_count":5052,"oa_works_count":208,"cited_by_count":735},{"year":1881,"works_count":4463,"oa_works_count":223,"cited_by_count":736},{"year":1880,"works_count":4802,"oa_works_count":168,"cited_by_count":913},{"year":1879,"works_count":4667,"oa_works_count":199,"cited_by_count":835},{"year":1878,"works_count":3987,"oa_works_count":175,"cited_by_count":969},{"year":1877,"works_count":3845,"oa_works_count":161,"cited_by_count":547},{"year":1876,"works_count":3901,"oa_works_count":267,"cited_by_count":621},{"year":1875,"works_count":3742,"oa_works_count":173,"cited_by_count":1158},{"year":1874,"works_count":3608,"oa_works_count":171,"cited_by_count":630},{"year":1873,"works_count":3439,"oa_works_count":172,"cited_by_count":665},{"year":1872,"works_count":3478,"oa_works_count":183,"cited_by_count":728},{"year":1871,"works_count":3030,"oa_works_count":153,"cited_by_count":554},{"year":1870,"works_count":3329,"oa_works_count":171,"cited_by_count":1366},{"year":1869,"works_count":2724,"oa_works_count":176,"cited_by_count":730},{"year":1868,"works_count":2729,"oa_works_count":155,"cited_by_count":378},{"year":1867,"works_count":2666,"oa_works_count":158,"cited_by_count":274},{"year":1866,"works_count":2166,"oa_works_count":178,"cited_by_count":529},{"year":1865,"works_count":2625,"oa_works_count":232,"cited_by_count":410},{"year":1864,"works_count":2402,"oa_works_count":224,"cited_by_count":1304},{"year":1863,"works_count":2424,"oa_works_count":209,"cited_by_count":222},{"year":1862,"works_count":2192,"oa_works_count":186,"cited_by_count":209},{"year":1861,"works_count":2685,"oa_works_count":193,"cited_by_count":365},{"year":1860,"works_count":3103,"oa_works_count":173,"cited_by_count":389},{"year":1859,"works_count":2281,"oa_works_count":131,"cited_by_count":838},{"year":1858,"works_count":2270,"oa_works_count":129,"cited_by_count":333},{"year":1857,"works_count":2690,"oa_works_count":130,"cited_by_count":516},{"year":1856,"works_count":2774,"oa_works_count":174,"cited_by_count":304},{"year":1855,"works_count":2047,"oa_works_count":128,"cited_by_count":241},{"year":1854,"works_count":2280,"oa_works_count":130,"cited_by_count":1072},{"year":1853,"works_count":2490,"oa_works_count":146,"cited_by_count":702},{"year":1852,"works_count":2307,"oa_works_count":156,"cited_by_count":413},{"year":1851,"works_count":2176,"oa_works_count":116,"cited_by_count":506},{"year":1850,"works_count":2150,"oa_works_count":153,"cited_by_count":410},{"year":1849,"works_count":1715,"oa_works_count":97,"cited_by_count":698},{"year":1848,"works_count":1892,"oa_works_count":130,"cited_by_count":462},{"year":1847,"works_count":1897,"oa_works_count":108,"cited_by_count":518},{"year":1846,"works_count":1761,"oa_works_count":101,"cited_by_count":296},{"year":1845,"works_count":1765,"oa_works_count":100,"cited_by_count":264},{"year":1844,"works_count":1732,"oa_works_count":111,"cited_by_count":359},{"year":1843,"works_count":1643,"oa_works_count":94,"cited_by_count":523},{"year":1842,"works_count":1772,"oa_works_count":70,"cited_by_count":229},{"year":1841,"works_count":1752,"oa_works_count":91,"cited_by_count":290},{"year":1840,"works_count":1891,"oa_works_count":100,"cited_by_count":454},{"year":1839,"works_count":1725,"oa_works_count":85,"cited_by_count":632},{"year":1838,"works_count":1529,"oa_works_count":93,"cited_by_count":238},{"year":1837,"works_count":1440,"oa_works_count":76,"cited_by_count":316},{"year":1836,"works_count":1573,"oa_works_count":88,"cited_by_count":246},{"year":1835,"works_count":1234,"oa_works_count":82,"cited_by_count":395},{"year":1834,"works_count":1372,"oa_works_count":76,"cited_by_count":90},{"year":1833,"works_count":1443,"oa_works_count":86,"cited_by_count":300},{"year":1832,"works_count":1280,"oa_works_count":65,"cited_by_count":355},{"year":1831,"works_count":1202,"oa_works_count":71,"cited_by_count":94},{"year":1830,"works_count":1305,"oa_works_count":70,"cited_by_count":226},{"year":1829,"works_count":1364,"oa_works_count":148,"cited_by_count":106},{"year":1828,"works_count":1254,"oa_works_count":190,"cited_by_count":432},{"year":1827,"works_count":1228,"oa_works_count":142,"cited_by_count":251},{"year":1826,"works_count":1105,"oa_works_count":124,"cited_by_count":140},{"year":1825,"works_count":1077,"oa_works_count":145,"cited_by_count":119},{"year":1824,"works_count":1074,"oa_works_count":122,"cited_by_count":198},{"year":1823,"works_count":910,"oa_works_count":114,"cited_by_count":62},{"year":1822,"works_count":943,"oa_works_count":104,"cited_by_count":240},{"year":1821,"works_count":836,"oa_works_count":94,"cited_by_count":125},{"year":1820,"works_count":883,"oa_works_count":120,"cited_by_count":119},{"year":1819,"works_count":948,"oa_works_count":111,"cited_by_count":76},{"year":1818,"works_count":773,"oa_works_count":62,"cited_by_count":216},{"year":1817,"works_count":615,"oa_works_count":56,"cited_by_count":139},{"year":1816,"works_count":623,"oa_works_count":50,"cited_by_count":322},{"year":1815,"works_count":576,"oa_works_count":33,"cited_by_count":228},{"year":1814,"works_count":533,"oa_works_count":41,"cited_by_count":161},{"year":1813,"works_count":527,"oa_works_count":49,"cited_by_count":181},{"year":1812,"works_count":566,"oa_works_count":43,"cited_by_count":50},{"year":1811,"works_count":523,"oa_works_count":39,"cited_by_count":73},{"year":1810,"works_count":645,"oa_works_count":58,"cited_by_count":332},{"year":1809,"works_count":487,"oa_works_count":49,"cited_by_count":104},{"year":1808,"works_count":497,"oa_works_count":48,"cited_by_count":129},{"year":1807,"works_count":446,"oa_works_count":30,"cited_by_count":57},{"year":1806,"works_count":471,"oa_works_count":23,"cited_by_count":36},{"year":1805,"works_count":424,"oa_works_count":27,"cited_by_count":459},{"year":1804,"works_count":354,"oa_works_count":26,"cited_by_count":67},{"year":1803,"works_count":415,"oa_works_count":23,"cited_by_count":75},{"year":1802,"works_count":447,"oa_works_count":25,"cited_by_count":137},{"year":1801,"works_count":343,"oa_works_count":16,"cited_by_count":80},{"year":1800,"works_count":858,"oa_works_count":61,"cited_by_count":76},{"year":1799,"works_count":316,"oa_works_count":10,"cited_by_count":25},{"year":1798,"works_count":311,"oa_works_count":6,"cited_by_count":18},{"year":1797,"works_count":244,"oa_works_count":8,"cited_by_count":77},{"year":1796,"works_count":363,"oa_works_count":12,"cited_by_count":4},{"year":1795,"works_count":247,"oa_works_count":13,"cited_by_count":21},{"year":1794,"works_count":326,"oa_works_count":14,"cited_by_count":6},{"year":1793,"works_count":309,"oa_works_count":8,"cited_by_count":18},{"year":1792,"works_count":333,"oa_works_count":20,"cited_by_count":16},{"year":1791,"works_count":314,"oa_works_count":11,"cited_by_count":6},{"year":1790,"works_count":313,"oa_works_count":22,"cited_by_count":8},{"year":1789,"works_count":302,"oa_works_count":13,"cited_by_count":8},{"year":1788,"works_count":272,"oa_works_count":7,"cited_by_count":5},{"year":1787,"works_count":229,"oa_works_count":9,"cited_by_count":12},{"year":1786,"works_count":229,"oa_works_count":7,"cited_by_count":26},{"year":1785,"works_count":202,"oa_works_count":7,"cited_by_count":87},{"year":1784,"works_count":215,"oa_works_count":8,"cited_by_count":2},{"year":1783,"works_count":225,"oa_works_count":10,"cited_by_count":5},{"year":1782,"works_count":258,"oa_works_count":11,"cited_by_count":9},{"year":1781,"works_count":266,"oa_works_count":7,"cited_by_count":69},{"year":1780,"works_count":289,"oa_works_count":8,"cited_by_count":5},{"year":1779,"works_count":172,"oa_works_count":7,"cited_by_count":4},{"year":1778,"works_count":192,"oa_works_count":10,"cited_by_count":7},{"year":1777,"works_count":216,"oa_works_count":8,"cited_by_count":18},{"year":1776,"works_count":206,"oa_works_count":6,"cited_by_count":8},{"year":1775,"works_count":230,"oa_works_count":7,"cited_by_count":75},{"year":1774,"works_count":202,"oa_works_count":2,"cited_by_count":5},{"year":1773,"works_count":176,"oa_works_count":7,"cited_by_count":7},{"year":1772,"works_count":151,"oa_works_count":5,"cited_by_count":5},{"year":1771,"works_count":186,"oa_works_count":6,"cited_by_count":6},{"year":1770,"works_count":202,"oa_works_count":13,"cited_by_count":60},{"year":1769,"works_count":174,"oa_works_count":3,"cited_by_count":0},{"year":1768,"works_count":204,"oa_works_count":7,"cited_by_count":3},{"year":1767,"works_count":182,"oa_works_count":3,"cited_by_count":7},{"year":1766,"works_count":214,"oa_works_count":6,"cited_by_count":64},{"year":1765,"works_count":156,"oa_works_count":10,"cited_by_count":49},{"year":1764,"works_count":181,"oa_works_count":13,"cited_by_count":1},{"year":1763,"works_count":164,"oa_works_count":5,"cited_by_count":12},{"year":1762,"works_count":159,"oa_works_count":6,"cited_by_count":6},{"year":1761,"works_count":144,"oa_works_count":0,"cited_by_count":0},{"year":1760,"works_count":139,"oa_works_count":3,"cited_by_count":48},{"year":1759,"works_count":129,"oa_works_count":3,"cited_by_count":2},{"year":1758,"works_count":120,"oa_works_count":3,"cited_by_count":5},{"year":1757,"works_count":129,"oa_works_count":6,"cited_by_count":2},{"year":1756,"works_count":119,"oa_works_count":5,"cited_by_count":2},{"year":1755,"works_count":121,"oa_works_count":5,"cited_by_count":0},{"year":1754,"works_count":104,"oa_works_count":2,"cited_by_count":5},{"year":1753,"works_count":120,"oa_works_count":5,"cited_by_count":3},{"year":1752,"works_count":111,"oa_works_count":0,"cited_by_count":5},{"year":1751,"works_count":127,"oa_works_count":2,"cited_by_count":5},{"year":1750,"works_count":153,"oa_works_count":7,"cited_by_count":1},{"year":1749,"works_count":86,"oa_works_count":3,"cited_by_count":4},{"year":1748,"works_count":90,"oa_works_count":4,"cited_by_count":2},{"year":1747,"works_count":83,"oa_works_count":4,"cited_by_count":0},{"year":1746,"works_count":87,"oa_works_count":2,"cited_by_count":0},{"year":1745,"works_count":92,"oa_works_count":1,"cited_by_count":4},{"year":1744,"works_count":91,"oa_works_count":5,"cited_by_count":0},{"year":1743,"works_count":66,"oa_works_count":6,"cited_by_count":3},{"year":1742,"works_count":81,"oa_works_count":1,"cited_by_count":0},{"year":1741,"works_count":94,"oa_works_count":2,"cited_by_count":0},{"year":1740,"works_count":96,"oa_works_count":1,"cited_by_count":2},{"year":1739,"works_count":98,"oa_works_count":4,"cited_by_count":0},{"year":1738,"works_count":87,"oa_works_count":6,"cited_by_count":0},{"year":1737,"works_count":85,"oa_works_count":4,"cited_by_count":0},{"year":1736,"works_count":70,"oa_works_count":0,"cited_by_count":0},{"year":1735,"works_count":75,"oa_works_count":5,"cited_by_count":1},{"year":1734,"works_count":79,"oa_works_count":3,"cited_by_count":14},{"year":1733,"works_count":77,"oa_works_count":2,"cited_by_count":8},{"year":1732,"works_count":91,"oa_works_count":4,"cited_by_count":1},{"year":1731,"works_count":79,"oa_works_count":4,"cited_by_count":0},{"year":1730,"works_count":88,"oa_works_count":4,"cited_by_count":1},{"year":1729,"works_count":90,"oa_works_count":4,"cited_by_count":0},{"year":1728,"works_count":87,"oa_works_count":2,"cited_by_count":0},{"year":1727,"works_count":74,"oa_works_count":2,"cited_by_count":3},{"year":1726,"works_count":85,"oa_works_count":4,"cited_by_count":25},{"year":1725,"works_count":91,"oa_works_count":6,"cited_by_count":6},{"year":1724,"works_count":92,"oa_works_count":6,"cited_by_count":1},{"year":1723,"works_count":91,"oa_works_count":2,"cited_by_count":4},{"year":1722,"works_count":112,"oa_works_count":1,"cited_by_count":0},{"year":1721,"works_count":90,"oa_works_count":5,"cited_by_count":0},{"year":1720,"works_count":90,"oa_works_count":0,"cited_by_count":5},{"year":1719,"works_count":66,"oa_works_count":1,"cited_by_count":1},{"year":1718,"works_count":66,"oa_works_count":3,"cited_by_count":1},{"year":1717,"works_count":81,"oa_works_count":5,"cited_by_count":1},{"year":1716,"works_count":89,"oa_works_count":3,"cited_by_count":0},{"year":1715,"works_count":84,"oa_works_count":1,"cited_by_count":1},{"year":1714,"works_count":141,"oa_works_count":1,"cited_by_count":8},{"year":1713,"works_count":60,"oa_works_count":1,"cited_by_count":0},{"year":1712,"works_count":58,"oa_works_count":3,"cited_by_count":1},{"year":1711,"works_count":72,"oa_works_count":7,"cited_by_count":0},{"year":1710,"works_count":87,"oa_works_count":5,"cited_by_count":0},{"year":1709,"works_count":66,"oa_works_count":2,"cited_by_count":0},{"year":1708,"works_count":64,"oa_works_count":4,"cited_by_count":0},{"year":1707,"works_count":62,"oa_works_count":3,"cited_by_count":2},{"year":1706,"works_count":69,"oa_works_count":2,"cited_by_count":1},{"year":1705,"works_count":70,"oa_works_count":5,"cited_by_count":3},{"year":1704,"works_count":66,"oa_works_count":6,"cited_by_count":1},{"year":1703,"works_count":71,"oa_works_count":4,"cited_by_count":2},{"year":1702,"works_count":61,"oa_works_count":0,"cited_by_count":1},{"year":1701,"works_count":207,"oa_works_count":15,"cited_by_count":0},{"year":1700,"works_count":101,"oa_works_count":5,"cited_by_count":0},{"year":1699,"works_count":41,"oa_works_count":4,"cited_by_count":0},{"year":1698,"works_count":52,"oa_works_count":0,"cited_by_count":3},{"year":1697,"works_count":38,"oa_works_count":5,"cited_by_count":0},{"year":1696,"works_count":39,"oa_works_count":3,"cited_by_count":0},{"year":1695,"works_count":45,"oa_works_count":6,"cited_by_count":0},{"year":1694,"works_count":36,"oa_works_count":2,"cited_by_count":0},{"year":1693,"works_count":32,"oa_works_count":2,"cited_by_count":0},{"year":1692,"works_count":32,"oa_works_count":2,"cited_by_count":0},{"year":1691,"works_count":43,"oa_works_count":3,"cited_by_count":0},{"year":1690,"works_count":56,"oa_works_count":4,"cited_by_count":0},{"year":1689,"works_count":53,"oa_works_count":3,"cited_by_count":0},{"year":1688,"works_count":43,"oa_works_count":5,"cited_by_count":0},{"year":1687,"works_count":59,"oa_works_count":16,"cited_by_count":0},{"year":1686,"works_count":31,"oa_works_count":1,"cited_by_count":0},{"year":1685,"works_count":29,"oa_works_count":1,"cited_by_count":1},{"year":1684,"works_count":39,"oa_works_count":0,"cited_by_count":0},{"year":1683,"works_count":44,"oa_works_count":4,"cited_by_count":0},{"year":1682,"works_count":32,"oa_works_count":1,"cited_by_count":0},{"year":1681,"works_count":25,"oa_works_count":3,"cited_by_count":0},{"year":1680,"works_count":41,"oa_works_count":3,"cited_by_count":0},{"year":1679,"works_count":49,"oa_works_count":3,"cited_by_count":0},{"year":1678,"works_count":23,"oa_works_count":0,"cited_by_count":0},{"year":1677,"works_count":31,"oa_works_count":0,"cited_by_count":1},{"year":1676,"works_count":28,"oa_works_count":1,"cited_by_count":0},{"year":1675,"works_count":41,"oa_works_count":0,"cited_by_count":0},{"year":1674,"works_count":34,"oa_works_count":0,"cited_by_count":0},{"year":1673,"works_count":51,"oa_works_count":3,"cited_by_count":0},{"year":1672,"works_count":46,"oa_works_count":1,"cited_by_count":0},{"year":1671,"works_count":43,"oa_works_count":5,"cited_by_count":0},{"year":1670,"works_count":39,"oa_works_count":0,"cited_by_count":1},{"year":1669,"works_count":29,"oa_works_count":1,"cited_by_count":0},{"year":1668,"works_count":26,"oa_works_count":4,"cited_by_count":0},{"year":1667,"works_count":48,"oa_works_count":6,"cited_by_count":6},{"year":1666,"works_count":31,"oa_works_count":1,"cited_by_count":0},{"year":1665,"works_count":46,"oa_works_count":5,"cited_by_count":0},{"year":1664,"works_count":34,"oa_works_count":1,"cited_by_count":0},{"year":1663,"works_count":28,"oa_works_count":1,"cited_by_count":0},{"year":1662,"works_count":40,"oa_works_count":9,"cited_by_count":0},{"year":1661,"works_count":23,"oa_works_count":1,"cited_by_count":0},{"year":1660,"works_count":46,"oa_works_count":3,"cited_by_count":0},{"year":1659,"works_count":35,"oa_works_count":3,"cited_by_count":0},{"year":1658,"works_count":40,"oa_works_count":5,"cited_by_count":0},{"year":1657,"works_count":39,"oa_works_count":1,"cited_by_count":0},{"year":1656,"works_count":37,"oa_works_count":2,"cited_by_count":0},{"year":1655,"works_count":58,"oa_works_count":1,"cited_by_count":1},{"year":1654,"works_count":37,"oa_works_count":1,"cited_by_count":1},{"year":1653,"works_count":29,"oa_works_count":0,"cited_by_count":0},{"year":1652,"works_count":37,"oa_works_count":2,"cited_by_count":0},{"year":1651,"works_count":27,"oa_works_count":0,"cited_by_count":0},{"year":1650,"works_count":57,"oa_works_count":2,"cited_by_count":0},{"year":1649,"works_count":49,"oa_works_count":4,"cited_by_count":0},{"year":1648,"works_count":48,"oa_works_count":6,"cited_by_count":1},{"year":1647,"works_count":28,"oa_works_count":1,"cited_by_count":0},{"year":1646,"works_count":33,"oa_works_count":2,"cited_by_count":0},{"year":1645,"works_count":31,"oa_works_count":3,"cited_by_count":0},{"year":1644,"works_count":30,"oa_works_count":1,"cited_by_count":0},{"year":1643,"works_count":31,"oa_works_count":5,"cited_by_count":1},{"year":1642,"works_count":39,"oa_works_count":1,"cited_by_count":0},{"year":1641,"works_count":24,"oa_works_count":3,"cited_by_count":0},{"year":1640,"works_count":41,"oa_works_count":3,"cited_by_count":0},{"year":1639,"works_count":24,"oa_works_count":2,"cited_by_count":0},{"year":1638,"works_count":23,"oa_works_count":1,"cited_by_count":0},{"year":1637,"works_count":16,"oa_works_count":2,"cited_by_count":0},{"year":1636,"works_count":18,"oa_works_count":0,"cited_by_count":0},{"year":1635,"works_count":30,"oa_works_count":1,"cited_by_count":0},{"year":1634,"works_count":25,"oa_works_count":1,"cited_by_count":2},{"year":1633,"works_count":30,"oa_works_count":2,"cited_by_count":0},{"year":1632,"works_count":23,"oa_works_count":2,"cited_by_count":0},{"year":1631,"works_count":37,"oa_works_count":4,"cited_by_count":0},{"year":1630,"works_count":34,"oa_works_count":0,"cited_by_count":0},{"year":1629,"works_count":43,"oa_works_count":5,"cited_by_count":0},{"year":1628,"works_count":19,"oa_works_count":0,"cited_by_count":1},{"year":1627,"works_count":25,"oa_works_count":0,"cited_by_count":0},{"year":1626,"works_count":41,"oa_works_count":5,"cited_by_count":0},{"year":1625,"works_count":41,"oa_works_count":5,"cited_by_count":0},{"year":1624,"works_count":29,"oa_works_count":0,"cited_by_count":2},{"year":1623,"works_count":29,"oa_works_count":3,"cited_by_count":0},{"year":1622,"works_count":24,"oa_works_count":5,"cited_by_count":0},{"year":1621,"works_count":23,"oa_works_count":2,"cited_by_count":1},{"year":1620,"works_count":31,"oa_works_count":2,"cited_by_count":0},{"year":1619,"works_count":23,"oa_works_count":4,"cited_by_count":0},{"year":1618,"works_count":27,"oa_works_count":2,"cited_by_count":5},{"year":1617,"works_count":22,"oa_works_count":1,"cited_by_count":0},{"year":1616,"works_count":40,"oa_works_count":1,"cited_by_count":0},{"year":1615,"works_count":50,"oa_works_count":4,"cited_by_count":0},{"year":1614,"works_count":43,"oa_works_count":2,"cited_by_count":0},{"year":1613,"works_count":21,"oa_works_count":3,"cited_by_count":0},{"year":1612,"works_count":28,"oa_works_count":3,"cited_by_count":0},{"year":1611,"works_count":45,"oa_works_count":0,"cited_by_count":0},{"year":1610,"works_count":22,"oa_works_count":0,"cited_by_count":0},{"year":1609,"works_count":22,"oa_works_count":1,"cited_by_count":0},{"year":1608,"works_count":16,"oa_works_count":1,"cited_by_count":0},{"year":1607,"works_count":17,"oa_works_count":3,"cited_by_count":0},{"year":1606,"works_count":22,"oa_works_count":0,"cited_by_count":0},{"year":1605,"works_count":14,"oa_works_count":0,"cited_by_count":0},{"year":1604,"works_count":17,"oa_works_count":0,"cited_by_count":0},{"year":1603,"works_count":21,"oa_works_count":1,"cited_by_count":1},{"year":1602,"works_count":16,"oa_works_count":2,"cited_by_count":0},{"year":1601,"works_count":187,"oa_works_count":15,"cited_by_count":0},{"year":1600,"works_count":76,"oa_works_count":5,"cited_by_count":0},{"year":1599,"works_count":17,"oa_works_count":2,"cited_by_count":3},{"year":1598,"works_count":15,"oa_works_count":1,"cited_by_count":0},{"year":1597,"works_count":13,"oa_works_count":2,"cited_by_count":0},{"year":1596,"works_count":27,"oa_works_count":4,"cited_by_count":0},{"year":1595,"works_count":13,"oa_works_count":1,"cited_by_count":0},{"year":1594,"works_count":15,"oa_works_count":1,"cited_by_count":2},{"year":1593,"works_count":18,"oa_works_count":2,"cited_by_count":0},{"year":1592,"works_count":19,"oa_works_count":2,"cited_by_count":0},{"year":1591,"works_count":18,"oa_works_count":0,"cited_by_count":0},{"year":1590,"works_count":15,"oa_works_count":1,"cited_by_count":0},{"year":1589,"works_count":20,"oa_works_count":1,"cited_by_count":0},{"year":1588,"works_count":24,"oa_works_count":3,"cited_by_count":0},{"year":1587,"works_count":16,"oa_works_count":0,"cited_by_count":0},{"year":1586,"works_count":20,"oa_works_count":0,"cited_by_count":0},{"year":1585,"works_count":21,"oa_works_count":1,"cited_by_count":0},{"year":1584,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1583,"works_count":18,"oa_works_count":2,"cited_by_count":0},{"year":1582,"works_count":13,"oa_works_count":2,"cited_by_count":0},{"year":1581,"works_count":10,"oa_works_count":1,"cited_by_count":0},{"year":1580,"works_count":11,"oa_works_count":1,"cited_by_count":0},{"year":1579,"works_count":17,"oa_works_count":1,"cited_by_count":1},{"year":1578,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1577,"works_count":11,"oa_works_count":1,"cited_by_count":0},{"year":1576,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1575,"works_count":15,"oa_works_count":2,"cited_by_count":0},{"year":1574,"works_count":27,"oa_works_count":4,"cited_by_count":0},{"year":1573,"works_count":19,"oa_works_count":0,"cited_by_count":1},{"year":1572,"works_count":14,"oa_works_count":1,"cited_by_count":0},{"year":1571,"works_count":8,"oa_works_count":0,"cited_by_count":0},{"year":1570,"works_count":15,"oa_works_count":2,"cited_by_count":0},{"year":1569,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1568,"works_count":14,"oa_works_count":1,"cited_by_count":0},{"year":1567,"works_count":13,"oa_works_count":0,"cited_by_count":1},{"year":1566,"works_count":19,"oa_works_count":2,"cited_by_count":2},{"year":1565,"works_count":14,"oa_works_count":0,"cited_by_count":0},{"year":1564,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1563,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1562,"works_count":19,"oa_works_count":0,"cited_by_count":0},{"year":1561,"works_count":28,"oa_works_count":1,"cited_by_count":15},{"year":1560,"works_count":25,"oa_works_count":1,"cited_by_count":0},{"year":1559,"works_count":10,"oa_works_count":0,"cited_by_count":0},{"year":1558,"works_count":28,"oa_works_count":0,"cited_by_count":0},{"year":1557,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1556,"works_count":8,"oa_works_count":0,"cited_by_count":0},{"year":1555,"works_count":10,"oa_works_count":0,"cited_by_count":0},{"year":1554,"works_count":13,"oa_works_count":0,"cited_by_count":0},{"year":1553,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1552,"works_count":19,"oa_works_count":0,"cited_by_count":0},{"year":1551,"works_count":16,"oa_works_count":1,"cited_by_count":0},{"year":1550,"works_count":19,"oa_works_count":1,"cited_by_count":0},{"year":1549,"works_count":18,"oa_works_count":0,"cited_by_count":0},{"year":1548,"works_count":13,"oa_works_count":1,"cited_by_count":0},{"year":1547,"works_count":14,"oa_works_count":0,"cited_by_count":0},{"year":1546,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1545,"works_count":11,"oa_works_count":1,"cited_by_count":0},{"year":1544,"works_count":17,"oa_works_count":1,"cited_by_count":0},{"year":1543,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1542,"works_count":9,"oa_works_count":0,"cited_by_count":0},{"year":1541,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1540,"works_count":17,"oa_works_count":1,"cited_by_count":0},{"year":1539,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1538,"works_count":4,"oa_works_count":1,"cited_by_count":0},{"year":1537,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1536,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1535,"works_count":16,"oa_works_count":1,"cited_by_count":0},{"year":1534,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1533,"works_count":9,"oa_works_count":0,"cited_by_count":0},{"year":1532,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1531,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1530,"works_count":11,"oa_works_count":2,"cited_by_count":0},{"year":1529,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1528,"works_count":14,"oa_works_count":0,"cited_by_count":0},{"year":1527,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1526,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1525,"works_count":7,"oa_works_count":1,"cited_by_count":0},{"year":1524,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1523,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1522,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1521,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1520,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1519,"works_count":7,"oa_works_count":1,"cited_by_count":0},{"year":1518,"works_count":6,"oa_works_count":1,"cited_by_count":0},{"year":1517,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1516,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1515,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1514,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1513,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1512,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1511,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1510,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1509,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1508,"works_count":6,"oa_works_count":1,"cited_by_count":0},{"year":1507,"works_count":4,"oa_works_count":1,"cited_by_count":0},{"year":1506,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1505,"works_count":9,"oa_works_count":2,"cited_by_count":0},{"year":1504,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1503,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1502,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1501,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1500,"works_count":50,"oa_works_count":5,"cited_by_count":2},{"year":1499,"works_count":19,"oa_works_count":1,"cited_by_count":0},{"year":1498,"works_count":22,"oa_works_count":0,"cited_by_count":0},{"year":1497,"works_count":18,"oa_works_count":0,"cited_by_count":0},{"year":1496,"works_count":12,"oa_works_count":1,"cited_by_count":0},{"year":1495,"works_count":12,"oa_works_count":1,"cited_by_count":0},{"year":1494,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1493,"works_count":16,"oa_works_count":0,"cited_by_count":0},{"year":1492,"works_count":17,"oa_works_count":2,"cited_by_count":0},{"year":1491,"works_count":13,"oa_works_count":0,"cited_by_count":0},{"year":1490,"works_count":19,"oa_works_count":1,"cited_by_count":0},{"year":1489,"works_count":9,"oa_works_count":0,"cited_by_count":0},{"year":1488,"works_count":12,"oa_works_count":1,"cited_by_count":0},{"year":1487,"works_count":10,"oa_works_count":0,"cited_by_count":0},{"year":1486,"works_count":9,"oa_works_count":1,"cited_by_count":0},{"year":1485,"works_count":13,"oa_works_count":2,"cited_by_count":0},{"year":1484,"works_count":7,"oa_works_count":1,"cited_by_count":0},{"year":1483,"works_count":13,"oa_works_count":3,"cited_by_count":0},{"year":1482,"works_count":14,"oa_works_count":0,"cited_by_count":0},{"year":1481,"works_count":8,"oa_works_count":0,"cited_by_count":0},{"year":1480,"works_count":10,"oa_works_count":1,"cited_by_count":0},{"year":1479,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1478,"works_count":9,"oa_works_count":1,"cited_by_count":0},{"year":1477,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1476,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1475,"works_count":13,"oa_works_count":3,"cited_by_count":0},{"year":1474,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1473,"works_count":3,"oa_works_count":1,"cited_by_count":0},{"year":1472,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1471,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1470,"works_count":9,"oa_works_count":3,"cited_by_count":0},{"year":1469,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1468,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1467,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1466,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1465,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":1463,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1462,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1461,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1460,"works_count":6,"oa_works_count":2,"cited_by_count":0},{"year":1459,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1458,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1456,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1455,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1453,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1450,"works_count":10,"oa_works_count":6,"cited_by_count":0},{"year":1449,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1448,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1447,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1446,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1445,"works_count":38,"oa_works_count":2,"cited_by_count":0},{"year":1444,"works_count":134,"oa_works_count":0,"cited_by_count":0},{"year":1443,"works_count":145,"oa_works_count":0,"cited_by_count":0},{"year":1442,"works_count":53,"oa_works_count":0,"cited_by_count":0},{"year":1441,"works_count":66,"oa_works_count":0,"cited_by_count":0},{"year":1440,"works_count":112,"oa_works_count":2,"cited_by_count":0},{"year":1439,"works_count":27,"oa_works_count":0,"cited_by_count":0},{"year":1438,"works_count":37,"oa_works_count":0,"cited_by_count":0},{"year":1437,"works_count":69,"oa_works_count":0,"cited_by_count":0},{"year":1436,"works_count":68,"oa_works_count":0,"cited_by_count":0},{"year":1435,"works_count":116,"oa_works_count":0,"cited_by_count":0},{"year":1434,"works_count":46,"oa_works_count":0,"cited_by_count":0},{"year":1433,"works_count":54,"oa_works_count":0,"cited_by_count":0},{"year":1432,"works_count":179,"oa_works_count":0,"cited_by_count":0},{"year":1431,"works_count":86,"oa_works_count":0,"cited_by_count":0},{"year":1430,"works_count":36,"oa_works_count":0,"cited_by_count":0},{"year":1429,"works_count":13,"oa_works_count":0,"cited_by_count":0},{"year":1428,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1427,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1426,"works_count":9,"oa_works_count":1,"cited_by_count":0},{"year":1425,"works_count":11,"oa_works_count":1,"cited_by_count":0},{"year":1424,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1423,"works_count":21,"oa_works_count":0,"cited_by_count":0},{"year":1422,"works_count":13,"oa_works_count":0,"cited_by_count":0},{"year":1421,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1420,"works_count":8,"oa_works_count":1,"cited_by_count":0},{"year":1419,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1418,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1417,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1416,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1415,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1414,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1413,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1412,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1411,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1410,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1409,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1408,"works_count":4,"oa_works_count":1,"cited_by_count":0},{"year":1407,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1406,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1405,"works_count":4,"oa_works_count":2,"cited_by_count":0},{"year":1404,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1403,"works_count":45,"oa_works_count":0,"cited_by_count":0},{"year":1402,"works_count":68,"oa_works_count":0,"cited_by_count":0},{"year":1401,"works_count":46,"oa_works_count":1,"cited_by_count":0},{"year":1400,"works_count":105,"oa_works_count":14,"cited_by_count":0},{"year":1399,"works_count":26,"oa_works_count":0,"cited_by_count":0},{"year":1398,"works_count":26,"oa_works_count":0,"cited_by_count":0},{"year":1397,"works_count":19,"oa_works_count":0,"cited_by_count":0},{"year":1396,"works_count":22,"oa_works_count":0,"cited_by_count":0},{"year":1395,"works_count":28,"oa_works_count":0,"cited_by_count":0},{"year":1394,"works_count":44,"oa_works_count":0,"cited_by_count":0},{"year":1393,"works_count":38,"oa_works_count":0,"cited_by_count":0},{"year":1392,"works_count":31,"oa_works_count":0,"cited_by_count":0},{"year":1391,"works_count":36,"oa_works_count":0,"cited_by_count":0},{"year":1390,"works_count":29,"oa_works_count":1,"cited_by_count":0},{"year":1389,"works_count":22,"oa_works_count":0,"cited_by_count":0},{"year":1388,"works_count":35,"oa_works_count":0,"cited_by_count":0},{"year":1387,"works_count":30,"oa_works_count":0,"cited_by_count":0},{"year":1386,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1385,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1384,"works_count":9,"oa_works_count":1,"cited_by_count":0},{"year":1383,"works_count":9,"oa_works_count":0,"cited_by_count":0},{"year":1382,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1381,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1380,"works_count":17,"oa_works_count":0,"cited_by_count":0},{"year":1379,"works_count":16,"oa_works_count":0,"cited_by_count":0},{"year":1378,"works_count":18,"oa_works_count":0,"cited_by_count":0},{"year":1377,"works_count":16,"oa_works_count":0,"cited_by_count":0},{"year":1376,"works_count":6,"oa_works_count":1,"cited_by_count":0},{"year":1375,"works_count":6,"oa_works_count":1,"cited_by_count":0},{"year":1374,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1373,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1372,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":1371,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1370,"works_count":24,"oa_works_count":0,"cited_by_count":0},{"year":1369,"works_count":14,"oa_works_count":0,"cited_by_count":0},{"year":1368,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1364,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1362,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1361,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1360,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1357,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1355,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1354,"works_count":11,"oa_works_count":0,"cited_by_count":0},{"year":1353,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1352,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1351,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1350,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1348,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1345,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1344,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1343,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1342,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1341,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1340,"works_count":6,"oa_works_count":0,"cited_by_count":0},{"year":1339,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1337,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1336,"works_count":4,"oa_works_count":1,"cited_by_count":0},{"year":1335,"works_count":4,"oa_works_count":1,"cited_by_count":0},{"year":1334,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1333,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1332,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1330,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1329,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1327,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1325,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1324,"works_count":5,"oa_works_count":1,"cited_by_count":0},{"year":1323,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1322,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1321,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1320,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1319,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1317,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1316,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1314,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1312,"works_count":3,"oa_works_count":1,"cited_by_count":0},{"year":1311,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1310,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1308,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1307,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":1306,"works_count":3,"oa_works_count":1,"cited_by_count":0},{"year":1305,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1303,"works_count":2,"oa_works_count":2,"cited_by_count":0},{"year":1302,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1301,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1300,"works_count":48,"oa_works_count":8,"cited_by_count":0},{"year":1299,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1298,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1296,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1292,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1291,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1290,"works_count":5,"oa_works_count":2,"cited_by_count":0},{"year":1289,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1287,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1285,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1284,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1280,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1279,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1276,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1275,"works_count":2,"oa_works_count":2,"cited_by_count":0},{"year":1274,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1272,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1270,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1269,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1267,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1265,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1258,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1257,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1255,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1254,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1250,"works_count":5,"oa_works_count":5,"cited_by_count":0},{"year":1249,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1243,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1240,"works_count":2,"oa_works_count":1,"cited_by_count":0},{"year":1237,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1236,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1235,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1234,"works_count":46,"oa_works_count":1,"cited_by_count":0},{"year":1233,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1232,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1231,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1228,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1225,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1223,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1222,"works_count":18,"oa_works_count":0,"cited_by_count":0},{"year":1221,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1220,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1218,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1217,"works_count":15,"oa_works_count":0,"cited_by_count":0},{"year":1214,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1213,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1212,"works_count":199,"oa_works_count":0,"cited_by_count":0},{"year":1211,"works_count":45,"oa_works_count":0,"cited_by_count":0},{"year":1206,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1204,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1203,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1202,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1201,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1200,"works_count":19,"oa_works_count":9,"cited_by_count":0},{"year":1197,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1196,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1193,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1192,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1190,"works_count":3,"oa_works_count":2,"cited_by_count":0},{"year":1177,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1152,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1151,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1150,"works_count":4,"oa_works_count":3,"cited_by_count":0},{"year":1146,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1140,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1134,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1128,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1126,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1122,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1112,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1111,"works_count":147,"oa_works_count":0,"cited_by_count":0},{"year":1105,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1103,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1101,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1100,"works_count":10,"oa_works_count":8,"cited_by_count":0},{"year":1099,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1098,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1096,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1094,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1093,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1087,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1080,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1079,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1077,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1075,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1070,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1069,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1068,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1066,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1065,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1064,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1061,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1060,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1057,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1050,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1047,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1045,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1029,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1027,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1025,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1024,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1023,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1022,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1021,"works_count":8,"oa_works_count":0,"cited_by_count":0},{"year":1020,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":1019,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1018,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1017,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1016,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1015,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":1014,"works_count":8,"oa_works_count":0,"cited_by_count":0},{"year":1013,"works_count":8,"oa_works_count":0,"cited_by_count":0},{"year":1012,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1011,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1010,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":1008,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1006,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":1004,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1002,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":1001,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":1000,"works_count":27,"oa_works_count":2,"cited_by_count":0},{"year":974,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":963,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":950,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":940,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":930,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":929,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":920,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":906,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":905,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":904,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":841,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":826,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":810,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":808,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":800,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":703,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":620,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":610,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":606,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":604,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":603,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":600,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":590,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":541,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":506,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":461,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":460,"works_count":1,"oa_works_count":1,"cited_by_count":0},{"year":418,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":414,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":412,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":410,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":400,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":381,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":350,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":343,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":320,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":312,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":303,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":301,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":223,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":222,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":221,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":218,"works_count":5,"oa_works_count":0,"cited_by_count":0},{"year":215,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":214,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":213,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":210,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":204,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":203,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":202,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":201,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":200,"works_count":5,"oa_works_count":1,"cited_by_count":0},{"year":144,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":123,"works_count":7,"oa_works_count":0,"cited_by_count":0},{"year":122,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":111,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":110,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":101,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":97,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":41,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":30,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":27,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":24,"works_count":3,"oa_works_count":0,"cited_by_count":0},{"year":23,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":19,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":15,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":9,"works_count":2,"oa_works_count":0,"cited_by_count":0},{"year":8,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":7,"works_count":12,"oa_works_count":0,"cited_by_count":0},{"year":6,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":5,"works_count":4,"oa_works_count":0,"cited_by_count":0},{"year":3,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":2,"works_count":1,"oa_works_count":0,"cited_by_count":0},{"year":1,"works_count":17,"oa_works_count":0,"cited_by_count":0},{"year":0,"works_count":21181,"oa_works_count":59,"cited_by_count":46}],"works_api_url":"https://api.openalex.org/works?filter=primary_location.source.id:S4210203682","updated_date":"2026-02-17T02:03:14","created_date":"2016-06-24T00:00:00"}],"group_by":[]}"#;
        let json_batch = create_attachments_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted = extract_tabular(
            "bytes",
            &[json_batch],
            &DataEncoding::None,
            &DataFormat::JsonSchema,
            &AvailableSubjects::OpenAlexResponseSources,
        )?;

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(
            test,
            [
                "SourceTable",
                "SourceAlternativeTitlesTable",
                "SourceCountsByYearTable",
                "SourceIdsTable",
                "SourceIssnTable",
                "SourceSummaryStatsTable"
            ]
        );
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(
            test,
            [
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular",
                "extract_tabular"
            ]
        );
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(
            test,
            [
                "SourceTable",
                "SourceAlternativeTitlesTable",
                "SourceCountsByYearTable",
                "SourceIdsTable",
                "SourceIssnTable",
                "SourceSummaryStatsTable"
            ]
        );
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);

        Ok(())
    }
}
