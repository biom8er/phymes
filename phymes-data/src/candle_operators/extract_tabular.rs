use std::collections::HashMap;

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, CsvFormat, DataFormat, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, JsonFormat, JsonSchemaTrait, MappableTrait, Table, TableBuilder, TableBuilderTrait, TableTrait, Tool, ToolType, open_alex
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Extract tabular data in either CSV or JSON format from Bytes
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExtractTabular {
    lhs_values: String,
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
        let format = config.format.clone().ok_or(anyhow!(
            "Missing `format` for `{}`.",
            Self::get_static_name()
        ))?;
        let schema = config.schema.clone().ok_or(anyhow!(
            "Missing `config` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(ExtractTabular { lhs_values, format, schema })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        extract_tabular(&self.lhs_values, lhs_args, &self.format, &self.schema)
    }
}

/// Extract tabular data in either CSV or JSON format from Bytes
#[instrument(skip(lhs_values, lhs_args))]
pub fn extract_tabular(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
    schema: &AvailableSubjects,
) -> Result<RecordBatch> {
    // Extract out the values
    let args_table = Table::get_builder()
        .with_name("extract_tabular")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table.get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter().flatten().collect::<Vec<_>>();

    // Parse the values depending upon the specified format
    let table = match format {
        DataFormat::Csv(csv_format) => Table::get_builder()
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
            Table::get_builder()
                .with_name("attachment")
                .with_csv(
                    &values_vec,
                    csv_format.delimiter,
                    csv_format.header,
                    csv_format.batch_size,
                )?
                .build()?
        }
        DataFormat::Json(json_format) => Table::get_builder()
            .with_name("attachment")
            .with_json(&values_vec, json_format.batch_size)?
            .build()?,
        DataFormat::JsonDefault => {
            let json_format = JsonFormat::default();
            Table::get_builder()
                .with_name("attachment")
                .with_json(&values_vec, json_format.batch_size)?
                .build()?
        }
        DataFormat::JsonSchema => match schema {
            AvailableSubjects::OpenAlexResponseWorks => match serde_json::from_slice::<open_alex::OpenAlexResponseWorks>(&values_vec) {
                Ok(open_alex_response) => {
                    let batch = open_alex_response.to_record_batch("extract_tabular")?;
                    Table::get_builder()
                        .with_name("OpenAlexResponseWorks")
                        .with_record_batches(vec![batch])?
                        .build()?
                }
                Err(err) => {
                    return Err(anyhow!("Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."));
                }
            }
            AvailableSubjects::OpenAlexResponseAuthors => match serde_json::from_slice::<open_alex::OpenAlexResponseAuthors>(&values_vec) {
                Ok(open_alex_response) => {
                    let batch = open_alex_response.to_record_batch("extract_tabular")?;
                    Table::get_builder()
                        .with_name("OpenAlexResponseAuthors")
                        .with_record_batches(vec![batch])?
                        .build()?
                }
                Err(err) => {
                    return Err(anyhow!("Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."));
                }
            }
            AvailableSubjects::OpenAlexResponseInstitutions => match serde_json::from_slice::<open_alex::OpenAlexResponseInstitution>(&values_vec) {
                Ok(open_alex_response) => {
                    let batch = open_alex_response.to_record_batch("extract_tabular")?;
                    Table::get_builder()
                        .with_name("OpenAlexResponseInstitutions")
                        .with_record_batches(vec![batch])?
                        .build()?
                }
                Err(err) => {
                    return Err(anyhow!("Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."));
                }
            }
            AvailableSubjects::OpenAlexResponseTopics => match serde_json::from_slice::<open_alex::OpenAlexResponseTopic>(&values_vec) {
                Ok(open_alex_response) => {
                    let batch = open_alex_response.to_record_batch("extract_tabular")?;
                    Table::get_builder()
                        .with_name("OpenAlexResponseTopics")
                        .with_record_batches(vec![batch])?
                        .build()?
                }
                Err(err) => {
                    return Err(anyhow!("Parse error `{err:?}` for format `{format}` and schema `{schema}` for extract_tabular operator."));
                }
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported format `{format}` and schema `{schema}` for extract_tabular operator."
                ));
            }
        }
        DataFormat::Ipc => TableBuilder::new_from_ipc_stream(&values_vec)?
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
    use bytes::Bytes;
    use phymes_core::{
        BuildableTrait, BuilderTrait, CsvFormat, DataFormat, JsonFormat, Table, TableBuilderTrait,
        TableTrait, create_blob_batch,
    };
    use phymes_diagnostics::create_timestamp_micros;

    use crate::candle_operators::extract_tabular::test_extract_tabular_data::make_scores_table;

    use super::*;

    #[test]
    fn test_extract_tabular_csv_format() {
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
            extract_tabular("bytes", &[csv_batch], &DataFormat::Csv(csv_format), &AvailableSubjects::Empty).unwrap();

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
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_json_format() {
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
            extract_tabular("bytes", &[json_batch], &DataFormat::Json(json_format), &AvailableSubjects::Empty).unwrap();

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
        assert_eq!(lhs_pk, ["a", "b", "c"]);
        let score = table.get_column_as_vec_primitive::<f64>("score").unwrap();
        assert_eq!(score, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_extract_tabular_schema_format() -> Result<()> {
        // OpenAlexResponseWorks
        // Make the tabular data
        let open_alex_response_str = "{\"meta\":{\"count\":11164054,\"db_response_time_ms\":25,\"page\":1,\"per_page\":1,\"groups_count\":null},\"results\":[{\"id\":\"https://openalex.org/W3038568908\",\"doi\":\"https://doi.org/10.1585/pfr.15.2402039\",\"title\":\"Radiation Resistant Camera System for Monitoring Deuterium Plasma Discharges in the Large Helical Device\",\"display_name\":\"Radiation Resistant Camera System for Monitoring Deuterium Plasma Discharges in the Large Helical Device\",\"publication_year\":2020,\"publication_date\":\"2020-06-08\",\"ids\":{\"openalex\":\"https://openalex.org/W3038568908\",\"doi\":\"https://doi.org/10.1585/pfr.15.2402039\",\"mag\":\"3038568908\"},\"language\":\"en\",\"primary_location\":{\"id\":\"doi:10.1585/pfr.15.2402039\",\"is_oa\":true,\"landing_page_url\":\"https://doi.org/10.1585/pfr.15.2402039\",\"pdf_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"source\":{\"id\":\"https://openalex.org/S46033839\",\"display_name\":\"Plasma and Fusion Research\",\"issn_l\":\"1880-6821\",\"issn\":[\"1880-6821\"],\"is_oa\":true,\"is_in_doaj\":false,\"is_core\":true,\"host_organization\":\"https://openalex.org/P4328135220\",\"host_organization_name\":\"Japan Society of Plasma Science and Nuclear Fusion Research\",\"host_organization_lineage\":[\"https://openalex.org/P4328135220\"],\"host_organization_lineage_names\":[\"Japan Society of Plasma Science and Nuclear Fusion Research\"],\"type\":\"journal\"},\"license\":null,\"license_id\":null,\"version\":\"publishedVersion\",\"is_accepted\":true,\"is_published\":true,\"raw_source_name\":\"Plasma and Fusion Research\",\"raw_type\":\"journal-article\"},\"type\":\"article\",\"indexed_in\":[\"crossref\"],\"open_access\":{\"is_oa\":true,\"oa_status\":\"diamond\",\"oa_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"any_repository_has_fulltext\":false},\"authorships\":[{\"author_position\":\"first\",\"author\":{\"id\":\"https://openalex.org/A5039600762\",\"display_name\":\"M. Shoji\",\"orcid\":\"https://orcid.org/0000-0003-0655-7347\"},\"institutions\":[{\"id\":\"https://openalex.org/I199525922\",\"display_name\":\"National Institutes of Natural Sciences\",\"ror\":\"https://ror.org/055n47h92\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\"]},{\"id\":\"https://openalex.org/I4210108322\",\"display_name\":\"National Institute for Fusion Science\",\"ror\":\"https://ror.org/01t3wyv61\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\",\"https://openalex.org/I4210108322\"]}],\"countries\":[\"JP\"],\"is_corresponding\":true,\"raw_author_name\":\"Mamoru SHOJI\",\"raw_affiliation_strings\":[\"National Institute for Fusion Science, National Institutes of Natural Sciences\"],\"affiliations\":[{\"raw_affiliation_string\":\"National Institute for Fusion Science, National Institutes of Natural Sciences\",\"institution_ids\":[\"https://openalex.org/I4210108322\",\"https://openalex.org/I199525922\"]}]},{\"author_position\":\"last\",\"author\":{\"id\":null,\"display_name\":\"LHD Experiment Group\",\"orcid\":null},\"institutions\":[{\"id\":\"https://openalex.org/I199525922\",\"display_name\":\"National Institutes of Natural Sciences\",\"ror\":\"https://ror.org/055n47h92\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\"]},{\"id\":\"https://openalex.org/I4210108322\",\"display_name\":\"National Institute for Fusion Science\",\"ror\":\"https://ror.org/01t3wyv61\",\"country_code\":\"JP\",\"type\":\"facility\",\"lineage\":[\"https://openalex.org/I1319490839\",\"https://openalex.org/I199525922\",\"https://openalex.org/I4210108322\"]}],\"countries\":[\"JP\"],\"is_corresponding\":false,\"raw_author_name\":\"LHD Experiment Group\",\"raw_affiliation_strings\":[\"National Institute for Fusion Science, National Institutes of Natural Sciences\"],\"affiliations\":[{\"raw_affiliation_string\":\"National Institute for Fusion Science, National Institutes of Natural Sciences\",\"institution_ids\":[\"https://openalex.org/I4210108322\",\"https://openalex.org/I199525922\"]}]}],\"institutions\":[],\"countries_distinct_count\":1,\"institutions_distinct_count\":2,\"corresponding_author_ids\":[\"https://openalex.org/A5039600762\"],\"corresponding_institution_ids\":[\"https://openalex.org/I199525922\",\"https://openalex.org/I4210108322\"],\"apc_list\":null,\"apc_paid\":null,\"fwci\":0.40325236,\"has_fulltext\":true,\"cited_by_count\":801216,\"citation_normalized_percentile\":{\"value\":0.86901083,\"is_in_top_1_percent\":false,\"is_in_top_10_percent\":false},\"cited_by_percentile_year\":{\"min\":89,\"max\":100},\"biblio\":{\"volume\":\"15\",\"issue\":\"0\",\"first_page\":\"2402039\",\"last_page\":\"2402039\"},\"is_retracted\":false,\"is_paratext\":false,\"is_xpac\":false,\"primary_topic\":{\"id\":\"https://openalex.org/T10346\",\"display_name\":\"Magnetic confinement fusion research\",\"score\":0.9991000294685364,\"subfield\":{\"id\":\"https://openalex.org/subfields/3106\",\"display_name\":\"Nuclear and High Energy Physics\"},\"field\":{\"id\":\"https://openalex.org/fields/31\",\"display_name\":\"Physics and Astronomy\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}},\"topics\":[{\"id\":\"https://openalex.org/T10346\",\"display_name\":\"Magnetic confinement fusion research\",\"score\":0.9991000294685364,\"subfield\":{\"id\":\"https://openalex.org/subfields/3106\",\"display_name\":\"Nuclear and High Energy Physics\"},\"field\":{\"id\":\"https://openalex.org/fields/31\",\"display_name\":\"Physics and Astronomy\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}},{\"id\":\"https://openalex.org/T11949\",\"display_name\":\"Nuclear Physics and Applications\",\"score\":0.9987999796867371,\"subfield\":{\"id\":\"https://openalex.org/subfields/3108\",\"display_name\":\"Radiation\"},\"field\":{\"id\":\"https://openalex.org/fields/31\",\"display_name\":\"Physics and Astronomy\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}},{\"id\":\"https://openalex.org/T10592\",\"display_name\":\"Fusion materials and technologies\",\"score\":0.998199999332428,\"subfield\":{\"id\":\"https://openalex.org/subfields/2505\",\"display_name\":\"Materials Chemistry\"},\"field\":{\"id\":\"https://openalex.org/fields/25\",\"display_name\":\"Materials Science\"},\"domain\":{\"id\":\"https://openalex.org/domains/3\",\"display_name\":\"Physical Sciences\"}}],\"keywords\":[{\"id\":\"https://openalex.org/keywords/radiation\",\"display_name\":\"Radiation\",\"score\":0.7057818174362183},{\"id\":\"https://openalex.org/keywords/plasma\",\"display_name\":\"Plasma\",\"score\":0.5598242878913879},{\"id\":\"https://openalex.org/keywords/materials-science\",\"display_name\":\"Materials science\",\"score\":0.5517664551734924},{\"id\":\"https://openalex.org/keywords/optics\",\"display_name\":\"Optics\",\"score\":0.5239154100418091},{\"id\":\"https://openalex.org/keywords/shield\",\"display_name\":\"Shield\",\"score\":0.5098416209220886},{\"id\":\"https://openalex.org/keywords/neutron\",\"display_name\":\"Neutron\",\"score\":0.4559711515903473},{\"id\":\"https://openalex.org/keywords/nuclear-engineering\",\"display_name\":\"Nuclear engineering\",\"score\":0.3836207985877991},{\"id\":\"https://openalex.org/keywords/physics\",\"display_name\":\"Physics\",\"score\":0.32291728258132935},{\"id\":\"https://openalex.org/keywords/nuclear-physics\",\"display_name\":\"Nuclear physics\",\"score\":0.13794386386871338},{\"id\":\"https://openalex.org/keywords/geology\",\"display_name\":\"Geology\",\"score\":0.05549171566963196}],\"concepts\":[{\"id\":\"https://openalex.org/C153385146\",\"wikidata\":\"https://www.wikidata.org/wiki/Q18335\",\"display_name\":\"Radiation\",\"level\":2,\"score\":0.7057818174362183},{\"id\":\"https://openalex.org/C82706917\",\"wikidata\":\"https://www.wikidata.org/wiki/Q10251\",\"display_name\":\"Plasma\",\"level\":2,\"score\":0.5598242878913879},{\"id\":\"https://openalex.org/C192562407\",\"wikidata\":\"https://www.wikidata.org/wiki/Q228736\",\"display_name\":\"Materials science\",\"level\":0,\"score\":0.5517664551734924},{\"id\":\"https://openalex.org/C120665830\",\"wikidata\":\"https://www.wikidata.org/wiki/Q14620\",\"display_name\":\"Optics\",\"level\":1,\"score\":0.5239154100418091},{\"id\":\"https://openalex.org/C138081364\",\"wikidata\":\"https://www.wikidata.org/wiki/Q852013\",\"display_name\":\"Shield\",\"level\":2,\"score\":0.5098416209220886},{\"id\":\"https://openalex.org/C152568617\",\"wikidata\":\"https://www.wikidata.org/wiki/Q2348\",\"display_name\":\"Neutron\",\"level\":2,\"score\":0.4559711515903473},{\"id\":\"https://openalex.org/C116915560\",\"wikidata\":\"https://www.wikidata.org/wiki/Q83504\",\"display_name\":\"Nuclear engineering\",\"level\":1,\"score\":0.3836207985877991},{\"id\":\"https://openalex.org/C121332964\",\"wikidata\":\"https://www.wikidata.org/wiki/Q413\",\"display_name\":\"Physics\",\"level\":0,\"score\":0.32291728258132935},{\"id\":\"https://openalex.org/C185544564\",\"wikidata\":\"https://www.wikidata.org/wiki/Q81197\",\"display_name\":\"Nuclear physics\",\"level\":1,\"score\":0.13794386386871338},{\"id\":\"https://openalex.org/C127313418\",\"wikidata\":\"https://www.wikidata.org/wiki/Q1069\",\"display_name\":\"Geology\",\"level\":0,\"score\":0.05549171566963196},{\"id\":\"https://openalex.org/C5900021\",\"wikidata\":\"https://www.wikidata.org/wiki/Q163082\",\"display_name\":\"Petrology\",\"level\":1,\"score\":0.0},{\"id\":\"https://openalex.org/C127413603\",\"wikidata\":\"https://www.wikidata.org/wiki/Q11023\",\"display_name\":\"Engineering\",\"level\":0,\"score\":0.0}],\"mesh\":[],\"locations_count\":1,\"locations\":[{\"id\":\"doi:10.1585/pfr.15.2402039\",\"is_oa\":true,\"landing_page_url\":\"https://doi.org/10.1585/pfr.15.2402039\",\"pdf_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"source\":{\"id\":\"https://openalex.org/S46033839\",\"display_name\":\"Plasma and Fusion Research\",\"issn_l\":\"1880-6821\",\"issn\":[\"1880-6821\"],\"is_oa\":true,\"is_in_doaj\":false,\"is_core\":true,\"host_organization\":\"https://openalex.org/P4328135220\",\"host_organization_name\":\"Japan Society of Plasma Science and Nuclear Fusion Research\",\"host_organization_lineage\":[\"https://openalex.org/P4328135220\"],\"host_organization_lineage_names\":[\"Japan Society of Plasma Science and Nuclear Fusion Research\"],\"type\":\"journal\"},\"license\":null,\"license_id\":null,\"version\":\"publishedVersion\",\"is_accepted\":true,\"is_published\":true,\"raw_source_name\":\"Plasma and Fusion Research\",\"raw_type\":\"journal-article\"}],\"best_oa_location\":{\"id\":\"doi:10.1585/pfr.15.2402039\",\"is_oa\":true,\"landing_page_url\":\"https://doi.org/10.1585/pfr.15.2402039\",\"pdf_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\",\"source\":{\"id\":\"https://openalex.org/S46033839\",\"display_name\":\"Plasma and Fusion Research\",\"issn_l\":\"1880-6821\",\"issn\":[\"1880-6821\"],\"is_oa\":true,\"is_in_doaj\":false,\"is_core\":true,\"host_organization\":\"https://openalex.org/P4328135220\",\"host_organization_name\":\"Japan Society of Plasma Science and Nuclear Fusion Research\",\"host_organization_lineage\":[\"https://openalex.org/P4328135220\"],\"host_organization_lineage_names\":[\"Japan Society of Plasma Science and Nuclear Fusion Research\"],\"type\":\"journal\"},\"license\":null,\"license_id\":null,\"version\":\"publishedVersion\",\"is_accepted\":true,\"is_published\":true,\"raw_source_name\":\"Plasma and Fusion Research\",\"raw_type\":\"journal-article\"},\"sustainable_development_goals\":[{\"score\":0.8799999952316284,\"display_name\":\"Affordable and clean energy\",\"id\":\"https://metadata.un.org/sdg/7\"}],\"awards\":[],\"funders\":[],\"has_content\":{\"grobid_xml\":true,\"pdf\":true},\"content_urls\":{\"pdf\":\"https://content.openalex.org/works/W3038568908.pdf\",\"grobid_xml\":\"https://content.openalex.org/works/W3038568908.grobid-xml\"},\"referenced_works_count\":8,\"referenced_works\":[\"https://openalex.org/W2069091362\",\"https://openalex.org/W2151240562\",\"https://openalex.org/W2527753843\",\"https://openalex.org/W2590699823\",\"https://openalex.org/W2783171299\",\"https://openalex.org/W2806477398\",\"https://openalex.org/W2922014310\",\"https://openalex.org/W2945236265\"],\"related_works\":[\"https://openalex.org/W2606430476\",\"https://openalex.org/W2069389872\",\"https://openalex.org/W2024680443\",\"https://openalex.org/W1992734408\",\"https://openalex.org/W2909752308\",\"https://openalex.org/W2074503354\",\"https://openalex.org/W2353473218\",\"https://openalex.org/W2060642378\",\"https://openalex.org/W2094345694\",\"https://openalex.org/W2889162861\"],\"abstract_inverted_index\":{\"Radiation\":[0],\"resistant\":[1,196],\"camera\":[2],\"system\":[3,18],\"was\":[4,98],\"constructed\":[5],\"for\":[6],\"monitoring\":[7],\"deuterium\":[8],\"plasma\":[9,44],\"discharges\":[10],\"in\":[11,42,52,69,83,118],\"the\":[12,43,47,62,88,91,94,105,108,112,115,119,124,129,132,139,142,145,151,159,163,174,178,187,191,194],\"Large\":[13],\"Helical\":[14],\"Device\":[15],\"(LHD).\":[16],\"This\":[17,181],\"has\":[19,134],\"contributed\":[20],\"to\":[21,32,123,162,186],\"safe\":[22],\"operation\":[23],\"during\":[24],\"two\":[25],\"experimental\":[26],\"campaigns\":[27],\"without\":[28],\"serious\":[29],\"problems\":[30],\"due\":[31],\"radiation\":[33,95,109,143,160,195],\"(neutrons\":[34],\"and\":[35,111],\"gamma-rays).\":[36],\"The\":[37,64],\"cameras\":[38,65,133],\"steadily\":[39],\"functioned\":[40],\"even\":[41],\"discharge\":[45],\"with\":[46,78,158],\"maximum\":[48],\"neutron\":[49],\"emission\":[50],\"rate\":[51],\"FY\":[53],\"2017,\":[54],\"though\":[55],\"some\":[56,169],\"bright\":[57,154,170],\"specks\":[58,155,171],\"temporarily\":[59],\"appeared\":[60],\"on\":[61,144,177],\"images.\":[63],\"have\":[66],\"been\":[67,135],\"installed\":[68],\"shield\":[70,92,120],\"boxes\":[71,76],\"which\":[72,103,165],\"consist\":[73],\"of\":[74,90,107,114,128,131,138,141,153,190,193],\"lead\":[75],\"covered\":[77],\"10%\":[79],\"borated\":[80],\"polyethylene\":[81],\"blocks\":[82],\"all\":[84],\"directions.\":[85],\"For\":[86],\"optimizing\":[87],\"design\":[89],\"box,\":[93],\"flux\":[96,110,161],\"distribution\":[97],\"calculated\":[99],\"by\":[100,173],\"MCNP-6\":[101],\"code,\":[102],\"reveals\":[104],\"reduction\":[106],\"change\":[113],\"energy\":[116],\"spectra\":[117],\"box.\":[121],\"Thanks\":[122],\"optimization,\":[125],\"significant\":[126],\"extension\":[127,189],\"lifetime\":[130,192],\"realized.\":[136],\"Investigation\":[137],\"influence\":[140],\"CCD\":[146],\"image\":[147,179],\"sensor\":[148],\"shows\":[149],\"that\":[150,168],\"number\":[152],\"generally\":[156],\"increases\":[157],\"camera,\":[164],\"also\":[166,183],\"indicates\":[167],\"disappear\":[172],\"self-annealing\":[175],\"process\":[176],\"sensor.\":[180],\"phenomenon\":[182],\"highly\":[184],\"contributes\":[185],\"further\":[188],\"cameras.\":[197]},\"counts_by_year\":[{\"year\":2026,\"cited_by_count\":1},{\"year\":2025,\"cited_by_count\":801210},{\"year\":2024,\"cited_by_count\":2},{\"year\":2022,\"cited_by_count\":2},{\"year\":2021,\"cited_by_count\":1}],\"updated_date\":\"2025-11-06T03:46:38.306776\",\"created_date\":\"2025-10-10T00:00:00\"}],\"group_by\":[]}\n";
        let json_batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted =
            extract_tabular("bytes", &[json_batch], &DataFormat::JsonSchema, &AvailableSubjects::OpenAlexResponseWorks)?;

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(test, ["WorkTable", "WorkAuthorshipTable", "WorkLocationTable", "WorkOpenAccessTable", "WorkBiblioTable", "WorkCitationPercentileTable", "WorkCitedByPercentileYearTable", "WorkCountsByYearTable", "WorkConceptTable", "WorkTopicTable", "WorkKeywordTable", "WorkSdgTagTable", "WorkCorrespondingAuthorTable", "WorkCorrespondingInstitutionTable", "WorkIndexedInTable", "WorkIdsTable", "WorkReferencedWorksTable", "WorkRelatedWorksTable"]);
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(test, ["extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular"]);
        let subjects = table.get_column_as_vec_str("subject");
        assert_eq!(subjects, ["WorkTable", "WorkAuthorshipTable", "WorkLocationTable", "WorkOpenAccessTable", "WorkBiblioTable", "WorkCitationPercentileTable", "WorkCitedByPercentileYearTable", "WorkCountsByYearTable", "WorkConceptTable", "WorkTopicTable", "WorkKeywordTable", "WorkSdgTagTable", "WorkCorrespondingAuthorTable", "WorkCorrespondingInstitutionTable", "WorkIndexedInTable", "WorkIdsTable", "WorkReferencedWorksTable", "WorkRelatedWorksTable"]);
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);
        let test_tables: Result<Vec<Table>> = table.get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .map(|b| TableBuilder::new_from_ipc_stream(&b)?
                .with_name("extracted_bytes")
                .build())
            .collect();
        let test = test_tables?.into_iter()            
            .zip(subjects)
            .map(|(t, s)| (s.to_string(), t)).collect::<HashMap<_, _>>();
        assert_eq!(test.get("WorkTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkAuthorshipTable").unwrap().count_rows(), 2);
        assert_eq!(test.get("WorkLocationTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkOpenAccessTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkBiblioTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkCitationPercentileTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkCitedByPercentileYearTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkCountsByYearTable").unwrap().count_rows(), 5);
        assert_eq!(test.get("WorkConceptTable").unwrap().count_rows(), 12);
        assert_eq!(test.get("WorkTopicTable").unwrap().count_rows(), 3);
        assert_eq!(test.get("WorkKeywordTable").unwrap().count_rows(), 10);
        assert!(test.get("WorkMeshTagTable").is_none());
        assert_eq!(test.get("WorkSdgTagTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkCorrespondingAuthorTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkCorrespondingInstitutionTable").unwrap().count_rows(), 2);
        assert_eq!(test.get("WorkIndexedInTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkIdsTable").unwrap().count_rows(), 1);
        assert_eq!(test.get("WorkReferencedWorksTable").unwrap().count_rows(), 8);
        assert_eq!(test.get("WorkRelatedWorksTable").unwrap().count_rows(), 10);
        
        // OpenAlexResponseAuthors
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":8329749,"db_response_time_ms":88,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/A5043579160","orcid":null,"display_name":"T. Tokuzawa","display_name_alternatives":["K. Tanaka","LHD Experiment Group","T Tokuzawa","T. Tokuzawa","TOKUZAWA, Tokihiko","Tokihiko TOKUZAWA","Tokihiko Tokuzawa","Tokihiko Tokuzawa Tokihiko Tokuzawa","Tokuzawa Tokihiko","Tokuzawa, T."],"works_count":671592,"cited_by_count":6540,"summary_stats":{"2yr_mean_citedness":9.238536019859872e-05,"h_index":40,"i10_index":184},"ids":{"openalex":"https://openalex.org/A5043579160","orcid":"https://orcid.org/0000-0001-5473-2109"},"affiliations":[{"institution":{"id":"https://openalex.org/I1289243028","ror":"https://ror.org/01qz5mb56","display_name":"Oak Ridge National Laboratory","country_code":"US","type":"facility","lineage":["https://openalex.org/I1289243028","https://openalex.org/I1330989302","https://openalex.org/I39565521","https://openalex.org/I4210159294"]},"years":[2008]},{"institution":{"id":"https://openalex.org/I135598925","ror":"https://ror.org/00p4k0j84","display_name":"Kyushu University","country_code":"JP","type":"education","lineage":["https://openalex.org/I135598925"]},"years":[2026,2025,2024,2023,2022,2014,2013]},{"institution":{"id":"https://openalex.org/I146399215","ror":"https://ror.org/02956yf07","display_name":"University of Tsukuba","country_code":"JP","type":"education","lineage":["https://openalex.org/I146399215"]},"years":[1999,1998,1997,1996,1995,1994,1993]},{"institution":{"id":"https://openalex.org/I199525922","ror":"https://ror.org/055n47h92","display_name":"National Institutes of Natural Sciences","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922"]},"years":[2026,2025,2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2012,2011,2010,2009,2008,2006,2005,2004,2003,2002,2001,2000]},{"institution":{"id":"https://openalex.org/I200475212","ror":"https://ror.org/0516ah480","display_name":"The Graduate University for Advanced Studies, SOKENDAI","country_code":"JP","type":"education","lineage":["https://openalex.org/I200475212"]},"years":[2026,2025,2024,2023,2022,2021,2020,2019,2018,2016,2015,2010,2008,2007,2006,2005,2003,2000]},{"institution":{"id":"https://openalex.org/I22299242","ror":"https://ror.org/02kpeqv85","display_name":"Kyoto University","country_code":"JP","type":"education","lineage":["https://openalex.org/I22299242"]},"years":[2005]},{"institution":{"id":"https://openalex.org/I2799567181","ror":"https://ror.org/03vn1ts68","display_name":"Princeton Plasma Physics Laboratory","country_code":"US","type":"facility","lineage":["https://openalex.org/I1330989302","https://openalex.org/I20089843","https://openalex.org/I2799567181","https://openalex.org/I39565521"]},"years":[2014]},{"institution":{"id":"https://openalex.org/I4200000001","ror":"https://ror.org/02nr0ka47","display_name":"OpenAlex","country_code":"CA","type":"nonprofit","lineage":["https://openalex.org/I4200000001"]},"years":[2025,2024,2023,2021,2018,2016,2015,2014,2013,2012,2006,2005,2004,2003,2002,1999]},{"institution":{"id":"https://openalex.org/I4210108322","ror":"https://ror.org/01t3wyv61","display_name":"National Institute for Fusion Science","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922","https://openalex.org/I4210108322"]},"years":[2026,2025,2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]},{"institution":{"id":"https://openalex.org/I4210110163","ror":"https://ror.org/01yk36x23","display_name":"Nippon Soken (Japan)","country_code":"JP","type":"company","lineage":["https://openalex.org/I4210110163"]},"years":[2020]},{"institution":{"id":"https://openalex.org/I4210125919","ror":"https://ror.org/02vtgg877","display_name":"Fusion (United States)","country_code":"US","type":"company","lineage":["https://openalex.org/I4210125919"]},"years":[2025,2006]},{"institution":{"id":"https://openalex.org/I4210149442","ror":"https://ror.org/05rwjyj14","display_name":"Fusion Academy","country_code":"US","type":"education","lineage":["https://openalex.org/I4210149442"]},"years":[2025,2021,2006]},{"institution":{"id":"https://openalex.org/I4210158445","ror":"https://ror.org/004tze884","display_name":"Institute of Natural Science","country_code":"KP","type":"education","lineage":["https://openalex.org/I4210158445"]},"years":[2024]},{"institution":{"id":"https://openalex.org/I4843557","ror":"https://ror.org/03e5eem51","display_name":"Budker Institute of Nuclear Physics","country_code":"RU","type":"facility","lineage":["https://openalex.org/I1313323035","https://openalex.org/I1313323035","https://openalex.org/I4210096333","https://openalex.org/I4210127387","https://openalex.org/I4843557"]},"years":[2010]},{"institution":{"id":"https://openalex.org/I50357001","ror":"https://ror.org/03ths8210","display_name":"Universidad Carlos III de Madrid","country_code":"ES","type":"education","lineage":["https://openalex.org/I50357001"]},"years":[2008]},{"institution":{"id":"https://openalex.org/I60134161","ror":"https://ror.org/04chrp450","display_name":"Nagoya University","country_code":"JP","type":"education","lineage":["https://openalex.org/I60134161"]},"years":[2022,2014,2005,2000]},{"institution":{"id":"https://openalex.org/I74801974","ror":"https://ror.org/057zh3y96","display_name":"The University of Tokyo","country_code":"JP","type":"education","lineage":["https://openalex.org/I74801974"]},"years":[2022,2012]}],"last_known_institutions":[{"id":"https://openalex.org/I135598925","ror":"https://ror.org/00p4k0j84","display_name":"Kyushu University","country_code":"JP","type":"education","lineage":["https://openalex.org/I135598925"]},{"id":"https://openalex.org/I199525922","ror":"https://ror.org/055n47h92","display_name":"National Institutes of Natural Sciences","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922"]},{"id":"https://openalex.org/I200475212","ror":"https://ror.org/0516ah480","display_name":"The Graduate University for Advanced Studies, SOKENDAI","country_code":"JP","type":"education","lineage":["https://openalex.org/I200475212"]},{"id":"https://openalex.org/I4210108322","ror":"https://ror.org/01t3wyv61","display_name":"National Institute for Fusion Science","country_code":"JP","type":"facility","lineage":["https://openalex.org/I1319490839","https://openalex.org/I199525922","https://openalex.org/I4210108322"]}],"topics":[{"id":"https://openalex.org/T10346","display_name":"Magnetic confinement fusion research","count":379,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10159","display_name":"Ionosphere and magnetosphere dynamics","count":176,"score":0.9998999834060669,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10781","display_name":"Plasma Diagnostics and Applications","count":102,"score":0.9998000264167786,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11367","display_name":"Particle accelerators and beam dynamics","count":102,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10592","display_name":"Fusion materials and technologies","count":96,"score":0.9997000098228455,"subfield":{"id":"https://openalex.org/subfields/2505","display_name":"Materials Chemistry"},"field":{"id":"https://openalex.org/fields/25","display_name":"Materials Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}}],"topic_share":[{"id":"https://openalex.org/T10346","display_name":"Magnetic confinement fusion research","value":0.0016848,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10592","display_name":"Fusion materials and technologies","value":0.0007482,"subfield":{"id":"https://openalex.org/subfields/2505","display_name":"Materials Chemistry"},"field":{"id":"https://openalex.org/fields/25","display_name":"Materials Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10384","display_name":"Laser-Plasma Interactions and Diagnostics","value":0.0005514,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10781","display_name":"Plasma Diagnostics and Applications","value":0.000435,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11367","display_name":"Particle accelerators and beam dynamics","value":0.0003862,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}}],"x_concepts":[{"id":"121332964","wikidata":"https://www.wikidata.org/wiki/Q413","display_name":"Physics","score":0.9342754483222961},{"id":"142757262","wikidata":"https://www.wikidata.org/wiki/Q76436","display_name":"Doppler effect","score":0.8120940923690796},{"id":"192562407","wikidata":"https://www.wikidata.org/wiki/Q228736","display_name":"Materials science","score":0.7516918182373047},{"id":"2779843651","wikidata":"https://www.wikidata.org/wiki/Q7390335","display_name":"SIGNAL (programming language)","score":0.6705514192581177},{"id":"165838908","wikidata":"https://www.wikidata.org/wiki/Q736777","display_name":"Calibration","score":0.6469423770904541}],"counts_by_year":[{"year":1993,"works_count":1,"oa_works_count":0,"cited_by_count":33},{"year":1994,"works_count":1,"oa_works_count":0,"cited_by_count":9},{"year":1995,"works_count":4,"oa_works_count":0,"cited_by_count":38},{"year":1996,"works_count":1,"oa_works_count":0,"cited_by_count":1},{"year":1997,"works_count":9,"oa_works_count":0,"cited_by_count":76},{"year":1998,"works_count":5,"oa_works_count":0,"cited_by_count":19},{"year":1999,"works_count":18,"oa_works_count":3,"cited_by_count":682},{"year":2000,"works_count":15,"oa_works_count":5,"cited_by_count":165},{"year":2001,"works_count":36,"oa_works_count":8,"cited_by_count":933},{"year":2002,"works_count":19,"oa_works_count":4,"cited_by_count":362},{"year":2003,"works_count":30,"oa_works_count":9,"cited_by_count":539},{"year":2004,"works_count":22,"oa_works_count":5,"cited_by_count":374},{"year":2005,"works_count":21,"oa_works_count":6,"cited_by_count":484},{"year":2006,"works_count":34,"oa_works_count":14,"cited_by_count":342},{"year":2007,"works_count":16,"oa_works_count":3,"cited_by_count":231},{"year":2008,"works_count":34,"oa_works_count":16,"cited_by_count":401},{"year":2009,"works_count":6,"oa_works_count":1,"cited_by_count":78},{"year":2010,"works_count":25,"oa_works_count":6,"cited_by_count":344},{"year":2011,"works_count":21,"oa_works_count":1,"cited_by_count":99},{"year":2012,"works_count":16,"oa_works_count":5,"cited_by_count":156},{"year":2013,"works_count":19,"oa_works_count":8,"cited_by_count":167},{"year":2014,"works_count":14,"oa_works_count":5,"cited_by_count":115},{"year":2015,"works_count":21,"oa_works_count":6,"cited_by_count":83},{"year":2016,"works_count":31,"oa_works_count":4,"cited_by_count":69},{"year":2017,"works_count":13,"oa_works_count":6,"cited_by_count":228},{"year":2018,"works_count":12,"oa_works_count":7,"cited_by_count":143},{"year":2019,"works_count":9,"oa_works_count":1,"cited_by_count":120},{"year":2020,"works_count":9,"oa_works_count":4,"cited_by_count":35},{"year":2021,"works_count":8,"oa_works_count":6,"cited_by_count":57},{"year":2022,"works_count":12,"oa_works_count":9,"cited_by_count":83},{"year":2023,"works_count":4,"oa_works_count":3,"cited_by_count":12},{"year":2024,"works_count":13,"oa_works_count":12,"cited_by_count":60},{"year":2025,"works_count":11,"oa_works_count":9,"cited_by_count":2},{"year":2026,"works_count":671078,"oa_works_count":671078,"cited_by_count":0}],"longest_name":"Tokihiko Tokuzawa Tokihiko Tokuzawa","parsed_longest_name":{"first":"tokihiko","middle":"tokuzawa tokihiko","last":"tokuzawa","suffix":"","nickname":""},"block_key":"t tokuzawa","works_api_url":"https://api.openalex.org/works?filter=author.id:A5043579160","updated_date":"2026-02-16T12:19:24","created_date":"2016-06-24T00:00:00"}],"group_by":[]}"#;
        let json_batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted =
            extract_tabular("bytes", &[json_batch], &DataFormat::JsonSchema, &AvailableSubjects::OpenAlexResponseAuthors)?;

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(test, ["AuthorTable", "AuthorDisplayNameAlternativesTable", "AuthorAffiliationTable", "AuthorLastKnownInstitutionsTable", "AuthorIdsTable", "AuthorSummaryStatsTable", "AuthorCountsByYearTable", "AuthorConceptTable"]);
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(test, ["extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular"]);
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(test, ["AuthorTable", "AuthorDisplayNameAlternativesTable", "AuthorAffiliationTable", "AuthorLastKnownInstitutionsTable", "AuthorIdsTable", "AuthorSummaryStatsTable", "AuthorCountsByYearTable", "AuthorConceptTable"]);
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);
        
        // OpenAlexResponseInstitutions
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":31340,"db_response_time_ms":4,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/I27837315","ror":"https://ror.org/00jmfr291","display_name":"University of Michigan","country_code":"US","type":"education","type_id":"https://openalex.org/institution-types/education","lineage":["https://openalex.org/I27837315"],"homepage_url":"https://www.umich.edu","image_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/University%20of%20Michigan%20logo.svg","image_thumbnail_url":"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/University%20of%20Michigan%20logo.svg&width=300","display_name_acronyms":["UM"],"display_name_alternatives":["UMich","University of Michigan","University of Michigan\u2013Ann Arbor","Universit\u00e9 du Michigan"],"repositories":[{"id":"https://openalex.org/S4306400393","display_name":"Deep Blue (University of Michigan)","host_organization":"https://openalex.org/I27837315","host_organization_name":"University of Michigan","host_organization_lineage":["https://openalex.org/I27837315"]},{"id":"https://openalex.org/S4306400708","display_name":"CINECA IRIS Institutional Research Information System (IRIS Istituto Nazionale di Ricerca Metrologica)","host_organization":"https://openalex.org/I27837315","host_organization_name":"University of Michigan","host_organization_lineage":["https://openalex.org/I27837315"]}],"works_count":941687,"cited_by_count":59852336,"summary_stats":{"2yr_mean_citedness":3.1827793123154584,"h_index":2006,"i10_index":619796},"ids":{"openalex":"https://openalex.org/I27837315","ror":"https://ror.org/00jmfr291","grid":"grid.214458.e","wikipedia":"http://en.wikipedia.org/wiki/University_of_Michigan","wikidata":null},"geo":{"city":"Ann Arbor","geonames_city_id":"4984247","region":"Michigan","country_code":"US","country":"United States","latitude":42.27756,"longitude":-83.74088},"international":{},"associated_institutions":[{"id":"https://openalex.org/I2801799315","ror":"https://ror.org/034npj057","display_name":"Hurley Medical Center","country_code":"US","type":"healthcare","relationship":"related"},{"id":"https://openalex.org/I4210092198","ror":"https://ror.org/01c3xc117","display_name":"University of Michigan\u2013Flint","country_code":"US","type":"education","relationship":"related"},{"id":"https://openalex.org/I4210104572","ror":"https://ror.org/015tnsz82","display_name":"Michigan Sea Grant","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4210114445","ror":"https://ror.org/01zcpa714","display_name":"Michigan Medicine","country_code":"US","type":"healthcare","relationship":"related"},{"id":"https://openalex.org/I4210130704","ror":"https://ror.org/035wtm547","display_name":"University of Michigan\u2013Dearborn","country_code":"US","type":"education","relationship":"related"},{"id":"https://openalex.org/I4210163254","ror":"https://ror.org/057mgcy61","display_name":"Michigan Space Grant Consortium","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4387153780","ror":"https://ror.org/02q7mkh03","display_name":"Inter-university Consortium for Political and Social Research","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4387153798","ror":"https://ror.org/00rx1p510","display_name":"University of Michigan Press","country_code":"US","type":"other","relationship":"child"},{"id":"https://openalex.org/I4387154935","ror":"https://ror.org/04pk7zz41","display_name":"Arctic Long Term Ecological Research","country_code":"US","type":"facility","relationship":"related"},{"id":"https://openalex.org/I4387930282","ror":"https://ror.org/02hhndj92","display_name":"University of Michigan Biological Station","country_code":"US","type":"facility","relationship":"child"},{"id":"https://openalex.org/I4402554065","ror":"https://ror.org/04xgzv028","display_name":"Cooperative Institute for Great Lakes Research","country_code":"US","type":"facility","relationship":"child"},{"id":"https://openalex.org/I4404532909","ror":"https://ror.org/02bkkgm47","display_name":"Center for Complex Particle Systems","country_code":"US","type":"education","relationship":"child"},{"id":"https://openalex.org/I4405258370","ror":"https://ror.org/00hv7q333","display_name":"Zettawatt-Equivalent Ultrashort pulse laser System","country_code":"US","type":"facility","relationship":"child"},{"id":"https://openalex.org/I4405258921","ror":"https://ror.org/01df9hd73","display_name":"Institute for Social Research","country_code":"US","type":"other","relationship":"child"}],"counts_by_year":[{"year":2027,"works_count":4,"oa_works_count":4,"cited_by_count":0},{"year":2026,"works_count":2487,"oa_works_count":1589,"cited_by_count":71},{"year":2025,"works_count":22243,"oa_works_count":14901,"cited_by_count":27126},{"year":2024,"works_count":23505,"oa_works_count":16530,"cited_by_count":110287},{"year":2023,"works_count":23035,"oa_works_count":16404,"cited_by_count":226656},{"year":2022,"works_count":456362,"oa_works_count":448843,"cited_by_count":333830},{"year":2021,"works_count":22297,"oa_works_count":15504,"cited_by_count":484898},{"year":2020,"works_count":25893,"oa_works_count":18305,"cited_by_count":707421},{"year":2019,"works_count":22929,"oa_works_count":15352,"cited_by_count":672563},{"year":2018,"works_count":20675,"oa_works_count":12227,"cited_by_count":719217},{"year":2017,"works_count":18315,"oa_works_count":9827,"cited_by_count":774510},{"year":2016,"works_count":16843,"oa_works_count":9158,"cited_by_count":801111},{"year":2015,"works_count":15713,"oa_works_count":8216,"cited_by_count":879665},{"year":2014,"works_count":15322,"oa_works_count":7746,"cited_by_count":767144},{"year":2013,"works_count":14809,"oa_works_count":7378,"cited_by_count":788449},{"year":2012,"works_count":14549,"oa_works_count":7305,"cited_by_count":810578},{"year":2011,"works_count":13207,"oa_works_count":5956,"cited_by_count":787692},{"year":2010,"works_count":12627,"oa_works_count":5548,"cited_by_count":844939}],"roles":[{"role":"funder","id":"https://openalex.org/F4320309652","works_count":32442},{"role":"institution","id":"https://openalex.org/I27837315","works_count":941687},{"role":"publisher","id":"https://openalex.org/P4310316579","works_count":1272}],"topics":[{"id":"https://openalex.org/T10048","display_name":"Particle physics theoretical and experimental studies","count":6010,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10325","display_name":"Astro and Planetary Science","count":4389,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11949","display_name":"Nuclear Physics and Applications","count":4109,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3108","display_name":"Radiation"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10039","display_name":"Stellar, planetary, and galactic studies","count":3994,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10235","display_name":"Health disparities and outcomes","count":3930,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3306","display_name":"Health"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10391","display_name":"Healthcare Policy and Management","count":3616,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10527","display_name":"High-Energy Particle Collisions Research","count":3600,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10804","display_name":"Health Systems, Economic Evaluations, Quality of Life","count":3410,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10224","display_name":"Quantum Chromodynamics and Particle Interactions","count":3330,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3106","display_name":"Nuclear and High Energy Physics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12646","display_name":"Inorganic Fluorides and Related Compounds","count":3260,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1604","display_name":"Inorganic Chemistry"},"field":{"id":"https://openalex.org/fields/16","display_name":"Chemistry"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10522","display_name":"Medical Imaging Techniques and Applications","count":3241,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2741","display_name":"Radiology, Nuclear Medicine and Imaging"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10251","display_name":"Solar and Space Plasma Dynamics","count":3076,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10026","display_name":"Galaxies: Formation, Evolution, Phenomena","count":3066,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10182","display_name":"Child and Adolescent Psychosocial and Emotional Development","count":3007,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3203","display_name":"Clinical Psychology"},"field":{"id":"https://openalex.org/fields/32","display_name":"Psychology"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10159","display_name":"Ionosphere and magnetosphere dynamics","count":2996,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10477","display_name":"Astrophysics and Star Formation Studies","count":2721,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10077","display_name":"Neuroscience and Neuropharmacology Research","count":2703,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2804","display_name":"Cellular and Molecular Neuroscience"},"field":{"id":"https://openalex.org/fields/28","display_name":"Neuroscience"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T10269","display_name":"Epigenetics and DNA Methylation","count":2665,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T12564","display_name":"Sensor Technology and Measurement Systems","count":2596,"score":0.9994000196456909,"subfield":{"id":"https://openalex.org/subfields/1705","display_name":"Computer Networks and Communications"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11930","display_name":"Cardiac, Anesthesia and Surgical Outcomes","count":2552,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2705","display_name":"Cardiology and Cardiovascular Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T12917","display_name":"Astronomy and Astrophysical Research","count":2517,"score":0.9998999834060669,"subfield":{"id":"https://openalex.org/subfields/3105","display_name":"Instrumentation"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10015","display_name":"Genomics and Phylogenetic Studies","count":2512,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T10543","display_name":"Prostate Cancer Treatment and Research","count":2469,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2740","display_name":"Pulmonary and Respiratory Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10645","display_name":"Cardiac Arrest and Resuscitation","count":2447,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/2711","display_name":"Emergency Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10521","display_name":"RNA and protein synthesis mechanisms","count":2422,"score":1.0,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}}],"topic_share":[{"id":"https://openalex.org/T14440","display_name":"Quality of Life Measurement","value":0.1774779,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T14320","display_name":"Optics and Image Analysis","value":0.039857,"subfield":{"id":"https://openalex.org/subfields/1404","display_name":"Management Information Systems"},"field":{"id":"https://openalex.org/fields/14","display_name":"Business, Management and Accounting"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T12281","display_name":"Animal testing and alternatives","value":0.0361171,"subfield":{"id":"https://openalex.org/subfields/3404","display_name":"Small Animals"},"field":{"id":"https://openalex.org/fields/34","display_name":"Veterinary"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10652","display_name":"Racial and Ethnic Identity Research","value":0.0281192,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T11170","display_name":"Biomimetic flight and propulsion mechanisms","value":0.0269944,"subfield":{"id":"https://openalex.org/subfields/2202","display_name":"Aerospace Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T11539","display_name":"Survey Methodology and Nonresponse","value":0.0260274,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T12646","display_name":"Inorganic Fluorides and Related Compounds","value":0.0243669,"subfield":{"id":"https://openalex.org/subfields/1604","display_name":"Inorganic Chemistry"},"field":{"id":"https://openalex.org/fields/16","display_name":"Chemistry"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10843","display_name":"Diversity and Career in Medicine","value":0.0241965,"subfield":{"id":"https://openalex.org/subfields/3318","display_name":"Gender Studies"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T13736","display_name":"Steroid Chemistry and Biochemistry","value":0.0239324,"subfield":{"id":"https://openalex.org/subfields/1312","display_name":"Molecular Biology"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T11890","display_name":"Scientific Measurement and Uncertainty Evaluation","value":0.0238833,"subfield":{"id":"https://openalex.org/subfields/1804","display_name":"Statistics, Probability and Uncertainty"},"field":{"id":"https://openalex.org/fields/18","display_name":"Decision Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10875","display_name":"Pesticide Exposure and Toxicity","value":0.0237792,"subfield":{"id":"https://openalex.org/subfields/1110","display_name":"Plant Science"},"field":{"id":"https://openalex.org/fields/11","display_name":"Agricultural and Biological Sciences"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T11033","display_name":"Contact Dermatitis and Allergies","value":0.0231777,"subfield":{"id":"https://openalex.org/subfields/2708","display_name":"Dermatology"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T12564","display_name":"Sensor Technology and Measurement Systems","value":0.0228602,"subfield":{"id":"https://openalex.org/subfields/1705","display_name":"Computer Networks and Communications"},"field":{"id":"https://openalex.org/fields/17","display_name":"Computer Science"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T13075","display_name":"Legal Systems and Judicial Processes","value":0.0208929,"subfield":{"id":"https://openalex.org/subfields/3320","display_name":"Political Science and International Relations"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T13648","display_name":"Cerebrovascular and genetic disorders","value":0.0203678,"subfield":{"id":"https://openalex.org/subfields/2728","display_name":"Neurology"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T10261","display_name":"Genetic Associations and Epidemiology","value":0.0198635,"subfield":{"id":"https://openalex.org/subfields/1311","display_name":"Genetics"},"field":{"id":"https://openalex.org/fields/13","display_name":"Biochemistry, Genetics and Molecular Biology"},"domain":{"id":"https://openalex.org/domains/1","display_name":"Life Sciences"}},{"id":"https://openalex.org/T13138","display_name":"Legal and Constitutional Studies","value":0.0195024,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10235","display_name":"Health disparities and outcomes","value":0.0190722,"subfield":{"id":"https://openalex.org/subfields/3306","display_name":"Health"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10026","display_name":"Galaxies: Formation, Evolution, Phenomena","value":0.0185358,"subfield":{"id":"https://openalex.org/subfields/3103","display_name":"Astronomy and Astrophysics"},"field":{"id":"https://openalex.org/fields/31","display_name":"Physics and Astronomy"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10391","display_name":"Healthcare Policy and Management","value":0.0184369,"subfield":{"id":"https://openalex.org/subfields/2002","display_name":"Economics and Econometrics"},"field":{"id":"https://openalex.org/fields/20","display_name":"Economics, Econometrics and Finance"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T13856","display_name":"Advanced Power Generation Technologies","value":0.0182906,"subfield":{"id":"https://openalex.org/subfields/2208","display_name":"Electrical and Electronic Engineering"},"field":{"id":"https://openalex.org/fields/22","display_name":"Engineering"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T10845","display_name":"Advanced Causal Inference Techniques","value":0.0181425,"subfield":{"id":"https://openalex.org/subfields/2613","display_name":"Statistics and Probability"},"field":{"id":"https://openalex.org/fields/26","display_name":"Mathematics"},"domain":{"id":"https://openalex.org/domains/3","display_name":"Physical Sciences"}},{"id":"https://openalex.org/T12174","display_name":"Hospital Admissions and Outcomes","value":0.0180397,"subfield":{"id":"https://openalex.org/subfields/2711","display_name":"Emergency Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}},{"id":"https://openalex.org/T14288","display_name":"Law, Rights, and Freedoms","value":0.0177982,"subfield":{"id":"https://openalex.org/subfields/3312","display_name":"Sociology and Political Science"},"field":{"id":"https://openalex.org/fields/33","display_name":"Social Sciences"},"domain":{"id":"https://openalex.org/domains/2","display_name":"Social Sciences"}},{"id":"https://openalex.org/T10543","display_name":"Prostate Cancer Treatment and Research","value":0.0173817,"subfield":{"id":"https://openalex.org/subfields/2740","display_name":"Pulmonary and Respiratory Medicine"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"}}],"is_super_system":false,"works_api_url":"https://api.openalex.org/works?filter=institutions.id:I27837315","updated_date":"2026-02-17T06:01:00","created_date":"2016-06-24T00:00:00"}],"group_by":[]}"#;
        let json_batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted =
            extract_tabular("bytes", &[json_batch], &DataFormat::JsonSchema, &AvailableSubjects::OpenAlexResponseInstitutions)?;

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(test, ["InstitutionTable", "InstitutionDisplayNameAcronymsTable", "InstitutionDisplayNameAlternativesTable", "InstitutionGeoTable", "InstitutionIdsTable", "InstitutionAssociatedInstitutionTable", "InstitutionRepositoryTable", "InstitutionRoleTable", "InstitutionInternationalNamesTable", "InstitutionSummaryStatsTable", "InstitutionCountsByYearTable", "InstitutionLineageTable"]);
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(test, ["extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular"]);
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(test, ["InstitutionTable", "InstitutionDisplayNameAcronymsTable", "InstitutionDisplayNameAlternativesTable", "InstitutionGeoTable", "InstitutionIdsTable", "InstitutionAssociatedInstitutionTable", "InstitutionRepositoryTable", "InstitutionRoleTable", "InstitutionInternationalNamesTable", "InstitutionSummaryStatsTable", "InstitutionCountsByYearTable", "InstitutionLineageTable"]);
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);

        // OpenAlexResponseTopics
        // Make the tabular data
        let open_alex_response_str = r#"{"meta":{"count":8,"db_response_time_ms":3,"page":1,"per_page":1,"groups_count":null},"results":[{"id":"https://openalex.org/T11636","display_name":"Artificial Intelligence in Healthcare and Education","description":"This cluster of papers explores the intersection of artificial intelligence and medicine, focusing on applications in healthcare, medical imaging, clinical decision support, and the ethical challenges associated with AI implementation. It delves into topics such as machine learning, big data, precision medicine, and the potential impact of AI on health equity.","keywords":["Artificial Intelligence","Machine Learning","Healthcare","Medical Imaging","Clinical Decision Support","Ethical Challenges","Big Data","Precision Medicine","Radiology","Health Equity"],"ids":{"openalex":"https://openalex.org/T11636","wikipedia":"https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare"},"subfield":{"id":"https://openalex.org/subfields/2718","display_name":"Health Informatics"},"field":{"id":"https://openalex.org/fields/27","display_name":"Medicine"},"domain":{"id":"https://openalex.org/domains/4","display_name":"Health Sciences"},"siblings":[],"relevance_score":9815.777,"works_count":105814,"cited_by_count":734445,"works_api_url":"https://api.openalex.org/works?filter=topics.id:T11636","updated_date":"2026-02-17T03:01:20","created_date":"2024-01-23T15:27:11"}],"group_by":[]}"#;
        let json_batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["json".to_string()],
            vec![Bytes::from(open_alex_response_str).to_vec()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )?;

        // Extract the tabular data
        let extracted =
            extract_tabular("bytes", &[json_batch], &DataFormat::JsonSchema, &AvailableSubjects::OpenAlexResponseTopics)?;

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])?
            .build()?;
        let test = table.get_column_as_vec_str("name");
        assert_eq!(test, ["TopicTable", "TopicDomainTable", "TopicFieldTable", "TopicSubfieldTable", "TopicIdsTable", "TopicKeywordTable"]);
        let test = table.get_column_as_vec_str("publisher");
        assert_eq!(test, ["extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular", "extract_tabular"]);
        let test = table.get_column_as_vec_str("subject");
        assert_eq!(test, ["TopicTable", "TopicDomainTable", "TopicFieldTable", "TopicSubfieldTable", "TopicIdsTable", "TopicKeywordTable"]);
        let test = table.get_column_as_vec_str("format");
        assert_eq!(test, ["Ipc", "Ipc", "Ipc", "Ipc", "Ipc", "Ipc"]);

        Ok(())
    }
}
