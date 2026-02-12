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
                Ok(open_alex_response_works) => {
                    let batch = open_alex_response_works.to_record_batch("extract_tabular")?;
                    Table::get_builder()
                        .with_name("attachment")
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
        Ok(())
    }
}
