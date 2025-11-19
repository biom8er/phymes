use std::{collections::HashMap, fmt::Display, io::Cursor};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, Function, FunctionParameters,
    JSONSchemaDefine, JSONSchemaType, MappableTrait, Table, TableBuilder,
    TableBuilderTrait, TableTrait, Tool, ToolType,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use quick_xml::{Reader, events::{BytesStart, Event}};

use crate::{ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Extract xml tags in either XML or OWL format from Bytes
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExtractXMLTags {
    lhs_values: String,
    format: DataFormat,
    as_columns: Vec<String>,
    tags: Vec<String>,
}

impl MappableTrait for ExtractXMLTags {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for ExtractXMLTags {
    fn get_description(&self) -> String {
        "Extract XML data in either XMl or OWL format from Bytes".to_string()
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

impl DataOperatorTrait for ExtractXMLTags {
    fn new(config: &DataConfig) -> Result<Self>
    where
        Self: Sized,
    {
        // Extract the members from the DataConfig
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
        let format = config.format.ok_or(anyhow!(
            "Missing `format` for `{}`.",
            Self::get_static_name()
        ))?;
        let as_columns = config
            .as_columns
            .as_ref()
            .cloned()
            .ok_or(anyhow!(
                "Missing `as_columns` for `{}`.",
                Self::get_static_name()
            ))?;
        let tags = config
            .tags
            .as_ref()
            .cloned()
            .ok_or(anyhow!(
                "Missing `tags` for `{}`.",
                Self::get_static_name()
            ))?;

        // Ensure that the array lengths for values, columns, and operators match
        if as_columns.len() != tags.len() {
            return Err(anyhow!(
                "as_columns length {} is not equal to the tags length {}",
                as_columns.len(),
                tags.len()
            ));
        }

        Ok(ExtractXMLTags { lhs_values, format, as_columns, tags })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        let as_columns = self
            .as_columns
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let tags = self
            .tags
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        extract_xml_tags(&self.lhs_values, lhs_args, &self.format, &as_columns, &tags)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum XMLTags {
    /// RDFS Name of the term
    #[default]
    #[value(name = "rdfs:label")]
    RdfsLabel,
    /// OBO Definition of the term
    #[value(name = "obo:IAO_0000115")]
    OboDefinition,
    /// OWL The Class ID
    #[value(name = "owl:Class")]
    OwlClass,
    /// OWL equivalent Class ID
    #[value(name = "owl:equivalentClass")]
    OwlEquivalentClass,
    /// RDFS Sub class of the class
    #[value(name = "rdfs:subclassOf")]
    RdfsSubClassOf,
    /// OBO subset
    #[value(name = "oboInOwl:inSubset")]
    OboInOwlInsubset,
    /// OBO synonyms
    #[value(name = "oboInOwl:hasRelatedSynonym")]
    OboInOwlHasRelatedSynonym,
    /// OBO synonyms
    #[value(name = "oboInOwl:hasExactSynonym")]
    OboInOwlHasExactSynonym,
    /// OBO Namespace
    #[value(name = "ooboInOwl:hasOBONamespace")]
    OboInOwlHasOboNamespace,
    /// OBO ID
    #[value(name = "ooboInOwl:id")]
    OboInOwlId,
    /// OWL The Property ID
    #[value(name = "owl:ObjectProperty")]
    OwlObjectProperty,
    /// OWL Same as
    #[value(name = "owl:sameAs")]
    OwlSameAs,
    /// OWL The equivalent Property ID
    #[value(name = "owl:equivalentProperty")]
    OwlEquivalentProperty,
    /// RDFS domain of the property
    #[value(name = "rdfs:domain")]
    RdfsDomain,
    /// RDFS range of the property
    #[value(name = "rdfs:range")]
    RdfsRange,
    /// Sub properties of the property
    #[value(name = "rdfs:subPropertyOf")]
    RdfsSubPropertyOf,
    /// RDFS see also
    #[value(name = "rdfs:seeAlso ")]
    RdfsSeeAlso,
    /// SKOS mapping property
    #[value(name = "skos:closeMatch")]
    SkosCloseMatch,
    /// SKOS mapping property
    #[value(name = "skos:exactMatch")]
    SkosExactMatch,
    /// SKOS mapping property
    #[value(name = "skos:broadMatch")]
    SkosBroadMatch,
    /// SKOS mapping property
    #[value(name = "skos:narrowMatch")]
    SkosNarrowMatch,
    /// SKOS mapping property
    #[value(name = "skos:relatedMatch")]
    SkosRelatedMatch,
    /// SKOS semantic relations
    #[value(name = "skos:semanticRelation")]
    SkosSemanticRelation,
    /// SKOS semantic relations
    #[value(name = "skos:broader")]
    SkosBroader,
    /// SKOS semantic relations
    #[value(name = "skos:narrower")]
    SkosNarrower,
    /// SKOS semantic relations
    #[value(name = "skos:related")]
    SkosRelated,
    /// SKOS semantic relations
    #[value(name = "skos:broaderTransitive")]
    SkosBroaderTransitive,
    /// SKOS semantic relations
    #[value(name = "skos:narrowerTransitive")]
    SkosNarrowerTransitive,
    /// Named tag provided by the user
    #[value(skip)]
    Custom(String)
}

impl XMLTags {
    /// Parse the XML tag
    pub fn parse<'a>(&self, e: &BytesStart<'a>, reader: &mut Reader<Cursor<&Vec<u8>>>, buf: &mut Vec<u8>) -> Vec<String> {
        match self {
            Self::RdfsLabel 
            | Self::RdfsSeeAlso
            | Self::OboDefinition
            | Self::OboInOwlHasOboNamespace
            | Self::OboInOwlId
            | Self::OboInOwlHasRelatedSynonym
            | Self::OboInOwlHasExactSynonym
            | Self::SkosCloseMatch
            | Self::SkosExactMatch
            | Self::SkosBroadMatch
            | Self::SkosNarrowMatch
            | Self::SkosRelatedMatch
            | Self::SkosSemanticRelation
            | Self::SkosBroader
            | Self::SkosNarrower
            | Self::SkosRelated
            | Self::SkosBroaderTransitive
            | Self::SkosNarrowerTransitive => {
                if let Ok(Event::Text(text)) = reader.read_event_into(buf) {
                    vec![String::from_utf8_lossy(&text as &[u8]).to_string()]
                } else {
                    vec![]
                }
            },
            Self::OwlClass | Self::OwlObjectProperty => {
                e.attributes().flatten().filter_map(|attr| if attr.key.as_ref() == b"rdf:about" {
                        Some(String::from_utf8_lossy(&attr.value).to_string())
                    } else {
                        None
                    }).collect::<Vec<_>>()
            },
            Self::RdfsDomain 
            | Self::RdfsRange 
            | Self::RdfsSubPropertyOf 
            | Self::RdfsSubClassOf
            | Self::OwlSameAs
            | Self::OwlEquivalentClass
            | Self::OwlEquivalentProperty
            | Self::OboInOwlInsubset => {
                e.attributes().flatten().filter_map(|attr| if attr.key.as_ref() == b"rdf:resource" {
                        Some(String::from_utf8_lossy(&attr.value).to_string())
                    } else {
                        None
                    }).collect::<Vec<_>>()
            },
            Self::Custom(_s) => todo!(),
        }
    }
}

impl Display for XMLTags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RdfsLabel => write!(f, "rdfs:label"),
            Self::OboDefinition => write!(f, "obo:IAO_0000115"),
            Self::OwlClass => write!(f, "owl:Class"),
            Self::OwlEquivalentClass => write!(f, "owl:equivalentClass"),
            Self::RdfsSubClassOf => write!(f, "rdfs:subclassOf"),
            Self::OboInOwlInsubset => write!(f, "oboInOwl:inSubset"),
            Self::OboInOwlHasRelatedSynonym => write!(f, "oboInOwl:hasRelatedSynonym"),
            Self::OboInOwlHasExactSynonym => write!(f, "oboInOwl:hasExactSynonym"),
            Self::OboInOwlHasOboNamespace => write!(f, "oboInOwl:hasOBONamespace"),
            Self::OboInOwlId => write!(f, "oboInOwl:id"),
            Self::OwlObjectProperty => write!(f, "owl:ObjectProperty"),
            Self::OwlSameAs => write!(f, "owl:sameAs"),
            Self::OwlEquivalentProperty => write!(f, "owl:equivalentProperty"),
            Self::RdfsDomain => write!(f, "rdfs:domain"),
            Self::RdfsRange => write!(f, "rdfs:range"),
            Self::RdfsSubPropertyOf => write!(f, "rdfs:subPropertyOf"),
            Self::RdfsSeeAlso => write!(f, "rdfs:seeAlso"),
            Self::SkosCloseMatch => write!(f, "skos:closeMatch"),
            Self::SkosExactMatch => write!(f, "skos:exactMatch"),
            Self::SkosBroadMatch => write!(f, "skos:broadMatch"),
            Self::SkosNarrowMatch => write!(f, "skos:narrowMatch"),
            Self::SkosRelatedMatch => write!(f, "skos:relatedMatch"),
            Self::SkosSemanticRelation => write!(f, "skos:semanticRelation"),
            Self::SkosBroader => write!(f, "skos:broader"),
            Self::SkosNarrower => write!(f, "skos:narrower"),
            Self::SkosRelated => write!(f, "skos:related"),
            Self::SkosBroaderTransitive => write!(f, "skos:broaderTransitive"),
            Self::SkosNarrowerTransitive => write!(f, "skos:narrowerTransitive"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Extract xml tags in either XML or OWL format from Bytes
/// 
/// # Arguments
/// * `lhs_values` - The column to extract data from (i.e., `bytes`)
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `format` - The format of the bytes
/// * `as_columns` - Slice of Strings for the columns of extracted data
/// * `tags` - The tags in the XML to extract
#[instrument(skip(lhs_values, lhs_args, format, as_columns, tags))]
pub fn extract_xml_tags(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
    as_columns: &[&str],
    tags: &[&str],
) -> Result<RecordBatch> {
    // Extract out the bytes
    let args_table = Table::get_builder()
        .with_name("extract_xml_tags")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table.get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter().flatten().collect::<Vec<_>>();

    // Read the XML document
    let mut cursor = Cursor::new(&values_vec);
    let mut reader = match format {
        DataFormat::Xml => Reader::from_reader(cursor),
        DataFormat::Owl => Reader::from_reader(cursor),
        _ => {
            return Err(anyhow!(
                "Unsupported format {format:?} for extract_xml_tags operator."
            ));
        }
    };

    // Extract out the tags
    let mut buf = Vec::new();
    let mut data = Vec::new();
    let mut current_object: Option<HashMap<String, Vec<String>>> = None;
    while let Ok(event) = reader.read_event_into(&mut buf) {
        match event {
            Event::Start(ref e) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or_default();
                if let Ok(xml_tag) = XMLTags::from_str(&tag, false) {
                    let parsed = xml_tag.parse(e, &mut reader, &mut buf);
                    if let Some(current_object) = current_object {
                        current_object.insert(xml_tag.to_string(), parsed);
                    } else {
                        let mut map = HashMap::new();
                        map.insert(xml_tag.to_string(), parsed);
                        current_object.replace(map);
                    }
                } else {
                    return Err(anyhow!("tag `{tag}` is not a supported XML tag."))
                }
            }
            Event::End(ref e) => {
                if let Some(current_object) = current_object.take() {
                    data.push(current_object);
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    // Determine the columns

    // Create the arrays with empty string as the default for missing

    let batch = table.get_record_batches_own().remove(0);
    Ok(batch)
}

pub mod test_extract_xml_tags {
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

    use crate::candle_operators::extract_xml_tags::test_extract_xml_tags::make_scores_table;

    use super::*;

    #[test]
    fn test_extract_xml_tags_csv_format() {
        let csv_format = CsvFormat::default();

        // Make the xml tags
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

        // Extract the xml tags
        let extracted =
            extract_xml_tags("bytes", &[csv_batch], &DataFormat::Csv(csv_format)).unwrap();

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
    fn test_extract_xml_tags_json_format() {
        let json_format = JsonFormat::default();

        // Make the xml tags
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

        // Extract the xml tags
        let extracted =
            extract_xml_tags("bytes", &[json_batch], &DataFormat::Json(json_format)).unwrap();

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
