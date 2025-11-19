use std::{collections::HashMap, fmt::Display, io::Cursor, sync::Arc};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, Function, FunctionParameters,
    JSONSchemaDefine, JSONSchemaType, MappableTrait, Table,
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
    tags: Vec<XMLTags>,
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
        extract_xml_tags(&self.lhs_values, lhs_args, &self.format, &as_columns, &self.tags)
    }
}

/// XML Tags with for OWL
/// 
/// # Notes
/// * rdf:about is the ID for the subject
/// * rdf:resource is the ID for the object
#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum XMLTags {
    /// RDF type or instance of a class
    #[value(name = "rdf:type")]
    RdfType,
    /// RDF class
    #[value(name = "rdf:description")]
    RdfDescription,
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
    /// OBO synonyms
    #[value(name = "oboInOwl:hasBroadSynonym")]
    OboInOwlHasBroadSynonym,
    /// OBO synonyms
    #[value(name = "oboInOwl:hasNarrowSynonym")]
    OboInOwlHasNarrowSynonym,
    /// OBO Namespace
    #[value(name = "ooboInOwl:hasOBONamespace")]
    OboInOwlHasOboNamespace,
    /// OBO ID
    #[value(name = "ooboInOwl:id")]
    OboInOwlId,
    /// OBO ID
    #[value(name = "ooboInOwl:hasAlternativeId")]
    OboInOwlHasAlternativeId,
    /// OWL The Property ID
    #[value(name = "owl:ObjectProperty")]
    OwlObjectProperty,
    /// OWL Same as
    #[value(name = "owl:sameAs")]
    OwlSameAs,
    /// OWL inverse of
    #[value(name = "owl:inverseOf ")]
    OwlInverseOf,
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
            | Self::OboInOwlHasAlternativeId
            | Self::OboInOwlHasRelatedSynonym
            | Self::OboInOwlHasExactSynonym
            | Self::OboInOwlHasBroadSynonym
            | Self::OboInOwlHasNarrowSynonym
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
            | Self::RdfType
            | Self::OwlSameAs
            | Self::OwlInverseOf
            | Self::OwlEquivalentClass
            | Self::OwlEquivalentProperty
            | Self::OboInOwlInsubset => {
                e.attributes().flatten().filter_map(|attr| if attr.key.as_ref() == b"rdf:resource" {
                        Some(String::from_utf8_lossy(&attr.value).to_string())
                    } else {
                        None
                    }).collect::<Vec<_>>()
            },
            Self::Custom(_s) => {
                if let Ok(Event::Text(text)) = reader.read_event_into(buf) {
                    vec![String::from_utf8_lossy(&text as &[u8]).to_string()]
                } else {
                    vec![]
                }
            },
        }
    }
    /// OWL common tags
    pub fn owl_common() -> Vec<XMLTags> {
        vec![
            Self::RdfType,
            Self::RdfsLabel,
            Self::RdfsSeeAlso,
            Self::OboDefinition,
            Self::OboInOwlHasOboNamespace,
            Self::OboInOwlId,
            Self::OboInOwlHasAlternativeId,
            Self::OboInOwlHasRelatedSynonym,
            Self::OboInOwlHasExactSynonym,
            Self::OboInOwlHasBroadSynonym,
            Self::OboInOwlHasNarrowSynonym,
            Self::OwlSameAs,
            Self::OboInOwlInsubset,
        ]
    }
    /// OWL Class tags
    pub fn owl_classes() -> Vec<XMLTags> {
        let mut tags = vec![
            Self::OwlClass,
            Self::RdfsSubClassOf
        ];
        tags.extend(Self::owl_common());
        tags
    }
    /// OWL Class tags
    pub fn owl_properties() -> Vec<XMLTags> {
        let mut tags = vec![
            Self::OwlObjectProperty,
            Self::OwlInverseOf,
            Self::RdfsSubPropertyOf,
            Self::RdfsDomain,
            Self::RdfsRange
        ];
        tags.extend(Self::owl_common());
        tags
    }
}

impl Display for XMLTags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RdfType => write!(f, "rdf:type"),
            Self::RdfsLabel => write!(f, "rdfs:label"),
            Self::OboDefinition => write!(f, "obo:IAO_0000115"),
            Self::OwlClass => write!(f, "owl:Class"),
            Self::OwlEquivalentClass => write!(f, "owl:equivalentClass"),
            Self::RdfsSubClassOf => write!(f, "rdfs:subclassOf"),
            Self::OboInOwlInsubset => write!(f, "oboInOwl:inSubset"),
            Self::OboInOwlHasRelatedSynonym => write!(f, "oboInOwl:hasRelatedSynonym"),
            Self::OboInOwlHasExactSynonym => write!(f, "oboInOwl:hasExactSynonym"),
            Self::OboInOwlHasBroadSynonym => write!(f, "oboInOwl:hasBroadSynonym"),
            Self::OboInOwlHasNarrowSynonym => write!(f, "oboInOwl:hasNarrowSynonym"),
            Self::OboInOwlHasOboNamespace => write!(f, "oboInOwl:hasOBONamespace"),
            Self::OboInOwlId => write!(f, "oboInOwl:id"),
            Self::OboInOwlHasAlternativeId => write!(f, "oboInOwl:hasAlternativeId"),
            Self::OwlObjectProperty => write!(f, "owl:ObjectProperty"),
            Self::OwlSameAs => write!(f, "owl:sameAs"),
            Self::OwlInverseOf => write!(f, "owl:inverseOf "),
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
/// * `xml_tags` - The tags in the XML to extract
/// * `xml_attributes` - The possible attributes of a tag in the XML to extract
/// 
/// # Notes
/// * Basic parsing of tags with a flat hierarchy is currently supported
/// * Hierarchical or nested structures are not yet supported
/// * See <https://github.com/phillord/horned-owl> for a full-fledged OWL parser
#[instrument(skip(lhs_values, lhs_args, format, as_columns, tags))]
pub fn extract_xml_tags(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
    as_columns: &[&str],
    tags: &[XMLTags],
) -> Result<RecordBatch> {
    // Extract out the bytes
    let args_table = Table::get_builder()
        .with_name("extract_xml_tags")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table.get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter().flatten().collect::<Vec<_>>();

    // Read the XML document
    let cursor = Cursor::new(&values_vec);
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
    let mut parse = Vec::new();
    let mut data = Vec::new();
    let mut subjects = Vec::new();
    let mut predicates: Vec<HashMap<String, Vec<String>>> = Vec::new();
    while let Ok(event) = reader.read_event_into(&mut buf) {
        match event {
            Event::Empty(ref e) => {
                // Parse the tag attribute objects
                let attributes = e.attributes()
                    .flatten()
                    .map(|attr| (String::from_utf8_lossy(attr.key.as_ref()).to_string(), String::from_utf8_lossy(&attr.value).to_string()))
                    .collect::<Vec<_>>();

                // Update the rows with the new triple(s)
            },
            Event::Text(ref e) => {
                // Parse the literal object
                todo!()
            },
            Event::Start(ref e) => {
                // Recurse the subject
                let start_tag = std::str::from_utf8(e.name().into_inner()).unwrap_or_default();
                subjects.push(start_tag);
                
                // Parse the tag attributes for the subject if any
                let attributes = e.attributes()
                    .flatten()
                    .map(|attr| (String::from_utf8_lossy(attr.key.as_ref()).to_string(), String::from_utf8_lossy(&attr.value).to_string()))
                    .collect::<Vec<_>>();

                // Initialize the predicates
                let obj: HashMap<String, Vec<String>> = HashMap::new();
                for (k, v) in attributes {
                    if let Some(val) = obj.get_mut(&k) {
                        val.push(v);
                    } else {
                        obj.insert(k, vec![v]);
                    }
                }
                predicates.push(obj);
            }
            Event::End(ref e) => {
                let end_tag = std::str::from_utf8(e.name().into_inner()).unwrap_or_default();
                if let Some(tag) = subjects.pop() {
                    assert_eq!(end_tag, tag);
                    if let Some(predicates) = predicates.pop() {

                        // Add the data row
                        for t in subjects {

                        }
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    // Initialize the columns
    let mut columns = HashMap::new();
    for tag in subjects.iter() {
        columns.insert(tag.to_string(), data.iter().map(|_| String::new()).collect::<Vec<_>>());
    }

    // Update the columns with the parsed tag values
    for (i, row) in data.into_iter().enumerate() {
        for (k, v) in row.into_iter() {
            if let Some(column) = columns.get_mut(&k) {
                if let Some(value) = column.get_mut(i) {
                    value.push_str(&v.join(";"));
                } else {
                    return Err(anyhow!("iterator `{i}` was not found in XML parsed columns with length {}", columns.len()));
                }
            } else {
                return Err(anyhow!("Key `{k}` was not found in XML parsed columns with keys {:?}", columns.keys()));
            }
        }
    }

    // Build the batch
    let mut batch_vec = Vec::new();
    for (i, (_tag, column)) in columns.into_iter().enumerate() {
        let arr: ArrayRef = Arc::new(StringArray::from(column));
        batch_vec.push((as_columns.get(i).unwrap().to_string(), arr));
    }

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_core::{
        BuildableTrait, BuilderTrait, DataFormat, Table, TableBuilderTrait,
        TableTrait, create_blob_batch,
    };
    use phymes_diagnostics::create_timestamp_micros;

    use super::*;

    #[test]
    fn test_extract_xml_tags_owl_class() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.example.com/iri#"
     xml:base="http://www.example.com/iri"
     xmlns:o="http://www.example.com/iri#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://www.example.com/iri">
        <owl:versionIRI rdf:resource="http://www.example.com/viri"/>
    </owl:Ontology>  

    <!-- http://purl.obolibrary.org/obo/GO_0010958 -->

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_0010958">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="http://purl.obolibrary.org/obo/GO_0065007"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002211"/>
                        <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/GO_0089718"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/GO_1903789"/>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002211"/>
                <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/GO_0089718"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <obo:IAO_0000115>Any process that modulates the frequency, rate or extent of amino acid import into a cell.</obo:IAO_0000115>
        <oboInOwl:created_by>tb</oboInOwl:created_by>
        <oboInOwl:creation_date>2009-05-06T11:33:12Z</oboInOwl:creation_date>
        <oboInOwl:hasBroadSynonym>regulation of amino acid import</oboInOwl:hasBroadSynonym>
        <oboInOwl:hasOBONamespace>biological_process</oboInOwl:hasOBONamespace>
        <oboInOwl:id>GO:0010958</oboInOwl:id>
        <rdfs:label>regulation of amino acid import across plasma membrane</rdfs:label>
    </owl:Class>

    <owl:Axiom>
        <owl:annotatedSource rdf:resource="http://purl.obolibrary.org/obo/GO_0010958"/>
        <owl:annotatedProperty rdf:resource="http://purl.obolibrary.org/obo/IAO_0000115"/>
        <owl:annotatedTarget>Any process that modulates the frequency, rate or extent of amino acid import into a cell.</owl:annotatedTarget>
        <oboInOwl:hasDbXref>GOC:dph</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>GOC:tb</oboInOwl:hasDbXref>
    </owl:Axiom>

    <!-- http://purl.obolibrary.org/obo/GO_0010968 -->

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_0010968">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="http://purl.obolibrary.org/obo/GO_0065007"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002211"/>
                        <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/GO_0007020"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/GO_0031113"/>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002211"/>
                <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/GO_0007020"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <obo:IAO_0000115>Any process that modulates the rate, frequency or extent of microtubule nucleation. Microtubule nucleation is the &apos;de novo&apos; formation of a microtubule, in which tubulin heterodimers form metastable oligomeric aggregates, some of which go on to support formation of a complete microtubule. Microtubule nucleation usually occurs from a specific site within a cell.</obo:IAO_0000115>
        <oboInOwl:created_by>tb</oboInOwl:created_by>
        <oboInOwl:creation_date>2009-05-20T11:51:21Z</oboInOwl:creation_date>
        <oboInOwl:hasOBONamespace>biological_process</oboInOwl:hasOBONamespace>
        <oboInOwl:id>GO:0010968</oboInOwl:id>
        <rdfs:label>regulation of microtubule nucleation</rdfs:label>
    </owl:Class>

    <owl:Axiom>
        <owl:annotatedSource rdf:resource="http://purl.obolibrary.org/obo/GO_0010968"/>
        <owl:annotatedProperty rdf:resource="http://purl.obolibrary.org/obo/IAO_0000115"/>
        <owl:annotatedTarget>Any process that modulates the rate, frequency or extent of microtubule nucleation. Microtubule nucleation is the &apos;de novo&apos; formation of a microtubule, in which tubulin heterodimers form metastable oligomeric aggregates, some of which go on to support formation of a complete microtubule. Microtubule nucleation usually occurs from a specific site within a cell.</owl:annotatedTarget>
        <oboInOwl:hasDbXref>GOC:dph</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>GOC:tb</oboInOwl:hasDbXref>
    </owl:Axiom>
</rdf:RDF>"#;

        // Make the xml data
        let batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["owl".to_string()],
            vec![owl.into()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the xml tags
        let as_columns = XMLTags::owl_classes()
            .into_iter()
            .map(|x| x.to_string().split(":").collect::<Vec<_>>().get(1).unwrap().to_string())
            .collect::<Vec<_>>();
        let extracted = extract_xml_tags(
            "bytes", 
            &[batch], 
            &DataFormat::Owl,
            &as_columns.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            &XMLTags::owl_classes()).unwrap();
        dbg!(&extracted);

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), XMLTags::owl_classes().len());
        assert_eq!(extracted.num_rows(), 2);

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table.get_column_as_vec_str("label");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("IAO_0000115");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("Class");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("equivalentClass");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("subclassOf");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("hasBroadSynonym");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("hasOBONamespace");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("id");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("hasRelatedSynonym");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("hasExactSynonym");
        assert_eq!(result, ["", ""]);
    }    

    #[test]
    fn test_extract_xml_tags_owl_properties() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.example.com/iri#"
     xml:base="http://www.example.com/iri"
     xmlns:o="http://www.example.com/iri#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://www.example.com/iri">
        <owl:versionIRI rdf:resource="http://www.example.com/viri"/>
    </owl:Ontology>

    <!-- http://purl.obolibrary.org/obo/RO_0002437 -->

    <owl:ObjectProperty rdf:about="http://purl.obolibrary.org/obo/RO_0002437">
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/RO_0002321"/>
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/RO_0002434"/>
        <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#SymmetricProperty"/>
        <rdfs:domain rdf:resource="http://purl.obolibrary.org/obo/BFO_0000040"/>
        <rdfs:range rdf:resource="http://purl.obolibrary.org/obo/BFO_0000040"/>
        <obo:IAO_0000115>An interaction relationship in which at least one of the partners is an organism and the other is either an organism or an abiotic entity with which the organism interacts.</obo:IAO_0000115>
        <obo:IAO_0000117 rdf:resource="https://orcid.org/0000-0002-6601-2165"/>
        <obo:IAO_0000118>interacts with on organism level</obo:IAO_0000118>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/ro/subsets#ro-eco"/>
        <rdfs:label>biotically interacts with</rdfs:label>
        <rdfs:seeAlso rdf:resource="http://dx.doi.org/10.1016/j.ecoinf.2014.08.005"/>
        <rdfs:seeAlso>http://eol.org/schema/terms/interactsWith</rdfs:seeAlso>
    </owl:ObjectProperty>

    <!-- http://purl.obolibrary.org/obo/RO_0002438 -->

    <owl:ObjectProperty rdf:about="http://purl.obolibrary.org/obo/RO_0002438">
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/RO_0002574"/>
        <obo:IAO_0000112>lions trophically interact with the zebras that they eat</obo:IAO_0000112>
        <obo:IAO_0000115>An interaction relationship in which the partners are related via a feeding relationship.</obo:IAO_0000115>
        <obo:IAO_0000117 rdf:resource="https://orcid.org/0000-0002-6601-2165"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/ro/subsets#ro-eco"/>
        <rdfs:label>trophically interacts with</rdfs:label>
        <rdfs:seeAlso rdf:resource="http://dx.doi.org/10.1016/j.ecoinf.2014.08.005"/>
    </owl:ObjectProperty>
</rdf:RDF>"#;

        // Make the xml data
        let batch = create_blob_batch(
            vec!["attachment".to_string()],
            vec!["owl".to_string()],
            vec![owl.into()],
            vec!["".to_string()],
            vec![create_timestamp_micros()],
        )
        .unwrap();

        // Extract the xml tags
        let as_columns = XMLTags::owl_classes()
            .into_iter()
            .map(|x| x.to_string().split(":").collect::<Vec<_>>().get(1).unwrap().to_string())
            .collect::<Vec<_>>();
        let extracted = extract_xml_tags(
            "bytes", 
            &[batch], 
            &DataFormat::Owl,
            &as_columns.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            &XMLTags::owl_classes()).unwrap();
        dbg!(&extracted);

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), XMLTags::owl_classes().len());
        assert_eq!(extracted.num_rows(), 2);

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table.get_column_as_vec_str("label");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("IAO_0000115");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("ObjectProperty");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("subPropertyOf");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("inverseOf");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("seeAlso");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("inSubset");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("id");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("hasExactSynonym");
        assert_eq!(result, ["", ""]);
    }
}
