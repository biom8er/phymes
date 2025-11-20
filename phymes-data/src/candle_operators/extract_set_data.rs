use std::{collections::HashMap, fmt::Display, io::Cursor, sync::Arc};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use arrow::{array::{ArrayRef, RecordBatch, StringArray}, datatypes::DataType};
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
pub struct ExtractSetData {
    lhs_values: String,
    format: DataFormat,
    as_columns: Vec<String>,
}

impl MappableTrait for ExtractSetData {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for ExtractSetData {
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

impl DataOperatorTrait for ExtractSetData {
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

        Ok(ExtractSetData { lhs_values, format, as_columns })
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
        extract_set_data(&self.lhs_values, lhs_args, &self.format, &as_columns)
    }
}

/// XML element
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct XMLElement {
    /// tag
    tag: String,
    /// named attributes for the tag
    attributes: HashMap<String, String>,
}

/// XML element type
#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum XMLType {
    #[default]
    #[value(name = "Element")]
    Element,
    #[value(name = "Text")]
    Text,
}

impl Display for XMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Element => write!(f, "Element"),
            Self::Text => write!(f, "Text"),
        }
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
            Self::RdfDescription => write!(f, "rdf:description"),
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

/// Helper function to parse the attributes of the XML tag into a serialized [XMLElement]
fn parse_xml_tag<'a>(e: &BytesStart<'a>) -> Result<String> {    
    // Parse the tag
    let start_tag = std::str::from_utf8(e.name().into_inner()).unwrap_or_default();

    // Parse the tag attribute objects
    let attributes = e.attributes()
        .flatten()
        .map(|attr| (String::from_utf8_lossy(attr.key.as_ref()).to_string(), String::from_utf8_lossy(&attr.value).to_string()))
        .collect::<HashMap<String, String>>();

    // Serialize the element
    let element = XMLElement {tag: start_tag.to_string(), attributes: attributes };
    let serialized = serde_json::to_string(&element)?;
    Ok(serialized)
}

/// Extract Set (or Graph data) in XML, HTML, or OWL format from Bytes
/// 
/// # Arguments
/// * `lhs_values` - The column to extract data from (i.e., `bytes`)
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `format` - The format of the bytes
/// * `subject_tags` - Slice of Strings for the subject tags to consider (e.g., rdf:Description)
///   owl:Axiom will only pull out the triples
///   owl:Class and owl:ObjectProperty will also pull out the annotations for the triples
/// * `subject_attributes` - Slice of Strings of attributes to identify the subject (i.e., rdf:about)
/// * `predicate_tags` - Slice of Strings for the predicate tags to consider (e.g., rdfs:label)
/// * `object_tags` - Slice of Strings for the object tags to consider when the object is not a text value
/// * `object_attributes` - Slice of Strings of attributes to identify the object within the predicate element (i.e., rdf:resource)
/// 
/// # Notes
/// * Basic parsing of tags with a flat hierarchy is currently supported
/// * Hierarchical or nested structures are not yet supported
/// * See <https://github.com/phillord/horned-owl> for a full-fledged OWL parser
#[instrument(skip(lhs_values, lhs_args, format, as_columns))]
pub fn extract_set_data(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
    as_columns: &[&str],
) -> Result<RecordBatch> {
    // Extract out the bytes
    let args_table = Table::get_builder()
        .with_name("extract_set_data")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table.get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter().flatten().collect::<Vec<_>>();

    // Read the XML document
    let cursor = Cursor::new(&values_vec);
    let mut reader = match format {
        DataFormat::Html => Reader::from_reader(cursor),
        DataFormat::Xml => Reader::from_reader(cursor),
        DataFormat::Owl => Reader::from_reader(cursor),
        _ => {
            return Err(anyhow!(
                "Unsupported format {format:?} for extract_set_data operator."
            ));
        }
    };

    // buffer for reading
    let mut buf = Vec::new();
    // Map of serialized elements and their children
    let mut relations = HashMap::<String, Vec<(XMLType, String)>>::new();
    // The current elements in scope
    let mut elements = Vec::<String>::new();

    // Extract out the tags
    while let Ok(event) = reader.read_event_into(&mut buf) {
        match event {
            Event::Empty(ref e) => {
                // Parse the tag
                let serialized = parse_xml_tag(e)?;

                // Update the relations children
                if let Some(last_element) = elements.last() {
                    if let Some(relation) = relations.get_mut(last_element) {
                        relation.push((XMLType::Element, serialized.clone()));
                    } else {                        
                        return Err(anyhow!("Key `{last_element}` was not found in XML parsed relations {:?}", relations.keys()));
                    }
                }

                // Add the new element to the relations
                let _ = relations.insert(serialized, Vec::new());
            },
            Event::Text(ref e) => {
                let text = String::from_utf8_lossy(&e as &[u8]);
                let text = text.trim();

                // Update the relations children if there is text
                if !text.is_empty() {
                    if let Some(last_element) = elements.last() {
                        if let Some(relation) = relations.get_mut(last_element) {
                            relation.push((XMLType::Text, String::from_utf8_lossy(&e as &[u8]).to_string()));
                        } else {
                            return Err(anyhow!("Key `{last_element}` was not found in XML parsed relations {:?}", relations.keys()));
                        }
                    }
                }
            },
            Event::Start(ref e) => {
                // Parse the tag
                let serialized = parse_xml_tag(e)?;

                // Update the relations children
                if let Some(last_element) = elements.last() {
                    if let Some(relation) = relations.get_mut(last_element) {
                        relation.push((XMLType::Element, serialized.clone()));
                    } else {                        
                        return Err(anyhow!("Key `{last_element}` was not found in XML parsed relations {:?}", relations.keys()));
                    }
                }

                // Add the new element to the relations
                let _ = relations.insert(serialized.clone(), Vec::new());

                // Add the new element to the current scope
                elements.push(serialized);
            }
            Event::End(ref _e) => {
                elements.pop();
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    // Initialize the columns
    let mut element_vec = Vec::new();
    let mut type_vec = Vec::new();
    let mut children_vec = Vec::new();
    for (element, children) in relations {

        // Preview the children
        let mut type_tmp = Vec::new();
        let mut children_tmp = Vec::new();
        for (t, c) in children {
            type_tmp.push(t);
            children_tmp.push(c);
        }

        // Join all text children for the case of multi-line text
        if type_tmp.iter().filter(|t| *t != &XMLType::Text).collect::<Vec<_>>().is_empty() {
            let children = children_tmp.join("");
            type_tmp.clear();
            children_tmp.clear();
            type_tmp.push(XMLType::Text);
            children_tmp.push(children);
        }

        // Update the columns
        let combined = type_tmp.into_iter().map(|t| t.to_string()).collect::<Vec<_>>()
            .into_iter().zip(children_tmp);
        for (t, c) in combined {
            element_vec.push(element.clone());
            type_vec.push(t);
            children_vec.push(c);
        }
    }

    // Build the batch
    let element_arr: ArrayRef = Arc::new(StringArray::from(element_vec));
    let type_arr: ArrayRef = Arc::new(StringArray::from(type_vec));
    let children_arr: ArrayRef = Arc::new(StringArray::from(children_vec));
    let batch = RecordBatch::try_from_iter(vec![
        ("element", element_arr),
        ("child_type", type_arr),
        ("child", children_arr),
    ])?;
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
    fn test_extract_set_data_owl_class() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.example.com/iri#"
 

    <!-- http://purl.obolibrary.org/obo/GO_0010958 -->


    <owl:Axiom>
        <owl:annotatedSource rdf:resource="http://purl.obolibrary.org/obo/GO_0010958"/>
        <owl:annotatedProperty rdf:resource="http://purl.obolibrary.org/obo/IAO_0000115"/>
        <owl:annotatedTarget>Any process that modulates the frequency, rate or extent of amino acid import into a cell.</owl:annotatedTarget>
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
        let extracted = extract_set_data(
            "bytes", 
            &[batch], 
            &DataFormat::Owl,
            &as_columns.iter().map(|s| s.as_str()).collect::<Vec<&str>>())
            .unwrap();
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
    fn test_extract_set_data_owl_properties() {
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
        let extracted = extract_set_data(
            "bytes", 
            &[batch], 
            &DataFormat::Owl,
            &as_columns.iter().map(|s| s.as_str()).collect::<Vec<&str>>())
            .unwrap();
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
