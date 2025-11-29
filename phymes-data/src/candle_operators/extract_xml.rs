use std::{collections::HashMap, fmt::Display, io::Cursor};

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use clap::ValueEnum;
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, Function, FunctionParameters, JSONSchemaDefine,
    JSONSchemaType, MappableTrait, OwlFormat, Table, TableBuilderTrait, TableTrait, Tool, ToolType,
    create_parse_owl_batch, create_parse_xml_batch,
};
use quick_xml::{
    Reader,
    events::{BytesStart, Event},
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    ToolTrait, candle_data::DataConfig, candle_operators::DataOperatorTrait,
    sort,
};

/// Extract xml tags in either XML or OWL format from Bytes
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExtractXML {
    lhs_values: String,
    format: DataFormat,
}

impl MappableTrait for ExtractXML {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for ExtractXML {
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

impl DataOperatorTrait for ExtractXML {
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
        let format = config.format.clone().ok_or(anyhow!(
            "Missing `format` for `{}`.",
            Self::get_static_name()
        ))?;

        Ok(ExtractXML { lhs_values, format })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        extract_xml(&self.lhs_values, lhs_args, &self.format, device)
    }
}

/// XML element
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct XMLElement {
    /// index the element was found in the document
    index: usize,
    /// tag of the element
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

/// Parse XML tag
fn parse_xml_tag<'a>(e: &BytesStart<'a>) -> String {
    let start_tag = std::str::from_utf8(e.name().into_inner()).unwrap_or_default();
    start_tag.to_string()
}

/// Parse XML Attributes
fn parse_xml_attrs<'a>(e: &BytesStart<'a>, attrs: &[&str]) -> HashMap<String, String> {
    // Parse the tag attribute objects
    e.attributes()
        .flatten()
        .filter_map(|attr| {
            let attr_str = String::from_utf8_lossy(attr.key.as_ref()).to_string();
            if attrs.is_empty() || attrs.contains(&attr_str.as_str()) {
                Some((attr_str, String::from_utf8_lossy(&attr.value).to_string()))
            } else {
                None
            }
        })
        .collect::<HashMap<String, String>>()
}

/// Helper function to parse the attributes of the XML tag into a serialized [XMLElement]
fn serialize_xml_tag<'a>(index: usize, e: &BytesStart<'a>) -> Result<String> {
    let start_tag = parse_xml_tag(e);
    let attributes = parse_xml_attrs(e, &[]);

    // Serialize the element
    let element = XMLElement {
        index,
        tag: start_tag,
        attributes,
    };
    let serialized = serde_json::to_string(&element)?;
    Ok(serialized)
}

fn parse_xml(bytes: &[u8], device: &Device) -> Result<RecordBatch> {
    let cursor = Cursor::new(bytes);
    let mut reader = Reader::from_reader(cursor);

    // buffer for reading
    let mut buf = Vec::new();
    // Map of serialized elements and their children
    let mut relations = HashMap::<String, Vec<(XMLType, String)>>::new();
    // The current elements in scope
    let mut elements = Vec::<String>::new();
    let mut index = 0;

    // Extract out the tags
    while let Ok(event) = reader.read_event_into(&mut buf) {
        match event {
            Event::Empty(ref e) => {
                // Parse the tag
                let serialized = serialize_xml_tag(index, e)?;

                // Update the relations children
                if let Some(last_element) = elements.last() {
                    if let Some(relation) = relations.get_mut(last_element) {
                        relation.push((XMLType::Element, serialized.clone()));
                    } else {
                        return Err(anyhow!(
                            "Key `{last_element}` was not found in XML parsed relations {:?}",
                            relations.keys()
                        ));
                    }
                }

                // Add the new element to the relations
                let _ = relations.insert(serialized, Vec::new());
                index += 1;
            }
            Event::Text(ref e) => {
                let text = String::from_utf8_lossy(e as &[u8]);
                let text = text.trim();

                // Update the relations children if there is text
                if !text.is_empty()
                    && let Some(last_element) = elements.last()
                {
                    if let Some(relation) = relations.get_mut(last_element) {
                        relation.push((
                            XMLType::Text,
                            String::from_utf8_lossy(e as &[u8]).to_string(),
                        ));
                    } else {
                        return Err(anyhow!(
                            "Key `{last_element}` was not found in XML parsed relations {:?}",
                            relations.keys()
                        ));
                    }
                }
            }
            Event::Start(ref e) => {
                // Parse the tag
                let serialized = serialize_xml_tag(index, e)?;

                // Update the relations children
                if let Some(last_element) = elements.last() {
                    if let Some(relation) = relations.get_mut(last_element) {
                        relation.push((XMLType::Element, serialized.clone()));
                    } else {
                        return Err(anyhow!(
                            "Key `{last_element}` was not found in XML parsed relations {:?}",
                            relations.keys()
                        ));
                    }
                }

                // Add the new element to the relations
                let _ = relations.insert(serialized.clone(), Vec::new());
                index += 1;

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
    let mut element_index_vec = Vec::new();
    let mut element_tag_vec = Vec::new();
    let mut element_attr_vec = Vec::new();
    let mut text_vec = Vec::new();
    let mut child_index_vec = Vec::new();
    let mut child_tag_vec = Vec::new();
    let mut child_attr_vec = Vec::new();
    for (element, children) in relations {
        // Preview the children
        let mut type_tmp = Vec::new();
        let mut children_tmp = Vec::new();
        for (t, c) in children {
            type_tmp.push(t);
            children_tmp.push(c);
        }

        // Join all text children for the case of multi-line text
        if type_tmp
            .iter()
            .filter(|t| *t != &XMLType::Text)
            .collect::<Vec<_>>()
            .is_empty()
        {
            let children = children_tmp.join("");
            type_tmp.clear();
            children_tmp.clear();
            type_tmp.push(XMLType::Text);
            children_tmp.push(children);
        }

        // Deserialize XML elements
        let xml_element: XMLElement = serde_json::from_str(&element)?;
        for (t, c) in type_tmp.into_iter().zip(children_tmp) {
            match t {
                XMLType::Text => {
                    element_index_vec.push(xml_element.index.to_owned() as u32);
                    element_attr_vec.push(serde_json::to_string(&xml_element.attributes)?);
                    element_tag_vec.push(xml_element.tag.to_owned());
                    text_vec.push(c);
                    child_index_vec.push(0_u32);
                    child_attr_vec.push(String::new());
                    child_tag_vec.push(String::new());
                }
                XMLType::Element => {
                    element_index_vec.push(xml_element.index.to_owned() as u32);
                    element_attr_vec.push(serde_json::to_string(&xml_element.attributes)?);
                    element_tag_vec.push(xml_element.tag.to_owned());
                    text_vec.push(String::new());
                    let child_element: XMLElement = serde_json::from_str(&c)?;
                    child_index_vec.push(child_element.index as u32);
                    child_attr_vec.push(serde_json::to_string(&child_element.attributes)?);
                    child_tag_vec.push(child_element.tag);
                }
            }
        }
    }

    // Build the batch
    let mut batch = create_parse_xml_batch(
        element_tag_vec,
        element_attr_vec,
        text_vec,
        child_tag_vec,
        child_attr_vec,
        element_index_vec,
        child_index_vec,
    )?;

    // Sort by the element index
    for column_name in ["child_index", "element_index"] {
        batch = sort(column_name, &[batch], true, device)?;
    }
    Ok(batch)
}

fn parse_owl(bytes: &[u8], format: &OwlFormat, device: &Device) -> Result<RecordBatch> {
    let cursor = Cursor::new(bytes);
    let mut reader = Reader::from_reader(cursor);

    // buffer for reading
    let mut buf = Vec::new();
    // The current subject in scope
    let mut s_tag: Vec<String> = Vec::new();
    let mut subject: Vec<String> = Vec::new();
    let mut predicate: Option<String> = None;
    let mut xml_type: Option<XMLType> = Some(XMLType::Element);
    // The extracted triples
    let mut subjects = Vec::new();
    let mut predicates = Vec::new();
    let mut objects = Vec::new();

    // Extract out the tags
    while let Ok(event) = reader.read_event_into(&mut buf) {
        match event {
            Event::Empty(ref e) => {
                let tag = parse_xml_tag(e);
                if format.subject_tags.contains(&tag) {
                    // do nothing
                } else if format.predicate_tags.contains(&tag) {
                    let attributes = parse_xml_attrs(
                        e,
                        &format
                            .predicate_attributes
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                    );
                    if !attributes.is_empty() {
                        if let Some(s) = subject.last() {
                            if let Some(_p) = predicate.as_ref() {
                                // do nothing
                            } else {
                                // Create the triple
                                let v = if let Some(v) =
                                    attributes.get(format.predicate_attributes.first().unwrap())
                                {
                                    v
                                } else {
                                    return Err(anyhow!(
                                        "Predicate attribute `{}` was not found in XML parsed attributes {:?}",
                                        format.predicate_attributes.first().unwrap(),
                                        attributes.keys()
                                    ));
                                };
                                subjects.push(s.to_owned());
                                predicates.push(tag);
                                objects.push(v.to_string());
                                xml_type.replace(XMLType::Element);
                            }
                        } else {
                            // ignore
                            // return Err(anyhow!(
                            //     "Found a predicate tag `{tag}` when there is no current subject."
                            // ));
                        }
                    } else {
                        // ignore recursive predicates for now
                    }
                }
            }
            Event::Text(ref e) => {
                let text = String::from_utf8_lossy(e as &[u8]);
                let text = text.trim();
                if !text.is_empty()
                    && let Some(s) = subject.last()
                    && let Some(p) = predicate.as_ref()
                {
                    // Handle the case of multi-line text
                    if let Some(t) = xml_type.as_ref() {
                        match t {
                            XMLType::Element => {
                                subjects.push(s.to_string());
                                predicates.push(p.to_string());
                                objects.push(text.to_string());
                                xml_type.replace(XMLType::Text);
                            }
                            XMLType::Text => {
                                if let Some(mut o) = objects.pop() {
                                    o.push_str(text);
                                    objects.push(o);
                                }
                            }
                        }
                    }
                }
            }
            Event::Start(ref e) => {
                let tag = parse_xml_tag(e);
                if format.subject_tags.contains(&tag) {
                    let attributes = parse_xml_attrs(
                        e,
                        &format
                            .subject_attributes
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                    );
                    if !attributes.is_empty() {
                        if let Some(s) = subject.last() {
                            if let Some(p) = predicate.as_ref() {
                                // Create triple where the object is another element
                                subjects.push(s.to_owned());
                                predicates.push(p.to_owned());
                                objects.push(tag);
                                xml_type.replace(XMLType::Element);
                            } else {
                                return Err(anyhow!(
                                    "Found another subject tag `{tag}` for current subject `{s}` when there was no predicate"
                                ));
                            }
                        } else {
                            // Create the new subject
                            let s = if let Some(v) =
                                attributes.get(format.subject_attributes.first().unwrap())
                            {
                                v
                            } else {
                                return Err(anyhow!(
                                    "Subject attribute `{}` was not found in XML parsed attributes {:?}",
                                    format.subject_attributes.first().unwrap(),
                                    attributes.keys()
                                ));
                            };
                            subject.push(s.to_string());
                            s_tag.push(tag);
                        }
                    } else {
                        // Buffer the subjects when the same subject tag is found
                        // since we are ignoring recursive predicates
                        if let Some(s) = s_tag.last()
                            && s == &tag
                        {
                            subject.push(tag.clone());
                            s_tag.push(tag);
                        }
                    }
                } else if format.predicate_tags.contains(&tag) {
                    if subject.len() == 1 {
                        if let Some(_p) = predicate.as_ref() {
                            // nothing todo
                        } else {
                            // Create the new predicate
                            predicate.replace(tag);
                            xml_type.replace(XMLType::Element);
                        }
                    } else {
                        // ignore
                        // return Err(anyhow!(
                        //     "Found a predicate tag `{tag}` when there is no current subject."
                        // ));
                    }
                }
            }
            Event::End(ref e) => {
                let tag = std::str::from_utf8(e.name().into_inner()).unwrap_or_default();
                if let Some(p) = predicate.take() {
                    if tag != p {
                        predicate.replace(p);
                    } else {
                        xml_type.replace(XMLType::Element);
                    }
                } else if let Some(s) = s_tag.last()
                    && tag == s
                {
                    subject.pop();
                    s_tag.pop();
                    xml_type.replace(XMLType::Element);
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    // Build the batch
    let mut batch = create_parse_owl_batch(subjects, predicates, objects)?;

    // Sorty by the subject and predicate
    for column_name in ["predicate", "subject"] {
        batch = sort(column_name, &[batch], true, device)?;
    }
    Ok(batch)
}

/// Extract Set (or Graph data) in XML, HTML, or OWL format from Bytes
///
/// # Arguments
/// * `lhs_values` - The column to extract data from (i.e., `bytes`)
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `format` - The format of the bytes
///
/// # Notes
/// * Hierarchical or nested children structures are supported
/// *
/// * See <https://github.com/phillord/horned-owl> for a full-fledged OWL parser
#[instrument(skip(lhs_values, lhs_args, format, device))]
pub fn extract_xml(
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
    device: &Device,
) -> Result<RecordBatch> {
    // Extract out the bytes
    let args_table = Table::get_builder()
        .with_name("extract_xml")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table
        .get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // Read the XML document
    match format {
        DataFormat::Html | DataFormat::Xml | DataFormat::OwlDefault => {
            parse_xml(&values_vec, device)
        }
        DataFormat::Owl(format) => parse_owl(&values_vec, format, device),
        DataFormat::OwlClass => parse_owl(&values_vec, &OwlFormat::owl_format_class(), device),
        DataFormat::OwlObjectProperty => parse_owl(
            &values_vec,
            &OwlFormat::owl_format_object_property(),
            device,
        ),
        DataFormat::OwlNamedIndividual => parse_owl(
            &values_vec,
            &OwlFormat::owl_format_named_individual(),
            device,
        ),
        DataFormat::OwlOntology => parse_owl(
            &values_vec,
            &OwlFormat::owl_format_ontology(),
            device,
        ),
        _ => Err(anyhow!(
            "Unsupported format {format:?} for extract_set_data operator."
        )),
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{
        BuildableTrait, BuilderTrait, DataFormat, Table, TableBuilderTrait, TableTrait,
        create_blob_batch, device,
    };
    use phymes_diagnostics::create_timestamp_micros;

    use super::*;

    #[test]
    fn test_extract_xml() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.example.com/iri#"
     xml:base="http://www.example.com/iri"
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
        <rdfs:label>regulation of amino acid import across plasma membrane</rdfs:label>
    </owl:Class>
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

        // Make the device
        let device = device(false).unwrap();

        // Extract the xml tags
        let extracted =
            extract_xml("bytes", &[batch], &DataFormat::OwlDefault, &device).unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 7);
        assert_eq!(extracted.num_rows(), 16);

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table
            .get_column_as_vec_primitive::<u32>("element_index")
            .unwrap();
        assert_eq!(result, [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11]);
        let result = table.get_column_as_vec_str("element_tag");
        assert_eq!(
            result,
            [
                "rdf:RDF",
                "rdf:RDF",
                "owl:Ontology",
                "owl:versionIRI",
                "owl:Class",
                "owl:Class",
                "owl:equivalentClass",
                "owl:Class",
                "owl:intersectionOf",
                "owl:intersectionOf",
                "rdf:Description",
                "owl:Restriction",
                "owl:Restriction",
                "owl:onProperty",
                "owl:someValuesFrom",
                "rdfs:label"
            ]
        );
        let _result = table.get_column_as_vec_str("element_attr");
        let result = table.get_column_as_vec_str("text");
        assert_eq!(
            result,
            [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "regulation of amino acid import across plasma membrane"
            ]
        );
        let result = table
            .get_column_as_vec_primitive::<u32>("child_index")
            .unwrap();
        assert_eq!(result[..8], [1, 3, 2, 0, 4, 11, 5, 6]); // DM: Sorting on the GPU and CPU changes after index 8
        let result = table.get_column_as_vec_str("child_tag");
        assert_eq!(
            result[..8],
            [
                "owl:Ontology",
                "owl:Class",
                "owl:versionIRI",
                "",
                "owl:equivalentClass",
                "rdfs:label",
                "owl:Class",
                "owl:intersectionOf"
            ]
        );
        let _result = table.get_column_as_vec_str("child_attr");
    }

    #[test]
    fn test_extract_owl_class() {
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

        // Make the device
        let device = device(false).unwrap();

        // Extract the xml tags
        let extracted =
            extract_xml("bytes", &[batch], &DataFormat::OwlClass, &device).unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 3);
        assert_eq!(extracted.num_rows(), 9);

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table.get_column_as_vec_str("subject");
        assert_eq!(
            result,
            [
                "http://purl.obolibrary.org/obo/GO_0010958",
                "http://purl.obolibrary.org/obo/GO_0010958",
                "http://purl.obolibrary.org/obo/GO_0010958",
                "http://purl.obolibrary.org/obo/GO_0010958",
                "http://purl.obolibrary.org/obo/GO_0010958",
                "http://purl.obolibrary.org/obo/GO_0010968",
                "http://purl.obolibrary.org/obo/GO_0010968",
                "http://purl.obolibrary.org/obo/GO_0010968",
                "http://purl.obolibrary.org/obo/GO_0010968"
            ]
        );
        let result = table.get_column_as_vec_str("predicate");
        assert_eq!(
            result,
            [
                "obo:IAO_0000115",
                "oboInOwl:hasBroadSynonym",
                "oboInOwl:hasOBONamespace",
                "oboInOwl:id",
                "rdfs:label",
                "obo:IAO_0000115",
                "oboInOwl:hasOBONamespace",
                "oboInOwl:id",
                "rdfs:label"
            ]
        );
        let result = table.get_column_as_vec_str("object");
        assert_eq!(
            result,
            [
                "Any process that modulates the frequency, rate or extent of amino acid import into a cell.",
                "regulation of amino acid import",
                "biological_process",
                "GO:0010958",
                "regulation of amino acid import across plasma membrane",
                "Any process that modulates the rate, frequency or extent of microtubule nucleation. Microtubule nucleation is thede novoformation of a microtubule, in which tubulin heterodimers form metastable oligomeric aggregates, some of which go on to support formation of a complete microtubule. Microtubule nucleation usually occurs from a specific site within a cell.",
                "biological_process",
                "GO:0010968",
                "regulation of microtubule nucleation"
            ]
        );
    }

    #[test]
    fn test_extract_owl_properties() {
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

        // Make the device
        let device = device(false).unwrap();

        // Extract the xml tags
        let extracted =
            extract_xml("bytes", &[batch], &DataFormat::OwlObjectProperty, &device).unwrap();

        // Check the dimensions of the extracted data
        assert_eq!(extracted.num_columns(), 3);
        assert_eq!(extracted.num_rows(), 13);

        // Check the contents of the extracted data
        let table = Table::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table.get_column_as_vec_str("subject");
        assert_eq!(
            result,
            [
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002437",
                "http://purl.obolibrary.org/obo/RO_0002438",
                "http://purl.obolibrary.org/obo/RO_0002438",
                "http://purl.obolibrary.org/obo/RO_0002438",
                "http://purl.obolibrary.org/obo/RO_0002438"
            ]
        );
        let result = table.get_column_as_vec_str("predicate");
        assert_eq!(
            result,
            [
                "obo:IAO_0000115",
                "rdf:type",
                "rdfs:domain",
                "rdfs:label",
                "rdfs:range",
                "rdfs:seeAlso",
                "rdfs:seeAlso",
                "rdfs:subPropertyOf",
                "rdfs:subPropertyOf",
                "obo:IAO_0000115",
                "rdfs:label",
                "rdfs:seeAlso",
                "rdfs:subPropertyOf"
            ]
        );
        let result = table.get_column_as_vec_str("object");
        assert_eq!(
            result,
            [
                "An interaction relationship in which at least one of the partners is an organism and the other is either an organism or an abiotic entity with which the organism interacts.",
                "http://www.w3.org/2002/07/owl#SymmetricProperty",
                "http://purl.obolibrary.org/obo/BFO_0000040",
                "biotically interacts with",
                "http://purl.obolibrary.org/obo/BFO_0000040",
                "http://dx.doi.org/10.1016/j.ecoinf.2014.08.005",
                "http://eol.org/schema/terms/interactsWith",
                "http://purl.obolibrary.org/obo/RO_0002321",
                "http://purl.obolibrary.org/obo/RO_0002434",
                "An interaction relationship in which the partners are related via a feeding relationship.",
                "trophically interacts with",
                "http://dx.doi.org/10.1016/j.ecoinf.2014.08.005",
                "http://purl.obolibrary.org/obo/RO_0002574"
            ]
        );
    }
}
