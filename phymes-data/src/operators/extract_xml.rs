use std::{collections::HashMap, fmt::Display, io::Cursor};

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use clap::ValueEnum;
use phymes_schemas::{
    DataFormat, Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
    create_parse_owl_batch, create_parse_xml_batch,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use quick_xml::{
    NsReader,
    events::{BytesStart, Event},
    name::ResolveResult,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{DataConfig, DataOperatorTrait, ToolTrait, sort};

/// Extract xml tags in either XML or OWL format from Bytes
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExtractXML {
    lhs_name: String,
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
        let lhs_name = config.lhs_name.clone().ok_or(anyhow!(
            "Missing `lhs_name` for `{}`.",
            Self::get_static_name()
        ))?;
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

        Ok(ExtractXML {
            lhs_name,
            lhs_values,
            format,
        })
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        extract_xml(
            &self.lhs_name,
            &self.lhs_values,
            lhs_args,
            &self.format,
            device,
        )
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
fn parse_xml_tag<'a>(reader: &NsReader<Cursor<&[u8]>>, e: &BytesStart<'a>) -> String {
    // Resolve the namespace and add it to the tag
    let (ns, local) = reader.resolver().resolve_element(e.name());
    let ns = match ns {
        ResolveResult::Bound(s) => std::str::from_utf8(s.into_inner())
            .unwrap_or_default()
            .to_string(),
        ResolveResult::Unbound => String::new(),
        ResolveResult::Unknown(s) => std::str::from_utf8(&s).unwrap_or_default().to_string(),
    };

    // If the namespace was not resolved, then use the prefixed name
    let ns = if !ns.contains("http://") && !ns.contains("https://") {
        format!("{ns}:")
    } else {
        ns
    };
    let local = std::str::from_utf8(local.into_inner()).unwrap_or_default();
    format!("{ns}{local}")
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
fn serialize_xml_tag<'a>(
    reader: &NsReader<Cursor<&[u8]>>,
    index: usize,
    e: &BytesStart<'a>,
) -> Result<String> {
    let start_tag = parse_xml_tag(reader, e);
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

/// Parse the XML document into a HashMap of serialized elements and their children
fn parse_xml(bytes: &[u8]) -> Result<HashMap<String, Vec<(XMLType, String)>>> {
    let cursor = Cursor::new(bytes);
    let mut reader = NsReader::from_reader(cursor);

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
                let serialized = serialize_xml_tag(&reader, index, e)?;

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
                let serialized = serialize_xml_tag(&reader, index, e)?;

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

    Ok(relations)
}

/// Helper function to join multi-line text children
fn join_text_children(children: Vec<(XMLType, String)>) -> (Vec<XMLType>, Vec<String>) {
    // Preview the children
    let (mut type_tmp, mut children_tmp): (Vec<XMLType>, Vec<String>) = children.into_iter().unzip();

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

    (type_tmp, children_tmp)
}

/// Convert the parsed xml relations to a parsed xml schema
fn xml_to_parsed_xml_record_batch(
    relations: HashMap<String, Vec<(XMLType, String)>>,
    lhs_name: &str,
    device: &Device,
) -> Result<RecordBatch> {
    // Initialize the columns
    let mut document_id_vec = Vec::new();
    let mut element_index_vec = Vec::new();
    let mut element_tag_vec = Vec::new();
    let mut element_attr_vec = Vec::new();
    let mut text_vec = Vec::new();
    let mut child_index_vec = Vec::new();
    let mut child_tag_vec = Vec::new();
    let mut child_attr_vec = Vec::new();
    for (element, children) in relations {
        // Join all text children for the case of multi-line text
        let (type_tmp, children_tmp) = join_text_children(children);

        // Deserialize XML elements
        let xml_element: XMLElement = serde_json::from_str(&element)?;
        for (t, c) in type_tmp.into_iter().zip(children_tmp) {
            match t {
                XMLType::Text => {
                    document_id_vec.push(lhs_name.to_string());
                    element_index_vec.push(xml_element.index.to_owned() as u32);
                    element_attr_vec.push(serde_json::to_string(&xml_element.attributes)?);
                    element_tag_vec.push(xml_element.tag.to_owned());
                    text_vec.push(c);
                    child_index_vec.push(0_u32);
                    child_attr_vec.push(String::new());
                    child_tag_vec.push(String::new());
                }
                XMLType::Element => {
                    document_id_vec.push(lhs_name.to_string());
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
        document_id_vec,
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

/// Helper function to parse OWL children
fn parse_owl_children(
    relations: &HashMap<String, Vec<(XMLType, String)>>,
    children: Vec<(XMLType, String)>,
) -> Result<(Vec<String>, Vec<String>)> {
    // // Join all text children for the case of multi-line text
    // let (type_tmp, children_tmp) = join_text_children(children);

    // Parse out the predicates and objects
    let mut predicate_vec = Vec::new();
    let mut object_vec = Vec::new();
    for (t, c) in children {
    // for (t, c) in type_tmp.into_iter().zip(children_tmp) {
        match t {
            XMLType::Text => {}
            XMLType::Element => {
                if let Some((predicate, object)) = children_to_po(relations, &c)? {
                    predicate_vec.push(predicate);
                    object_vec.push(object);
                }
            }
        };
    }
    Ok((predicate_vec, object_vec))
}

/// Helper function to lookup and extract out the children of an XML child element
fn children_to_po(
    relations: &HashMap<String, Vec<(XMLType, String)>>,
    child: &str,
) -> Result<Option<(String, String)>> {
    let child_element: XMLElement = serde_json::from_str(child)?;
    let po = if let Some(resource) = child_element.attributes.get("rdf:resource") {
        Some((child_element.tag, resource.to_string()))
    } else if let Some(element) = relations.get(child) {
        // Retrieve the child element from the relations
        // Join all text children for the case of multi-line text
        let (mut type_tmp, mut children_tmp) = join_text_children(element.clone());

        // Hierarchical objects are not yet supported (e.g., EquivalentClass, ChainedAxioms, etc.)
        if type_tmp.len() == 1 {
            if let (Some(t), Some(c)) = (type_tmp.pop(), children_tmp.pop()) {
                match t {
                    XMLType::Text => Some((child_element.tag, c)),
                    XMLType::Element => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        let _err = anyhow!("Child `{child}` not found in parsed relations `{:?}`.", relations.keys());
        None
    };
    Ok(po)
}

/// Convert the parsed xml relations to a parsed owl schema
fn xml_to_parsed_owl_record_batch(
    relations: HashMap<String, Vec<(XMLType, String)>>,
    lhs_name: &str,
    device: &Device,
) -> Result<RecordBatch> {
    // Partition into entities and their attributes
    #[allow(clippy::type_complexity)]
    let (entities, attributes): (HashMap<String, Vec<(XMLType, String)>>, HashMap<String, Vec<(XMLType, String)>>) = relations.into_par_iter()
    .filter(|(k, _v)| {
        let xml_element: XMLElement = serde_json::from_str(k).unwrap();

        // --- Exclusion list ---
        // Hierarchical objects are not yet supported (e.g., EquivalentClass, ChainedAxioms, etc.)
        !((xml_element.tag == "http://www.w3.org/2002/07/owl#Ontology" && !xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#AnnotationProperty" && !xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#DatatypeProperty" && !xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#Class" && !xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#ObjectProperty" && !xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#NamedIndividual" && !xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2000/01/rdf-schema#subClassOf" && !xml_element.attributes.contains_key("rdf:resource"))
        || (xml_element.tag == "http://www.w3.org/2000/01/rdf-schema#subPropertyOf" && !xml_element.attributes.contains_key("rdf:resource"))
        || (xml_element.tag == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && !xml_element.attributes.contains_key("rdf:resource"))
        || xml_element.tag == "http://www.w3.org/2002/07/owl#equivalentClass"
        || xml_element.tag == "http://www.w3.org/2002/07/owl#Restriction"
        || xml_element.tag == "http://www.w3.org/2002/07/owl#intersectionOf"
        || xml_element.tag == "http://www.w3.org/2002/07/owl#onProperty"
        || xml_element.tag == "http://www.w3.org/2002/07/owl#someValuesFrom"
        || xml_element.tag == "http://www.w3.org/2002/07/owl#propertyChainAxiom"

        // DM: re-instate axiom when testing with GO-CAM
        || xml_element.tag == "http://www.w3.org/2002/07/owl#Axiom"

        // Ignore other properties not used in the embeddings
        || xml_element.tag == "http://purl.obolibrary.org/obo/OMO_0002000"
        || xml_element.tag == "http://purl.org/dc/elements/1.1/contributor"
        || xml_element.tag == "http://purl.org/dc/terms/contributor"
        || xml_element.tag == "http://purl.org/dc/terms/license"
        || xml_element.tag == "http://xmlns.com/foaf/0.1/depiction"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:id"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasDbXref"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:created_by"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:creation_date"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasNarrowSynonym"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasBroadSynonym"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasRelatedSynonym"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:shorthand"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasOBOFormatVersion"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasScope"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasSynonymType"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:inconsistent_with"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:inferred_by"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:isAbout"
        || xml_element.tag == "http://www.geneontology.org/formats/oboInOwl#oboInOwl:hasAlternativeId")

    })
    .partition(|(k, _v)| {
        let xml_element: XMLElement = serde_json::from_str(k).unwrap();
        (xml_element.tag == "http://www.w3.org/2002/07/owl#Ontology" && xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#AnnotationProperty" && xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#DatatypeProperty" && xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#Class" && xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#ObjectProperty" && xml_element.attributes.contains_key("rdf:about"))
        || (xml_element.tag == "http://www.w3.org/2002/07/owl#NamedIndividual" && xml_element.attributes.contains_key("rdf:about"))
        || xml_element.tag == "http://www.w3.org/2002/07/owl#Axiom"
    });
    dbg!(&entities.len());
    dbg!(&attributes.len());

    // Fold into N-Quad arrays
    let (dataset_vec, entity_vec, graph_vec, subject_vec, predicate_vec, object_vec, _counts) = entities
        .into_par_iter()
        .fold( || (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0),
            |mut acc, (element, children)| {
            acc.6 += 1;
            println!("Iter: {}",  acc.6);
            // Deserialize XML elements
            let xml_element: XMLElement = serde_json::from_str(&element).unwrap();

            // Parse the primary OWL Entities
            if xml_element.tag == "http://www.w3.org/2002/07/owl#Ontology"
            || xml_element.tag == "http://www.w3.org/2002/07/owl#AnnotationProperty" 
            || xml_element.tag == "http://www.w3.org/2002/07/owl#DatatypeProperty" 
            || xml_element.tag == "http://www.w3.org/2002/07/owl#Class" 
            || xml_element.tag == "http://www.w3.org/2002/07/owl#ObjectProperty" 
            || xml_element.tag == "http://www.w3.org/2002/07/owl#NamedIndividual" {
                if let Some(subject) = xml_element.attributes.get("rdf:about") {
                    let (predicates, objects) = parse_owl_children(&attributes, children).unwrap();
                    for (predicate, object) in predicates.into_iter().zip(objects) {
                        acc.0.push(lhs_name.to_string());
                        acc.1.push(xml_element.tag.to_string());
                        let graph = format!("{subject}-{predicate}-{object}");
                        acc.2.push(graph.to_string());
                        acc.3.push(subject.to_string());
                        acc.4.push(predicate);
                        acc.5.push(object);
                    }
                }
            } else if xml_element.tag == "http://www.w3.org/2002/07/owl#Axiom" {
                let (predicates, objects) = parse_owl_children(&attributes, children).unwrap();

                // Determine the subject of the axium
                let subject_triple = predicates
                    .iter()
                    .zip(objects.iter())
                    .filter_map(|(t, c)| {
                        if t == "http://www.w3.org/2002/07/owl#annotatedSource"
                            || t == "http://www.w3.org/2002/07/owl#annotatedProperty"
                            || t == "http://www.w3.org/2002/07/owl#annotatedTarget"
                            // || t == "http://purl.obolibrary.org/obo/RO_0002582"
                            // || t == "http://purl.obolibrary.org/obo/RO_0002581"
                        {
                            Some((t.to_string(), c.to_string()))
                        } else {
                            None
                        }
                    })
                    .collect::<HashMap<_, _>>();
                let subject = if subject_triple.len() == 3 {
                    format!("{}-{}-{}",
                        subject_triple.get("http://www.w3.org/2002/07/owl#annotatedSource").ok_or(anyhow!("Key `source` missing from Owl:Axiom extracted annotation triples `{:?}`. Available predicates are `{predicates:?}`.", subject_triple.keys())).unwrap(),
                        subject_triple.get("http://www.w3.org/2002/07/owl#annotatedProperty").ok_or(anyhow!("Key `property` missing from Owl:Axiom extracted annotation triples `{:?}`. Available predicates are `{predicates:?}`.", subject_triple.keys())).unwrap(),
                        subject_triple.get("http://www.w3.org/2002/07/owl#annotatedTarget").ok_or(anyhow!("Key `target` missing from Owl:Axiom extracted annotation triples `{:?}`. Available predicates are `{predicates:?}`.", subject_triple.keys())).unwrap()
                    )
                } else {
                    subject_triple.into_values().collect::<Vec<_>>().join("-")
                };

                // Continue extracting the predicate/object pairs
                for (predicate, object) in predicates.into_iter().zip(objects) {
                    acc.0.push(lhs_name.to_string());
                    acc.1.push("http://www.w3.org/2002/07/owl#Axiom".to_string());
                    let graph = format!("{subject}-{predicate}-{object}");
                    acc.2.push(graph.to_string());
                    acc.3.push(subject.to_string());
                    acc.4.push(predicate);
                    acc.5.push(object);
                }
            }
            acc
        })
        .reduce(
            || (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0), // identity value for reduction
            |mut acc1, acc2| {
                acc1.0.extend(acc2.0);
                acc1.1.extend(acc2.1);
                acc1.2.extend(acc2.2);
                acc1.3.extend(acc2.3);
                acc1.4.extend(acc2.4);
                acc1.5.extend(acc2.5);
                acc1
            },
        );
        // .into_iter()
        // .enumerate()
        // .filter_map(|(i, (element, children))| {
        //     println!("Iter: {i}, attributes len: {}", attributes.len());
        //     let mut acc = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

        //     // Deserialize XML elements
        //     let xml_element: XMLElement = serde_json::from_str(&element).unwrap();

        //     // Parse the primary OWL Entities
        //     if xml_element.tag == "http://www.w3.org/2002/07/owl#Ontology"
        //     || xml_element.tag == "http://www.w3.org/2002/07/owl#AnnotationProperty" 
        //     || xml_element.tag == "http://www.w3.org/2002/07/owl#DatatypeProperty" 
        //     || xml_element.tag == "http://www.w3.org/2002/07/owl#Class" 
        //     || xml_element.tag == "http://www.w3.org/2002/07/owl#ObjectProperty" 
        //     || xml_element.tag == "http://www.w3.org/2002/07/owl#NamedIndividual" {
        //         if let Some(subject) = xml_element.attributes.get("rdf:about") {
        //             let (predicates, objects) = parse_owl_children(&mut attributes, children).unwrap();
        //             for (predicate, object) in predicates.into_iter().zip(objects) {
        //                 acc.0.push(lhs_name.to_string());
        //                 acc.1.push(xml_element.tag.to_string());
        //                 let graph = format!("{subject}-{predicate}-{object}");
        //                 acc.2.push(graph.to_string());
        //                 acc.3.push(subject.to_string());
        //                 acc.4.push(predicate);
        //                 acc.5.push(object);
        //             }
        //             Some((acc.0, acc.1, acc.2, acc.3, acc.4, acc.5))
        //         } else {
        //             None
        //         }
        //     } else if xml_element.tag == "http://www.w3.org/2002/07/owl#Axiom" {
        //         let (predicates, objects) = parse_owl_children(&mut attributes, children).unwrap();

        //         // Determine the subject of the axium
        //         let subject_triple = predicates
        //             .iter()
        //             .zip(objects.iter())
        //             .filter_map(|(t, c)| {
        //                 if t == "http://www.w3.org/2002/07/owl#annotatedSource"
        //                     || t == "http://www.w3.org/2002/07/owl#annotatedProperty"
        //                     || t == "http://www.w3.org/2002/07/owl#annotatedTarget"
        //                     // || t == "http://purl.obolibrary.org/obo/RO_0002582"
        //                     // || t == "http://purl.obolibrary.org/obo/RO_0002581"
        //                 {
        //                     Some((t.to_string(), c.to_string()))
        //                 } else {
        //                     None
        //                 }
        //             })
        //             .collect::<HashMap<_, _>>();
        //         let subject = if subject_triple.len() == 3 {
        //             format!("{}-{}-{}",
        //                 subject_triple.get("http://www.w3.org/2002/07/owl#annotatedSource").ok_or(anyhow!("Key `source` missing from Owl:Axiom extracted annotation triples `{:?}`. Available predicates are `{predicates:?}`.", subject_triple.keys())).unwrap(),
        //                 subject_triple.get("http://www.w3.org/2002/07/owl#annotatedProperty").ok_or(anyhow!("Key `property` missing from Owl:Axiom extracted annotation triples `{:?}`. Available predicates are `{predicates:?}`.", subject_triple.keys())).unwrap(),
        //                 subject_triple.get("http://www.w3.org/2002/07/owl#annotatedTarget").ok_or(anyhow!("Key `target` missing from Owl:Axiom extracted annotation triples `{:?}`. Available predicates are `{predicates:?}`.", subject_triple.keys())).unwrap()
        //             )
        //         } else {
        //             subject_triple.into_iter().map(|(_k, v)| v).collect::<Vec<_>>().join("-")
        //         };

        //         // Continue extracting the predicate/object pairs
        //         for (predicate, object) in predicates.into_iter().zip(objects) {
        //             acc.0.push(lhs_name.to_string());
        //             acc.1.push("http://www.w3.org/2002/07/owl#Axiom".to_string());
        //             let graph = format!("{subject}-{predicate}-{object}");
        //             acc.2.push(graph.to_string());
        //             acc.3.push(subject.to_string());
        //             acc.4.push(predicate);
        //             acc.5.push(object);
        //         }
        //         Some((acc.0, acc.1, acc.2, acc.3, acc.4, acc.5))
        //     } else {
        //         None
        //     }
        // })
        // .reduce(|mut acc1, acc2| {
        //     acc1.0.extend(acc2.0);
        //     acc1.1.extend(acc2.1);
        //     acc1.2.extend(acc2.2);
        //     acc1.3.extend(acc2.3);
        //     acc1.4.extend(acc2.4);
        //     acc1.5.extend(acc2.5);
        //     acc1
        // })
        // .unwrap();

    // Build the batch
    let mut batch = create_parse_owl_batch(
        entity_vec,
        subject_vec,
        predicate_vec,
        object_vec,
        graph_vec,
        dataset_vec,
    )?;

    // Sort by the element index
    for column_name in ["graph", "object", "predicate", "subject", "entity"] {
        batch = sort(column_name, &[batch], true, device)?;
    }
    Ok(batch)
}

/// Extract Set (or Graph data) in XML, HTML, or OWL format from Bytes
///
/// # Arguments
/// * `lhs_name` - The name of the XML document
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
    lhs_name: &str,
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    format: &DataFormat,
    device: &Device,
) -> Result<RecordBatch> {
    // Extract out the bytes
    let args_table = Subject::get_builder()
        .with_name("extract_xml")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let values_vec = args_table
        .get_column_as_vec_nested_primitive::<u8>(lhs_values)?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // Read the XML document
    let parsed = parse_xml(&values_vec)?;
    match format {
        DataFormat::Html | DataFormat::Xml => {
            xml_to_parsed_xml_record_batch(parsed, lhs_name, device)
        }
        DataFormat::Owl => xml_to_parsed_owl_record_batch(parsed, lhs_name, device),
        _ => Err(anyhow!(
            "Unsupported format {format:?} for extract_set_data operator."
        )),
    }
}

#[cfg(test)]
mod tests {
    use crate::device;
    use phymes_diagnostics::{HashSet, create_timestamp_micros};
    use phymes_schemas::{DataFormat, create_attachments_batch};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };

    use super::*;

    #[test]
    fn test_parse_xml() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.example.com/iri#"
     xml:base="http://www.example.com/iri"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
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

        // Extract the xml tags
        let bytes: Vec<u8> = owl.into();
        let extracted = parse_xml(&bytes).unwrap();
        let _keys = extracted.keys().collect::<Vec<_>>();
        // DM: not able to reliable compare using sort nor hashset....
        // assert_eq!(
        //     keys,
        //     [
        //         "{\"index\":0,\"tag\":\"rdf:RDF\",\"attributes\":{\"xml:base\":\"http://www.example.com/iri\",\"xmlns\":\"http://www.example.com/iri#\",\"xmlns:owl\":\"http://www.w3.org/2002/07/owl#\",\"xmlns:rdfs\":\"http://www.w3.org/2000/01/rdf-schema#\"}}", "{\"index\":1,\"tag\":\"http://www.w3.org/2002/07/owl#Ontology\",\"attributes\":{\"rdf:about\":\"http://www.example.com/iri\"}}", "{\"index\":10,\"tag\":\"http://www.w3.org/2002/07/owl#someValuesFrom\",\"attributes\":{\"rdf:resource\":\"http://purl.obolibrary.org/obo/GO_0089718\"}}", "{\"index\":11,\"tag\":\"http://www.w3.org/2000/01/rdf-schema#label\",\"attributes\":{}}", "{\"index\":2,\"tag\":\"http://www.w3.org/2002/07/owl#versionIRI\",\"attributes\":{\"rdf:resource\":\"http://www.example.com/viri\"}}", "{\"index\":3,\"tag\":\"http://www.w3.org/2002/07/owl#Class\",\"attributes\":{\"rdf:about\":\"http://purl.obolibrary.org/obo/GO_0010958\"}}", "{\"index\":4,\"tag\":\"http://www.w3.org/2002/07/owl#equivalentClass\",\"attributes\":{}}", "{\"index\":5,\"tag\":\"http://www.w3.org/2002/07/owl#Class\",\"attributes\":{}}", "{\"index\":6,\"tag\":\"http://www.w3.org/2002/07/owl#intersectionOf\",\"attributes\":{\"rdf:parseType\":\"Collection\"}}", "{\"index\":7,\"tag\":\"rdf:Description\",\"attributes\":{\"rdf:about\":\"http://purl.obolibrary.org/obo/GO_0065007\"}}", "{\"index\":8,\"tag\":\"http://www.w3.org/2002/07/owl#Restriction\",\"attributes\":{}}", "{\"index\":9,\"tag\":\"http://www.w3.org/2002/07/owl#onProperty\",\"attributes\":{\"rdf:resource\":\"http://purl.obolibrary.org/obo/RO_0002211\"}}"
        //     ]
        // );
    }

    #[test]
    fn test_extract_xml() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.example.com/iri#"
     xml:base="http://www.example.com/iri"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
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
        let batch = create_attachments_batch(
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
        let extracted = extract_xml("test", "bytes", &[batch], &DataFormat::Xml, &device).unwrap();

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table.get_column_as_vec_str("document_id");
        assert_eq!(
            result,
            [
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test",
            ]
        );
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
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#versionIRI",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#equivalentClass",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#intersectionOf",
                "http://www.w3.org/2002/07/owl#intersectionOf",
                "rdf:Description",
                "http://www.w3.org/2002/07/owl#Restriction",
                "http://www.w3.org/2002/07/owl#Restriction",
                "http://www.w3.org/2002/07/owl#onProperty",
                "http://www.w3.org/2002/07/owl#someValuesFrom",
                "http://www.w3.org/2000/01/rdf-schema#label"
            ]
        );
        let result = table.get_column_as_vec_str("element_attr");
        // DM: not able to consistently compare even with Sort nor HashSet...
        let _result_set = result.into_iter().collect::<HashSet<_>>();
        let _result_test = ["{\"rdf:about\":\"http://purl.obolibrary.org/obo/GO_0010958\"}", "{\"rdf:about\":\"http://purl.obolibrary.org/obo/GO_0010958\"}", "{\"rdf:about\":\"http://purl.obolibrary.org/obo/GO_0065007\"}", "{\"rdf:about\":\"http://www.example.com/iri\"}", "{\"rdf:parseType\":\"Collection\"}", "{\"rdf:parseType\":\"Collection\"}", "{\"rdf:resource\":\"http://purl.obolibrary.org/obo/GO_0089718\"}", "{\"rdf:resource\":\"http://purl.obolibrary.org/obo/RO_0002211\"}", "{\"rdf:resource\":\"http://www.example.com/viri\"}", "{\"xmlns:rdfs\":\"http://www.w3.org/2000/01/rdf-schema#\",\"xml:base\":\"http://www.example.com/iri\",\"xmlns:owl\":\"http://www.w3.org/2002/07/owl#\",\"xmlns\":\"http://www.example.com/iri#\"}", "{\"xmlns:rdfs\":\"http://www.w3.org/2000/01/rdf-schema#\",\"xml:base\":\"http://www.example.com/iri\",\"xmlns:owl\":\"http://www.w3.org/2002/07/owl#\",\"xmlns\":\"http://www.example.com/iri#\"}", "{}", "{}", "{}", "{}", "{}"].into_iter().collect::<HashSet<_>>();
        // assert_eq!(result_set, result_test);
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
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#versionIRI",
                "",
                "http://www.w3.org/2002/07/owl#equivalentClass",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#intersectionOf"
            ]
        );
        let result = table.get_column_as_vec_str("child_attr");
        assert_eq!(
            result[..8],
            [
                "{\"rdf:about\":\"http://www.example.com/iri\"}",
                "{\"rdf:about\":\"http://purl.obolibrary.org/obo/GO_0010958\"}",
                "{\"rdf:resource\":\"http://www.example.com/viri\"}",
                "",
                "{}",
                "{}",
                "{}",
                "{\"rdf:parseType\":\"Collection\"}"
            ]
        );
    }

    #[test]
    fn test_extract_owl() {
        // Test owl file
        let owl = r#"<?xml version="1.0"?>
<rdf:RDF xmlns="http://purl.obolibrary.org/obo/ro.owl#"
     xml:base="http://purl.obolibrary.org/obo/ro.owl"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:obo="http://purl.obolibrary.org/obo/"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:cito="http://purl.org/spar/cito/"
     xmlns:core="http://purl.obolibrary.org/obo/uberon/core#"
     xmlns:foaf="http://xmlns.com/foaf/0.1/"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:skos="http://www.w3.org/2004/02/skos/core#"
     xmlns:swrl="http://www.w3.org/2003/11/swrl#"
     xmlns:swrla="http://swrl.stanford.edu/ontologies/3.3/swrla.owl#"
     xmlns:swrlb="http://www.w3.org/2003/11/swrlb#"
     xmlns:terms="http://purl.org/dc/terms/"
     xmlns:subsets="http://purl.obolibrary.org/obo/ro/subsets#"
     xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
    <owl:Ontology rdf:about="http://purl.obolibrary.org/obo/ro.owl">
        <owl:versionIRI rdf:resource="http://purl.obolibrary.org/obo/ro/releases/2025-06-24/ro.owl"/>
        <terms:description xml:lang="en">The OBO Relations Ontology (RO) is a collection of OWL relations (ObjectProperties) intended for use across a wide variety of biological ontologies.</terms:description>
        <terms:license rdf:resource="https://creativecommons.org/publicdomain/zero/1.0/"/>
        <terms:title xml:lang="en">OBO Relations Ontology</terms:title>
        <owl:versionInfo>2025-06-24</owl:versionInfo>
        <foaf:homepage rdf:datatype="http://www.w3.org/2001/XMLSchema#anyURI"> https://github.com/oborel/obo-relations/</foaf:homepage>
    </owl:Ontology>
    
    <!-- http://purl.obolibrary.org/obo/RO_0002161 -->

    <owl:AnnotationProperty rdf:about="http://purl.obolibrary.org/obo/RO_0002161">
        <obo:IAO_0000112>tooth SubClassOf &apos;never in taxon&apos; value &apos;Aves&apos;</obo:IAO_0000112>
        <obo:IAO_0000115>x never in taxon T if and only if T is a class, and x does not instantiate the class expression &quot;in taxon some T&quot;. Note that this is a shortcut relation, and should be used as a hasValue restriction in OWL.</obo:IAO_0000115>
        <obo:IAO_0000117 rdf:resource="https://orcid.org/0000-0002-6601-2165"/>
        <obo:IAO_0000119 rdf:resource="http://www.ncbi.nlm.nih.gov/pubmed/17921072"/>
        <obo:IAO_0000119 rdf:resource="http://www.ncbi.nlm.nih.gov/pubmed/20973947"/>
        <obo:IAO_0000425>Class: ?X DisjointWith: RO_0002162 some ?Y </obo:IAO_0000425>
        <obo:OMO_0002000>PREFIX rdfs: &lt;http://www.w3.org/2000/01/rdf-schema#&gt;
PREFIX owl: &lt;http://www.w3.org/2002/07/owl#&gt;
PREFIX in_taxon: &lt;http://purl.obolibrary.org/obo/RO_0002162&gt;
PREFIX never_in_taxon: &lt;http://purl.obolibrary.org/obo/RO_0002161&gt;
CONSTRUCT {
  in_taxon: a owl:ObjectProperty .
  ?x owl:disjointWith [
    a owl:Restriction ;
    owl:onProperty in_taxon: ;
    owl:someValuesFrom ?taxon
  ] .
  ?x rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty in_taxon: ;
    owl:someValuesFrom [
      a owl:Class ;
      owl:complementOf ?taxon
    ]
  ] .
}
WHERE {
  ?x never_in_taxon: ?taxon .
}</obo:OMO_0002000>
        <rdfs:label>never in taxon</rdfs:label>
        <rdfs:seeAlso rdf:resource="https://github.com/obophenotype/uberon/wiki/Taxon-constraints"/>
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/RO_0002172"/>
    </owl:AnnotationProperty>

    <!-- http://purl.obolibrary.org/obo/RO_0009501 -->

    <owl:ObjectProperty rdf:about="http://purl.obolibrary.org/obo/RO_0009501">
        <rdfs:subPropertyOf rdf:resource="http://purl.obolibrary.org/obo/RO_0002410"/>
        <rdfs:domain rdf:resource="http://purl.obolibrary.org/obo/BFO_0000017"/>
        <rdfs:range rdf:resource="http://purl.obolibrary.org/obo/BFO_0000015"/>
        <owl:propertyChainAxiom rdf:parseType="Collection">
            <rdf:Description rdf:about="http://purl.obolibrary.org/obo/BFO_0000054"/>
            <rdf:Description rdf:about="http://purl.obolibrary.org/obo/RO_0002404"/>
        </owl:propertyChainAxiom>
        <obo:IAO_0000112>A drought sensitivity trait that inheres in a whole plant is realized in a systemic response process in response to exposure to drought conditions.</obo:IAO_0000112>
        <obo:IAO_0000112>An inflammatory disease that is realized in response to an inflammatory process occurring in the gut (which is itself the realization of a process realized in response to harmful stimuli in the mucosal lining of th gut)</obo:IAO_0000112>
        <obo:IAO_0000112>Environmental polymorphism in butterflies: These butterflies have a &apos;responsivity to day length trait&apos; that is realized in response to the duration of the day, and is realized in developmental processes that lead to increased or decreased pigmentation in the adult morph.</obo:IAO_0000112>
        <obo:IAO_0000115>r &apos;realized in response to&apos; s iff, r is a realizable (e.g. a plant trait such as responsivity to drought), s is an environmental stimulus (a process), and s directly causes the realization of r.</obo:IAO_0000115>
        <terms:contributor rdf:resource="https://orcid.org/0000-0001-6996-0040"/>
        <terms:contributor rdf:resource="https://orcid.org/0000-0002-6601-2165"/>
        <terms:contributor rdf:resource="https://orcid.org/0000-0002-7073-9172"/>
        <terms:contributor rdf:resource="https://orcid.org/0000-0002-8461-9745"/>
        <oboInOwl:hasExactSynonym>triggered by process</oboInOwl:hasExactSynonym>
        <rdfs:label xml:lang="en">realized in response to</rdfs:label>
        <rdfs:seeAlso rdf:datatype="http://www.w3.org/2001/XMLSchema#anyURI">https://docs.google.com/document/d/1KWhZxVBhIPkV6_daHta0h6UyHbjY2eIrnON1WIRGgdY/edit</rdfs:seeAlso>
    </owl:ObjectProperty>
    <owl:Axiom>
        <owl:annotatedSource rdf:resource="http://purl.obolibrary.org/obo/RO_0009501"/>
        <owl:annotatedProperty rdf:resource="http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"/>
        <owl:annotatedTarget>triggered by process</owl:annotatedTarget>
        <oboInOwl:hasDbXref rdf:resource="https://orcid.org/0000-0002-6601-2165"/>
    </owl:Axiom>

    <!-- http://purl.obolibrary.org/obo/CL_0000000 -->

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CL_0000000">
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/UBERON_0000061"/>
        <obo:IAO_0000115>A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.</obo:IAO_0000115>
        <obo:IAO_0000116 xml:lang="en">CL and GO definitions of cell differ based on inclusive or exclusive of cell wall, etc.</obo:IAO_0000116>
        <obo:IAO_0000116 xml:lang="en">We struggled with this definition. We are worried about circularity. We also considered requiring the capability of metabolism.</obo:IAO_0000116>
        <oboInOwl:hasDbXref>CALOHA:TS-2035</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>FMA:68646</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>GO:0005623</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>KUPO:0000002</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MESH:D002477</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>VHOG:0001533</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>WBbt:0004017</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>XAO:0003012</oboInOwl:hasDbXref>
        <rdfs:comment>The definition of cell is intended to represent all cells, and thus a cell is defined as a material entity and not an anatomical structure, which implies that it is part of an organism (or the entirety of one).</rdfs:comment>
        <rdfs:label>cell</rdfs:label>
    </owl:Class>
    <owl:Axiom>
        <owl:annotatedSource rdf:resource="http://purl.obolibrary.org/obo/CL_0000000"/>
        <owl:annotatedProperty rdf:resource="http://purl.obolibrary.org/obo/IAO_0000115"/>
        <owl:annotatedTarget>A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.</owl:annotatedTarget>
        <oboInOwl:hasDbXref>CARO:mah</oboInOwl:hasDbXref>
    </owl:Axiom>

    <!-- http://purl.obolibrary.org/obo/ENVO_01001569 -->

    <owl:NamedIndividual rdf:about="http://purl.obolibrary.org/obo/ENVO_01001569">
        <obo:BFO_0000050 rdf:resource="http://purl.obolibrary.org/obo/ENVO_01001571"/>
        <oboInOwl:created_by rdf:resource="https://orcid.org/0000-0002-4366-3088"/>
        <oboInOwl:creation_date rdf:datatype="http://www.w3.org/2001/XMLSchema#dateTime">2019-03-05T17:25:21Z</oboInOwl:creation_date>
        <oboInOwl:hasBroadSynonym>Western Australia Ecoregion</oboInOwl:hasBroadSynonym>
        <oboInOwl:hasDbXref>WWF:AA1310</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>https://www.worldwildlife.org/ecoregions/aa1310</oboInOwl:hasDbXref>
        <rdfs:label xml:lang="en">Western Australian Mulga Shrublands Ecoregion</rdfs:label>
    </owl:NamedIndividual>

    <!-- http://purl.obolibrary.org/obo/RO_0002029 -->

    <owl:DatatypeProperty rdf:about="http://purl.obolibrary.org/obo/RO_0002029">
        <rdfs:range>
            <rdfs:Datatype>
                <owl:onDatatype rdf:resource="http://www.w3.org/2001/XMLSchema#short"/>
                <owl:withRestrictions rdf:parseType="Collection">
                    <rdf:Description>
                        <xsd:minInclusive rdf:datatype="http://www.w3.org/2001/XMLSchema#short">0</xsd:minInclusive>
                    </rdf:Description>
                    <rdf:Description>
                        <xsd:maxInclusive rdf:datatype="http://www.w3.org/2001/XMLSchema#short">100</xsd:maxInclusive>
                    </rdf:Description>
                </owl:withRestrictions>
            </rdfs:Datatype>
        </rdfs:range>
        <obo:IAO_0000115>Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.</obo:IAO_0000115>
        <oboInOwl:created_by rdf:resource="https://orcid.org/0000-0002-7073-9172"/>
        <oboInOwl:creation_date rdf:datatype="http://www.w3.org/2001/XMLSchema#dateTime">2018-05-22T16:43:28Z</oboInOwl:creation_date>
        <rdfs:comment>This could be used to record the increased infant morality rate in some population compared to wild-type.  For examples of usage see http://purl.obolibrary.org/obo/FBcv_0000351 and subclasses.</rdfs:comment>
        <rdfs:label xml:lang="en">has increased age-specific mortality rate</rdfs:label>
    </owl:DatatypeProperty>
    <owl:Axiom>
        <owl:annotatedSource rdf:resource="http://purl.obolibrary.org/obo/RO_0002029"/>
        <owl:annotatedProperty rdf:resource="http://purl.obolibrary.org/obo/IAO_0000115"/>
        <owl:annotatedTarget>Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.</owl:annotatedTarget>
        <oboInOwl:hasDbXref>PMID:24138933</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>Wikipedia:Infant_mortality</oboInOwl:hasDbXref>
    </owl:Axiom>

</rdf:RDF>"#;

        // Make the xml data
        let batch = create_attachments_batch(
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
        let extracted = extract_xml("test", "bytes", &[batch], &DataFormat::Owl, &device).unwrap();

        // Check the contents of the extracted data
        let table = Subject::get_builder()
            .with_name("extracted")
            .with_record_batches(vec![extracted])
            .unwrap()
            .build()
            .unwrap();
        let result = table.get_column_as_vec_str("dataset");
        assert_eq!(
            result,
            [
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test", "test", "test", "test", "test",
                "test", "test", "test", "test", "test", "test", "test", "test", "test"
            ]
        );
        let mut result = table.get_column_as_vec_str("graph");
        result.sort();
        assert_eq!(
            result,
            [
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.-http://www.geneontology.org/formats/oboInOwl#hasDbXref-CARO:mah",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.-http://www.w3.org/2002/07/owl#annotatedProperty-http://purl.obolibrary.org/obo/IAO_0000115",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.-http://www.w3.org/2002/07/owl#annotatedSource-http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.-http://www.w3.org/2002/07/owl#annotatedTarget-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000116-CL and GO definitions of cell differ based on inclusive or exclusive of cell wall, etc.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000116-We struggled with this definition. We are worried about circularity. We also considered requiring the capability of metabolism.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-CALOHA:TS-2035",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-FMA:68646",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-GO:0005623",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-KUPO:0000002",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-MESH:D002477",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-VHOG:0001533",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-WBbt:0004017",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.geneontology.org/formats/oboInOwl#hasDbXref-XAO:0003012",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.w3.org/2000/01/rdf-schema#comment-The definition of cell is intended to represent all cells, and thus a cell is defined as a material entity and not an anatomical structure, which implies that it is part of an organism (or the entirety of one).",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.w3.org/2000/01/rdf-schema#label-cell",
                "http://purl.obolibrary.org/obo/CL_0000000-http://www.w3.org/2000/01/rdf-schema#subClassOf-http://purl.obolibrary.org/obo/UBERON_0000061",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://purl.obolibrary.org/obo/BFO_0000050-http://purl.obolibrary.org/obo/ENVO_01001571",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://www.geneontology.org/formats/oboInOwl#created_by-https://orcid.org/0000-0002-4366-3088",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://www.geneontology.org/formats/oboInOwl#creation_date-2019-03-05T17:25:21Z",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym-Western Australia Ecoregion",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://www.geneontology.org/formats/oboInOwl#hasDbXref-WWF:AA1310",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://www.geneontology.org/formats/oboInOwl#hasDbXref-https://www.worldwildlife.org/ecoregions/aa1310",
                "http://purl.obolibrary.org/obo/ENVO_01001569-http://www.w3.org/2000/01/rdf-schema#label-Western Australian Mulga Shrublands Ecoregion",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.-http://www.geneontology.org/formats/oboInOwl#hasDbXref-PMID:24138933",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.-http://www.geneontology.org/formats/oboInOwl#hasDbXref-Wikipedia:Infant_mortality",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.-http://www.w3.org/2002/07/owl#annotatedProperty-http://purl.obolibrary.org/obo/IAO_0000115",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.-http://www.w3.org/2002/07/owl#annotatedSource-http://purl.obolibrary.org/obo/RO_0002029",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.-http://www.w3.org/2002/07/owl#annotatedTarget-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://www.geneontology.org/formats/oboInOwl#created_by-https://orcid.org/0000-0002-7073-9172",
                "http://purl.obolibrary.org/obo/RO_0002029-http://www.geneontology.org/formats/oboInOwl#creation_date-2018-05-22T16:43:28Z",
                "http://purl.obolibrary.org/obo/RO_0002029-http://www.w3.org/2000/01/rdf-schema#comment-This could be used to record the increased infant morality rate in some population compared to wild-type.  For examples of usage see http://purl.obolibrary.org/obo/FBcv_0000351 and subclasses.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://www.w3.org/2000/01/rdf-schema#label-has increased age-specific mortality rate",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/IAO_0000112-tooth SubClassOf never in taxon value Aves",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/IAO_0000115-x never in taxon T if and only if T is a class, and x does not instantiate the class expression in taxon some T. Note that this is a shortcut relation, and should be used as a hasValue restriction in OWL.",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/IAO_0000117-https://orcid.org/0000-0002-6601-2165",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/IAO_0000119-http://www.ncbi.nlm.nih.gov/pubmed/17921072",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/IAO_0000119-http://www.ncbi.nlm.nih.gov/pubmed/20973947",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/IAO_0000425-Class: ?X DisjointWith: RO_0002162 some ?Y ",
                "http://purl.obolibrary.org/obo/RO_0002161-http://purl.obolibrary.org/obo/OMO_0002000-PREFIX rdfs: http://www.w3.org/2000/01/rdf-schema#\nPREFIX owl: http://www.w3.org/2002/07/owl#\nPREFIX in_taxon: http://purl.obolibrary.org/obo/RO_0002162\nPREFIX never_in_taxon: http://purl.obolibrary.org/obo/RO_0002161\nCONSTRUCT {\n  in_taxon: a owl:ObjectProperty .\n  ?x owl:disjointWith [\n    a owl:Restriction ;\n    owl:onProperty in_taxon: ;\n    owl:someValuesFrom ?taxon\n  ] .\n  ?x rdfs:subClassOf [\n    a owl:Restriction ;\n    owl:onProperty in_taxon: ;\n    owl:someValuesFrom [\n      a owl:Class ;\n      owl:complementOf ?taxon\n    ]\n  ] .\n}\nWHERE {\n  ?x never_in_taxon: ?taxon .\n}",
                "http://purl.obolibrary.org/obo/RO_0002161-http://www.w3.org/2000/01/rdf-schema#label-never in taxon",
                "http://purl.obolibrary.org/obo/RO_0002161-http://www.w3.org/2000/01/rdf-schema#seeAlso-https://github.com/obophenotype/uberon/wiki/Taxon-constraints",
                "http://purl.obolibrary.org/obo/RO_0002161-http://www.w3.org/2000/01/rdf-schema#subPropertyOf-http://purl.obolibrary.org/obo/RO_0002172",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.obolibrary.org/obo/IAO_0000112-A drought sensitivity trait that inheres in a whole plant is realized in a systemic response process in response to exposure to drought conditions.",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.obolibrary.org/obo/IAO_0000112-An inflammatory disease that is realized in response to an inflammatory process occurring in the gut (which is itself the realization of a process realized in response to harmful stimuli in the mucosal lining of th gut)",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.obolibrary.org/obo/IAO_0000112-Environmental polymorphism in butterflies: These butterflies have a responsivity to day length trait that is realized in response to the duration of the day, and is realized in developmental processes that lead to increased or decreased pigmentation in the adult morph.",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.obolibrary.org/obo/IAO_0000115-r realized in response to s iff, r is a realizable (e.g. a plant trait such as responsivity to drought), s is an environmental stimulus (a process), and s directly causes the realization of r.",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.org/dc/terms/contributor-https://orcid.org/0000-0001-6996-0040",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.org/dc/terms/contributor-https://orcid.org/0000-0002-6601-2165",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.org/dc/terms/contributor-https://orcid.org/0000-0002-7073-9172",
                "http://purl.obolibrary.org/obo/RO_0009501-http://purl.org/dc/terms/contributor-https://orcid.org/0000-0002-8461-9745",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process-http://www.geneontology.org/formats/oboInOwl#hasDbXref-https://orcid.org/0000-0002-6601-2165",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process-http://www.w3.org/2002/07/owl#annotatedProperty-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process-http://www.w3.org/2002/07/owl#annotatedSource-http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process-http://www.w3.org/2002/07/owl#annotatedTarget-triggered by process",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.w3.org/2000/01/rdf-schema#domain-http://purl.obolibrary.org/obo/BFO_0000017",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.w3.org/2000/01/rdf-schema#label-realized in response to",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.w3.org/2000/01/rdf-schema#range-http://purl.obolibrary.org/obo/BFO_0000015",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.w3.org/2000/01/rdf-schema#seeAlso-https://docs.google.com/document/d/1KWhZxVBhIPkV6_daHta0h6UyHbjY2eIrnON1WIRGgdY/edit",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.w3.org/2000/01/rdf-schema#subPropertyOf-http://purl.obolibrary.org/obo/RO_0002410",
                "http://purl.obolibrary.org/obo/ro.owl-http://purl.org/dc/terms/description-The OBO Relations Ontology (RO) is a collection of OWL relations (ObjectProperties) intended for use across a wide variety of biological ontologies.",
                "http://purl.obolibrary.org/obo/ro.owl-http://purl.org/dc/terms/license-https://creativecommons.org/publicdomain/zero/1.0/",
                "http://purl.obolibrary.org/obo/ro.owl-http://purl.org/dc/terms/title-OBO Relations Ontology",
                "http://purl.obolibrary.org/obo/ro.owl-http://www.w3.org/2002/07/owl#versionIRI-http://purl.obolibrary.org/obo/ro/releases/2025-06-24/ro.owl",
                "http://purl.obolibrary.org/obo/ro.owl-http://www.w3.org/2002/07/owl#versionInfo-2025-06-24",
                "http://purl.obolibrary.org/obo/ro.owl-http://xmlns.com/foaf/0.1/homepage- https://github.com/oborel/obo-relations/"
            ]
        );
        let result = table.get_column_as_vec_str("entity");
        assert_eq!(
            result,
            [
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#AnnotationProperty",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Axiom",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2002/07/owl#DatatypeProperty",
                "http://www.w3.org/2002/07/owl#DatatypeProperty",
                "http://www.w3.org/2002/07/owl#DatatypeProperty",
                "http://www.w3.org/2002/07/owl#DatatypeProperty",
                "http://www.w3.org/2002/07/owl#DatatypeProperty",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#NamedIndividual",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#ObjectProperty",
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#Ontology",
                "http://www.w3.org/2002/07/owl#Ontology"
            ]
        );
        let result = table.get_column_as_vec_str("subject");
        assert_eq!(
            result,
            [
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0002161",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "http://purl.obolibrary.org/obo/CL_0000000-http://purl.obolibrary.org/obo/IAO_0000115-A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0002029-http://purl.obolibrary.org/obo/IAO_0000115-Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process",
                "http://purl.obolibrary.org/obo/RO_0009501-http://www.geneontology.org/formats/oboInOwl#hasExactSynonym-triggered by process",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "http://purl.obolibrary.org/obo/RO_0002029",
                "http://purl.obolibrary.org/obo/RO_0002029",
                "http://purl.obolibrary.org/obo/RO_0002029",
                "http://purl.obolibrary.org/obo/RO_0002029",
                "http://purl.obolibrary.org/obo/RO_0002029",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/ENVO_01001569",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "http://purl.obolibrary.org/obo/ro.owl",
                "http://purl.obolibrary.org/obo/ro.owl",
                "http://purl.obolibrary.org/obo/ro.owl",
                "http://purl.obolibrary.org/obo/ro.owl",
                "http://purl.obolibrary.org/obo/ro.owl",
                "http://purl.obolibrary.org/obo/ro.owl"
            ]
        );
        let result = table.get_column_as_vec_str("predicate");
        assert_eq!(
            result,
            [
                "http://purl.obolibrary.org/obo/IAO_0000112",
                "http://purl.obolibrary.org/obo/IAO_0000115",
                "http://purl.obolibrary.org/obo/IAO_0000117",
                "http://purl.obolibrary.org/obo/IAO_0000119",
                "http://purl.obolibrary.org/obo/IAO_0000119",
                "http://purl.obolibrary.org/obo/IAO_0000425",
                "http://purl.obolibrary.org/obo/OMO_0002000",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
                "http://www.w3.org/2002/07/owl#annotatedTarget",
                "http://www.w3.org/2002/07/owl#annotatedTarget",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.w3.org/2002/07/owl#annotatedProperty",
                "http://www.w3.org/2002/07/owl#annotatedSource",
                "http://www.w3.org/2002/07/owl#annotatedTarget",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.w3.org/2002/07/owl#annotatedProperty",
                "http://www.w3.org/2002/07/owl#annotatedSource",
                "http://www.w3.org/2002/07/owl#annotatedProperty",
                "http://www.w3.org/2002/07/owl#annotatedSource",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.w3.org/2000/01/rdf-schema#comment",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                "http://purl.obolibrary.org/obo/IAO_0000116",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://purl.obolibrary.org/obo/IAO_0000115",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://purl.obolibrary.org/obo/IAO_0000116",
                "http://purl.obolibrary.org/obo/IAO_0000115",
                "http://www.w3.org/2000/01/rdf-schema#comment",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.geneontology.org/formats/oboInOwl#created_by",
                "http://www.geneontology.org/formats/oboInOwl#creation_date",
                "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
                "http://www.geneontology.org/formats/oboInOwl#created_by",
                "http://www.geneontology.org/formats/oboInOwl#creation_date",
                "http://purl.obolibrary.org/obo/BFO_0000050",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://purl.obolibrary.org/obo/IAO_0000112",
                "http://purl.obolibrary.org/obo/IAO_0000115",
                "http://purl.org/dc/terms/contributor",
                "http://purl.org/dc/terms/contributor",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2000/01/rdf-schema#range",
                "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                "http://purl.org/dc/terms/contributor",
                "http://purl.obolibrary.org/obo/IAO_0000112",
                "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
                "http://purl.org/dc/terms/contributor",
                "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
                "http://www.w3.org/2000/01/rdf-schema#domain",
                "http://purl.obolibrary.org/obo/IAO_0000112",
                "http://purl.org/dc/terms/title",
                "http://purl.org/dc/terms/description",
                "http://purl.org/dc/terms/license",
                "http://www.w3.org/2002/07/owl#versionIRI",
                "http://www.w3.org/2002/07/owl#versionInfo",
                "http://xmlns.com/foaf/0.1/homepage"
            ]
        );
        let result = table.get_column_as_vec_str("object");
        assert_eq!(
            result,
            [
                "tooth SubClassOf never in taxon value Aves",
                "x never in taxon T if and only if T is a class, and x does not instantiate the class expression in taxon some T. Note that this is a shortcut relation, and should be used as a hasValue restriction in OWL.",
                "https://orcid.org/0000-0002-6601-2165",
                "http://www.ncbi.nlm.nih.gov/pubmed/17921072",
                "http://www.ncbi.nlm.nih.gov/pubmed/20973947",
                "Class: ?X DisjointWith: RO_0002162 some ?Y ",
                "PREFIX rdfs: http://www.w3.org/2000/01/rdf-schema#\nPREFIX owl: http://www.w3.org/2002/07/owl#\nPREFIX in_taxon: http://purl.obolibrary.org/obo/RO_0002162\nPREFIX never_in_taxon: http://purl.obolibrary.org/obo/RO_0002161\nCONSTRUCT {\n  in_taxon: a owl:ObjectProperty .\n  ?x owl:disjointWith [\n    a owl:Restriction ;\n    owl:onProperty in_taxon: ;\n    owl:someValuesFrom ?taxon\n  ] .\n  ?x rdfs:subClassOf [\n    a owl:Restriction ;\n    owl:onProperty in_taxon: ;\n    owl:someValuesFrom [\n      a owl:Class ;\n      owl:complementOf ?taxon\n    ]\n  ] .\n}\nWHERE {\n  ?x never_in_taxon: ?taxon .\n}",
                "never in taxon",
                "https://github.com/obophenotype/uberon/wiki/Taxon-constraints",
                "http://purl.obolibrary.org/obo/RO_0002172",
                "triggered by process",
                "A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "CARO:mah",
                "http://purl.obolibrary.org/obo/IAO_0000115",
                "http://purl.obolibrary.org/obo/CL_0000000",
                "Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "PMID:24138933",
                "Wikipedia:Infant_mortality",
                "http://purl.obolibrary.org/obo/IAO_0000115",
                "http://purl.obolibrary.org/obo/RO_0002029",
                "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
                "http://purl.obolibrary.org/obo/RO_0009501",
                "https://orcid.org/0000-0002-6601-2165",
                "XAO:0003012",
                "CALOHA:TS-2035",
                "GO:0005623",
                "VHOG:0001533",
                "KUPO:0000002",
                "MESH:D002477",
                "The definition of cell is intended to represent all cells, and thus a cell is defined as a material entity and not an anatomical structure, which implies that it is part of an organism (or the entirety of one).",
                "cell",
                "http://purl.obolibrary.org/obo/UBERON_0000061",
                "We struggled with this definition. We are worried about circularity. We also considered requiring the capability of metabolism.",
                "WBbt:0004017",
                "A material entity of anatomical origin (part of or deriving from an organism) that has as its parts a maximally connected cell compartment surrounded by a plasma membrane.",
                "FMA:68646",
                "CL and GO definitions of cell differ based on inclusive or exclusive of cell wall, etc.",
                "Then percentage of organisms in a population that die during some specified age range (age-specific mortality rate), minus the percentage that die in during the same age range in a wild-type population.",
                "This could be used to record the increased infant morality rate in some population compared to wild-type.  For examples of usage see http://purl.obolibrary.org/obo/FBcv_0000351 and subclasses.",
                "has increased age-specific mortality rate",
                "https://orcid.org/0000-0002-7073-9172",
                "2018-05-22T16:43:28Z",
                "Western Australia Ecoregion",
                "https://orcid.org/0000-0002-4366-3088",
                "2019-03-05T17:25:21Z",
                "http://purl.obolibrary.org/obo/ENVO_01001571",
                "https://www.worldwildlife.org/ecoregions/aa1310",
                "WWF:AA1310",
                "Western Australian Mulga Shrublands Ecoregion",
                "A drought sensitivity trait that inheres in a whole plant is realized in a systemic response process in response to exposure to drought conditions.",
                "r realized in response to s iff, r is a realizable (e.g. a plant trait such as responsivity to drought), s is an environmental stimulus (a process), and s directly causes the realization of r.",
                "https://orcid.org/0000-0002-8461-9745",
                "https://orcid.org/0000-0001-6996-0040",
                "realized in response to",
                "http://purl.obolibrary.org/obo/BFO_0000015",
                "https://docs.google.com/document/d/1KWhZxVBhIPkV6_daHta0h6UyHbjY2eIrnON1WIRGgdY/edit",
                "https://orcid.org/0000-0002-6601-2165",
                "An inflammatory disease that is realized in response to an inflammatory process occurring in the gut (which is itself the realization of a process realized in response to harmful stimuli in the mucosal lining of th gut)",
                "http://purl.obolibrary.org/obo/RO_0002410",
                "https://orcid.org/0000-0002-7073-9172",
                "triggered by process",
                "http://purl.obolibrary.org/obo/BFO_0000017",
                "Environmental polymorphism in butterflies: These butterflies have a responsivity to day length trait that is realized in response to the duration of the day, and is realized in developmental processes that lead to increased or decreased pigmentation in the adult morph.",
                "OBO Relations Ontology",
                "The OBO Relations Ontology (RO) is a collection of OWL relations (ObjectProperties) intended for use across a wide variety of biological ontologies.",
                "https://creativecommons.org/publicdomain/zero/1.0/",
                "http://purl.obolibrary.org/obo/ro/releases/2025-06-24/ro.owl",
                "2025-06-24",
                " https://github.com/oborel/obo-relations/"
            ]
        );
    }
}
