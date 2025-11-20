use std::fmt::Display;

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone)]
pub struct CsvFormat {
    pub delimiter: u8,
    pub header: bool,
    pub batch_size: usize,
}

impl Default for CsvFormat {
    fn default() -> Self {
        CsvFormat {
            delimiter: b',',
            header: true,
            batch_size: 1024,
        }
    }
}

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone)]
pub struct JsonFormat {
    pub batch_size: usize,
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat { batch_size: 1024 }
    }
}

/// How to extract out the OWL triples
/// 
/// # Notes
/// * the resultant schema is subject, predicate, object
/// * subject = serialized {attr: val}
/// * predicate = predicate tag
/// * object = serialized {attr: val} or text
#[derive(Parser, Debug, PartialEq, Eq, Serialize, Deserialize, Clone, Default)]
pub struct OwlFormat {
    /// Slice of Strings for the subject tags to consider (e.g., rdf:Description)
    /// empty indicates all
    pub subject_tags: Vec<String>,
    /// Slice of Strings of attributes to identify the subject (i.e., rdf:about)
    /// empty indicates all
    pub subject_attributes: Vec<String>,
    /// Slice of Strings for the predicate tags to consider (e.g., rdfs:label)
    /// empty indicates all
    pub predicate_tags: Vec<String>,
    /// Slice of Strings of attributes to identify the object within the predicate element (i.e., rdf:resource) if specified
    /// empty indicates all
    pub predicate_attributes: Vec<String>,
}

impl OwlFormat {
    pub fn owl_format_class() -> Self {
        let subject_tags = ["owl:Class"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subject_attributes = ["rdf:about"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let predicate_tags = [
            "rdf:type",
            "rdfs:label",
            "rdfs:seeAlso",
            "obo:IAO_0000115",
            "oboInOwl:hasOBONamespace",
            "oboInOwl:id",
            "oboInOwl:hasRelatedSynonym",
            "oboInOwl:hasExactSynonym",
            "oboInOwl:hasBroadSynonym",
            "oboInOwl:hasNarrowSynonym",
            "owl:sameAs"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let predicate_attributes = ["rdf:resource"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        OwlFormat { 
            subject_tags, 
            subject_attributes, 
            predicate_tags, 
            predicate_attributes
        }
    }
    pub fn owl_format_object_property() -> Self {
        let subject_tags = ["owl:ObjectProperty"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subject_attributes = ["rdf:about"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let predicate_tags = [
            "rdf:type",
            "rdfs:label",
            "rdfs:seeAlso",
            "obo:IAO_0000115",
            "oboInOwl:hasOBONamespace",
            "oboInOwl:id",
            "oboInOwl:hasRelatedSynonym",
            "oboInOwl:hasExactSynonym",
            "oboInOwl:hasBroadSynonym",
            "oboInOwl:hasNarrowSynonym",
            "owl:sameAs",
            "owl:inverseOf ",
            "rdfs:subPropertyOf",
            "rdfs:domain",
            "rdfs:range"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let predicate_attributes = ["rdf:resource"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        OwlFormat { 
            subject_tags, 
            subject_attributes, 
            predicate_tags, 
            predicate_attributes
        }
    }
    pub fn owl_format_named_individual() -> Self {
        let subject_tags = ["owl:NamedIndividual"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subject_attributes = ["rdf:about"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let predicate_tags = Vec::new();
        let predicate_attributes = ["rdf:resource"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        OwlFormat { 
            subject_tags, 
            subject_attributes, 
            predicate_tags, 
            predicate_attributes
        }
    }
    pub fn owl_common() -> Vec<String> {
        [
            "rdf:type",
            "rdfs:label",
            "rdfs:seeAlso",
            "obo:IAO_0000115",
            "oboInOwl:hasOBONamespace",
            "oboInOwl:id",
            "oboInOwl:hasAlternativeId",
            "oboInOwl:hasRelatedSynonym",
            "oboInOwl:hasExactSynonym",
            "oboInOwl:hasBroadSynonym",
            "oboInOwl:hasNarrowSynonym",
            "owl:sameAs",
            "oboInOwl:inSubset",
        ].into_iter().map(|s| s.to_string())
        .collect::<Vec<_>>()
    }
    pub fn owl_class() -> Vec<String> {
        ["owl:Class",
            "rdfs:subclassOf",
        ].into_iter().map(|s| s.to_string())
        .chain(Self::owl_common())
        .collect::<Vec<_>>()
    }
    pub fn owl_object_property() -> Vec<String> {
        ["owl:ObjectProperty",
            "owl:inverseOf ",
            "rdfs:subPropertyOf",
            "rdfs:domain",
            "rdfs:range",
        ].into_iter().map(|s| s.to_string())
        .chain(Self::owl_common())
        .collect::<Vec<_>>()
    }
    pub fn owl_named_individual() -> Vec<String> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, ValueEnum, Deserialize, Default)]
pub enum DataFormat {
    /// Comma Seperated Values string
    #[value(name = "CsvDefault")]
    CsvDefault,
    #[clap(skip)]
    #[value(name = "Csv")]
    Csv(CsvFormat),
    /// Json attachment
    #[value(name = "JsonDefault")]
    JsonDefault,
    #[clap(skip)]
    #[value(name = "Json")]
    Json(JsonFormat),
    /// Pdf attachment
    #[value(name = "Pdf")]
    Pdf,
    /// JSON Object written to Bytes
    #[value(name = "Bytes")]
    Bytes,
    /// Arrow IPC stream
    #[value(name = "Ipc")]
    Ipc,
    /// The raw [RecordBatch]
    ///
    /// [RecordBatch]: arrow::record_batch::RecordBatch
    #[default]
    #[value(name = "None")]
    None,
    /// HTML format
    #[value(name = "Html")]
    Html,
    /// HTML format
    #[value(name = "Txt")]
    Txt,
    /// XML format
    #[value(name = "Xml")]
    Xml,
    /// OWL format
    #[value(name = "OwlDefault")]
    OwlDefault,
    /// OWL format
    #[value(name = "OwlClass")]
    OwlClass,
    #[value(name = "OwlObjectProperty")]
    OwlObjectProperty,
    #[value(name = "OwlNamedIndividual")]
    OwlNamedIndividual,
    /// OWL format
    #[clap(skip)]
    #[value(name = "Owl")]
    Owl(OwlFormat),
}

impl DataFormat {
    /// Convert from a filename extension
    pub fn from_extension(extension: &str) -> Result<Self> {
        let format = match extension {
            "csv" => DataFormat::CsvDefault,
            "json" => DataFormat::JsonDefault,
            "pdf" => DataFormat::Pdf,
            "bytes" => DataFormat::Bytes,
            "ipc" => DataFormat::Ipc,
            "html" => DataFormat::Html,
            "txt" => DataFormat::Txt,
            "Xml" => DataFormat::Xml,
            "OwlDefault" => DataFormat::OwlDefault,
            "OwlClass" => DataFormat::OwlClass,
            "OwlObjectProperty" => DataFormat::OwlObjectProperty,
            "OwlNamedIndividual" => DataFormat::OwlNamedIndividual,
            _ => {
                return Err(anyhow!(
                    "File extension {extension} was not recognized. Supported extensions are .csv, .json, .pdf, .bytes, .ipc, .txt, .xml,, .owl, and .html"
                ));
            }
        };
        Ok(format)
    }

    /// The file extension for the format
    pub fn to_extension(&self) -> &str {
        match self {
            Self::Csv(_) | Self::CsvDefault => "csv",
            Self::Json(_) | Self::JsonDefault => "json",
            Self::Bytes => "bytes",
            Self::Ipc => "ipc",
            Self::Pdf => "pdf",
            Self::Html => "html",
            Self::Txt => "txt",
            Self::Xml => "Xml",
            Self::Owl(_) | Self::OwlDefault | Self::OwlClass | Self::OwlObjectProperty | Self::OwlNamedIndividual => "Owl",
            Self::None => "",
        }
    }
}

impl Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv(_) => write!(f, "Csv"),
            Self::CsvDefault => write!(f, "CsvDefault"),
            Self::Json(_) => write!(f, "Json"),
            Self::JsonDefault => write!(f, "JsonDefault"),
            Self::Bytes => write!(f, "Bytes"),
            Self::Ipc => write!(f, "Ipc"),
            Self::Pdf => write!(f, "Pdf"),
            Self::Html => write!(f, "Html"),
            Self::Txt => write!(f, "Txt"),
            Self::Xml => write!(f, "Xml"),
            Self::Owl(_) => write!(f, "Owl"),
            Self::OwlDefault => write!(f, "OwlDefault"),
            Self::OwlClass => write!(f, "OwlClass"),
            Self::OwlObjectProperty => write!(f, "OwlObjectProperty"),
            Self::OwlNamedIndividual => write!(f, "OwlNamedIndividual"),
            Self::None => write!(f, "None"),
        }
    }
}
