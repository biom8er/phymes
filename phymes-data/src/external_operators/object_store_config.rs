use std::fmt::Display;

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_core::{AvailableSubjects, DataFormat, MappableTrait, ObjectStorageBackend, Table, TableTrait};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{DataConfigTrait, HTTPClientRequestSchemas};

/// The Object Store operation types
/// 
/// # Todo
/// - Support for other operations besides "Get" and "PutMultipart"
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum ObjectStoreOptsType {
    #[default]
    #[value(name = "Get")]
    Get,
    #[value(name = "GetStream")]
    GetStream,
    /// Get followed by `meta` without reading bytes
    #[value(name = "GetMeta")]
    GetMeta,
    #[value(name = "GetRanges")]
    GetRanges,
    #[value(name = "Put")]
    Put,
    #[value(name = "PutMultipart")]
    PutMultipart,
    #[value(name = "List")]
    List,
    #[value(name = "Delete")]
    Delete,
    #[value(name = "Copy")]
    Copy,
    #[value(name = "Rename")]
    Rename,
}
impl Display for ObjectStoreOptsType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Get => write!(f, "Get"),
            Self::GetStream => write!(f, "GetStream"),
            Self::GetMeta => write!(f, "GetMeta"),
            Self::GetRanges => write!(f, "GetRanges"),
            Self::Put => write!(f, "Put"),
            Self::PutMultipart => write!(f, "PutMultipart"),
            Self::List => write!(f, "List"),
            Self::Delete => write!(f, "Delete"),
            Self::Copy => write!(f, "Copy"),
            Self::Rename => write!(f, "Rename"),
        }
    }
}

/// Object store configuration
/// 
/// # Todo
/// - Other throttle configs
///   See <https://docs.rs/object_store/latest/object_store/throttle/struct.ThrottleConfig.html>
/// 
/// - Other Get/Put/... specific configurations
///   See individutal `options` per operation <https://docs.rs/object_store/latest/object_store/trait.ObjectStore.html?
/// 
/// - Other options including projections (column subsetting), filtering, etc.
/// 
/// # Notes
/// - Config-driven requests: `locations` must be specified
/// - Message-driven requests: `subject_name` must be specified
#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct ObjectStoreConfig {
    /// The timeout in seconds
    #[arg(long, default_value_t = 15)]
    pub timeout: usize,

    /// The object store operations type
    #[arg(long, default_value_t = ObjectStoreOptsType::Get)]
    pub ops_type: ObjectStoreOptsType,

    /// The object store backend
    #[arg(long)]
    pub backend: ObjectStorageBackend,

    /// The object store bucket (also called `container` for Azure or `root` for LocalFs; None for InMemory)
    #[arg(long)]
    pub bucket: Option<String>,

    /// Serialized JSON value representing a HashMap of ObjectStore configurations
    /// See AWS, GCP, and Azure documentation for valid Key/Value pairs
    #[arg(long)]
    pub config: Option<Map<String,Value>>,

    /// The (partition) location(s) within the object store that the data are in
    /// DM: in the future this could be a URL that contains the full address similar to AWS Redshift Manifest
    ///     See <https://docs.aws.amazon.com/redshift/latest/dg/loading-data-files-using-manifest.html>
    /// 
    /// # Notes
    /// - Same as `bucket`
    #[arg(long)]
    pub locations: Option<Vec<String>>,

    /// The length of the data chunks to send
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,

    /// The name of the streaming subject to write to the object store 
    ///   OR the name of the streaming manifest file with subjects to read
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_name: Option<String>,
}

impl DataConfigTrait for ObjectStoreConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_table(table: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // Check for the required fields
        let column_names = table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<HashSet<_>>();
        if !(column_names.contains("timeout")
            && column_names.contains("ops_type")
            && column_names.contains("backend")
            && column_names.contains("bucket"))
        {
            return Err(anyhow!(
                "Table {} is missing required Field for `timeout`, `ops_type`, `backend`, and `bucket` in ObjectStoreConfig.",
                table.get_name()
            ));
        }

        // Try to build the config
        match table.to_struct::<ObjectStoreConfig>() {
            Ok(config_vec) => match config_vec.first() {
                Some(config) => Ok(config.to_owned()),
                None => Err(anyhow!(
                    "No config data found for ObjectStoreConfig with subject {}",
                    table.get_name()
                )),
            },
            Err(err) => Err(anyhow!(
                "ObjectStoreConfig could not be built for subject {}. {err}",
                table.get_name()
            )),
        }
    }
}
