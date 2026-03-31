use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::anyhow;
use clap::ValueEnum;
#[cfg(not(target_arch = "wasm32"))]
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, memory::InMemory};
#[cfg(feature = "api")]
use object_store::{
    aws::{AmazonS3Builder, AmazonS3ConfigKey},
    azure::MicrosoftAzureBuilder,
    gcp::GoogleCloudStorageBuilder,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
#[cfg(feature = "api")]
use std::str::FromStr;
use std::{fmt::Display, sync::Arc};

#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default, PartialEq)]
pub enum ObjectStorageBackend {
    #[cfg(feature = "api")]
    #[value(name = "Aws")]
    Aws,
    #[cfg(feature = "api")]
    #[value(name = "Gcp")]
    Gcp,
    #[cfg(feature = "api")]
    #[value(name = "Azure")]
    Azure,
    #[cfg(not(target_arch = "wasm32"))]
    #[value(name = "LocalFs")]
    LocalFs,
    #[default]
    #[value(name = "InMemory")]
    InMemory,
}
impl Display for ObjectStorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "api")]
            Self::Aws => write!(f, "Aws"),
            #[cfg(feature = "api")]
            Self::Gcp => write!(f, "Gcp"),
            #[cfg(feature = "api")]
            Self::Azure => write!(f, "Azure"),
            #[cfg(not(target_arch = "wasm32"))]
            Self::LocalFs => write!(f, "LocalFs"),
            Self::InMemory => write!(f, "InMemory"),
        }
    }
}

pub fn make_store(
    backend: &ObjectStorageBackend,
    _bucket: Option<&String>,
    _config: Option<&Map<String, Value>>,
) -> Result<Arc<dyn ObjectStore>> {
    let store: Arc<dyn ObjectStore> = match backend {
        #[cfg(feature = "api")]
        ObjectStorageBackend::Aws => {
            let mut builder = AmazonS3Builder::from_env()
                .with_bucket_name(_bucket.ok_or(anyhow!("Missing `bucket` name for {backend}"))?);
            if let Some(config) = _config {
                for (k, v) in config {
                    let key = AmazonS3ConfigKey::from_str(k)?;
                    builder = builder.with_config(key, v.as_str().unwrap_or_default());
                }
            }
            Arc::new(builder.build()?)
        }
        #[cfg(feature = "api")]
        ObjectStorageBackend::Gcp => Arc::new(
            GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(_bucket.ok_or(anyhow!("Missing `bucket` name for {backend}"))?)
                .build()?,
        ),
        #[cfg(feature = "api")]
        ObjectStorageBackend::Azure => Arc::new(
            MicrosoftAzureBuilder::from_env()
                .with_container_name(_bucket.ok_or(anyhow!("Missing `bucket` name for {backend}"))?)
                .build()?,
        ),
        #[cfg(not(target_arch = "wasm32"))]
        ObjectStorageBackend::LocalFs => Arc::new(LocalFileSystem::new_with_prefix(
            _bucket.ok_or(anyhow!("Missing `bucket` name for {backend}"))?,
        )?),
        ObjectStorageBackend::InMemory => Arc::new(InMemory::new()),
    };

    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn make_in_memory_store() {
        let store = make_store(&ObjectStorageBackend::InMemory, None, None).unwrap();
        let _ = store.list(None);
    }
}
