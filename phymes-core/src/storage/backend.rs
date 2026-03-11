// src/storage.rs
use anyhow::{Result, anyhow};
use clap::ValueEnum;
use object_store::{ObjectStore, local::LocalFileSystem, memory::InMemory};
#[cfg(feature = "api")]
use object_store::{
    aws::AmazonS3Builder, azure::MicrosoftAzureBuilder, gcp::GoogleCloudStorageBuilder,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};


#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
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
            Self::LocalFs => write!(f, "LocalFs"),
            Self::InMemory => write!(f, "InMemory"),
        }
    }
}

pub async fn make_store(cfg: ObjectStorageBackend, bucket: Option<String>) -> Result<Arc<dyn ObjectStore>> {
    let store: Arc<dyn ObjectStore> = match cfg {
        #[cfg(feature = "api")]
        ObjectStorageBackend::Aws => Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(bucket.ok_or(anyhow!("Missing `bucket` name for {cfg}"))?)
                .build()?,
        ),
        #[cfg(feature = "api")]
        ObjectStorageBackend::Gcp => Arc::new(
            GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket.ok_or(anyhow!("Missing `bucket` name for {cfg}"))?)
                .build()?,
        ),
        #[cfg(feature = "api")]
        ObjectStorageBackend::Azure => Arc::new(
            MicrosoftAzureBuilder::from_env()
                .with_container_name(bucket.ok_or(anyhow!("Missing `bucket` name for {cfg}"))?)
                .build()?,
        ),
        ObjectStorageBackend::LocalFs => Arc::new(LocalFileSystem::new_with_prefix(bucket.ok_or(anyhow!("Missing `bucket` name for {cfg}"))?)?),
        ObjectStorageBackend::InMemory => Arc::new(InMemory::new()),
    };

    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn make_in_memory_store() {
        let store = make_store(ObjectStorageBackend::InMemory, None).await.unwrap();
        let _ = store.list(None);
    }
}
