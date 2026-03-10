// src/storage.rs
use std::sync::Arc;
use anyhow::Result;
use object_store::{ObjectStore, memory::InMemory, local::LocalFileSystem};
#[cfg(feature = "api")]
use object_store::{aws::AmazonS3Builder, azure::MicrosoftAzureBuilder, gcp::GoogleCloudStorageBuilder};

pub enum StorageBackendConfig {
    #[cfg(feature = "api")]
    Aws { bucket: String },
    #[cfg(feature = "api")]
    Gcp { bucket: String },
    #[cfg(feature = "api")]
    Azure { container: String },
    LocalFs { root: String },
    InMemory,
}

pub async fn make_store(cfg: StorageBackendConfig) -> Result<Arc<dyn ObjectStore>> {

    let store: Arc<dyn ObjectStore> = match cfg {
        #[cfg(feature = "api")]
        StorageBackendConfig::Aws { bucket } => Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?,
        ),
        #[cfg(feature = "api")]
        StorageBackendConfig::Gcp { bucket } => Arc::new(
            GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket)
                .build()?,
        ),
        #[cfg(feature = "api")]
        StorageBackendConfig::Azure { container } => Arc::new(
            MicrosoftAzureBuilder::from_env()
                .with_container_name(container)
                .build()?,
        ),
        StorageBackendConfig::LocalFs { root } => {
            Arc::new(LocalFileSystem::new_with_prefix(root)?)
        }
        StorageBackendConfig::InMemory => Arc::new(InMemory::new()),
    };

    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn make_in_memory_store() {
        let store = make_store(StorageBackendConfig::InMemory).await.unwrap();
        let _ = store.list(None);
    }
}
