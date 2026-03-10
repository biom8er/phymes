use bytes::Bytes;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use std::sync::Arc;

use crate::storage::IpcWriterOld;

/// Drives a multipart upload using chunks produced by IpcWriter.
///
/// Your application controls when to call write_batch() and finish().
pub async fn upload_multipart(
    store: Arc<dyn ObjectStore>,
    path: Path,
    writer: &mut IpcWriterOld,
) -> anyhow::Result<()> {
    let mut mp = store.put_multipart(&path).await?;

    // Drain chunks produced so far
    while let Some(chunk) = writer.poll_chunk() {
        mp.put_part(Bytes::from(chunk).into()).await?;
    }

    // Finish the IPC writer
    writer.finish()?;

    // Drain remaining chunks
    while let Some(chunk) = writer.poll_chunk() {
        mp.put_part(Bytes::from(chunk).into()).await?;
    }

    mp.complete().await?;
    Ok(())
}
