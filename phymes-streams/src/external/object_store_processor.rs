use std::{
    collections::VecDeque,
    io::Write,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use bytes::Bytes;
use chrono::Utc;
use futures::{FutureExt, Stream, StreamExt};
use object_store::{
    GetOptions, GetResult, MultipartUpload, ObjectMeta, ObjectStore, ObjectStoreExt, PutOptions,
    PutPayload, PutResult, WriteMultipart, path::Path,
};
use parking_lot::Mutex;
use phymes_core::{
    BuilderTrait, ChunkedWriter, MappableTrait, ObjectStorageBackend, OnChunk, RecordBatchStream,
    RuntimeEnv, RuntimeEnvTrait, SendableRecordBatchStream, SubjectBuilder, SubjectBuilderTrait,
    SubjectTrait, make_store,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait, create_timestamp_micros,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, create_bytes_fields, create_object_store_batch,
    create_object_store_meta_batch, create_values_fields,
};
use serde_json::{Map, Value, json};

use crate::{ObjectStoreConfig, ObjectStoreOptsType};

/// The state of the Object Store API request.
#[allow(clippy::type_complexity)]
pub enum ObjectStoreState {
    NotStarted,
    StorageReaderGetResult(
        Pin<Box<dyn Future<Output = Result<GetResult, object_store::Error>> + Send>>,
    ),
    StorageReaderBytesResult(
        Pin<Box<dyn Future<Output = Result<Bytes, object_store::Error>> + Send>>,
    ),
    StorageReaderStreamResult(
        Pin<Box<dyn Stream<Item = Result<Bytes, object_store::Error>> + Send>>,
    ),
    StorageReaderStreamList(
        Pin<Box<dyn Stream<Item = Result<ObjectMeta, object_store::Error>> + Send>>,
    ),
    StorageWriterMultipart(
        Pin<Box<dyn Future<Output = Result<Box<dyn MultipartUpload>, object_store::Error>> + Send>>,
    ),
    StorageWriterPutResult(
        Pin<Box<dyn Future<Output = Result<PutResult, object_store::Error>> + Send>>,
    ),
    StorageWriterStreamDelete(
        Pin<Box<dyn Stream<Item = Result<Path, object_store::Error>> + Send>>,
    ),
    Done,
}

pub struct ObjectStoreStream {
    /// Output schema
    schema: SchemaRef,
    /// The messages containing the lhs and rhs
    /// which we cannot determine until we intialize the config
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    runtime_env: Arc<RuntimeEnv>,
    /// Store
    store: Option<Arc<dyn ObjectStore>>,
    /// Path
    path: Option<Path>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<ObjectStoreConfig>,
    /// State of the OpenAI API request
    state: ObjectStoreState,
    /// The polled record batches from the input
    /// Can be manifests files to get or subjects to put
    record_batches: Option<VecDeque<Map<String, Value>>>,
    /// The location to write the data or location to read the data from
    locations: Option<VecDeque<String>>,
    /// The metadata for the current object
    meta: Option<ObjectMeta>,
}

impl ObjectStoreStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::ObjectStore.to_schema(),
            messages,
            diagnostic_builder,
            config_stream,
            runtime_env,
            store: None,
            path: None,
            config: None,
            state: ObjectStoreState::NotStarted,
            record_batches: None,
            locations: None,
            meta: None,
        })
    }
}

impl Stream for ObjectStoreStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Iterate through each state (depending upon the opts type) until the API request is completed
        match &mut self.state {
            ObjectStoreState::NotStarted => {
                // Initialize the config
                if self.config.is_none() {
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    let config_table = SubjectBuilder::new()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    if config_table
                        .get_schema()
                        .fields()
                        .contains(&create_values_fields())
                    {
                        let config_json = config_table.get_column_as_vec_str("values").join("");
                        let config = serde_json::from_str::<ObjectStoreConfig>(&config_json)?;
                        self.config.replace(config);
                    } else if config_table
                        .get_schema()
                        .fields()
                        .contains(&create_bytes_fields())
                    {
                        let config_json = config_table
                            .get_column_as_vec_nested_primitive::<u8>("bytes")?
                            .into_iter()
                            .map(|b| String::from_utf8(b).unwrap())
                            .collect::<Vec<_>>()
                            .join("");
                        let config = serde_json::from_str::<ObjectStoreConfig>(&config_json)?;
                        self.config.replace(config);
                    } else {
                        let config = ObjectStoreConfig::from_table(&config_table)?;
                        self.config.replace(config);
                    }
                }

                // Collect the request data
                if self.record_batches.is_none()
                    && self.locations.is_none()
                    && let Some(subject_name) = self.config.as_ref().unwrap().subject_name.clone()
                {
                    match remove_message_by_subject(&subject_name, &mut self.messages) {
                        // Poll the next batches in a streaming fashion
                        Some(mut fut) => {
                            if let Some(Ok(batch)) =
                                ready!(fut.get_message_mut().poll_next_unpin(cx))
                            {
                                let json_object = SubjectBuilder::default()
                                    .with_name("")
                                    .with_record_batches(vec![batch])?
                                    .build()?
                                    .to_json_object()?;
                                self.record_batches.replace(json_object.into());
                            }
                            self.messages.insert(fut.get_name().to_string(), fut);
                        }
                        // Extract the data from the config
                        None => {
                            if let Some(location) = self.config.as_mut().unwrap().locations.take() {
                                self.locations.replace(location.into());
                            } else {
                                self.state = ObjectStoreState::Done;
                                return Poll::Ready(Some(Err(anyhow!(
                                    "Subject `{subject_name}` was not found in the messages. The available message subjects are `{:?}`",
                                    self.messages.keys()
                                ))));
                            }
                        }
                    }
                } else if self.record_batches.is_none()
                    && self.locations.is_none()
                    && let Some(location) = self.config.as_mut().unwrap().locations.take()
                {
                    // Extract the data from the config
                    self.locations.replace(location.into());
                }

                // The poll ends when there are no more batches
                if self.record_batches.is_none() && self.locations.is_none() {
                    self.state = ObjectStoreState::Done;
                    return Poll::Ready(None);
                }

                // Create the object store or use the runtime environment
                if self.store.is_none()
                    && self.config.as_ref().unwrap().backend == ObjectStorageBackend::InMemory
                {
                    let store = self.runtime_env.object_store().clone();
                    self.store.replace(store);
                } else if self.store.is_none() {
                    let bucket = self.config.as_ref().unwrap().bucket.clone();
                    let config = if let Some(config_str) =
                        self.config.as_ref().unwrap().backend_config.as_ref()
                    {
                        let config_map = serde_json::from_str::<Map<String, Value>>(config_str)?;
                        Some(config_map)
                    } else {
                        None
                    };
                    let store = make_store(
                        &self.config.as_ref().unwrap().backend,
                        bucket.as_ref(),
                        config.as_ref(),
                    )?;
                    self.store.replace(store);
                }

                // Get the location prioritizing the subject messages over the config
                // Note: Delete supports bulk operations while all others support single path/row operations
                let locations = if self.config.as_ref().unwrap().ops_type
                    == ObjectStoreOptsType::Delete
                {
                    if let Some(batch) = self.record_batches.take() {
                        let locations_result: Result<Vec<Path>> = batch.into_iter()
                            .map(|row| {
                                let location = row.get("location").ok_or(anyhow!("Missing column `location` in RecordBatch for ObjectStoreStream."))?;
                                let location = location.as_str().ok_or(anyhow!("Value for key location `{location}` could not be parsed as a String for ObjectSToreStream."))?.to_string();
                                Ok(Path::from(location))
                            })
                            .collect();
                        match locations_result {
                            Ok(locations) => locations,
                            Err(err) => {
                                self.state = ObjectStoreState::Done;
                                return Poll::Ready(Some(Err(anyhow!(err))));
                            }
                        }
                    } else if let Some(locations) = self.locations.take() {
                        locations.into_iter().map(Path::from).collect::<Vec<_>>()
                    } else {
                        self.state = ObjectStoreState::Done;
                        let err = "Location not provided for ObjectStoreStream.";
                        return Poll::Ready(Some(Err(anyhow!(err))));
                    }
                } else {
                    let location = if let Some(batch) = self.record_batches.take() {
                        let location = if let Some(row) = batch.front() {
                            let location = row.get("location").ok_or(anyhow!(
                                "Missing column `location` in RecordBatch for ObjectStoreStream."
                            ))?;
                            location.as_str().ok_or(anyhow!("Value for key location `{location}` could not be parsed as a String for ObjectSToreStream."))?.to_string()
                        } else {
                            self.state = ObjectStoreState::Done;
                            let err =
                                "Locations from RecordBatches is empty for ObjectStoreStream.";
                            return Poll::Ready(Some(Err(anyhow!(err))));
                        };
                        self.record_batches.replace(batch);
                        location
                    } else if let Some(mut locations) = self.locations.take() {
                        if let Some(location) = locations.pop_front() {
                            if !locations.is_empty() {
                                self.locations.replace(locations);
                            }
                            location
                        } else {
                            self.state = ObjectStoreState::Done;
                            let err = "Locations from Config is empty for ObjectStoreStream.";
                            return Poll::Ready(Some(Err(anyhow!(err))));
                        }
                    } else {
                        self.state = ObjectStoreState::Done;
                        let err = "Location not provided for ObjectStoreStream.";
                        return Poll::Ready(Some(Err(anyhow!(err))));
                    };

                    // Save the path for subsequent polling
                    let path = Path::from(location);
                    self.path.replace(path.clone());
                    vec![path]
                };

                // Determine the opts type
                match self.config.as_ref().unwrap().ops_type {
                    ObjectStoreOptsType::Get
                    | ObjectStoreOptsType::GetStream
                    | ObjectStoreOptsType::GetMeta => {
                        // Check if there are more batches for the next round
                        if let Some(mut batch) = self.record_batches.take() {
                            let _ = batch.pop_front();
                            if !batch.is_empty() {
                                self.record_batches.replace(batch);
                            }
                        }

                        // Get operation
                        let store = self.store.as_ref().unwrap().clone();
                        let path = self.path.as_ref().unwrap().clone();

                        // Add in any addition `GetOptions`
                        if let Some(_options) = self.config.as_ref().unwrap().get_options.as_ref() {
                            // let options = serde_json::from_str::<GetOptions>(options)?;
                            let fut = Box::pin(async move {
                                store.get_opts(&path, GetOptions::default()).await
                            });
                            self.state = ObjectStoreState::StorageReaderGetResult(fut);
                            self.poll_next(cx)
                        } else {
                            // DM: the `async move` trick to capture the lifetime only works for futures
                            let fut = Box::pin(async move { store.get(&path).await });
                            self.state = ObjectStoreState::StorageReaderGetResult(fut);
                            self.poll_next(cx)
                        }
                    }
                    ObjectStoreOptsType::List => {
                        // Check if there are more batches for the next round
                        if let Some(mut batch) = self.record_batches.take() {
                            let _ = batch.pop_front();
                            if !batch.is_empty() {
                                self.record_batches.replace(batch);
                            }
                        }

                        // List operation
                        let store = self.store.as_ref().unwrap().clone();
                        let path = if self.path.as_ref().unwrap().is_root() {
                            None
                        } else {
                            Some(self.path.as_ref().unwrap().clone())
                        };
                        let stream = store.list(path.as_ref());
                        self.state = ObjectStoreState::StorageReaderStreamList(stream);
                        self.poll_next(cx)
                    }
                    ObjectStoreOptsType::PutMultipart => {
                        let store = self.store.as_ref().unwrap().clone();
                        let path = self.path.as_ref().unwrap().clone();
                        let fut = Box::pin(async move { store.put_multipart(&path).await });
                        self.state = ObjectStoreState::StorageWriterMultipart(fut);
                        self.poll_next(cx)
                    }
                    ObjectStoreOptsType::Put => {
                        // Pop front the next batch of data
                        let row =
                            self.record_batches
                                .as_mut()
                                .unwrap()
                                .pop_front()
                                .ok_or(anyhow!(
                                    "Missing rows for RecordBatch for ObjectStoreStream."
                                ))?;
                        let bytes = row.get("bytes").ok_or(anyhow!("Missing column `bytes` for RecordBatch for ObjectStoreStream."))?
                            .as_array().ok_or(anyhow!("Value for key `bytes` could not be parsed as an array for ObjectStoreStream."))?
                            .iter()
                            .map(|v| v.as_u64().unwrap_or_default() as u8)
                            .collect::<Vec<_>>();
                        let location = row.get("location")
                            .ok_or(anyhow!("Missing column `location` in RecordBatch for ObjectStoreStream."))?
                            .as_str()
                            .ok_or(anyhow!("Value for key `location` could not be parsed as a String for ObjectSToreStream."))?
                            .to_string();
                        let meta = ObjectMeta {
                            e_tag: None,
                            version: None,
                            location: Path::from(location),
                            last_modified: Utc::now(),
                            size: bytes.len() as u64,
                        };
                        self.meta.replace(meta);
                        let payload = PutPayload::from_bytes(Bytes::from(bytes));

                        // Check if there are more batches for the next round
                        if let Some(batch) = self.record_batches.take()
                            && !batch.is_empty()
                        {
                            self.record_batches.replace(batch);
                        }

                        // Put operation
                        let store = self.store.as_ref().unwrap().clone();
                        let path = self.path.as_ref().unwrap().clone();

                        // Add in additional `PutOptions`
                        if let Some(_options) = self.config.as_ref().unwrap().put_options.as_ref() {
                            // let options = serde_json::from_str::<PutOptions>(options)?;
                            let fut = Box::pin(async move {
                                store.put_opts(&path, payload, PutOptions::default()).await
                            });
                            self.state = ObjectStoreState::StorageWriterPutResult(fut);
                            self.poll_next(cx)
                        } else {
                            let fut = Box::pin(async move { store.put(&path, payload).await });
                            self.state = ObjectStoreState::StorageWriterPutResult(fut);
                            self.poll_next(cx)
                        }
                    }
                    ObjectStoreOptsType::Delete => {
                        // Convert locations to futures
                        let paths = futures::stream::iter(locations.into_iter().map(Ok)).boxed();

                        // Delete operation
                        let store = self.store.as_ref().unwrap().clone();
                        let stream = store.delete_stream(paths);
                        self.state = ObjectStoreState::StorageWriterStreamDelete(stream);
                        self.poll_next(cx)
                    }
                    _ => {
                        self.state = ObjectStoreState::Done;
                        let err = format!(
                            "Object store operation `{}` is not yet supported.",
                            self.config.as_ref().unwrap().ops_type
                        );
                        Poll::Ready(Some(Err(anyhow!(err))))
                    }
                }
            }
            ObjectStoreState::StorageReaderGetResult(fut) => {
                match ready!(fut.as_mut().poll_unpin(cx)) {
                    Ok(result) => {
                        match self.config.as_ref().unwrap().ops_type {
                            ObjectStoreOptsType::Get => {
                                // Extract out the metadata
                                self.meta.replace(result.meta.clone());

                                // Ready the stream for polling
                                let fut = Box::pin(result.bytes());
                                self.state = ObjectStoreState::StorageReaderBytesResult(fut);
                                self.poll_next(cx)
                            }
                            ObjectStoreOptsType::GetStream => {
                                // Extract out the metadata
                                self.meta.replace(result.meta.clone());

                                // Ready the stream for polling
                                let stream = Box::pin(result.into_stream());
                                self.state = ObjectStoreState::StorageReaderStreamResult(stream);
                                self.poll_next(cx)
                            }
                            ObjectStoreOptsType::GetMeta => {
                                // Initialize the metrics
                                let baseline_metrics =
                                    if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                        Some(
                                            diagnostic_builder
                                                .clone()
                                                .to_child("ObjectStoreStream")?
                                                .baseline_metrics(
                                                    line!(),
                                                    file!(),
                                                    "poll_next.ObjectStoreState::ToBytes",
                                                ),
                                        )
                                    } else {
                                        None
                                    };
                                let _timer = baseline_metrics.as_ref().map(|baseline_metrics| {
                                    baseline_metrics.elapsed_compute().timer()
                                });

                                // Make the object store meta batch
                                let location =
                                    vec![self.meta.as_ref().unwrap().location.to_string()];
                                let bucket = vec![
                                    self.config
                                        .as_ref()
                                        .unwrap()
                                        .bucket
                                        .clone()
                                        .unwrap_or_default(),
                                ];
                                let last_modified = vec![
                                    self.meta.as_ref().unwrap().last_modified.timestamp_micros(),
                                ];
                                let size = vec![self.meta.as_ref().unwrap().size as u32];
                                let version = vec![
                                    self.meta
                                        .as_ref()
                                        .unwrap()
                                        .version
                                        .clone()
                                        .unwrap_or_default(),
                                ];
                                let e_tag = vec![
                                    self.meta
                                        .as_ref()
                                        .unwrap()
                                        .e_tag
                                        .clone()
                                        .unwrap_or_default(),
                                ];
                                let batch = create_object_store_meta_batch(
                                    location,
                                    bucket,
                                    e_tag,
                                    version,
                                    size,
                                    last_modified,
                                )?;

                                // Record the poll
                                self.state = ObjectStoreState::NotStarted;
                                self.schema = AvailableSubjects::ObjectStoreMeta.to_schema();
                                let poll = Poll::Ready(Some(Ok(batch)));
                                if let Some(baseline_metrics) = &baseline_metrics {
                                    baseline_metrics.record_poll(poll)
                                } else {
                                    poll
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    Err(err) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                }
            }
            ObjectStoreState::StorageReaderBytesResult(fut) => {
                match ready!(fut.as_mut().poll_unpin(cx)) {
                    Ok(bytes) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("ObjectStoreStream")?
                                        .baseline_metrics(
                                            line!(),
                                            file!(),
                                            "poll_next.ObjectStoreState::ToBytes",
                                        ),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Make the object store batch
                        let location = vec![self.meta.as_ref().unwrap().location.to_string()];
                        let bucket = vec![
                            self.config
                                .as_ref()
                                .unwrap()
                                .bucket
                                .clone()
                                .unwrap_or_default(),
                        ];
                        let last_modified =
                            vec![self.meta.as_ref().unwrap().last_modified.timestamp_micros()];
                        let metadata_json = json!({"size": self.meta.as_ref().unwrap().size, "version": self.meta.as_ref().unwrap().version.clone().unwrap_or_default(), "e_tag": self.meta.as_ref().unwrap().e_tag.clone().unwrap_or_default()});
                        let metadata = vec![serde_json::to_string(&metadata_json)?];
                        let bytes = vec![bytes.to_vec()];
                        let batch = create_object_store_batch(
                            location,
                            bucket,
                            metadata,
                            last_modified,
                            bytes,
                        )?;

                        // Record the poll
                        self.state = ObjectStoreState::NotStarted;
                        self.schema = AvailableSubjects::ObjectStore.to_schema();
                        let poll = Poll::Ready(Some(Ok(batch)));
                        if let Some(baseline_metrics) = &baseline_metrics {
                            baseline_metrics.record_poll(poll)
                        } else {
                            poll
                        }
                    }
                    Err(err) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                }
            }
            ObjectStoreState::StorageReaderStreamResult(stream) => {
                match ready!(stream.as_mut().poll_next_unpin(cx)) {
                    Some(Ok(bytes)) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("ObjectStoreStream")?
                                        .baseline_metrics(
                                            line!(),
                                            file!(),
                                            "poll_next.ObjectStoreState::ToBytes",
                                        ),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Make the object store batch
                        let location = vec![self.meta.as_ref().unwrap().location.to_string()];
                        let bucket = vec![
                            self.config
                                .as_ref()
                                .unwrap()
                                .bucket
                                .clone()
                                .unwrap_or_default(),
                        ];
                        let last_modified =
                            vec![self.meta.as_ref().unwrap().last_modified.timestamp_micros()];
                        let metadata_json = json!({"size": self.meta.as_ref().unwrap().size, "version": self.meta.as_ref().unwrap().version.clone().unwrap_or_default(), "e_tag": self.meta.as_ref().unwrap().e_tag.clone().unwrap_or_default()});
                        let metadata = vec![serde_json::to_string(&metadata_json)?];
                        let bytes = vec![bytes.to_vec()];
                        let batch = create_object_store_batch(
                            location,
                            bucket,
                            metadata,
                            last_modified,
                            bytes,
                        )?;

                        // Record the poll
                        self.schema = AvailableSubjects::ObjectStore.to_schema();
                        let poll = Poll::Ready(Some(Ok(batch)));
                        if let Some(baseline_metrics) = &baseline_metrics {
                            baseline_metrics.record_poll(poll)
                        } else {
                            poll
                        }
                    }
                    Some(Err(err)) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                    None => {
                        self.state = ObjectStoreState::NotStarted;
                        self.poll_next(cx)
                    }
                }
            }
            ObjectStoreState::StorageReaderStreamList(stream) => {
                match ready!(stream.as_mut().poll_next_unpin(cx)) {
                    Some(Ok(meta)) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("ObjectStoreStream")?
                                        .baseline_metrics(
                                            line!(),
                                            file!(),
                                            "poll_next.ObjectStoreState::ToBytes",
                                        ),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Make the object store metadata
                        let location = vec![meta.location.to_string()];
                        let bucket = vec![
                            self.config
                                .as_ref()
                                .unwrap()
                                .bucket
                                .clone()
                                .unwrap_or_default(),
                        ];
                        let last_modified = vec![meta.last_modified.timestamp_micros()];
                        let size = vec![meta.size as u32];
                        let version = vec![meta.version.unwrap_or_default()];
                        let e_tag = vec![meta.e_tag.unwrap_or_default()];
                        let batch = create_object_store_meta_batch(
                            location,
                            bucket,
                            e_tag,
                            version,
                            size,
                            last_modified,
                        )?;

                        // Record the poll
                        self.schema = AvailableSubjects::ObjectStoreMeta.to_schema();
                        let poll = Poll::Ready(Some(Ok(batch)));
                        if let Some(baseline_metrics) = &baseline_metrics {
                            baseline_metrics.record_poll(poll)
                        } else {
                            poll
                        }
                    }
                    Some(Err(err)) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                    None => {
                        self.state = ObjectStoreState::NotStarted;
                        self.poll_next(cx)
                    }
                }
            }
            ObjectStoreState::StorageWriterMultipart(fut) => {
                match ready!(fut.as_mut().poll_unpin(cx)) {
                    Ok(mp) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("ObjectStoreStream")?
                                        .baseline_metrics(
                                            line!(),
                                            file!(),
                                            "poll_next.ObjectStoreState::ToBytes",
                                        ),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Write the bytes to the buffer
                        let row = self.record_batches.as_mut().unwrap().pop_front().unwrap();
                        let bytes = row.get("bytes").ok_or(anyhow!("Missing column `bytes` for RecordBatch for ObjectStoreStream."))?
                        .as_array().ok_or(anyhow!("Value for key `bytes` could not be parsed as an array for ObjectStoreStream."))?
                        .iter()
                        .map(|v| v.as_u64().unwrap_or_default() as u8)
                        .collect::<Vec<_>>();
                        let location = row.get("location")
                        .ok_or(anyhow!("Missing column `location` in RecordBatch for ObjectStoreStream."))?
                        .as_str()
                        .ok_or(anyhow!("Value for key `location` could not be parsed as a String for ObjectSToreStream."))?
                        .to_string();
                        let pending = Arc::new(Mutex::new(VecDeque::new()));
                        let on_chunk = OnChunk::new(&pending);
                        let chunk_size = self.config.as_mut().unwrap().chunk_size.ok_or(
                            anyhow!("Missing `chunk_size` for multipart put in ObjectStoreStream."),
                        )?;
                        let mut chunk_writer = ChunkedWriter::new(chunk_size, on_chunk);
                        let size = chunk_writer.write(&bytes)?;
                        let meta = ObjectMeta {
                            e_tag: None,
                            version: None,
                            location: Path::from(location),
                            last_modified: Utc::now(),
                            size: size as u64,
                        };
                        self.meta.replace(meta);

                        // Check if there are more batches for the next round
                        if let Some(batch) = self.record_batches.take()
                            && !batch.is_empty()
                        {
                            self.record_batches.replace(batch);
                        }

                        // Put the chunks
                        let mut write = WriteMultipart::new(mp);
                        while let Some(chunk) = pending.lock().pop_front() {
                            write.write(&chunk);
                        }
                        let fut = write.finish();
                        self.state = ObjectStoreState::StorageWriterPutResult(Box::pin(fut));
                        self.poll_next(cx)
                    }
                    Err(err) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                }
            }
            ObjectStoreState::StorageWriterPutResult(fut) => {
                match ready!(fut.as_mut().poll_unpin(cx)) {
                    Ok(results) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("ObjectStoreStream")?
                                        .baseline_metrics(
                                            line!(),
                                            file!(),
                                            "poll_next.ObjectStoreState::ToBytes",
                                        ),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Make the object store meta batch
                        let location = vec![self.meta.as_ref().unwrap().location.to_string()];
                        let bucket = vec![
                            self.config
                                .as_ref()
                                .unwrap()
                                .bucket
                                .clone()
                                .unwrap_or_default(),
                        ];
                        let last_modified =
                            vec![self.meta.as_ref().unwrap().last_modified.timestamp_micros()];
                        let size = vec![self.meta.as_ref().unwrap().size as u32];
                        let version = vec![results.version.unwrap_or_default()];
                        let e_tag = vec![results.e_tag.unwrap_or_default()];
                        let batch = create_object_store_meta_batch(
                            location,
                            bucket,
                            e_tag,
                            version,
                            size,
                            last_modified,
                        )?;

                        // Record the poll
                        self.state = ObjectStoreState::NotStarted;
                        self.schema = AvailableSubjects::ObjectStoreMeta.to_schema();
                        let poll = Poll::Ready(Some(Ok(batch)));
                        if let Some(baseline_metrics) = &baseline_metrics {
                            baseline_metrics.record_poll(poll)
                        } else {
                            poll
                        }
                    }
                    Err(err) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                }
            }
            ObjectStoreState::StorageWriterStreamDelete(stream) => {
                match ready!(stream.as_mut().poll_next_unpin(cx)) {
                    Some(Ok(location)) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("ObjectStoreStream")?
                                        .baseline_metrics(
                                            line!(),
                                            file!(),
                                            "poll_next.ObjectStoreState::ToBytes",
                                        ),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Make the object store metadata
                        let location = vec![location.to_string()];
                        let bucket = vec![
                            self.config
                                .as_ref()
                                .unwrap()
                                .bucket
                                .clone()
                                .unwrap_or_default(),
                        ];
                        let last_modified = vec![create_timestamp_micros()];
                        let size = vec![0];
                        let version = vec![String::new()];
                        let e_tag = vec![String::new()];
                        let batch = create_object_store_meta_batch(
                            location,
                            bucket,
                            e_tag,
                            version,
                            size,
                            last_modified,
                        )?;

                        // Record the poll
                        self.schema = AvailableSubjects::ObjectStoreMeta.to_schema();
                        let poll = Poll::Ready(Some(Ok(batch)));
                        if let Some(baseline_metrics) = &baseline_metrics {
                            baseline_metrics.record_poll(poll)
                        } else {
                            poll
                        }
                    }
                    Some(Err(err)) => {
                        self.state = ObjectStoreState::Done;
                        Poll::Ready(Some(Err(err.into())))
                    }
                    None => {
                        self.state = ObjectStoreState::NotStarted;
                        self.poll_next(cx)
                    }
                }
            }
            ObjectStoreState::Done => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for ObjectStoreStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
