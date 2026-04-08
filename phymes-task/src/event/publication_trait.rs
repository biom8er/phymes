use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field, Schema},
};
use futures::StreamExt;
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, ObjectStorageBackend,
    RecordBatchStreamAdapter, RuntimeEnv, SendableRecordBatchStream, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_event::Publication;
use phymes_schemas::{AvailableSchemaTrait, AvailableSubjects, DataEncoding, DataFormat};
use phymes_message::{MessageBuilderTrait, SendableRecordBatchStreamMessage, make_random_id};
use phymes_data::{AvailableOperators,  DataColumnOperator, DataConfig, DataJoinOperator, DataStreamManager};
use phymes_streams::{CandleDataStream, ObjectStoreConfig, ObjectStoreOptsType, ObjectStoreStream};
use phymes_diagnostics::HashMap;

use crate::list_subject;

/// Generate the object store path
///
/// # Todo
/// * Handle more complex partitioning schemes
pub fn make_object_store_path(
    session_name: &str,
    subject_name: &str,
    step: u32,
    publisher: &str,
    partition: u32,
) -> String {
    let hash = make_random_id().unwrap();
    format!(
        "session={session_name}/subject={subject_name}/superstep={step}/publisher={publisher}/partition={partition}/{subject_name}-{hash}.ipc"
    )
}

/// Generate the a vector of record batches of locations to put the subject
pub fn make_object_store_paths_record_batch(
    session_name: &str,
    subject_name: &str,
    step: u32,
    publisher: &str,
    n_batches: u32,
) -> Vec<RecordBatch> {
    (0..n_batches)
        .map(|i| {
            let location = make_object_store_path(session_name, subject_name, step, publisher, i);
            let pk: Vec<u32> = vec![0];
            let location: ArrayRef = Arc::new(StringArray::from(vec![location]));
            let pk: ArrayRef = Arc::new(UInt32Array::from(pk));
            RecordBatch::try_from_iter(vec![("location_updated", location), ("pk", pk)]).unwrap()
        })
        .collect::<Vec<_>>()
}

/// Pipeline stream to `extend` the subject in object storage
pub fn extend_subject(
    runtime_env: &Arc<RuntimeEnv>,
    session_name: &str,
    sn: &str,
    new: Vec<RecordBatch>,
    step: u32,
    publisher: &str,
) -> Result<SendableRecordBatchStream> {
    // 1. Create the locations RecordBatches
    let ln = "locations";
    let locations = Subject::get_builder()
        .with_name(ln)
        .with_record_batches(make_object_store_paths_record_batch(
            session_name,
            sn,
            step,
            publisher,
            new.len() as u32,
        ))?
        .build()?
        .to_record_batch_stream();

    // 2. Pack the tabular data
    let config = DataConfig {
        lhs_name: Some(sn.to_string()),
        encoding: Some(DataEncoding::default()),
        format: Some(DataFormat::Ipc),
        schema: Some(AvailableSubjects::ObjectStore),
        doc_name: Some(sn.to_string()),
        cpu: false,
        operator: AvailableOperators::PackTabular,
        lhs_stream: DataStreamManager::Stream,
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("DataConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(
                Subject::get_builder()
                    .with_name(sn)
                    .with_record_batches(new)?
                    .build()?
                    .to_record_batch_stream(),
            )
            .build()?,
    );
    let stream = Box::pin(CandleDataStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);

    // 3. Add PK to subject stream (and drop all other columns that are not needed)
    let config = DataConfig {
        lhs_name: Some(sn.to_string()),
        lhs_values: Some(vec![
            "location".to_string(),
            "bytes".to_string(),
            "pk".to_string(),
        ]),
        rhs_values: None,
        as_columns: Some(vec![
            "location".to_string(),
            "bytes".to_string(),
            "pk".to_string(),
        ]),
        column_operators: Some(vec![
            DataColumnOperator::None,
            DataColumnOperator::None,
            DataColumnOperator::Zeros,
        ]),
        cast_operators: None,
        cast_datatypes: Some(vec![
            DataType::Utf8.to_string(),
            "List-UInt8".to_string(),
            DataType::UInt32.to_string(),
        ]),
        cast_templates: None,
        cpu: false,
        operator: AvailableOperators::Select,
        lhs_stream: DataStreamManager::Stream,
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("DataConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(stream)
            .build()?,
    );
    let stream = Box::pin(CandleDataStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);

    // 4. Join on locations stream
    let config = DataConfig {
        lhs_name: Some(sn.to_string()),
        rhs_name: Some(ln.to_string()),
        lhs_pk: Some("pk".to_string()),
        rhs_pk: Some("pk".to_string()),
        lhs_fk: Some("pk".to_string()),
        rhs_fk: Some("pk".to_string()),
        cpu: false,
        operator: AvailableOperators::Join,
        join_operators: Some(DataJoinOperator::Inner),
        lhs_stream: DataStreamManager::Stream,
        rhs_stream: Some(DataStreamManager::Stream),
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("DataConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(stream)
            .build()?,
    );
    let _ = message.insert(
        ln.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(ln)
            .with_publisher("")
            .with_subject(ln)
            .with_update(&Publication::None)
            .with_message(locations)
            .build()?,
    );
    let stream = Box::pin(CandleDataStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);

    // 5. Replace the locations column
    let config = DataConfig {
        lhs_name: Some(sn.to_string()),
        lhs_values: Some(vec!["location_updated".to_string(), "bytes".to_string()]),
        as_columns: Some(vec!["location".to_string(), "bytes".to_string()]),
        cpu: false,
        operator: AvailableOperators::Select,
        lhs_stream: DataStreamManager::Stream,
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("DataConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(stream)
            .build()?,
    );
    let stream = Box::pin(CandleDataStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);

    // 6. Put into the object store
    let config = ObjectStoreConfig {
        timeout: 5,
        ops_type: ObjectStoreOptsType::Put,
        backend: ObjectStorageBackend::InMemory, // Force use of the runtime_env
        locations: None,
        subject_name: Some(sn.to_string()),
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("ObjectStoreConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(stream)
            .build()?,
    );
    let stream = Box::pin(ObjectStoreStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);
    Ok(stream)
}

/// Pipeline stream to `clear` the subject data from object storage (optionally limiting to jsut the last partition)
pub fn clear_subject(
    runtime_env: &Arc<RuntimeEnv>,
    session_name: &str,
    sn: &str,
    last: bool,
) -> Result<SendableRecordBatchStream> {
    // 1. List the locations
    let stream = list_subject(runtime_env, session_name, sn, last)?;

    // 2. Delete the partitions at the locations listed
    let config = ObjectStoreConfig {
        timeout: 5,
        ops_type: ObjectStoreOptsType::Delete,
        backend: ObjectStorageBackend::InMemory, // Force use of the runtime_env
        locations: None,
        subject_name: Some(sn.to_string()),
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("ObjectStoreConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(stream)
            .build()?,
    );
    let stream = Box::pin(ObjectStoreStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);
    Ok(stream)
}

/// Update an subject with record batches coming from a new table
pub trait PublicationTrait {
    fn publish_to_subject(
        &self,
        runtime_env: &Arc<RuntimeEnv>,
        new: Vec<RecordBatch>,
        step: u32,
        publisher: &str,
        session_name: &str,
    ) -> Result<Option<SendableRecordBatchStream>>;
}

impl PublicationTrait for Publication {
    fn publish_to_subject(
        &self,
        runtime_env: &Arc<RuntimeEnv>,
        new: Vec<RecordBatch>,
        step: u32,
        publisher: &str,
        session_name: &str,
    ) -> Result<Option<SendableRecordBatchStream>> {
        match self {
            Self::Extend { subject_name: sn } => {
                let stream = extend_subject(runtime_env, session_name, sn, new, step, publisher)?;
                Ok(Some(stream))
            }
            Publication::ExtendChunks {
                subject_name: sn,
                col_name: cn,
            } => {
                let chunks = new
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(cn.as_str())
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let new_first_row = create_record_batch_from_first_row(
                    new.first().unwrap(),
                    cn.as_str(),
                    chunks.as_str(),
                )?;
                let stream = extend_subject(
                    runtime_env,
                    session_name,
                    sn,
                    vec![new_first_row],
                    step,
                    publisher,
                )?;
                Ok(Some(stream))
            }
            Publication::ExtendBytes {
                subject_name: sn,
                col_name: cn,
                serialize_format: sf,
            } => {
                let new_batches_res: Result<Vec<Vec<RecordBatch>>> = new.into_iter()
                    .map(|batch| {
                        let new_table = SubjectBuilder::default()
                            .with_name("ExtendBytes")
                            .with_record_batches(vec![batch])?
                            .build()?;
                        match sf {
                            DataFormat::Ipc => {
                                let bytes = new_table.get_column_as_vec_nested_primitive::<u8>(cn)?
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>();
                                let batches = SubjectBuilder::new_from_ipc_stream(&bytes)?
                                    .with_name("ExtendBytesIpc")
                                    .build()?
                                    .get_record_batches_own();
                                Ok(batches)
                            }
                            DataFormat::Bytes => {
                                let bytes = new_table.get_column_as_vec_nested_primitive::<u8>(cn)?
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>();
                                let schema = AvailableSubjects::Bytes.to_schema();
                                let batches = SubjectBuilder::new()
                                    .with_schema(schema)
                                    .with_name("ExtendBytesBytes")
                                    .with_bytes(&bytes)?
                                    .build()?
                                    .get_record_batches_own();
                                Ok(batches)
                            }
                            _ => Err(anyhow!(
                                "Serialization format {sf} for table name {} and update table target {sn} is not supported.",
                                self.get_name(),
                            ))
                        }
                    })
                    .collect();
                let new_batches = new_batches_res?.into_iter().flatten().collect::<Vec<_>>();
                let stream =
                    extend_subject(runtime_env, session_name, sn, new_batches, step, publisher)?;
                Ok(Some(stream))
            }
            Publication::Replace { subject_name: sn } => {
                // Delete all RecordBatches
                let clear = clear_subject(runtime_env, session_name, sn, false)?;

                // Extend the subject with the new record batches
                let stream = extend_subject(runtime_env, session_name, sn, new, step, publisher)?;
                let stream = Box::pin(RecordBatchStreamAdapter::new(
                    Arc::clone(&stream.schema()),
                    clear.chain(stream),
                ));
                Ok(Some(stream))
            }
            Publication::ReplaceLast { subject_name: sn } => {
                // Delete the last RecordBatch
                let clear = clear_subject(runtime_env, session_name, sn, true)?;

                // Extend the subject with the new record batches
                let stream = extend_subject(runtime_env, session_name, sn, new, step, publisher)?;
                let stream = Box::pin(RecordBatchStreamAdapter::new(
                    Arc::clone(&stream.schema()),
                    clear.chain(stream),
                ));
                Ok(Some(stream))
            }
            Publication::None => Ok(None),
            Publication::Custom(_) => Ok(None),
        }
    }
}

fn get_first_row(batch: &RecordBatch) -> Result<Vec<String>> {
    let mut first_row = Vec::new();
    for column in batch.columns() {
        let value = match column.data_type() {
            DataType::Utf8 => {
                let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                array.value(0).to_string()
            }
            DataType::UInt8 => {
                let array = column.as_any().downcast_ref::<UInt8Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::UInt16 => {
                let array = column.as_any().downcast_ref::<Int16Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::UInt32 => {
                let array = column.as_any().downcast_ref::<UInt32Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::UInt64 => {
                let array = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Int8 => {
                let array = column.as_any().downcast_ref::<Int8Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Int16 => {
                let array = column.as_any().downcast_ref::<Int16Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Int32 => {
                let array = column.as_any().downcast_ref::<Int32Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Int64 => {
                let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Float32 => {
                let array = column.as_any().downcast_ref::<Float32Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Float64 => {
                let array = column.as_any().downcast_ref::<Float64Array>().unwrap();
                array.value(0).to_string()
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported data type {} for array.",
                    column.data_type()
                ));
            }
        };
        first_row.push(value);
    }
    Ok(first_row)
}

/// Create a new record batch from the first row BUT replace the streamed chunks row
fn create_record_batch_from_first_row(
    batch: &RecordBatch,
    name: &str,
    new_content: &str,
) -> Result<RecordBatch> {
    let first_row = get_first_row(batch)?;
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut fields: Vec<Field> = Vec::new();

    for (i, column) in batch.columns().iter().enumerate() {
        let field = batch.schema().field(i).clone();
        fields.push(field.clone());

        let array: ArrayRef = match column.data_type() {
            DataType::Utf8 => {
                if field.name().eq(name) {
                    let values = vec![new_content];
                    Arc::new(StringArray::from(values)) as ArrayRef
                } else {
                    let values = vec![first_row[i].clone()];
                    Arc::new(StringArray::from(values)) as ArrayRef
                }
            }
            DataType::Int8 => {
                let values = vec![first_row[i].parse::<i8>().unwrap()];
                Arc::new(Int8Array::from(values)) as ArrayRef
            }
            DataType::Int16 => {
                let values = vec![first_row[i].parse::<i16>().unwrap()];
                Arc::new(Int16Array::from(values)) as ArrayRef
            }
            DataType::Int32 => {
                let values = vec![first_row[i].parse::<i32>().unwrap()];
                Arc::new(Int32Array::from(values)) as ArrayRef
            }
            DataType::Int64 => {
                let values = vec![first_row[i].parse::<i64>().unwrap()];
                Arc::new(Int64Array::from(values)) as ArrayRef
            }
            DataType::UInt8 => {
                let values = vec![first_row[i].parse::<u8>().unwrap()];
                Arc::new(UInt8Array::from(values)) as ArrayRef
            }
            DataType::UInt16 => {
                let values = vec![first_row[i].parse::<u16>().unwrap()];
                Arc::new(UInt16Array::from(values)) as ArrayRef
            }
            DataType::UInt32 => {
                let values = vec![first_row[i].parse::<u32>().unwrap()];
                Arc::new(UInt32Array::from(values)) as ArrayRef
            }
            DataType::UInt64 => {
                let values = vec![first_row[i].parse::<u64>().unwrap()];
                Arc::new(UInt64Array::from(values)) as ArrayRef
            }
            DataType::Float32 => {
                let values = vec![first_row[i].parse::<f32>().unwrap()];
                Arc::new(Float32Array::from(values)) as ArrayRef
            }
            DataType::Float64 => {
                let values = vec![first_row[i].parse::<f64>().unwrap()];
                Arc::new(Float64Array::from(values)) as ArrayRef
            }
            _ => return Err(anyhow!("Unsupported type")),
        };
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Schema;

    use futures::TryStreamExt;
    use phymes_core::test_subject;
    use phymes_event::Subscription;
    use phymes_schemas::create_bytes_record_batch;

    use crate::SubscriptionTrait;

    use super::*;

    #[test]
    fn test_create_record_batch_from_first_row_string() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "column1",
            DataType::Utf8,
            false,
        )]));
        let array = Arc::new(StringArray::from(vec!["value1", "value2"])) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let new_batch = create_record_batch_from_first_row(&batch, "column1", "new")?;
        let new_array = new_batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(new_array.value(0), "new");
        Ok(())
    }

    #[test]
    fn test_create_record_batch_from_first_row_int32() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "column1",
            DataType::Int32,
            false,
        )]));
        let array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let new_batch = create_record_batch_from_first_row(&batch, "column1", "new")?;
        let new_array = new_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(new_array.value(0), 1);
        Ok(())
    }

    #[test]
    fn test_create_record_batch_from_first_row_float32() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "column1",
            DataType::Float32,
            false,
        )]));
        let array = Arc::new(Float32Array::from(vec![1.1, 2.2])) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let new_batch = create_record_batch_from_first_row(&batch, "column1", "new")?;
        let new_array = new_batch
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(new_array.value(0), 1.1);
        Ok(())
    }

    #[tokio::test]
    async fn test_table_publication_extend_update() -> Result<()> {
        let subject_name = "test_table";
        let old = test_subject::make_test_subject(subject_name, 4, 0, 3)?;
        let runtime_env = Arc::new(RuntimeEnv::default());
        let _publication: Vec<RecordBatch> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, old.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let new = test_subject::make_test_subject(subject_name, 1, 0, 1)?;
        let _publication: Vec<RecordBatch> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, new.get_record_batches_own(), 1, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 13);
        Ok(())
    }

    #[tokio::test]
    async fn test_table_publication_replace_update() -> Result<()> {
        let subject_name = "test_table";
        let old = test_subject::make_test_subject(subject_name, 4, 0, 3)?;
        let runtime_env = Arc::new(RuntimeEnv::default());
        let _publication: Vec<RecordBatch> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, old.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let new = test_subject::make_test_subject(subject_name, 1, 0, 1)?;
        let _publication: Vec<RecordBatch> = Publication::Replace {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, new.get_record_batches_own(), 1, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_table_publication_none_update() -> Result<()> {
        let subject_name = "test_table";
        let old = test_subject::make_test_subject(subject_name, 4, 0, 3)?;
        let runtime_env = Arc::new(RuntimeEnv::default());
        let _publication: Vec<RecordBatch> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, old.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let new = test_subject::make_test_subject(subject_name, 1, 0, 1)?;
        let publication = Publication::None.publish_to_subject(
            &runtime_env,
            new.get_record_batches_own(),
            1,
            "",
            "",
        )?;
        assert!(publication.is_none());
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 12);
        Ok(())
    }

    #[tokio::test]
    async fn test_table_publication_extend_chunks_update() -> Result<()> {
        let subject_name = "messages";
        let old = test_subject::make_test_subject_chat(subject_name)?;
        let runtime_env = Arc::new(RuntimeEnv::default());
        let _publication: Vec<RecordBatch> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, old.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;

        // Example streamed chunks
        let role_1: ArrayRef = Arc::new(StringArray::from(vec![
            "assistant".to_string(),
            "assistant".to_string(),
        ]));
        let role_2: ArrayRef = Arc::new(StringArray::from(vec![
            "assistant".to_string(),
            "assistant".to_string(),
        ]));
        let content_1: ArrayRef =
            Arc::new(StringArray::from(vec!["0".to_string(), "1".to_string()]));
        let content_2: ArrayRef =
            Arc::new(StringArray::from(vec!["2".to_string(), "3".to_string()]));
        let timestamap_1: ArrayRef = Arc::new(Int64Array::from(vec![0, 0]));
        let timestamap_2: ArrayRef = Arc::new(Int64Array::from(vec![0, 0]));
        let new_1 = RecordBatch::try_from_iter(vec![
            ("role", role_1),
            ("content", content_1),
            ("timestamp", timestamap_1),
        ])?;
        let new_2 = RecordBatch::try_from_iter(vec![
            ("role", role_2),
            ("content", content_2),
            ("timestamp", timestamap_2),
        ])?;

        let _publication: Vec<RecordBatch> = Publication::ExtendChunks {
            subject_name: subject_name.to_string(),
            col_name: "content".to_string(),
        }
        .publish_to_subject(&runtime_env, vec![new_1, new_2], 1, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.count_rows(), 5);
        assert_eq!(
            subject.get_column_as_vec_str("role"),
            ["user", "assistant", "user", "assistant", "assistant"]
        );
        assert_eq!(
            subject.get_column_as_vec_str("content"),
            [
                "Hi!",
                "Hello how can I help?",
                "What is Deep Learning?",
                "magic!",
                "0123"
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_table_publication_extend_bytes_update() -> Result<()> {
        let subject_name = "test_table";
        let old = test_subject::make_test_subject(subject_name, 1, 0, 1)?;
        let runtime_env = Arc::new(RuntimeEnv::default());
        let _publication: Vec<RecordBatch> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, old.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;

        // IPC format
        let new = vec![
            create_bytes_record_batch(vec![
                test_subject::make_test_subject("test_table", 2, 0, 2)?.to_ipc_stream()?,
            ])?,
            create_bytes_record_batch(vec![
                test_subject::make_test_subject("test_table", 2, 0, 2)?.to_ipc_stream()?,
            ])?,
        ];
        let _publication: Vec<RecordBatch> = Publication::ExtendBytes {
            subject_name: "test_table".to_string(),
            col_name: "bytes".to_string(),
            serialize_format: DataFormat::Ipc,
        }
        .publish_to_subject(&runtime_env, new, 1, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;

        assert_eq!(subject.count_rows(), 9);
        assert_eq!(subject.get_record_batches().len(), 5);
        Ok(())
    }
}
