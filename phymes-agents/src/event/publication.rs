use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field, Schema},
};
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, DataEncoding, DataFormat, MappableTrait, MessageBuilderTrait, ObjectStorageBackend, Publication, RuntimeEnv, SendableRecordBatchStreamMessage, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait
};
use phymes_data::{AvailableCandleOperators, CandleDataStream, DataConfig, DataStreamManager, ObjectStoreConfig, ObjectStoreOptsType, ObjectStoreStream};
use phymes_diagnostics::HashMap;

/// Generate the object store path
/// 
/// # Todo
/// * Handle more complex partitioning schemes
fn make_object_store_path(subject_name: &str, step: u32, partition: u32) -> String {
    format!("{subject_name}/superstep={step}/partition={partition}/{subject_name}.ipc")
}

/// Update an subject with record batches coming from a new table
pub trait TablePublicationTrait {
    fn publish_to_subject(&self, runtime_env: &Arc<RuntimeEnv>, new: Vec<RecordBatch>, step: u32) -> Result<()>;
}

impl TablePublicationTrait for Publication {
    fn publish_to_subject(&self, runtime_env: &Arc<RuntimeEnv>, new: Vec<RecordBatch>, step: u32) -> Result<()> {
        match self {
            Self::Extend { subject_name: sn } => {
                // 1. Create the locations column
                let locations = (0..new.len()).map(|i| make_object_store_path(sn, step, i as u32)).collect::<Vec<_>>();

                // 2. Pack the tabular data
                let config = DataConfig {
                    lhs_name: Some(sn.to_string()),
                    encoding: Some(DataEncoding::default()),
                    format: Some(DataFormat::Ipc),
                    schema: Some(AvailableSubjects::ObjectStore),
                    doc_name: Some(sn.to_string()),
                    cpu: false,
                    operator: AvailableCandleOperators::PackTabular,
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
                        .with_message(Subject::get_builder()
                            .with_name(sn)
                            .with_record_batches(new)?
                            .build()?
                            .to_record_batch_stream())
                        .build()?,
                ); 
                let stream = Box::pin(CandleDataStream::new(
                    message,
                    config_table.to_record_batch_stream(),
                    Arc::clone(&runtime_env),
                    None
                )?);

                // 3. Replace the locations column
                // DM: new operator?

                // 4. Put into the object store
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
                    Arc::clone(&runtime_env),
                    None
                )?);
                Ok(())
            }
            Publication::ExtendChunks {
                subject_name: sn,
                col_name: cn,
            } => {
                if self.get_name() != sn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        sn
                    ));
                }
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
                self.get_record_batches_mut().push(new_first_row);
                Ok(())
            }
            Publication::ExtendBytes {
                subject_name: sn,
                col_name: cn,
                serialize_format: sf,
            } => {
                if self.get_name() != sn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {sn}.",
                        self.get_name(),
                    ));
                }
                let new_batches_res: Result<Vec<Vec<RecordBatch>>> = new.into_iter()
                    .map(|batch| {
                        let new_table = SubjectBuilder::default()
                            .with_name("ExtendBytes")
                            .with_record_batches(vec![batch])?
                            .build()?;
                        match sf {
                            DataFormat::Ipc => {
                                let bytes = new_table.get_column_as_vec_nested_primitive::<u8>(&cn)?
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
                                let bytes = new_table.get_column_as_vec_nested_primitive::<u8>(&cn)?
                                    .into_iter()
                                    .flatten()
                                    .collect::<Vec<_>>();
                                let batches = SubjectBuilder::new()
                                    .with_schema(self.get_schema())
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
                if !self.get_schema().eq(&new_batches.first().unwrap().schema()) {
                    Err(anyhow!(
                        "Mismatch between schema {:?} and batches {:?} when attempting to update table {}.",
                        self.get_schema(),
                        &new_batches.first().unwrap(),
                        self.get_name()
                    ))
                } else {
                    self.get_record_batches_mut().extend(new_batches);
                    Ok(())
                }
            }
            Publication::Replace { subject_name: sn } => {
                if self.get_name() != sn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        sn
                    ));
                }
                for batch in new.iter() {
                    if !self.get_schema().eq(&batch.schema()) {
                        return Err(anyhow!(
                            "Mismatch between schema {:?} and batches {:?} when attempting to update table {}.",
                            self.get_schema(),
                            &batch.schema(),
                            self.get_name()
                        ));
                    }
                }
                self.get_record_batches_mut().clear();
                self.get_record_batches_mut().extend(new);
                Ok(())
            }
            Publication::ReplaceLast { subject_name: sn } => {
                if self.get_name() != sn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        sn
                    ));
                }
                let last = new.last().unwrap();
                if !self.get_schema().eq(&last.schema()) {
                    return Err(anyhow!(
                        "Mismatch between schema {:?} and batches {:?} when attempting to update table {}.",
                        self.get_schema(),
                        &last.schema(),
                        self.get_name()
                    ));
                }
                self.get_record_batches_mut().last().replace(last);
                Ok(())
            }
            Publication::None => Ok(()),
            Publication::Custom(_) => Ok(()),
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

    use phymes_core::{create_bytes_record_batch, test_subject};

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

    #[test]
    fn test_table_publication_table_wrong_table_name() -> Result<()> {
        let mut old = test_subject::make_test_subject("test_table", 4, 0, 3)?;
        let new = test_subject::make_test_subject("test_table", 1, 0, 1)?;
        match old.publish_to_table(
            new.clone().get_record_batches_own(),
            Publication::Extend {
                subject_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        match old.publish_to_table(
            new.clone().get_record_batches_own(),
            Publication::Replace {
                subject_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        match old.publish_to_table(
            new.clone().get_record_batches_own(),
            Publication::ReplaceLast {
                subject_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        match old.publish_to_table(
            new.clone().get_record_batches_own(),
            Publication::ExtendChunks {
                subject_name: "missing".to_string(),
                col_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        Ok(())
    }

    #[test]
    fn test_table_publication_extend_update() -> Result<()> {
        let mut old = test_subject::make_test_subject("test_table", 4, 0, 3)?;
        let new = test_subject::make_test_subject("test_table", 1, 0, 1)?;
        old.publish_to_table(
            new.get_record_batches_own(),
            Publication::Extend {
                subject_name: "test_table".to_string(),
            },
        )?;
        assert_eq!(old.count_rows(), 13);
        Ok(())
    }

    #[test]
    fn test_table_publication_replace_update() -> Result<()> {
        let mut old = test_subject::make_test_subject("test_table", 4, 0, 3)?;
        let new = test_subject::make_test_subject("test_table", 1, 0, 1)?;
        old.publish_to_table(
            new.get_record_batches_own(),
            Publication::Replace {
                subject_name: "test_table".to_string(),
            },
        )?;
        assert_eq!(old.count_rows(), 1);
        Ok(())
    }

    #[test]
    fn test_table_publication_none_update() -> Result<()> {
        let mut old = test_subject::make_test_subject("test_table", 4, 0, 3)?;
        let new = test_subject::make_test_subject("test_table", 1, 0, 1)?;
        old.publish_to_table(new.get_record_batches_own(), Publication::None)?;
        assert_eq!(old.count_rows(), 12);
        Ok(())
    }

    #[test]
    fn test_table_publication_extend_chunks_update() -> Result<()> {
        let mut old = test_subject::make_test_subject_chat("messages")?;
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
        let new_1 = RecordBatch::try_from_iter(vec![("role", role_1), ("content", content_1)])?;
        let new_2 = RecordBatch::try_from_iter(vec![("role", role_2), ("content", content_2)])?;
        old.publish_to_table(
            vec![new_1, new_2],
            Publication::ExtendChunks {
                subject_name: "messages".to_string(),
                col_name: "content".to_string(),
            },
        )?;
        assert_eq!(old.count_rows(), 5);
        assert_eq!(
            old.get_column_as_vec_str("role"),
            ["user", "assistant", "user", "assistant", "assistant"]
        );
        assert_eq!(
            old.get_column_as_vec_str("content"),
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

    #[test]
    fn test_table_publication_extend_bytes_update() -> Result<()> {
        // IPC format
        let mut old = test_subject::make_test_subject("test_table", 1, 0, 1)?;
        let new = vec![
            create_bytes_record_batch(vec![
                test_subject::make_test_subject("test_table", 2, 0, 2)?.to_ipc_stream()?,
            ])?,
            create_bytes_record_batch(vec![
                test_subject::make_test_subject("test_table", 2, 0, 2)?.to_ipc_stream()?,
            ])?,
        ];

        old.publish_to_table(
            new,
            Publication::ExtendBytes {
                subject_name: "test_table".to_string(),
                col_name: "bytes".to_string(),
                serialize_format: DataFormat::Ipc,
            },
        )?;
        assert_eq!(old.count_rows(), 9);
        assert_eq!(old.get_record_batches().len(), 5);

        // Bytes format
        let mut old = test_subject::make_test_subject("test_table", 1, 0, 1)?;
        let new = vec![
            create_bytes_record_batch(vec![
                test_subject::make_test_subject("test_table", 2, 0, 2)?.to_bytes()?.to_vec(),
            ])?,
            create_bytes_record_batch(vec![
                test_subject::make_test_subject("test_table", 2, 0, 2)?.to_bytes()?.to_vec(),
            ])?,
        ];

        old.publish_to_table(
            new,
            Publication::ExtendBytes {
                subject_name: "test_table".to_string(),
                col_name: "bytes".to_string(),
                serialize_format: DataFormat::Bytes,
            },
        )?;
        assert_eq!(old.count_rows(), 9);
        assert_eq!(old.get_record_batches().len(), 3);
        Ok(())
    }
}
