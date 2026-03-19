use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field, Schema},
};
use serde::{Deserialize, Serialize};

use crate::{
    BuilderTrait, DataFormat, SubjectBuilder, SubjectBuilderTrait, MappableTrait, Subject, SubjectTrait
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum Publication {
    /// Push a new vector of record batches onto the table
    Extend { subject_name: String },
    /// Push a new vector of record batches onto the table
    /// after joining the chunks along the named column
    ExtendChunks {
        subject_name: String,
        col_name: String,
    },
    /// Push a new vector of record batches onto the table
    /// after deserializing bytes from a specified format
    /// DM: intended for internal routing of messages
    ExtendBytes {
        subject_name: String,
        col_name: String,
        serialize_format: DataFormat,
    },
    /// Replace the existing vector of record batches with a new one
    Replace { subject_name: String },
    /// Replace only the last record batch
    ReplaceLast { subject_name: String },
    /// No updates
    #[default]
    None,
    /// Custom update function
    Custom(String),
}

impl Publication {
    /// Short name for the [Publication] that omits the `subject_name` and other information
    pub fn short_name(&self) -> &str {
        match self {
            Self::Extend { subject_name: _tn } => "Extend",
            Self::ExtendChunks {
                subject_name: _tn,
                col_name: _cn,
            } => "ExtendChunks",
            Self::ExtendBytes {
                subject_name: _tn,
                col_name: _cn,
                serialize_format: _sf,
            } => "ExtendBytes",
            Self::Replace { subject_name: _tn } => "Replace",
            Self::ReplaceLast { subject_name: _tn } => "ReplaceLast",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }

    /// Full name for the [Publication] that includes the `subject_name` and other information
    pub fn full_name(&self) -> String {
        match self {
            Self::Extend { subject_name: tn } => format!("extend-{tn}"),
            Self::ExtendChunks {
                subject_name: tn,
                col_name: cn,
            } => format!("extend-chunks-{tn}-{cn}"),
            Self::ExtendBytes {
                subject_name: tn,
                col_name: cn,
                serialize_format: sf,
            } => format!("extend-values-{tn}-{cn}-{sf}"),
            Self::Replace { subject_name: tn } => format!("replace-{tn}"),
            Self::ReplaceLast { subject_name: tn } => format!("replace-last-{tn}"),
            Self::None => "none".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    /// The `subject_name` of the variant
    pub fn subject_name(&self) -> &str {
        match self {
            Self::Extend { subject_name: tn } => tn,
            Self::ExtendChunks {
                subject_name: tn,
                col_name: _cn,
            } => tn,
            Self::ExtendBytes {
                subject_name: tn,
                col_name: _cn,
                serialize_format: _sf,
            } => tn,
            Self::Replace { subject_name: tn } => tn,
            Self::ReplaceLast { subject_name: tn } => tn,
            Self::None => "",
            Self::Custom(_name) => "",
        }
    }

    /// New [Publication] from a short name identifying the variant and the `subject_name`
    pub fn from_str_fuzzy(name: &str, subject: &str) -> Result<Publication> {
        let publication = if name.contains("ExtendChunks") {
            Publication::ExtendChunks {
                subject_name: subject.to_string(),
                col_name: "content".to_string(),
            }
        } else if name.contains("Extend") {
            Publication::Extend {
                subject_name: subject.to_string(),
            }
        } else if name.contains("ReplaceLast") {
            Publication::ReplaceLast {
                subject_name: subject.to_string(),
            }
        } else if name.contains("Replace") {
            Publication::Replace {
                subject_name: subject.to_string(),
            }
        } else if name.contains("None") {
            Publication::None {}
        } else {
            return Err(anyhow!(
                "Variant for ArrowTablePublish {name} with subject {subject} was not recognized."
            ));
        };
        Ok(publication)
    }

    /// New [Publication] from a short name identifying the variant, the subject `subject_name`
    ///   and the mermaid.js flowchart diagram link type
    pub fn from_str_mermaid(line: &str, subject: &str) -> Result<Publication> {
        if line.contains("|") & line.contains("-->") & line.contains("ExtendChunks") {
            Ok(Publication::ExtendChunks {
                subject_name: subject.to_string(),
                col_name: "content".to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("Extend") {
            Ok(Publication::Extend {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("ReplaceLast") {
            Ok(Publication::ReplaceLast {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("Replace") {
            Ok(Publication::Replace {
                subject_name: subject.to_string(),
            })
        } else if line.contains("None") {
            Ok(Publication::None {})
        } else {
            Err(anyhow!(
                "Variant for Publication with subject {subject} was not recognized in string slice {line}."
            ))
        }
    }
}

impl Display for Publication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extend { subject_name: _ } => write!(f, "Extend"),
            Self::Replace { subject_name: _ } => write!(f, "Replace"),
            Self::ReplaceLast { subject_name: _ } => write!(f, "ReplaceLast"),
            Self::None => write!(f, "None"),
            Self::ExtendChunks {
                subject_name: _,
                col_name: _,
            } => write!(f, "ExtendChunks"),
            Self::ExtendBytes {
                subject_name: _,
                col_name: _,
                serialize_format: _,
            } => write!(f, "ExtendBytes"),
            Self::Custom(_s) => write!(f, "Custom"),
        }
    }
}

impl MappableTrait for Publication {
    fn get_name(&self) -> &str {
        self.short_name()
    }
}

/// Update an arrow table with record batches coming from a new table
pub trait TablePublicationTrait: SubjectTrait {
    fn publish_to_table(&mut self, new: Vec<RecordBatch>, update: Publication) -> Result<()>;
}

impl TablePublicationTrait for Subject {
    fn publish_to_table(&mut self, new: Vec<RecordBatch>, update: Publication) -> Result<()> {
        match update {
            Publication::Extend { subject_name: tn } => {
                if self.get_name() != tn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        tn
                    ));
                }
                for batch in new.into_iter() {
                    if !self.get_schema().eq(&batch.schema()) {
                        return Err(anyhow!(
                            "Mismatch between schema {:?} and batches {:?} when attempting to update table {}.",
                            self.get_schema(),
                            &batch.schema(),
                            self.get_name()
                        ));
                    } else {
                        self.get_record_batches_mut().push(batch);
                    }
                }
                Ok(())
            }
            Publication::ExtendChunks {
                subject_name: tn,
                col_name: cn,
            } => {
                if self.get_name() != tn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        tn
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
                subject_name: tn,
                col_name: cn,
                serialize_format: sf,
            } => {
                if self.get_name() != tn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {tn}.",
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
                                "Serialization format {sf} for table name {} and update table target {tn} is not supported.",
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
            Publication::Replace { subject_name: tn } => {
                if self.get_name() != tn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        tn
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
            Publication::ReplaceLast { subject_name: tn } => {
                if self.get_name() != tn {
                    return Err(anyhow!(
                        "Mismatch between table name {} and update table target {}.",
                        self.get_name(),
                        tn
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

    use crate::{create_bytes_record_batch, test_subject::{make_test_subject, make_test_subject_chat}};

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
        let mut old = make_test_subject("test_table", 4, 0, 3)?;
        let new = make_test_subject("test_table", 1, 0, 1)?;
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
        let mut old = make_test_subject("test_table", 4, 0, 3)?;
        let new = make_test_subject("test_table", 1, 0, 1)?;
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
        let mut old = make_test_subject("test_table", 4, 0, 3)?;
        let new = make_test_subject("test_table", 1, 0, 1)?;
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
        let mut old = make_test_subject("test_table", 4, 0, 3)?;
        let new = make_test_subject("test_table", 1, 0, 1)?;
        old.publish_to_table(new.get_record_batches_own(), Publication::None)?;
        assert_eq!(old.count_rows(), 12);
        Ok(())
    }

    #[test]
    fn test_table_publication_extend_chunks_update() -> Result<()> {
        let mut old = make_test_subject_chat("messages")?;
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
        let mut old = make_test_subject("test_table", 1, 0, 1)?;
        let new = vec![
            create_bytes_record_batch(vec![
                make_test_subject("test_table", 2, 0, 2)?.to_ipc_stream()?,
            ])?,
            create_bytes_record_batch(vec![
                make_test_subject("test_table", 2, 0, 2)?.to_ipc_stream()?,
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
        let mut old = make_test_subject("test_table", 1, 0, 1)?;
        let new = vec![
            create_bytes_record_batch(vec![
                make_test_subject("test_table", 2, 0, 2)?.to_bytes()?.to_vec(),
            ])?,
            create_bytes_record_batch(vec![
                make_test_subject("test_table", 2, 0, 2)?.to_bytes()?.to_vec(),
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

    #[test]
    fn test_table_publication_from_str_mermaid() -> Result<()> {
        let line = "message_parser-publish-->|ExtendChunks|AssistantMessages-subject";
        let subject = "AssistantMessages";
        let publication = Publication::ExtendChunks {
            subject_name: subject.to_string(),
            col_name: "content".to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parser-publish-->|Extend|AssistantMessages-subject";
        let publication = Publication::Extend {
            subject_name: subject.to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parser-publish-->|ReplaceLast|AssistantMessages-subject";
        let publication = Publication::ReplaceLast {
            subject_name: subject.to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parser-publish-->|Replace|AssistantMessages-subject";
        let publication = Publication::Replace {
            subject_name: subject.to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        Ok(())
    }
}
