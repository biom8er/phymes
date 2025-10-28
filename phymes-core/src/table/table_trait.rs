use crate::session::{BuildableTrait, BuilderTrait, MappableTrait};

use super::{
    stream::{SendableIPCRecordBatchStream, SendableRecordBatchStream},
    stream_adapter::RecordBatchStreamAdapter,
};

use arrow::json::reader::infer_json_schema;
use arrow::json::{ArrayWriter, LineDelimitedWriter, ReaderBuilder};
use arrow::{
    array::StringBuilder,
    compute::cast,
    ipc::{
        reader::{FileReader, StreamReader},
        writer::{FileWriter, StreamWriter},
    },
};
use arrow::{
    array::{
        Array,
        ArrayRef,
        BooleanArray, //Float16Array,
        Float32Array,
        Float64Array,
        StringArray,
        UInt8Array,
        UInt16Array,
        UInt32Array,
        UInt64Array,
    },
    csv::{WriterBuilder, reader::Format},
    datatypes::DataType,
};
use arrow::{
    array::{ArrayData, Float32Builder, ListBuilder},
    buffer::Buffer,
    datatypes::{Field, Schema},
};
use arrow::{
    array::{
        FixedSizeListArray, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray,
        ListArray,
    },
    compute::{concat_batches, kernels::concat},
};
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use num_traits::{Bounded, Num, NumCast};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::io::{Cursor, Seek};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use futures::TryStreamExt;
use serde_json::{Map, Value};
use tracing::{Level, event, instrument};

/// Traits for a columnar table where all [RecordBatch]es are guaranteed to have the same [Schema]
pub trait TableTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
    fn get_schema(&self) -> SchemaRef;
    fn get_record_batches(&self) -> &Vec<RecordBatch>;
    fn get_record_batches_own(self) -> Vec<RecordBatch>;

    /// Write record batches to IPC file
    #[instrument(level = "trace")]
    fn to_ipc_file(&self, file: &mut File) -> Result<()> {
        if self.get_record_batches().is_empty() {
            return Err(anyhow!(
                "Cannot write empty record batches to IPC file since they cannot be read back in."
            ));
        }
        let mut writer = FileWriter::try_new(file, &self.get_schema().clone()).unwrap();
        for batch in self.get_record_batches() {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        Ok(())
    }

    /// Write record batches to CSV
    fn to_csv_file(&self, file: &mut File, delimiter: u8, header: bool) -> Result<()> {
        let builder = WriterBuilder::new()
            .with_header(header)
            .with_delimiter(delimiter)
            .with_quote(b'\'')
            .with_null("NULL".to_string())
            .with_time_format("%r".to_string());
        let mut writer = builder.build(file);
        for batch in self.get_record_batches() {
            writer.write(batch).unwrap();
        }
        drop(writer);
        Ok(())
    }

    /// Write record batches to CSV
    fn to_csv(&self, delimiter: u8, header: bool) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let builder = WriterBuilder::new()
            .with_header(header)
            .with_delimiter(delimiter)
            .with_quote(b'\'')
            .with_null("NULL".to_string())
            .with_time_format("%r".to_string());
        let mut writer = builder.build(&mut bytes);
        for batch in self.get_record_batches() {
            writer.write(batch).unwrap();
        }
        let data = writer.into_inner().to_vec();
        Ok(data)
    }

    /// Write record batches to IPC stream
    fn to_ipc_stream(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let mut writer =
            StreamWriter::try_new(Cursor::new(&mut bytes), &self.get_schema().clone())?;
        for batch in self.get_record_batches() {
            writer.write(batch)?;
        }
        writer.finish().unwrap();
        drop(writer);
        Ok(bytes)
    }

    /// Write record batches to JSON
    #[instrument(level = "trace")]
    fn to_json(&self) -> Result<Vec<u8>> {
        let buf = Vec::new();
        let mut writer = LineDelimitedWriter::new(buf);
        for batch in self.get_record_batches() {
            writer.write(batch)?;
        }
        writer.finish().unwrap();
        let json_data = writer.into_inner();
        Ok(json_data)
    }

    /// Write record batches to JSON
    fn to_json_object(&self) -> Result<Vec<Map<String, Value>>> {
        let buf = Vec::new();
        let mut writer = ArrayWriter::new(buf);
        for batch in self.get_record_batches() {
            writer.write(batch)?;
        }
        writer.finish()?;
        let json_data = writer.into_inner();
        let json_rows: Vec<Map<String, Value>> = serde_json::from_reader(json_data.as_slice())?;
        Ok(json_rows)
    }

    /// Convert to a sendable record batch stream
    fn to_record_batch_stream(&self) -> SendableRecordBatchStream {
        let stream = futures::stream::iter(self.get_record_batches().clone().into_iter().map(Ok));
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.get_schema()),
            stream,
        ))
    }

    /// Convert to a sendable record batch stream
    fn to_record_batch_stream_last_record_batch(&self) -> SendableRecordBatchStream {
        let last_record_batch = if let Some(batch) = self.get_record_batches().last() {
            vec![batch.clone()]
        } else {
            Vec::new()
        };
        let stream = futures::stream::iter(last_record_batch.into_iter().map(Ok));
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.get_schema()),
            stream,
        ))
    }

    /// Convert to a byte stream
    fn to_bytes(&self) -> Result<Bytes> {
        let object = self.to_json_object()?;
        let content = serde_json::to_string(&object)?;
        let buf = Bytes::from(content);
        Ok(buf)
    }

    /// Convert to a vector of structs
    fn to_struct<T>(&self) -> Result<Vec<T>>
    where
        T: Sized + for<'a> Deserialize<'a>,
    {
        let buf = Vec::new();
        let mut writer = ArrayWriter::new(buf);
        for batch in self.get_record_batches() {
            writer.write(batch)?;
        }
        writer.finish()?;
        let json_data = writer.into_inner();
        let content = match serde_json::from_reader::<_, Vec<T>>(json_data.as_slice()) {
            Ok(content) => content,
            Err(err) => return Err(anyhow!("{err}")),
        };
        Ok(content)
    }

    /// Count the number of rows
    fn count_rows(&self) -> usize {
        self.get_record_batches()
            .iter()
            .map(|batches| batches.num_rows())
            .sum::<usize>()
    }

    /// Get a column as a vector of strings
    fn get_column_as_vec_str(&self, column_name: &str) -> Vec<&str> {
        self.get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name(column_name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    /// Get a column as a vector of strings
    fn get_column_as_vec_string(&self, column_name: &str) -> Result<Option<Vec<String>>> {
        match self.get_column_data_type(column_name)? {
            DataType::Utf8 => {
                let vec_str = self
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column_name)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default().to_string())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(Some(vec_str))
            }
            DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Boolean
            | DataType::Null => {
                // Cast the column to a String
                let arr = cast(&self.get_column_as_array(column_name), &DataType::Utf8)?;
                let vec_str = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default().to_string())
                    .collect::<Vec<_>>();
                Ok(Some(vec_str))
            }
            DataType::List(_) | DataType::FixedSizeList(_, _) => {
                // Convert the column to JSON
                let arr = self.get_column_as_array(column_name);
                let batch = RecordBatch::try_from_iter(vec![(column_name, arr)])?;
                let buf = Vec::new();
                let mut writer = ArrayWriter::new(buf);
                writer.write(&batch)?;
                writer.finish()?;
                let json_data = writer.into_inner();
                let json_rows: Vec<Map<String, Value>> =
                    serde_json::from_reader(json_data.as_slice())?;
                let vec_str = json_rows
                    .into_iter()
                    .map(|m| serde_json::to_string(m.get(column_name).unwrap()).unwrap())
                    .collect::<Vec<_>>();
                Ok(Some(vec_str))
            }
            _ => Ok(None),
        }
    }

    /// Get the type of the column
    fn get_column_data_type(&self, column_name: &str) -> Result<DataType> {
        let data_type = self
            .get_schema()
            .field_with_name(column_name)?
            .data_type()
            .clone();
        Ok(data_type)
    }

    /// Get a column as a vector of non-primitive types
    fn get_array_as_vec_nonprimitive<T>(arr: &Arc<dyn Array>, column_name: &str) -> Result<Vec<T>>
    where
        T: From<String> + 'static,
    {
        let data_type = arr.data_type();
        if data_type.is_primitive() {
            return Err(anyhow!("Column {column_name} is a primitive type"));
        }
        use std::any::TypeId;
        match data_type {
            DataType::Utf8 => {
                if TypeId::of::<T>() != TypeId::of::<String>() {
                    return Err(anyhow!(
                        "Expected String data type for column {column_name}"
                    ));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| T::from(s.unwrap_or_default().to_string()))
                    .collect::<Vec<T>>();
                Ok(arr_vec)
            }
            DataType::LargeUtf8 => {
                if TypeId::of::<T>() != TypeId::of::<String>() {
                    return Err(anyhow!("Expected Int16 data type for column {column_name}"));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| T::from(s.unwrap_or_default().to_string()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            _ => Err(anyhow!(
                "Unsupported data type {data_type} for column {column_name}"
            )),
        }
    }

    /// Get a column as a vector of primitive types
    fn get_array_as_vec_primitive<T>(arr: &Arc<dyn Array>, column_name: &str) -> Result<Vec<T>>
    where
        T: Num + Bounded + NumCast + Send + Sync + 'static,
    {
        let data_type = arr.data_type();
        if !data_type.is_primitive() {
            return Err(anyhow!("Column {column_name} is not a primitive type"));
        }
        use std::any::TypeId;
        match data_type {
            // DataType::Boolean => {
            //     if TypeId::of::<T>() != TypeId::of::<bool>() {
            //         return Err(anyhow!(
            //             "Expected bool data type for column {}",
            //             column_name
            //         ));
            //     }
            //     let arr_vec = arr
            //         .as_any()
            //         .downcast_ref::<BooleanArray>()
            //         .unwrap()
            //         .iter()
            //         .filter_map(|s| NumCast::from(s.unwrap_or_default() as bool))
            //         .collect::<Vec<_>>();
            //     Ok(arr_vec)
            // }
            DataType::Int8 => {
                if TypeId::of::<T>() != TypeId::of::<i8>() {
                    return Err(anyhow!("Expected Int8 data type for column {column_name}"));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::Int16 => {
                if TypeId::of::<T>() != TypeId::of::<i16>() {
                    return Err(anyhow!("Expected Int16 data type for column {column_name}"));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::Int32 => {
                if TypeId::of::<T>() != TypeId::of::<i32>() {
                    return Err(anyhow!("Expected Int32 data type for column {column_name}"));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::Int64 => {
                if TypeId::of::<T>() != TypeId::of::<i64>() {
                    return Err(anyhow!("Expected Int64 data type for column {column_name}"));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::UInt8 => {
                if TypeId::of::<T>() != TypeId::of::<u8>() {
                    return Err(anyhow!("Expected UInt8 data type for column {column_name}"));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::UInt16 => {
                if TypeId::of::<T>() != TypeId::of::<u16>() {
                    return Err(anyhow!(
                        "Expected UInt16 data type for column {column_name}"
                    ));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::UInt32 => {
                if TypeId::of::<T>() != TypeId::of::<u32>() {
                    return Err(anyhow!(
                        "Expected UInt32 data type for column {column_name}"
                    ));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::UInt64 => {
                if TypeId::of::<T>() != TypeId::of::<u64>() {
                    return Err(anyhow!(
                        "Expected UInt64 data type for column {column_name}"
                    ));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::Float32 => {
                if TypeId::of::<T>() != TypeId::of::<f32>() {
                    return Err(anyhow!(
                        "Expected Float32 data type for column {column_name}"
                    ));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::Float64 => {
                if TypeId::of::<T>() != TypeId::of::<f64>() {
                    return Err(anyhow!(
                        "Expected Float64 data type for column {column_name}"
                    ));
                }
                let arr_vec = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .filter_map(|s| NumCast::from(s.unwrap_or_default()))
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            _ => Err(anyhow!(
                "Unsupported data type {data_type} for column {column_name}"
            )),
        }
    }

    fn get_column_as_vec_nested_nonprimitive<T>(&self, column_name: &str) -> Result<Vec<Vec<T>>>
    where
        T: From<String> + 'static,
    {
        let data_type = self.get_column_data_type(column_name)?;
        if !data_type.is_nested() {
            return Err(anyhow!("Column {column_name} is not a nested type"));
        }
        match data_type {
            DataType::FixedSizeList(_field, _size) => {
                // DM: deal with each primitive data type
                let arr_vec = self
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column_name)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                Self::get_array_as_vec_nonprimitive::<T>(&s.unwrap(), column_name)
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::List(_field) => {
                // DM: deal with each primitive data type
                let arr_vec = self
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column_name)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                Self::get_array_as_vec_nonprimitive::<T>(&s.unwrap(), column_name)
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            _ => Err(anyhow!(
                "Unsupported data type {data_type} for column {column_name}"
            )),
        }
    }

    fn get_column_as_vec_nested_primitive<T>(&self, column_name: &str) -> Result<Vec<Vec<T>>>
    where
        T: Num + Bounded + NumCast + Send + Sync + 'static,
    {
        let data_type = self.get_column_data_type(column_name)?;
        if !data_type.is_nested() {
            return Err(anyhow!("Column {column_name} is not a nested type"));
        }
        match data_type {
            DataType::FixedSizeList(_field, _size) => {
                // DM: deal with each primitive data type
                let arr_vec = self
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column_name)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                Self::get_array_as_vec_primitive::<T>(&s.unwrap(), column_name)
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            DataType::List(_field) => {
                // DM: deal with each primitive data type
                let arr_vec = self
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column_name)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .unwrap()
                            .iter()
                            .map(|s| {
                                Self::get_array_as_vec_primitive::<T>(&s.unwrap(), column_name)
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(arr_vec)
            }
            _ => Err(anyhow!(
                "Unsupported data type {data_type} for column {column_name}"
            )),
        }
    }

    /// Get a column as a vector of primitive types
    fn get_column_as_vec_nonprimitive<T>(&self, column_name: &str) -> Result<Vec<T>>
    where
        T: From<String> + 'static,
    {
        let mut result = Vec::new();
        for batch in self.get_record_batches() {
            if let Some(array) = batch.column_by_name(column_name) {
                let vec = Self::get_array_as_vec_nonprimitive::<T>(&array.clone(), column_name)?;
                result.extend(vec);
            } else {
                return Err(anyhow!("Column {column_name} not found"));
            }
        }
        Ok(result)
    }

    /// Get a column as a vector of primitive types
    fn get_column_as_vec_primitive<T>(&self, column_name: &str) -> Result<Vec<T>>
    where
        T: Num + Bounded + NumCast + Send + Sync + 'static,
    {
        let mut result = Vec::new();
        for batch in self.get_record_batches() {
            if let Some(array) = batch.column_by_name(column_name) {
                let vec = Self::get_array_as_vec_primitive::<T>(&array.clone(), column_name)?;
                result.extend(vec);
            } else {
                return Err(anyhow!("Column {column_name} not found"));
            }
        }
        Ok(result)
    }

    /// Get a column as an arrow array
    fn get_column_as_array(&self, column_name: &str) -> Arc<dyn Array> {
        let array_refs = self
            .get_record_batches()
            .iter()
            .map(|batch| batch.column_by_name(column_name).unwrap().as_ref())
            .collect::<Vec<_>>();
        concat::concat(&array_refs).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    name: String,
    schema: SchemaRef,
    pub(crate) record_batches: Vec<RecordBatch>,
}

impl Default for Table {
    fn default() -> Self {
        Self {
            name: "".to_string(),
            schema: Arc::new(Schema::empty()),
            record_batches: Vec::new(),
        }
    }
}

impl Table {
    /// Concatenate multiple record batches into a single record batch
    pub fn concat_record_batches(mut self) -> Result<Self> {
        let concatenated = concat_batches(&self.schema, &self.record_batches)?;
        self.record_batches = vec![concatenated];
        Ok(self)
    }
}

impl MappableTrait for Table {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for Table {
    type T = TableBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl TableTrait for Table {
    fn get_schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn get_record_batches(&self) -> &Vec<RecordBatch> {
        &self.record_batches
    }

    fn get_record_batches_own(self) -> Vec<RecordBatch> {
        self.record_batches
    }
}

pub trait TableBuilderTrait: BuilderTrait + Debug + Send + Sync {
    /// The schema for all record batches in the table
    fn with_schema(self, schema: SchemaRef) -> Self;

    /// Add record batches
    fn with_record_batches(self, batches: Vec<RecordBatch>) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a IPC file
    fn new_from_ipc_file(file: &File) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a CSV file
    fn with_csv_file(
        self,
        file: &File,
        delimiter: u8,
        header: bool,
        batch_size: usize,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a IPC stream
    fn new_from_ipc_stream(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a JSON
    fn with_json(self, bytes: &[u8], batch_size: usize) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a CSV file
    fn with_csv(self, bytes: &[u8], delimiter: u8, header: bool, batch_size: usize) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a JSON array of values
    fn with_json_values(self, json_values: &[Value]) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a SendableRecordBatchStream
    #[allow(async_fn_in_trait)]
    async fn new_from_sendable_record_batch_stream(
        stream: SendableRecordBatchStream,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a SendableIPCRecordBatchStream
    #[allow(async_fn_in_trait)]
    async fn new_from_sendable_ipc_record_batch_stream(
        stream: SendableIPCRecordBatchStream,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream table with the provided batches
    /// from a vector of structs
    fn with_struct<T>(self, s: &[T]) -> Result<Self>
    where
        Self: Sized,
        T: Sized + Serialize;

    /// Create a new stream table with the provided batches
    /// from a JSON object in byte format
    fn with_bytes(self, bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct TableBuilder {
    pub name: Option<String>,
    pub schema: Option<SchemaRef>,
    pub record_batches: Option<Vec<RecordBatch>>,
}

impl BuilderTrait for TableBuilder {
    type T = Table;
    fn new() -> Self {
        Self {
            name: None,
            schema: None,
            record_batches: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        match (self.name, self.schema, self.record_batches) {
            (Some(name), Some(schema), Some(record_batches)) => Ok(Self::T {
                name,
                schema,
                record_batches,
            }),
            _ => Err(anyhow!(
                "Please define the name, schema, and record batches before trying to build!"
            )),
        }
    }
}

impl TableBuilderTrait for TableBuilder {
    fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    fn with_record_batches(mut self, batches: Vec<RecordBatch>) -> Result<Self> {
        // Handle the case of no schema
        if self.schema.is_none() {
            if let Some(batch) = batches.first() {
                self.schema = Some(batch.schema());
            } else {
                return Err(anyhow!("Missing schema and batches!"));
            }
        };

        // Check the batch schemas are consistent
        let schema = self.schema.clone().unwrap();
        for batch in batches.iter() {
            if !schema.eq(&batch.schema()) {
                return Err(anyhow!("Mismatch between schema and batches!"));
            }
        }
        self.record_batches = Some(batches);
        Ok(self)
    }

    #[instrument(level = "trace")]
    fn new_from_ipc_file(file: &File) -> Result<Self> {
        match FileReader::try_new(file, None) {
            Ok(mut reader) => {
                let mut record_batches = vec![];
                while let Some(Ok(read_batch)) = reader.next() {
                    record_batches.push(read_batch);
                }
                if record_batches.is_empty() {
                    return Err(anyhow!(
                        "Cannot make a new ArrowTable from IPC File with empty record batches."
                    ));
                }
                let schema = record_batches.first().unwrap().schema();
                Self::new()
                    .with_schema(schema.clone())
                    .with_record_batches(record_batches)
            }
            Err(e) => Err(anyhow!("Error trying to read IPC File {e:?}.")),
        }
    }

    fn with_csv_file(
        mut self,
        mut file: &File,
        delimiter: u8,
        header: bool,
        batch_size: usize,
    ) -> Result<Self> {
        // Handle the case of no schema
        let format = Format::default()
            .with_header(header)
            .with_delimiter(delimiter);
        let schema = match self.schema {
            Some(ref schema) => schema.clone(),
            None => {
                let (schema, _) = format.infer_schema(file, None)?;
                file.rewind().unwrap();
                let schema = Arc::new(schema);
                self.schema = Some(schema.clone());
                schema
            } // None => return Err(anyhow!("Please define the schema before adding record batches!"))
        };

        // Read in the file
        let mut csv = arrow::csv::ReaderBuilder::new(schema.clone())
            .with_batch_size(batch_size)
            .with_format(format)
            .build(file)?;
        let mut batches = Vec::new();
        while let Some(Ok(batch)) = csv.next() {
            batches.push(batch);
        }

        self.record_batches = Some(batches);
        Ok(self)
    }

    #[instrument(level = "trace")]
    fn new_from_ipc_stream(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        let mut reader = StreamReader::try_new(cursor, None)?;
        let mut record_batches = Vec::new();
        while let Some(Ok(read_batch)) = reader.next() {
            record_batches.push(read_batch);
        }
        let schema = match record_batches.first() {
            Some(batch) => batch.schema(),
            None => return Err(anyhow!("Failed to read the IPC stream.")),
        };
        Self::new()
            .with_schema(schema.clone())
            .with_record_batches(record_batches)
    }

    fn with_json(mut self, bytes: &[u8], batch_size: usize) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        // Infer the schema if not already provided
        let schema = match self.schema.as_ref() {
            Some(schema) => schema.clone(),
            None => {
                // Attempt to infer the schema
                let (schema, _) = infer_json_schema(&mut cursor, None)?;
                cursor.rewind().unwrap();

                // Change nullable = true to false
                let mut fields = Vec::new();
                for field in schema.fields() {
                    let data_type = match field.data_type() {
                        DataType::FixedSizeList(f, s) => DataType::FixedSizeList(
                            Arc::new(Field::new_list_field(f.data_type().clone(), false)),
                            *s,
                        ),
                        DataType::List(f) => DataType::List(Arc::new(Field::new_list_field(
                            f.data_type().clone(),
                            false,
                        ))),
                        _ => field.data_type().clone(),
                    };
                    fields.push(Field::new(field.name(), data_type, false));
                }

                // Make the Schema
                let schema = Arc::new(Schema::new(fields));
                self.schema = Some(schema.clone());
                schema
            }
        };

        // Read in the batches
        let mut reader = ReaderBuilder::new(schema)
            .with_batch_size(batch_size)
            .build(cursor)?;
        let mut record_batches = Vec::new();
        while let Some(Ok(read_batch)) = reader.next() {
            record_batches.push(read_batch);
        }
        self.record_batches = Some(record_batches);
        Ok(self)
    }

    fn with_csv(
        mut self,
        bytes: &[u8],
        delimiter: u8,
        header: bool,
        batch_size: usize,
    ) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        // Handle the case of no schema
        let format = Format::default()
            .with_header(header)
            .with_delimiter(delimiter);
        let schema = match self.schema {
            Some(ref schema) => schema.clone(),
            None => {
                // Attempt to infer the schema
                let (schema, _) = format.infer_schema(&mut cursor, None)?;
                cursor.rewind().unwrap();

                // Change nullable = true to false
                let mut fields = Vec::new();
                for field in schema.fields() {
                    let data_type = match field.data_type() {
                        DataType::FixedSizeList(f, s) => DataType::FixedSizeList(
                            Arc::new(Field::new_list_field(f.data_type().clone(), false)),
                            *s,
                        ),
                        DataType::List(f) => DataType::List(Arc::new(Field::new_list_field(
                            f.data_type().clone(),
                            false,
                        ))),
                        _ => field.data_type().clone(),
                    };
                    fields.push(Field::new(field.name(), data_type, false));
                }

                // Make the Schema
                let schema = Arc::new(Schema::new(fields));
                self.schema = Some(schema.clone());
                schema
            }
        };

        // Read in the CSV
        let mut csv = arrow::csv::ReaderBuilder::new(schema.clone())
            .with_batch_size(batch_size)
            .with_format(format)
            .build_buffered(&mut cursor)?;
        let mut record_batches = vec![];
        while let Some(Ok(read_batch)) = csv.next() {
            record_batches.push(read_batch);
        }
        self.record_batches = Some(record_batches);
        Ok(self)
    }

    fn with_json_values(mut self, json_values: &[Value]) -> Result<Self> {
        if self.schema.is_none() {
            return Err(anyhow!(
                "Please define the schema before adding record batches!"
            ));
        }

        // Prepare the data arrays
        let mut batch_vec = Vec::with_capacity(self.schema.as_ref().unwrap().fields().len());
        let n_rows = json_values.len();

        // Create the arrays
        for field in self.schema.as_ref().unwrap().fields() {
            match field.data_type() {
                DataType::Utf8 => {
                    let mut array_vec = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::String(val)) = map.get(field.name())
                        {
                            array_vec.push(val.to_owned());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::UInt8 => {
                    let mut array_vec: Vec<u8> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_u64().unwrap().try_into().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(UInt8Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::UInt16 => {
                    let mut array_vec: Vec<u16> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_u64().unwrap().try_into().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(UInt16Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::UInt32 => {
                    let mut array_vec: Vec<u32> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_u64().unwrap().try_into().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(UInt32Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::UInt64 => {
                    let mut array_vec: Vec<u64> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_u64().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(UInt64Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Int8 => {
                    let mut array_vec: Vec<i8> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_i64().unwrap().try_into().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(Int8Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Int16 => {
                    let mut array_vec: Vec<i16> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_i64().unwrap().try_into().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(Int16Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Int32 => {
                    let mut array_vec: Vec<i32> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_i64().unwrap().try_into().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(Int32Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Int64 => {
                    let mut array_vec: Vec<i64> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_i64().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(Int64Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Float16 => {
                    // let mut array_vec: Vec<f16> = Vec::with_capacity(n_rows);
                    // for value in json_values {
                    //     if let Value::Object(map) = value {
                    //         if let Some(Value::Number(val)) = map.get(field.name()) {
                    //             array_vec.push(val.as_f64().unwrap() as f16);
                    //         }
                    //     }
                    // }
                    // let array_ref: ArrayRef = Arc::new(Float16Array::from(array_vec));
                    // batch_vec.push((field.name(), array_ref));
                    return Err(anyhow!(
                        "Unstable/Unsupported type {:?} found when converting JSON object to RecordBatch",
                        field.data_type()
                    ));
                }
                DataType::Float32 => {
                    let mut array_vec: Vec<f32> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_f64().unwrap() as f32);
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(Float32Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Float64 => {
                    let mut array_vec: Vec<f64> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Number(val)) = map.get(field.name())
                        {
                            array_vec.push(val.as_f64().unwrap());
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(Float64Array::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::Boolean => {
                    let mut array_vec: Vec<bool> = Vec::with_capacity(n_rows);
                    for value in json_values {
                        if let Value::Object(map) = value
                            && let Some(Value::Bool(val)) = map.get(field.name())
                        {
                            array_vec.push(*val);
                        }
                    }
                    let array_ref: ArrayRef = Arc::new(BooleanArray::from(array_vec));
                    batch_vec.push((field.name(), array_ref));
                }
                DataType::FixedSizeList(f, s) => match f.data_type() {
                    DataType::Float32 => {
                        let mut array_vec: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                let mut inner_vec = Vec::with_capacity(*s as usize);
                                for v in val {
                                    if let Value::Number(num) = v {
                                        inner_vec.push(num.as_f64().unwrap() as f32);
                                    }
                                }
                                array_vec.push(inner_vec);
                            }
                        }
                        let list_values = array_vec.into_iter().flatten().collect::<Vec<_>>();
                        let value_data = ArrayData::builder(f.data_type().clone())
                            .len(list_values.len())
                            .add_buffer(Buffer::from_vec(list_values))
                            .build()
                            .unwrap();
                        let list_data_type = DataType::FixedSizeList(
                            Arc::new(Field::new_list_field(f.data_type().clone(), false)),
                            *s,
                        );
                        let list_data = ArrayData::builder(list_data_type)
                            .len(n_rows)
                            .add_child_data(value_data)
                            .build()
                            .unwrap();
                        let array_ref: ArrayRef = Arc::new(FixedSizeListArray::from(list_data));
                        batch_vec.push((field.name(), array_ref));
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported type {:?} found when converting JSON object to RecordBatch",
                            field.data_type()
                        ));
                    }
                },
                DataType::List(f) => match f.data_type() {
                    DataType::Float32 => {
                        let value_builder = Float32Builder::new();
                        let mut list_builder = ListBuilder::new(value_builder)
                            .with_field(Field::new_list_field(DataType::Float32, false));
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                let mut values = Vec::new();
                                for v in val {
                                    if let Value::Number(num) = v {
                                        values.push(num.as_f64().unwrap() as f32);
                                    }
                                }
                                list_builder.values().append_slice(&values);
                                list_builder.append(true);
                            }
                        }
                        let array_ref: ArrayRef = Arc::new(list_builder.finish());
                        batch_vec.push((field.name(), array_ref));
                    }
                    DataType::Utf8 => {
                        let value_builder = StringBuilder::new();
                        let mut list_builder = ListBuilder::new(value_builder)
                            .with_field(Field::new_list_field(DataType::Utf8, false));
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                for v in val {
                                    if let Value::String(str) = v {
                                        list_builder.values().append_value(str);
                                    }
                                }
                                list_builder.append(true);
                            }
                        }
                        let array_ref: ArrayRef = Arc::new(list_builder.finish());
                        batch_vec.push((field.name(), array_ref));
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported type {:?} found when converting JSON object to RecordBatch",
                            field.data_type()
                        ));
                    }
                },
                _ => {
                    return Err(anyhow!(
                        "Unsupported type {:?} found when converting JSON object to RecordBatch",
                        field.data_type()
                    ));
                }
            }
        }
        let batch = RecordBatch::try_from_iter(batch_vec)?;
        self.record_batches = Some(vec![batch]);
        Ok(self)
    }

    async fn new_from_sendable_record_batch_stream(
        stream: SendableRecordBatchStream,
    ) -> Result<Self> {
        // The stream schema maybe different than the actual schema if it is dynamically updated
        let stream_schema = stream.schema();

        // Collect the record batches
        let record_batches: Vec<RecordBatch> = stream.try_collect::<Vec<_>>().await?;
        let schema = record_batches.first().unwrap().schema();
        if !schema.eq(&stream_schema) {
            event!(
                Level::WARN,
                "Schema mismatch between stream {:?} and record batch {:?}",
                stream_schema,
                schema
            );
        }

        // Use the record batch schema
        Self::new()
            .with_schema(schema)
            .with_record_batches(record_batches)
    }

    async fn new_from_sendable_ipc_record_batch_stream(
        stream: SendableIPCRecordBatchStream,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let _schema = stream.schema();
        let bytes: Vec<u8> = stream
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Self::new_from_ipc_stream(&bytes) //This only works for single record batch streams!
    }

    fn with_struct<T>(self, s: &[T]) -> Result<Self>
    where
        Self: Sized,
        T: Sized + Serialize,
    {
        let mut values = Vec::new();
        for row in s {
            values.push(serde_json::to_value(row)?);
        }
        self.with_json_values(&values)
    }

    fn with_bytes(self, bytes: &[u8]) -> Result<Self>
    where
        Self: Sized,
    {
        let values: Vec<serde_json::Value> = serde_json::from_slice(bytes)?;
        self.with_json_values(&values)
    }
}

/// Mock objects and functions for table testing
pub mod test_table {
    use super::*;
    use arrow::{
        array::{ArrayData, ArrayRef, FixedSizeListArray, StringArray, UInt32Array},
        buffer::Buffer,
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    };
    use chrono::{DateTime, Utc};

    /// Test table struct
    #[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
    pub struct TestTable {
        id: u32,
        collection: String,
        title: String,
        text: String,
        metadata: String,
        score: f32,
        // DM: nested types are not yet supported in JSON Reader
        // embedding: Vec<Vec<f32>>,
    }

    /// Make a test record batch schema with fields for id, title, text, metadata, score, and embeddings
    #[allow(dead_code)]
    pub fn make_test_table_schema(embed_end: u32) -> Result<SchemaRef> {
        let id = Field::new("id", DataType::UInt32, false);
        let collection = Field::new("collection", DataType::Utf8, false);
        let title = Field::new("title", DataType::Utf8, false);
        let text = Field::new("text", DataType::Utf8, false);
        let metadata = Field::new("metadata", DataType::Utf8, false);
        let score = Field::new("score", DataType::Float32, false);

        // Construct a value array
        let schema = if embed_end > 0 {
            let list_data_type = DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                embed_end.try_into().unwrap(),
            );
            let embedding = Field::new("embedding", list_data_type, false);

            Schema::new(vec![
                id, collection, title, text, metadata, score, embedding,
            ])
        } else {
            Schema::new(vec![id, collection, title, text, metadata, score])
        };

        Ok(Arc::new(schema))
    }

    /// Make a test record batch with fields for id, title, text, metadata, score, and embeddings
    pub fn make_test_record_batch(seq_end: u32, embed_end: u32) -> Result<RecordBatch> {
        let seq_start: u32 = 0;
        let embed_start: u32 = 0;
        let embed_length = embed_end - embed_start;
        let seq_length = seq_end - seq_start;
        let total_length = embed_length * seq_length;

        let id: ArrayRef = Arc::new(UInt32Array::from((seq_start..seq_end).collect::<Vec<_>>()));
        let collection: ArrayRef = Arc::new(StringArray::from(
            (seq_start..seq_end)
                .map(|i| format!("collection{i}"))
                .collect::<Vec<_>>(),
        ));
        let title: ArrayRef = Arc::new(StringArray::from(
            (seq_start..seq_end)
                .map(|i| format!("title{i}"))
                .collect::<Vec<_>>(),
        ));
        let text: ArrayRef = Arc::new(StringArray::from(
            (seq_start..seq_end)
                .map(|i| format!("text{i}"))
                .collect::<Vec<_>>(),
        ));
        let metadata: ArrayRef = Arc::new(StringArray::from(
            (seq_start..seq_end)
                .map(|i| format!("metadata{i}"))
                .collect::<Vec<_>>(),
        ));
        let score: ArrayRef = Arc::new(Float32Array::from(
            (seq_start..seq_end).map(|i| i as f32).collect::<Vec<_>>(),
        ));

        // Construct a value array
        let batch = if embed_end > 0 {
            let value_data = ArrayData::builder(DataType::Float32)
                .len(total_length.try_into().unwrap())
                .add_buffer(Buffer::from_slice_ref(
                    (0..total_length).collect::<Vec<_>>(),
                ))
                .build()
                .unwrap();
            let list_data_type = DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                embed_length.try_into().unwrap(),
            );
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(seq_length.try_into().unwrap())
                .add_child_data(value_data.clone())
                .build()
                .unwrap();
            let embedding: ArrayRef = Arc::new(FixedSizeListArray::from(list_data));

            RecordBatch::try_from_iter(vec![
                ("id", id),
                ("collection", collection),
                ("title", title),
                ("text", text),
                ("metadata", metadata),
                ("score", score),
                ("embedding", embedding),
            ])?
        } else {
            RecordBatch::try_from_iter(vec![
                ("id", id),
                ("collection", collection),
                ("title", title),
                ("text", text),
                ("metadata", metadata),
                ("score", score),
            ])?
        };
        Ok(batch)
    }

    /// Make a test table with fields for id, title, text, metadata, score, and embeddings
    /// and with each record batch replicated per batch
    pub fn make_test_table(
        name: &str,
        seq_end: u32,
        embed_end: u32,
        n_batches: usize,
    ) -> Result<Table> {
        let batch = make_test_record_batch(seq_end, embed_end)?;
        let schema = batch.schema();
        let batches: Vec<RecordBatch> = (0..n_batches).map(|_| batch.clone()).collect();
        TableBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(batches)?
            .build()
    }

    #[allow(dead_code)]
    pub enum TestTableSizes {
        XS,
        S,
        M,
        L,
        XL,
    }

    #[allow(dead_code)]
    impl TestTableSizes {
        /// Return the operation based on the name
        pub fn new_from_name(name: &str) -> Option<Self> {
            if name == "xs" {
                Some(Self::XS)
            } else if name == "s" {
                Some(Self::S)
            } else if name == "m" {
                Some(Self::M)
            } else if name == "l" {
                Some(Self::L)
            } else if name == "xl" {
                Some(Self::XL)
            } else {
                None
            }
        }

        /// Get the test table by the test table size
        pub fn get_test_table(&self, name: &str) -> Result<Table> {
            match self {
                Self::XS => make_test_table(name, 1, 1512, 1),
                Self::S => make_test_table(name, 100, 1512, 1),
                Self::M => make_test_table(name, 1000, 1512, 1),
                Self::L => make_test_table(name, 1000, 1512, 1000),
                Self::XL => make_test_table(name, 1000000, 1512, 1000000),
            }
        }

        /// Get the name of the test table
        pub fn get_name(&self) -> &str {
            match self {
                Self::XS => "xs",
                Self::S => "s",
                Self::M => "m",
                Self::L => "l",
                Self::XL => "xl",
            }
        }

        /// Get the test table by the name of the test table size
        pub fn get_test_table_by_name(&self, name: &str) -> Result<Table> {
            if name == Self::XS.get_name() {
                Self::XS.get_test_table(name)
            } else if name == Self::S.get_name() {
                Self::S.get_test_table(name)
            } else if name == Self::M.get_name() {
                Self::M.get_test_table(name)
            } else if name == Self::L.get_name() {
                Self::L.get_test_table(name)
            } else if name == Self::XL.get_name() {
                Self::XL.get_test_table(name)
            } else {
                Err(anyhow!("Test table size name {name} is not supported."))
            }
        }
    }

    pub fn make_test_table_chat(name: &str) -> Result<Table> {
        let role: ArrayRef = Arc::new(StringArray::from(vec![
            "user".to_string(),
            "assistant".to_string(),
            "user".to_string(),
            "assistant".to_string(),
        ]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![
            "Hi!".to_string(),
            "Hello how can I help?".to_string(),
            "What is Deep Learning?".to_string(),
            "magic!".to_string(),
        ]));
        let timestamap: ArrayRef = Arc::new(Int64Array::from(vec![
            "2025-08-03T12:34:56Z"
                .parse::<DateTime<Utc>>()
                .unwrap()
                .timestamp(),
            "2025-08-06T12:55:56Z"
                .parse::<DateTime<Utc>>()
                .unwrap()
                .timestamp(),
            "2025-08-05T12:50:56Z"
                .parse::<DateTime<Utc>>()
                .unwrap()
                .timestamp(),
            "2025-08-04T12:40:56Z"
                .parse::<DateTime<Utc>>()
                .unwrap()
                .timestamp(),
        ]));

        let batch = RecordBatch::try_from_iter(vec![
            ("role", role),
            ("content", content),
            ("timestamp", timestamap),
        ])?;

        let schema = batch.schema();
        TableBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(vec![batch])?
            .build()
    }

    #[allow(dead_code)]
    pub fn make_test_table_tool(name: &str) -> Result<Table> {
        let tool_id: ArrayRef = Arc::new(StringArray::from(vec![
            "tool1".to_string(),
            "no_tool".to_string(),
        ]));
        let tool: ArrayRef = Arc::new(StringArray::from(vec![
            r#"{"type": "function","function": {"name": "tool1", "description": "description1", "parameters": {"type": "object","properties": {"parameter1": {"type": "string", "description": "Param1 description"}, "parameter2": {"type": "string", "enum_values": ["A", "B"], "description": "An Enum."}}, "required": ["parameter1", "parameter2"]}}}"#.to_string(),
            r#"{"type": "function","function": {"name": "no_tool", "description": "Open ended response with no specific tool selected", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The response content"}}, "required": ["content"]}}}"#.to_string(),
        ]));

        let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_id), ("tool", tool)])?;

        let schema = batch.schema();
        TableBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(vec![batch])?
            .build()
    }
}

#[cfg(test)]
mod tests {
    use crate::table::test_table::{TestTable, make_test_table, make_test_table_schema};

    use super::*;

    // Todo: additional tests for builder members

    use arrow::array::UInt32Array;
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempfile;

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_to_from_ipc_file() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 8, 3)?;

        // Create a file inside of `env::temp_dir()`.
        let mut file = tempfile()?;

        // Write data to IPC file
        test_table.to_ipc_file(&mut file)?;
        let test_table_read = TableBuilder::new_from_ipc_file(&file)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_to_from_csv_file() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        // Create a file inside of `env::temp_dir()`.
        let mut file = tempfile()?;

        // Write data to IPC file
        test_table.to_csv_file(&mut file, b',', true)?;

        // Read in the file with schema
        file.rewind().unwrap();
        let test_table_read = TableBuilder::new()
            .with_schema(test_table.get_schema())
            .with_csv_file(&file, b',', true, 4)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );

        // Read in the file without schema
        file.rewind().unwrap();
        let test_table_read = TableBuilder::new()
            .with_csv_file(&file, b',', true, 4)?
            .with_name("test_table")
            .build()?;

        // Test each columns since
        // JSON reader coerces UInt32 to Int64
        let test_table_title = test_table.get_column_as_vec_str("title");
        let test_table_read_title = test_table_read.get_column_as_vec_str("title");
        assert_eq!(test_table_title, test_table_read_title);

        let test_table_id: Vec<u32> = test_table.get_column_as_vec_primitive("id")?;
        let test_table_read_id: Vec<i64> = test_table_read.get_column_as_vec_primitive("id")?;
        assert_eq!(
            test_table_id,
            test_table_read_id
                .into_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>()
        );

        Ok(())
    }

    #[test]
    fn test_to_from_ipc_stream() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let bytes = test_table.to_ipc_stream()?;
        let test_table_read = TableBuilder::new_from_ipc_stream(&bytes)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );

        Ok(())
    }

    #[test]
    fn test_to_from_json() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        // Write data to json
        let bytes = test_table.to_json()?;
        let test_table_read = TableBuilder::new()
            .with_schema(test_table.get_schema().clone())
            .with_json(&bytes, 4)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );

        // Test again but without the schema
        let test_table_read = TableBuilder::new()
            .with_json(&bytes, 4)?
            .with_name("test_table")
            .build()?;

        // Test each columns since
        // JSON reader coerces UInt32 to Int64
        let test_table_title = test_table.get_column_as_vec_str("title");
        let test_table_read_title = test_table_read.get_column_as_vec_str("title");
        assert_eq!(test_table_title, test_table_read_title);

        let test_table_id: Vec<u32> = test_table.get_column_as_vec_primitive("id")?;
        let test_table_read_id: Vec<i64> = test_table_read.get_column_as_vec_primitive("id")?;
        assert_eq!(
            test_table_id,
            test_table_read_id
                .into_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>()
        );
        Ok(())
    }

    #[test]
    fn test_to_from_csv_str() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        // Write data to json
        let bytes = test_table.to_csv(b',', true)?;
        let test_table_read = TableBuilder::new()
            .with_schema(test_table.get_schema().clone())
            .with_csv(&bytes, b',', true, 4)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );

        // Test again but without the schema
        let test_table_read = TableBuilder::new()
            .with_csv(&bytes, b',', true, 4)?
            .with_name("test_table")
            .build()?;

        // Test each columns since
        // JSON reader coerces UInt32 to Int64
        let test_table_title = test_table.get_column_as_vec_str("title");
        let test_table_read_title = test_table_read.get_column_as_vec_str("title");
        assert_eq!(test_table_title, test_table_read_title);

        let test_table_id: Vec<u32> = test_table.get_column_as_vec_primitive("id")?;
        let test_table_read_id: Vec<i64> = test_table_read.get_column_as_vec_primitive("id")?;
        assert_eq!(
            test_table_id,
            test_table_read_id
                .into_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>()
        );

        Ok(())
    }

    #[test]
    fn test_to_from_json_object() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        // Write data to JSON object
        let json_rows = test_table.to_json_object()?;

        assert_eq!(
            serde_json::Value::Object(json_rows[0].clone()),
            serde_json::json!({"collection": "collection0".to_string(),
                "id": 0, "metadata": "metadata0".to_string(), "score": 0.0, "text": "text0".to_string(), "title": "title0".to_string()
            }),
        );

        Ok(())
    }

    #[test]
    fn test_to_from_json_values() -> Result<()> {
        // Create the test table
        let a: ArrayRef = Arc::new(StringArray::from(vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
        ]));
        let b: ArrayRef = Arc::new(UInt32Array::from(vec![0, 0, 0]));
        let c: ArrayRef = Arc::new(UInt16Array::from(vec![0, 0, 0]));
        let batch = RecordBatch::try_from_iter(vec![("a", a), ("b", b), ("c", c)])?;
        let test_table = TableBuilder::new()
            .with_name("test_table")
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the values
        let json_str = r#"[{"a": "a", "b": 0, "c": 0}, {"a": "a", "b": 0, "c": 0}, {"a": "a", "b": 0, "c": 0}]"#;
        let json_values: Vec<Value> = serde_json::from_str(json_str)?;

        // Build a new table from json
        let test_table_read = TableBuilder::new()
            .with_schema(test_table.get_schema())
            .with_json_values(&json_values)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );

        Ok(())
    }

    #[test]
    fn test_to_from_bytes() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        // Write data to Bytes
        let json_bytes = test_table.to_bytes()?;
        let json_str = String::from_utf8_lossy(json_bytes.as_ref()).into_owned();
        let json_rows: Vec<Map<String, Value>> = serde_json::from_str(json_str.as_str())?;

        assert_eq!(
            serde_json::Value::Object(json_rows[0].clone()),
            serde_json::json!({"collection": "collection0".to_string(),
                "id": 0, "metadata": "metadata0".to_string(), "score": 0.0, "text": "text0".to_string(), "title": "title0".to_string()
            }),
        );

        // Build a new table from json
        let test_table_read = TableBuilder::new()
            .with_schema(test_table.get_schema())
            .with_bytes(&json_bytes)?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.concat_record_batches()?.get_record_batches(),
            test_table_read.get_record_batches()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_to_from_record_batch_stream() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let stream = test_table.to_record_batch_stream();
        let test_table_read = TableBuilder::new_from_sendable_record_batch_stream(stream)
            .await?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_to_from_record_batch_stream_last_record_batch() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let stream = test_table.to_record_batch_stream_last_record_batch();
        let test_table_read = TableBuilder::new_from_sendable_record_batch_stream(stream)
            .await?
            .with_name("test_table")
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(test_table_read.get_record_batches().len(), 1);
        assert_eq!(
            test_table.get_record_batches().last().unwrap(),
            test_table_read.get_record_batches().first().unwrap()
        );

        Ok(())
    }

    #[test]
    fn test_to_from_struct() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        // Write data struct
        let s = test_table.to_struct::<TestTable>()?;
        let test_table_read = TableBuilder::new()
            .with_schema(make_test_table_schema(0)?)
            .with_name("test_table")
            .with_struct::<TestTable>(&s)?
            .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.concat_record_batches()?.get_record_batches(),
            test_table_read.get_record_batches()
        );

        Ok(())
    }

    #[test]
    fn test_concat_record_batches() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        let concat_table = test_table.concat_record_batches()?;

        let concat_table_batches = concat_table.get_record_batches();
        assert_eq!(concat_table_batches.len(), 1);
        assert_eq!(concat_table_batches.first().unwrap().num_rows(), 12);

        Ok(())
    }

    #[test]
    fn test_count_rows() -> Result<()> {
        let test_table = make_test_table("test_table", 4, 0, 3)?;

        let n_rows = test_table.count_rows();
        assert_eq!(n_rows, 12);

        Ok(())
    }
}
