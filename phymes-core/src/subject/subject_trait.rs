use std::{
    fmt::Debug,
    io::{Cursor, Seek, Write},
    pin::Pin,
    sync::Arc,
};

use crate::{
    BuildableTrait, BuilderTrait, IpcWriter, MappableTrait, RecordBatchStreamAdapter,
    SendableRecordBatchStream, StorageStreamWriterTrait, StorageWriterTrait, SubjectBuilder,
};

use anyhow::{Result, anyhow};
use arrow::{
    array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, LargeStringArray, ListArray, StringArray, UInt8Array, UInt16Array,
        UInt32Array, UInt64Array,
    },
    compute::{cast, concat_batches, kernels::concat},
    csv::WriterBuilder,
    datatypes::{DataType, Schema, SchemaRef},
    ipc::writer::{FileWriter, StreamWriter},
    json::{ArrayWriter, LineDelimitedWriter},
    record_batch::RecordBatch,
};
use bytes::Bytes;
use num_traits::{Bounded, Num, NumCast};
use object_store::{ObjectStore, path::Path};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Traits for a columnar table where all [RecordBatch]es are guaranteed to have the same [Schema]
pub trait SubjectTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
    fn get_schema(&self) -> SchemaRef;
    fn get_record_batches(&self) -> &Vec<RecordBatch>;
    fn get_record_batches_own(self) -> Vec<RecordBatch>;
    fn get_record_batches_mut(&mut self) -> &mut Vec<RecordBatch>;

    /// Generate a default object store path (without partitions)
    fn default_ipc_object_store_path(&self) -> Path {
        Path::from(format!("{}/{}.ipc", self.get_name(), self.get_name()))
    }

    /// Write record batches to IPC object store, consuming self
    fn to_ipc_object_store<'a>(
        &'a self,
        store: &'a Arc<dyn ObjectStore>,
        path: Option<&'a Path>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if self.get_record_batches().is_empty() {
                return Err(anyhow!(
                    "Cannot write empty record batches to IPC file since they cannot be read back in."
                ));
            }
            let mut writer = IpcWriter::new_with_config(self.get_schema().clone())?;
            for batch in self.get_record_batches() {
                writer.write_batch(batch)?;
            }
            writer.finish_batch()?;
            if let Some(path) = path {
                writer.put(store, path).await?;
            } else {
                let path = self.default_ipc_object_store_path();
                writer.put(store, &path).await?;
            }
            Ok(())
        })
    }

    /// Write record batches to IPC file
    fn to_ipc_file<F>(&self, file: &mut F) -> Result<()>
    where
        F: Write + Seek,
    {
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
    fn to_csv_file<F>(&self, file: &mut F, delimiter: u8, header: bool) -> Result<()>
    where
        F: Write + Seek,
    {
        // Convert nested columns to String
        let batches = self
            .get_record_batches()
            .iter()
            .map(|batch| {
                let binding = batch.schema();
                let batches = binding
                    .fields()
                    .iter()
                    .map(|f| {
                        if f.data_type().is_nested() {
                            let arr = batch.column_by_name(f.name()).unwrap();
                            let vec_str = Self::get_array_as_vec_string(arr, f.name()).unwrap();
                            let arr: ArrayRef = Arc::new(StringArray::from(vec_str));
                            (f.name(), arr)
                        } else {
                            let arr = batch.column_by_name(f.name()).unwrap();
                            (f.name(), arr.to_owned())
                        }
                    })
                    .collect::<Vec<_>>();
                RecordBatch::try_from_iter(batches).unwrap()
            })
            .collect::<Vec<_>>();

        // Write to CSV
        let builder = WriterBuilder::new()
            .with_header(header)
            .with_delimiter(delimiter)
            .with_quote(b'\'')
            .with_null("NULL".to_string())
            .with_time_format("%r".to_string());
        let mut writer = builder.build(file);
        for batch in batches {
            writer.write(&batch).unwrap();
        }
        drop(writer);
        Ok(())
    }

    /// Write record batches to CSV
    fn to_csv(&self, delimiter: u8, header: bool) -> Result<Vec<u8>> {
        // Convert nested columns to String
        let batches = self
            .get_record_batches()
            .iter()
            .map(|batch| {
                let binding = batch.schema();
                let batches = binding
                    .fields()
                    .iter()
                    .map(|f| {
                        if f.data_type().is_nested() {
                            let arr = batch.column_by_name(f.name()).unwrap();
                            let vec_str = Self::get_array_as_vec_string(arr, f.name()).unwrap();
                            let arr: ArrayRef = Arc::new(StringArray::from(vec_str));
                            (f.name(), arr)
                        } else {
                            let arr = batch.column_by_name(f.name()).unwrap();
                            (f.name(), arr.to_owned())
                        }
                    })
                    .collect::<Vec<_>>();
                RecordBatch::try_from_iter(batches).unwrap()
            })
            .collect::<Vec<_>>();

        // Write to CSV
        let mut bytes = Vec::new();
        let builder = WriterBuilder::new()
            .with_header(header)
            .with_delimiter(delimiter)
            .with_quote(b'\'')
            .with_null("NULL".to_string())
            .with_time_format("%r".to_string());
        let mut writer = builder.build(&mut bytes);
        for batch in batches {
            writer.write(&batch).unwrap();
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
        writer.finish()?;
        drop(writer);
        Ok(bytes)
    }

    /// Write record batches to JSON
    fn to_json(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let mut writer = LineDelimitedWriter::new(Cursor::new(&mut bytes));
        for batch in self.get_record_batches() {
            writer.write(batch)?;
        }
        writer.finish()?;
        Ok(bytes)
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

    /// Convert to a sendable record batch stream after taking ownership
    fn to_record_batch_stream_own(self) -> SendableRecordBatchStream
    where
        Self: Sized,
    {
        let schema = Arc::clone(&self.get_schema());
        let stream = futures::stream::iter(self.get_record_batches_own().into_iter().map(Ok));
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    /// Convert to a sendable record batch stream after taking ownership
    fn to_record_batch_stream_last_record_batch_own(self) -> SendableRecordBatchStream
    where
        Self: Sized,
    {
        let schema = Arc::clone(&self.get_schema());
        let last_record_batch = if let Some(batch) = self.get_record_batches_own().pop() {
            vec![batch]
        } else {
            Vec::new()
        };
        let stream = futures::stream::iter(last_record_batch.into_iter().map(Ok));
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    /// Convert to a sendable record batch stream by consuming the batches but leaving the object
    fn to_record_batch_stream_drain(&mut self) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.get_schema());
        let batches = self.get_record_batches_mut().drain(0..).collect::<Vec<_>>();
        let stream = futures::stream::iter(batches.into_iter().map(Ok));
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    /// Convert to a sendable record batch stream after taking ownership
    fn to_record_batch_stream_last_record_batch_pop(&mut self) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.get_schema());
        let last_record_batch = if let Some(batch) = self.get_record_batches_mut().pop() {
            vec![batch]
        } else {
            Vec::new()
        };
        let stream = futures::stream::iter(last_record_batch.into_iter().map(Ok));
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
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

    /// Get an array as a vector of strings
    fn get_array_as_vec_string(arr: &Arc<dyn Array>, column_name: &str) -> Result<Vec<String>> {
        match arr.data_type() {
            DataType::Utf8 => {
                let vec_str = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default().to_string())
                    .collect::<Vec<_>>();
                Ok(vec_str)
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
                let arr = cast(arr, &DataType::Utf8)?;
                let vec_str = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default().to_string())
                    .collect::<Vec<_>>();
                Ok(vec_str)
            }
            DataType::List(_) | DataType::FixedSizeList(_, _) => {
                // Convert the column to JSON
                let batch = RecordBatch::try_from_iter(vec![(column_name, arr.to_owned())])?;
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
                Ok(vec_str)
            }
            _ => Err(anyhow!(
                "Unsupported data type {} for column {column_name} when trying to convert to String.",
                arr.data_type()
            )),
        }
    }

    /// Get a column as a vector of strings
    fn get_column_as_vec_string(&self, column_name: &str) -> Result<Vec<String>> {
        let arr = self.get_column_as_array(column_name)?;
        Self::get_array_as_vec_string(&arr, column_name)
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
    fn get_column_as_array(&self, column_name: &str) -> Result<Arc<dyn Array>> {
        if self.get_record_batches().len() > 1 {
            let array_refs = self
                .get_record_batches()
                .iter()
                .map(|batch| batch.column_by_name(column_name).unwrap().as_ref())
                .collect::<Vec<_>>();
            let concatenated = concat::concat(&array_refs).unwrap();
            Ok(concatenated)
        } else if let Some(batch) = self.get_record_batches().first() {
            let arr = batch.column_by_name(column_name).unwrap();
            Ok(Arc::clone(arr))
        } else {
            Err(anyhow!(
                "Cannot get column {column_name} as an Array because there are no RecordBatches."
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Subject {
    pub(crate) name: String,
    pub(crate) schema: SchemaRef,
    pub(crate) record_batches: Vec<RecordBatch>,
}

impl Default for Subject {
    fn default() -> Self {
        Self {
            name: "".to_string(),
            schema: Arc::new(Schema::empty()),
            record_batches: Vec::new(),
        }
    }
}

impl PartialEq for Subject {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.schema == other.schema
            && self.record_batches == other.record_batches
    }
}

impl Subject {
    /// Concatenate multiple record batches into a single record batch
    pub fn concat_record_batches(mut self) -> Result<Self> {
        let concatenated = concat_batches(&self.schema, &self.record_batches)?;
        self.record_batches = vec![concatenated];
        Ok(self)
    }
}

impl MappableTrait for Subject {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for Subject {
    type T = SubjectBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl SubjectTrait for Subject {
    fn get_schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn get_record_batches(&self) -> &Vec<RecordBatch> {
        &self.record_batches
    }
    fn get_record_batches_own(self) -> Vec<RecordBatch> {
        self.record_batches
    }
    fn get_record_batches_mut(&mut self) -> &mut Vec<RecordBatch> {
        &mut self.record_batches
    }
}

/// Mock objects and functions for table testing
pub mod test_subject {
    use crate::SubjectBuilderTrait;

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
    pub struct TestSubject {
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
    pub fn make_test_subject_schema(embed_end: u32) -> Result<SchemaRef> {
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
    pub fn make_test_subject(
        name: &str,
        seq_end: u32,
        embed_end: u32,
        n_batches: usize,
    ) -> Result<Subject> {
        let batch = make_test_record_batch(seq_end, embed_end)?;
        let schema = batch.schema();
        let batches: Vec<RecordBatch> = (0..n_batches).map(|_| batch.clone()).collect();
        SubjectBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(batches)?
            .build()
    }

    #[allow(dead_code)]
    pub enum TestSubjectSizes {
        XS,
        S,
        M,
        L,
        XL,
    }

    #[allow(dead_code)]
    impl TestSubjectSizes {
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
        pub fn get_test_subject(&self, name: &str) -> Result<Subject> {
            match self {
                Self::XS => make_test_subject(name, 1, 1512, 1),
                Self::S => make_test_subject(name, 100, 1512, 1),
                Self::M => make_test_subject(name, 1000, 1512, 1),
                Self::L => make_test_subject(name, 1000, 1512, 1000),
                Self::XL => make_test_subject(name, 1000000, 1512, 1000000),
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
        pub fn get_test_subject_by_name(&self, name: &str) -> Result<Subject> {
            if name == Self::XS.get_name() {
                Self::XS.get_test_subject(name)
            } else if name == Self::S.get_name() {
                Self::S.get_test_subject(name)
            } else if name == Self::M.get_name() {
                Self::M.get_test_subject(name)
            } else if name == Self::L.get_name() {
                Self::L.get_test_subject(name)
            } else if name == Self::XL.get_name() {
                Self::XL.get_test_subject(name)
            } else {
                Err(anyhow!("Test table size name {name} is not supported."))
            }
        }
    }

    pub fn make_test_subject_chat(name: &str) -> Result<Subject> {
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
        SubjectBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(vec![batch])?
            .build()
    }

    #[allow(dead_code)]
    pub fn make_test_subject_tool(name: &str) -> Result<Subject> {
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
        SubjectBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(vec![batch])?
            .build()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        SubjectBuilderTrait,
        test_subject::{TestSubject, make_test_subject, make_test_subject_schema},
    };

    use super::*;

    // Todo: additional tests for builder members

    use arrow::array::UInt32Array;
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempfile;

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_to_from_ipc_file() -> Result<()> {
        use crate::SubjectBuilderTrait;

        let test_table = make_test_subject("test_table", 4, 8, 3)?;

        // Create a file inside of `env::temp_dir()`.
        let mut file = tempfile()?;

        // Write data to IPC file
        test_table.to_ipc_file(&mut file)?;
        let test_table_read = SubjectBuilder::new_from_ipc_file(&file)?
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
        use crate::SubjectBuilderTrait;

        let test_table = make_test_subject("test_table", 4, 0, 3)?;

        // Create a file inside of `env::temp_dir()`.
        let mut file = tempfile()?;

        // Write data to CSV file
        test_table.to_csv_file(&mut file, b',', true)?;

        // Read in the file with schema
        file.rewind().unwrap();
        let test_table_read = SubjectBuilder::new()
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
        let test_table_read = SubjectBuilder::new()
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

        // Test that we can write csv with nested fields
        let test_table = make_test_subject("test_table", 4, 8, 3)?;
        let mut file = tempfile()?;

        // Write data to CSV file
        test_table.to_csv_file(&mut file, b',', true)?;

        Ok(())
    }

    #[test]
    fn test_to_from_ipc_stream() -> Result<()> {
        let test_table = make_test_subject("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let bytes = test_table.to_ipc_stream()?;
        let test_table_read = SubjectBuilder::new_from_ipc_stream(&bytes)?
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
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

        // Write data to json
        let bytes = test_table.to_json()?;
        let test_table_read = SubjectBuilder::new()
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
        let test_table_read = SubjectBuilder::new()
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
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

        // Write data to json
        let bytes = test_table.to_csv(b',', true)?;
        let test_table_read = SubjectBuilder::new()
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
        let test_table_read = SubjectBuilder::new()
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

        // Test that we can write csv with nested fields
        let test_table = make_test_subject("test_table", 4, 8, 3)?;

        // Write data to CSV file
        test_table.to_csv(b',', true)?;

        Ok(())
    }

    #[test]
    fn test_to_from_json_object() -> Result<()> {
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

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
        let test_table = SubjectBuilder::new()
            .with_name("test_table")
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the values
        let json_str = r#"[{"a": "a", "b": 0, "c": 0}, {"a": "a", "b": 0, "c": 0}, {"a": "a", "b": 0, "c": 0}]"#;
        let json_values: Vec<Value> = serde_json::from_str(json_str)?;

        // Build a new table from json
        let test_table_read = SubjectBuilder::new()
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
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

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
        let test_table_read = SubjectBuilder::new()
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
        let test_table = make_test_subject("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let stream = test_table.to_record_batch_stream();
        let test_table_read = SubjectBuilder::new_from_sendable_record_batch_stream(stream)
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
        let test_table = make_test_subject("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let stream = test_table.to_record_batch_stream_last_record_batch();
        let test_table_read = SubjectBuilder::new_from_sendable_record_batch_stream(stream)
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
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

        // Write data struct
        let s = test_table.to_struct::<TestSubject>()?;
        let test_table_read = SubjectBuilder::new()
            .with_schema(make_test_subject_schema(0)?)
            .with_name("test_table")
            .with_struct::<TestSubject>(&s)?
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
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

        let concat_table = test_table.concat_record_batches()?;

        let concat_table_batches = concat_table.get_record_batches();
        assert_eq!(concat_table_batches.len(), 1);
        assert_eq!(concat_table_batches.first().unwrap().num_rows(), 12);

        Ok(())
    }

    #[test]
    fn test_count_rows() -> Result<()> {
        let test_table = make_test_subject("test_table", 4, 0, 3)?;

        let n_rows = test_table.count_rows();
        assert_eq!(n_rows, 12);

        Ok(())
    }
}
