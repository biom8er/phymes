use crate::session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait};

use super::{
    stream::{SendableIPCRecordBatchStream, SendableRecordBatchStream},
    stream_adapter::RecordBatchStreamAdapter,
};

use arrow::compute::concat_batches;
use arrow::datatypes::Schema;
use arrow::ipc::{
    reader::{FileReader, StreamReader},
    writer::{FileWriter, StreamWriter},
};
use arrow::json::reader::infer_json_schema;
use arrow::json::{ArrayWriter, LineDelimitedWriter, ReaderBuilder};
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
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use std::fmt::Debug;
use std::fs::File;
use std::io::{Cursor, Seek};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use futures::TryStreamExt;
use serde_json::{Map, Value};
use tracing::instrument;

/// Traits for an arrow table
/// All record batches are guaranteed to have the same schema
/// An optional config can be added that can be consumed by downstream task on the table
/// The user needs to overload all default methods needed in their implementation
pub trait ArrowTableTrait: MappableTrait + BuildableTrait + Debug + Send + Sync {
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
    #[instrument(level = "trace")]
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
    #[instrument(level = "trace")]
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
    #[instrument(level = "trace")]
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
    #[instrument(level = "trace")]
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

    /// Count the number of rows
    fn count_rows(&self) -> usize {
        self.get_record_batches()
            .iter()
            .map(|batches| batches.num_rows())
            .sum::<usize>()
    }

    fn get_column_as_str_vec(&self, column_name: &str) -> Vec<&str> {
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
}

#[derive(Debug, Clone)]
pub struct ArrowTable {
    name: String,
    schema: SchemaRef,
    pub(crate) record_batches: Vec<RecordBatch>,
}

impl Default for ArrowTable {
    fn default() -> Self {
        Self {
            name: "".to_string(),
            schema: Arc::new(Schema::empty()),
            record_batches: Vec::new(),
        }
    }
}

impl ArrowTable {
    /// Concatenate multiple record batches into a single record batch
    pub fn concat_record_batches(mut self) -> Result<Self> {
        let concatenated = concat_batches(&self.schema, &self.record_batches)?;
        self.record_batches = vec![concatenated];
        Ok(self)
    }
}

impl MappableTrait for ArrowTable {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for ArrowTable {
    type T = ArrowTableBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl ArrowTableTrait for ArrowTable {
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

pub trait ArrowTableBuilderTrait: BuilderTrait + Debug + Send + Sync {
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
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct ArrowTableBuilder {
    pub name: Option<String>,
    pub schema: Option<SchemaRef>,
    pub record_batches: Option<Vec<RecordBatch>>,
}

impl BuilderTrait for ArrowTableBuilder {
    type T = ArrowTable;
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

impl ArrowTableBuilderTrait for ArrowTableBuilder {
    fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    fn with_record_batches(mut self, batches: Vec<RecordBatch>) -> Result<Self> {
        // Handle the case of no schema
        if self.schema.is_none() {
            let schema = batches.first().unwrap().schema();
            self.schema = Some(schema);
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

    #[instrument(level = "trace")]
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
        let mut record_batches = vec![];
        while let Some(Ok(read_batch)) = reader.next() {
            record_batches.push(read_batch);
        }
        let schema = record_batches.first().unwrap().schema();
        Self::new()
            .with_schema(schema.clone())
            .with_record_batches(record_batches)
    }

    #[instrument(level = "trace")]
    fn with_json(mut self, bytes: &[u8], batch_size: usize) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);
        // Potentially remove the need to define a schema first...
        let schema = self.schema.clone().unwrap_or_else(|| {
            let (schema, _) = infer_json_schema(&mut cursor, None).unwrap();
            cursor.rewind().unwrap();
            let schema = Arc::new(schema);
            self.schema = Some(schema.clone());
            schema
        });
        let mut reader = ReaderBuilder::new(schema)
            .with_batch_size(batch_size)
            .build(cursor)?;
        let mut record_batches = vec![];
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
                let (schema, _) = format.infer_schema(&mut cursor, None)?;
                cursor.rewind().unwrap();
                let schema = Arc::new(schema);
                self.schema = Some(schema.clone());
                schema
            } // None => return Err(anyhow!("Please define the schema before adding record batches!"))
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
        // Prepare the data arrays
        let mut batch_vec = Vec::with_capacity(self.schema.as_ref().unwrap().fields().len());
        let n_rows = json_values.len();

        // Create the arrays
        for field in self.schema.as_ref().unwrap().fields() {
            if field.data_type() == &DataType::Utf8 {
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
            } else if field.data_type() == &DataType::UInt8 {
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
            } else if field.data_type() == &DataType::UInt16 {
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
            } else if field.data_type() == &DataType::UInt32 {
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
            } else if field.data_type() == &DataType::UInt64 {
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
            } else if field.data_type() == &DataType::Float16 {
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
            } else if field.data_type() == &DataType::Float32 {
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
            } else if field.data_type() == &DataType::Float64 {
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
            } else if field.data_type() == &DataType::Boolean {
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
            } else {
                return Err(anyhow!(
                    "Unsupported type {:?} found when converting JSON object to RecordBatch",
                    field.data_type()
                ));
            }
        }
        let batch = RecordBatch::try_from_iter(batch_vec)?;
        self.record_batches = Some(vec![batch]);
        Ok(self)
    }

    async fn new_from_sendable_record_batch_stream(
        stream: SendableRecordBatchStream,
    ) -> Result<Self> {
        let schema = stream.schema();
        let record_batches: Vec<RecordBatch> = stream.try_collect::<Vec<_>>().await?;
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

    pub fn make_test_table_schema(embed_end: u32) -> Result<SchemaRef> {
        let ids = Field::new("ids", DataType::UInt32, false);
        let collection = Field::new("collection", DataType::Utf8, false);
        let title = Field::new("title", DataType::Utf8, false);
        let text = Field::new("text", DataType::Utf8, false);
        let metadata = Field::new("metadata", DataType::Utf8, false);

        // Construct a value array
        let schema = if embed_end > 0 {
            let list_data_type = DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::UInt32, false)),
                embed_end.try_into().unwrap(),
            );
            let embedding = Field::new("embedding", list_data_type, false);

            Schema::new(vec![ids, collection, title, text, metadata, embedding])
        } else {
            Schema::new(vec![ids, collection, title, text, metadata])
        };

        Ok(Arc::new(schema))
    }

    pub fn make_test_record_batch(seq_end: u32, embed_end: u32) -> Result<RecordBatch> {
        let seq_start: u32 = 0;
        let embed_start: u32 = 0;
        let embed_length = embed_end - embed_start;
        let seq_length = seq_end - seq_start;
        let total_length = embed_length * seq_length;

        let ids: ArrayRef = Arc::new(UInt32Array::from((seq_start..seq_end).collect::<Vec<_>>()));
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

        // Construct a value array
        let batch = if embed_end > 0 {
            let value_data = ArrayData::builder(DataType::UInt32)
                .len(total_length.try_into().unwrap())
                .add_buffer(Buffer::from_slice_ref(
                    (0..total_length).collect::<Vec<_>>(),
                ))
                .build()
                .unwrap();
            let list_data_type = DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::UInt32, false)),
                embed_length.try_into().unwrap(),
            );
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(seq_length.try_into().unwrap())
                .add_child_data(value_data.clone())
                .build()
                .unwrap();
            let embedding: ArrayRef = Arc::new(FixedSizeListArray::from(list_data));

            RecordBatch::try_from_iter(vec![
                ("ids", ids),
                ("collection", collection),
                ("title", title),
                ("text", text),
                ("metadata", metadata),
                ("embedding", embedding),
            ])?
        } else {
            RecordBatch::try_from_iter(vec![
                ("ids", ids),
                ("collection", collection),
                ("title", title),
                ("text", text),
                ("metadata", metadata),
            ])?
        };
        Ok(batch)
    }

    pub fn make_test_table(
        name: &str,
        seq_end: u32,
        embed_end: u32,
        n_batches: usize,
    ) -> Result<ArrowTable> {
        let batch = make_test_record_batch(seq_end, embed_end)?;
        let schema = batch.schema();
        let batches: Vec<RecordBatch> = (0..n_batches).map(|_| batch.clone()).collect();
        ArrowTableBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(batches)?
            .build()
    }

    pub fn make_test_table_chat(name: &str) -> Result<ArrowTable> {
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
        let timestamap: ArrayRef = Arc::new(StringArray::from(vec![
            "Fri Jul 11 09:16:02 2025".to_string(),
            "Fri Jul 11 09:16:20 2025".to_string(),
            "Fri Jul 11 09:16:20 2025".to_string(),
            "Fri Jul 11 09:16:21 2025".to_string(),
        ]));

        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content), ("timestamp", timestamap)])?;

        let schema = batch.schema();
        ArrowTableBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(vec![batch])?
            .build()
    }

    pub fn make_test_table_tool(name: &str) -> Result<ArrowTable> {
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
        ArrowTableBuilder::new()
            .with_name(name)
            .with_schema(schema.clone())
            .with_record_batches(vec![batch])?
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Todo: additional tests for builder members

    use arrow::array::{Int64Array, UInt32Array};
    #[cfg(not(target_family = "wasm"))]
    use tempfile::tempfile;

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_to_from_ipc_file() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Create a file inside of `env::temp_dir()`.
        let mut file = tempfile()?;

        // Write data to IPC file
        test_table.to_ipc_file(&mut file)?;
        let test_table_read = ArrowTableBuilder::new_from_ipc_file(&file)?
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
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        // Create a file inside of `env::temp_dir()`.
        let mut file = tempfile()?;

        // Write data to IPC file
        test_table.to_csv_file(&mut file, b',', true)?;

        // Read in the file with schema
        file.rewind().unwrap();
        let test_table_read = ArrowTableBuilder::new()
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
        let test_table_read = ArrowTableBuilder::new()
            .with_csv_file(&file, b',', true, 4)?
            .with_name("test_table")
            .build()?;

        // Test each columns since
        // JSON reader coerces UInt32 to Int64
        let test_table_title = test_table.get_column_as_str_vec("title");
        let test_table_read_title = test_table_read.get_column_as_str_vec("title");
        assert_eq!(test_table_title, test_table_read_title);

        let test_table_id = test_table
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("ids")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let test_table_read_id = test_table_read
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("ids")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(test_table_id, test_table_read_id);

        Ok(())
    }

    #[test]
    fn test_to_from_ipc_stream() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let bytes = test_table.to_ipc_stream()?;
        let test_table_read = ArrowTableBuilder::new_from_ipc_stream(&bytes)?
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
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        // Write data to json
        let bytes = test_table.to_json()?;
        let test_table_read = ArrowTableBuilder::new()
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
        let test_table_read = ArrowTableBuilder::new()
            .with_json(&bytes, 4)?
            .with_name("test_table")
            .build()?;

        // Test each columns since
        // JSON reader coerces UInt32 to Int64
        let test_table_title = test_table.get_column_as_str_vec("title");
        let test_table_read_title = test_table_read.get_column_as_str_vec("title");
        assert_eq!(test_table_title, test_table_read_title);

        let test_table_id = test_table
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("ids")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let test_table_read_id = test_table_read
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("ids")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(test_table_id, test_table_read_id);

        Ok(())
    }

    #[test]
    fn test_to_from_csv_str() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        // Write data to json
        let bytes = test_table.to_csv(b',', true)?;
        let test_table_read = ArrowTableBuilder::new()
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
        let test_table_read = ArrowTableBuilder::new()
            .with_csv(&bytes, b',', true, 4)?
            .with_name("test_table")
            .build()?;

        // Test each columns since
        // JSON reader coerces UInt32 to Int64
        let test_table_title = test_table.get_column_as_str_vec("title");
        let test_table_read_title = test_table_read.get_column_as_str_vec("title");
        assert_eq!(test_table_title, test_table_read_title);

        let test_table_id = test_table
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("ids")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let test_table_read_id = test_table_read
            .get_record_batches()
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("ids")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default() as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(test_table_id, test_table_read_id);

        Ok(())
    }

    #[test]
    fn test_to_from_json_object() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        // Write data to JSON object
        let json_rows = test_table.to_json_object()?;

        assert_eq!(
            serde_json::Value::Object(json_rows[0].clone()),
            serde_json::json!({"collection": "collection0".to_string(),
                "ids": 0, "metadata": "metadata0".to_string(), "text": "text0".to_string(), "title": "title0".to_string()
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
        let test_table = ArrowTableBuilder::new()
            .with_name("test_table")
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the values
        let json_str = r#"[{"a": "a", "b": 0, "c": 0}, {"a": "a", "b": 0, "c": 0}, {"a": "a", "b": 0, "c": 0}]"#;
        let json_values: Vec<Value> = serde_json::from_str(json_str)?;

        // Build a new table from json
        let test_table_read = ArrowTableBuilder::new()
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
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        // Write data to Bytes
        let json_bytes = test_table.to_bytes()?;
        let json_str = String::from_utf8_lossy(json_bytes.as_ref()).into_owned();
        let json_rows: Vec<Map<String, Value>> = serde_json::from_str(json_str.as_str())?;

        assert_eq!(
            serde_json::Value::Object(json_rows[0].clone()),
            serde_json::json!({"collection": "collection0".to_string(),
                "ids": 0, "metadata": "metadata0".to_string(), "text": "text0".to_string(), "title": "title0".to_string()
            }),
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_to_from_record_batch_stream() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let stream = test_table.to_record_batch_stream();
        let test_table_read = ArrowTableBuilder::new_from_sendable_record_batch_stream(stream)
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
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Write data to IPC file
        let stream = test_table.to_record_batch_stream_last_record_batch();
        let test_table_read = ArrowTableBuilder::new_from_sendable_record_batch_stream(stream)
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
    fn test_concat_record_batches() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        let concat_table = test_table.concat_record_batches()?;

        let concat_table_batches = concat_table.get_record_batches();
        assert_eq!(concat_table_batches.len(), 1);
        assert_eq!(concat_table_batches.first().unwrap().num_rows(), 12);

        Ok(())
    }

    #[test]
    fn test_count_rows() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 0, 3)?;

        let n_rows = test_table.count_rows();
        assert_eq!(n_rows, 12);

        Ok(())
    }
}
