use std::{
    fmt::Debug,
    fs::File,
    io::{Cursor, Read, Seek},
    pin::Pin,
    sync::Arc,
};

use crate::{BuilderTrait, SendableIPCRecordBatchStream, SendableRecordBatchStream, Subject};
use anyhow::{Result, anyhow};
use arrow::{
    array::{
        ArrayData, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float32Builder,
        Float64Array, Float64Builder, Int8Array, Int16Array, Int32Array, Int64Array, Int64Builder,
        ListBuilder, StringArray, StringBuilder, UInt8Array, UInt8Builder, UInt16Array,
        UInt32Array, UInt32Builder, UInt64Array,
    },
    buffer::Buffer,
    csv::reader::Format,
    datatypes::{DataType, Field, Schema, SchemaRef},
    ipc::reader::{FileReader, StreamReader},
    json::{ReaderBuilder, reader::infer_json_schema},
    record_batch::RecordBatch,
};
use futures::{Stream, StreamExt, TryStreamExt};
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, path::Path};
use serde::Serialize;
use serde_json::Value;
use tracing::{Level, event};

pub trait SubjectBuilderTrait: BuilderTrait + Debug + Send + Sync {
    /// The schema for all record batches in the subject
    fn with_schema(self, schema: SchemaRef) -> Self;

    /// Add record batches
    fn with_record_batches(self, batches: Vec<RecordBatch>) -> Result<Self>
    where
        Self: Sized;

    /// List the partitions in the object storage
    fn list_partitions_object_store<'a>(
        &'a self,
        store: &'a Arc<dyn ObjectStore>,
    ) -> Pin<Box<dyn Stream<Item = Result<ObjectMeta, object_store::Error>> + Send>>;

    /// Delete the partitions in the object storage (with the option to specify which partitions)
    fn delete_partitions_object_store<'a>(
        &'a self,
        store: &'a Arc<dyn ObjectStore>,
        paths: Option<Vec<Path>>,
    ) -> Pin<Box<dyn Stream<Item = Result<Path, object_store::Error>> + Send>>;

    /// Create a new stream subject with the provided object storage
    /// from IPC format    
    fn new_from_ipc_object_store<'a>(
        store: &'a Arc<dyn ObjectStore>,
        path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<Self>> + Send + 'a>>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a IPC file
    fn new_from_ipc_file<F>(file: F) -> Result<Self>
    where
        F: Read + Seek,
        Self: Sized;

    /// Create a new stream subject with the provided batches
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

    /// Create a new stream subject with the provided batches
    /// from a IPC stream
    fn new_from_ipc_stream(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a JSON
    fn with_json(self, bytes: &[u8], batch_size: usize) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a CSV file
    fn with_csv(self, bytes: &[u8], delimiter: u8, header: bool, batch_size: usize) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a JSON array of values
    fn with_json_values(self, json_values: &[Value]) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a SendableRecordBatchStream
    #[allow(async_fn_in_trait)]
    async fn new_from_sendable_record_batch_stream(
        stream: SendableRecordBatchStream,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a SendableIPCRecordBatchStream
    #[allow(async_fn_in_trait)]
    async fn new_from_sendable_ipc_record_batch_stream(
        stream: SendableIPCRecordBatchStream,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Create a new stream subject with the provided batches
    /// from a vector of structs
    fn with_struct<T>(self, s: &[T]) -> Result<Self>
    where
        Self: Sized,
        T: Sized + Serialize;

    /// Create a new stream subject with the provided batches
    /// from a JSON object in byte format
    fn with_bytes(self, bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct SubjectBuilder {
    pub name: Option<String>,
    pub schema: Option<SchemaRef>,
    pub record_batches: Option<Vec<RecordBatch>>,
}

impl SubjectBuilder {
    pub fn from_ipc_stream_to_record_batches(bytes: &[u8]) -> Result<Vec<RecordBatch>> {
        let cursor = Cursor::new(bytes);
        let mut reader = StreamReader::try_new(cursor, None)?;
        let mut record_batches = Vec::new();
        while let Some(Ok(read_batch)) = reader.next() {
            record_batches.push(read_batch);
        }
        Ok(record_batches)
    }
}

impl BuilderTrait for SubjectBuilder {
    type T = Subject;
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

impl SubjectBuilderTrait for SubjectBuilder {
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
                return Err(anyhow!(
                    "Missing schema and batches for subject {}!",
                    self.name.unwrap_or_default()
                ));
            }
        };

        // Check the batch schemas are consistent
        let schema = self.schema.clone().unwrap();
        for batch in batches.iter() {
            if !schema.eq(&batch.schema()) {
                return Err(anyhow!(
                    "Mismatch between schema and batches for subject {}!",
                    self.name.unwrap_or_default()
                ));
            }
        }
        self.record_batches = Some(batches);
        Ok(self)
    }

    fn list_partitions_object_store<'a>(
        &'a self,
        store: &'a Arc<dyn ObjectStore>,
    ) -> Pin<Box<dyn Stream<Item = Result<ObjectMeta, object_store::Error>> + Send>> {
        if let Some(name) = self.name.as_ref() {
            let path = Some(Path::from(name.to_owned()));
            store.list(path.as_ref())
        } else {
            let err = object_store::Error::Generic { store: "", source: anyhow!("Provide a name for the subject before trying to list all of the tables partitions in object storage.").into_boxed_dyn_error() };
            futures::stream::iter(vec![Err(err)]).boxed()
        }
    }

    fn delete_partitions_object_store<'a>(
        &'a self,
        store: &'a Arc<dyn ObjectStore>,
        paths: Option<Vec<Path>>,
    ) -> Pin<Box<dyn Stream<Item = Result<Path, object_store::Error>> + Send>> {
        let locations = if let Some(paths) = paths {
            futures::stream::iter(paths.into_iter().map(Ok)).boxed()
        } else {
            self.list_partitions_object_store(store)
                .map_ok(|m| m.location)
                .boxed()
        };
        store.delete_stream(locations)
    }

    fn new_from_ipc_object_store<'a>(
        store: &'a Arc<dyn ObjectStore>,
        path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<Self>> + Send + 'a>> {
        Box::pin(async move {
            let result = store.get(path).await?;
            let mut stream = result.into_stream();
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.try_next().await? {
                bytes.extend(chunk);
            }
            Self::new_from_ipc_stream(&bytes)
        })
    }

    fn new_from_ipc_file<F>(file: F) -> Result<Self>
    where
        F: Read + Seek,
    {
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

    fn new_from_ipc_stream(bytes: &[u8]) -> Result<Self> {
        let record_batches = Self::from_ipc_stream_to_record_batches(bytes)?;
        let schema = match record_batches.first() {
            Some(batch) => batch.schema(),
            None => {
                return Err(anyhow!(
                    "Failed to read the IPC stream for data with bytes {}. Ensure that there are no NULL values.",
                    bytes.len()
                ));
            }
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
                    DataType::UInt8 => {
                        let value_builder = UInt8Builder::new();
                        let mut list_builder = ListBuilder::new(value_builder)
                            .with_field(Field::new_list_field(DataType::UInt8, false));
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                let mut values = Vec::new();
                                for v in val {
                                    if let Value::Number(num) = v {
                                        values.push(num.as_u64().unwrap() as u8);
                                    }
                                }
                                list_builder.values().append_slice(&values);
                                list_builder.append(true);
                            }
                        }
                        let array_ref: ArrayRef = Arc::new(list_builder.finish());
                        batch_vec.push((field.name(), array_ref));
                    }
                    DataType::UInt32 => {
                        let value_builder = UInt32Builder::new();
                        let mut list_builder = ListBuilder::new(value_builder)
                            .with_field(Field::new_list_field(DataType::UInt32, false));
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                let mut values = Vec::new();
                                for v in val {
                                    if let Value::Number(num) = v {
                                        values.push(num.as_u64().unwrap() as u32);
                                    }
                                }
                                list_builder.values().append_slice(&values);
                                list_builder.append(true);
                            }
                        }
                        let array_ref: ArrayRef = Arc::new(list_builder.finish());
                        batch_vec.push((field.name(), array_ref));
                    }
                    DataType::Int64 => {
                        let value_builder = Int64Builder::new();
                        let mut list_builder = ListBuilder::new(value_builder)
                            .with_field(Field::new_list_field(DataType::Int64, false));
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                let mut values = Vec::new();
                                for v in val {
                                    if let Value::Number(num) = v {
                                        values.push(num.as_i64().unwrap());
                                    }
                                }
                                list_builder.values().append_slice(&values);
                                list_builder.append(true);
                            }
                        }
                        let array_ref: ArrayRef = Arc::new(list_builder.finish());
                        batch_vec.push((field.name(), array_ref));
                    }
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
                    DataType::Float64 => {
                        let value_builder = Float64Builder::new();
                        let mut list_builder = ListBuilder::new(value_builder)
                            .with_field(Field::new_list_field(DataType::Float64, false));
                        for value in json_values {
                            if let Value::Object(map) = value
                                && let Some(Value::Array(val)) = map.get(field.name())
                            {
                                let mut values = Vec::new();
                                for v in val {
                                    if let Value::Number(num) = v {
                                        values.push(num.as_f64().unwrap());
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
