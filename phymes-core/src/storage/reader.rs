use anyhow::Result;
use arrow::{array::RecordBatch, csv::reader::Format, datatypes::{DataType, Field, Schema, SchemaRef}, ipc::reader::StreamReader, json::{Reader, ReaderBuilder, reader::infer_json_schema}};
use bytes::Bytes;
use futures::Stream;
use object_store::{GetResult, ObjectStore, ObjectStoreExt, path::Path};
use std::{
    io::{BufRead, Cursor, Read, Seek},
    pin::Pin,
    sync::Arc,
};

/// Get the results from object storage
pub fn storage_reader_get_result<'a>(store: &'a Arc<dyn ObjectStore>, path: &'a Path,
) -> Pin<Box<dyn Future<Output = Result<GetResult, object_store::Error>> + Send + 'a>> {
    Box::pin(store.get(path))
}

/// Stream the results from object storage after polling `get_result`
pub fn storage_reader_stream_result(result: GetResult,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, object_store::Error>> + Send>> {
    Box::pin(result.into_stream())
}

/// Trait for reading from object storage
pub trait StorageReader {
    type SR;

    /// Build the [StorageReader] after polling `get_result` and `stream_result`
    fn new(reader: Self::SR) -> Self
    where
        Self: Sized;
}

pub trait StorageStreamReader<R> {
    fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>>;
}

/// Read IPC from storage
pub struct IpcReader<R> {
    reader: StreamReader<R>,
}

impl IpcReader<Cursor<Vec<u8>>> {
    pub fn new_with_bytes(bytes: &[u8], projection: Option<Vec<usize>>) -> Result<Self> {
        let cursor = Cursor::new(bytes.to_vec());
        let reader = StreamReader::try_new(cursor, projection)?;
        Ok(Self { reader })
    }

}

impl<R: Read> StorageStreamReader<R> for IpcReader<R> {
    fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

impl<R: Read> StorageReader for IpcReader<R> {
    type SR = StreamReader<R>;
    
    fn new(reader: Self::SR) -> Self {
        Self { reader }
    }
}

/// Read JSON from storage
pub struct JsonReader<R> {
    reader: Reader<R>,
}

impl JsonReader<Cursor<Vec<u8>>> {
    pub fn new_with_bytes(bytes: &[u8], batch_size: usize, schema: Option<SchemaRef>) -> Result<Self> {
        let mut cursor = Cursor::new(bytes.to_vec());

        // Infer the schema if not already provided
        let schema = match schema {
            Some(schema) => schema,
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
                Arc::new(Schema::new(fields))
            }
        };

        // Read in the batches
        let reader = ReaderBuilder::new(schema)
            .with_batch_size(batch_size)
            .build(cursor)?;
        
        Ok(Self { reader })
    }

}

impl<R: Read + BufRead> StorageStreamReader<R> for JsonReader<R> {
    fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

impl<R: Read> StorageReader for JsonReader<R> {
    type SR = Reader<R>;
    
    fn new(reader: Self::SR) -> Self {
        Self { reader }
    }
}

/// Read Csv from storage
pub struct CsvReader<R> {
    reader: arrow::csv::reader::BufReader<std::io::BufReader<R>>,
}

impl CsvReader<Cursor<Vec<u8>>> {
    pub fn new_with_bytes(bytes: &[u8], header: bool, delimiter: u8, batch_size: usize, schema: Option<SchemaRef>) -> Result<Self> {
        let mut cursor = Cursor::new(bytes.to_vec());

        // Handle the case of no schema
        let format = Format::default()
            .with_header(header)
            .with_delimiter(delimiter);
        let schema = match schema {
            Some(schema) => schema,
            None => {
                let (schema, _) = format.infer_schema(&mut cursor, None)?;
                cursor.rewind().unwrap();
                Arc::new(schema)
            }
        };

        // Read in the file
        let reader = arrow::csv::ReaderBuilder::new(schema)
            .with_batch_size(batch_size)
            .with_format(format)
            .build(cursor)?;
        
        Ok(Self { reader })
    }

}

impl<R: Read + BufRead> StorageStreamReader<R> for CsvReader<R> {
    fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

impl<R: Read> StorageReader for CsvReader<R> {
    type SR = arrow::csv::reader::BufReader<std::io::BufReader<R>>;
    
    fn new(reader: Self::SR) -> Self {
        Self { reader }
    }
}