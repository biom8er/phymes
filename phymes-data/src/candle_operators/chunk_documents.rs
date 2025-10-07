use arrow::{
    array::{ArrayRef, Float32Array, StringArray, UInt32Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use candle_core::Device;
use phymes_core::{
    schemas::{chat_completion, types},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::table_trait::{Table, TableBuilderTrait, TableTrait},
};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

use crate::{candle_data::data_config::DataConfig, candle_operators::data_operator::DataOperatorTrait};

/// Chunk documents by splitting a StringArray column in a [RecordBatch] into multiple rows based on a defined criteria
#[derive(Debug)]
pub struct ChunkDocuments {
    lhs_pk: String,
    lhs_values: String,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl MappableTrait for ChunkDocuments {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for ChunkDocuments {
    fn new(config: &DataConfig) -> Self {
        let lhs_pk = config.lhs_pk.to_owned();
        let lhs_values = config.lhs_values.first().unwrap().to_string();
        let chunk_size = config.chunk_size.unwrap_or(512);
        let chunk_overlap = config.chunk_overlap.unwrap_or(64);

        ChunkDocuments {
            lhs_pk,
            lhs_values,
            chunk_size,
            chunk_overlap,
        }
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        chunk_documents(
            &self.lhs_pk,
            &self.lhs_values,
            lhs_args,
            self.chunk_size,
            self.chunk_overlap,
            device,
        )
    }
    fn get_description() -> String {
        "Chunk documents by splitting the document text".to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_pk".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "The primary key column identifier for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::Array),
                description: Some(
                    "A list of value column identifiers for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_static_name().to_string(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_pk".to_string(),
                    "lhs_values".to_string(),
                ]),
            },
        };
        let tool = chat_completion::Tool {
            r#type: chat_completion::ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

/**
Chunk documents by splitting a StringArray column in a [RecordBatch]
  into multiple rows based on a defined criteria

# Notes
* inspired by <https://python.langchain.com/api_reference/_modules/langchain_text_splitters/character.html#RecursiveCharacterTextSplitter>
* inspired by <https://python.langchain.com/api_reference/_modules/langchain_text_splitters/character.html#CharacterTextSplitter>
* A column named `chunk_id` of type UInt32 is added to ensure uniqueness with lhs_pk and chunk_id

# Arguments

* `lhs` - `RecordBatch`
* `lhs_pk` - Left hand side primary key
* `lhs_value` - Left hand value key
* `chunk_size` - the length of the document chunks
* `chunk_overlap` - the length of overlap between document chunks
* `device` - The compute device

*/
#[instrument(skip(lhs_pk, lhs_values, lhs_args, chunk_size, chunk_overlap, _device))]
pub fn chunk_documents(
    lhs_pk: &str,
    lhs_values: &str,
    lhs_args: &[RecordBatch],
    chunk_size: usize,
    chunk_overlap: usize,
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs into an ArrowTable
    let lhs_table = Table::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?;

    // Extract out the document text
    let text = lhs_table
        .get_record_batches()
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(lhs_values)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .map(|s| chunk_str(s, chunk_size, chunk_overlap))
        .collect::<Vec<_>>();

    // Wrap the output into a record batch
    let mut batch_vec = Vec::new();

    // Extract the rest of the columns according to type
    // Create new columns expanding when text vec size > 1
    // Ignore the lhs_values and chunk_id columns
    let columns: Vec<String> = lhs_table
        .get_schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != lhs_values) & (field.name() != "chunk_id") {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_ref: ArrayRef = match lhs_table.get_column_data_type(column)? {
            DataType::UInt8 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::UInt8Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::UInt8Array::from(array_vec))
            }
            DataType::UInt16 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::UInt16Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::UInt16Array::from(array_vec))
            }
            DataType::UInt32 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<UInt32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(UInt32Array::from(array_vec))
            }
            DataType::UInt64 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::UInt64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::UInt64Array::from(array_vec))
            }
            DataType::Int8 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::Int8Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::Int8Array::from(array_vec))
            }
            DataType::Int16 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::Int16Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::Int16Array::from(array_vec))
            }
            DataType::Int32 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::Int32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::Int32Array::from(array_vec))
            }
            DataType::Int64 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::Int64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::Int64Array::from(array_vec))
            }
            DataType::Float32 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(Float32Array::from(array_vec))
            }
            DataType::Float64 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<arrow::array::Float64Array>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(arrow::array::Float64Array::from(array_vec))
            }
            DataType::Utf8 => {
                let array_vec = lhs_table
                    .get_record_batches()
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column_by_name(column)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                    .enumerate()
                    .flat_map(|(index, s)| {
                        let mut ar = Vec::new();
                        (0..text.get(index).unwrap().len()).for_each(|_i| ar.push(s.to_string()));
                        ar
                    })
                    .collect::<Vec<_>>();
                Arc::new(StringArray::from(array_vec))
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported data type for column {}: {}",
                    column,
                    lhs_table.get_column_data_type(column)?.to_string()
                ));
            }
        };
        batch_vec.push((column, array_ref));
    }

    // Migrate the primary key to the chunk_id
    let array_vec = lhs_args
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(lhs_pk)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .enumerate()
        .flat_map(|(index, s)| {
            let mut ar = Vec::new();
            (0..text.get(index).unwrap().len()).for_each(|i| {
                let s_new = format!("{s}_{i}");
                ar.push(s_new)
            });
            ar
        })
        .collect::<Vec<_>>();
    let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
    let chunk_id = "chunk_id".to_string();
    batch_vec.insert(0, (&chunk_id, array_ref));

    // flatten the text column
    let array_ref: ArrayRef = Arc::new(StringArray::from(
        text.into_iter().flatten().collect::<Vec<_>>(),
    ));
    let text_name = lhs_values.to_string();
    batch_vec.push((&text_name, array_ref));

    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

/// Chunk a string according to chunk size with overlap for the remainder
///
/// # Notes
/// * This function attempts to protect against non-character boundaries
///
/// # Arguments
/// * `text` - The text to chunk
/// * `chunk_size` - The size of each chunk
/// * `chunk_overlap` - The overlap between chunks
///
/// # Returns
/// A vector of strings representing the chunks
pub fn chunk_str(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut doc = text.to_string();
    while doc.len() > chunk_size {
        if let Some((s1, s2)) = doc.split_at_checked(chunk_size) {
            chunks.push(s1.to_string());
            if doc.is_char_boundary(chunk_size - chunk_overlap) {
                doc = [
                    s1[chunk_size - chunk_overlap..chunk_size].to_string(),
                    s2[..].to_string(),
                ]
                .join("");
            } else {
                let chunk_size = chunk_size - 1;
                doc = [
                    s1[chunk_size - chunk_overlap..chunk_size].to_string(),
                    s2[..].to_string(),
                ]
                .join("");
            }
        } else {
            let chunk_size = chunk_size - 1;
            let (s1, s2) = doc.split_at_checked(chunk_size).unwrap();
            chunks.push(s1.to_string());
            if doc.is_char_boundary(chunk_size - chunk_overlap + 1) {
                doc = [
                    s1[chunk_size - chunk_overlap + 1..chunk_size].to_string(),
                    s2[..].to_string(),
                ]
                .join("");
            } else {
                doc = [
                    s1[chunk_size - chunk_overlap..chunk_size].to_string(),
                    s2[..].to_string(),
                ]
                .join("");
            }
        }
    }
    chunks.push(doc);
    chunks
}

#[cfg(test)]
mod tests {
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_chunk_str() {
        let text = "Per Martin-Löf";
        let chunk_size = 13;
        let chunk_overlap = 2;
        let chunks = chunk_str(text, chunk_size, chunk_overlap);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks.first().unwrap(), "Per Martin-L");
        assert_eq!(chunks.get(1).unwrap(), "Löf");
    }

    #[test]
    fn test_chunk_documents() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["01234597890123456789", "0123459789"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("text", lhs_text_array),
            ("metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["01234597890123456789", "0123459789"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("text", lhs_text_array),
            ("metadata", lhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Chunk the documents
        let result = chunk_documents("lhs_pk", "text", &[batch_1, batch_2], 10, 2, &device)?;

        let lhs_id = result
            .column_by_name("lhs_pk")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, vec!["0", "0", "0", "1", "2", "2", "2", "3"]);
        let metadata = result
            .column_by_name("metadata")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata, vec![1, 1, 1, 2, 3, 3, 3, 4]);
        let text = result
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            text,
            vec![
                "0123459789",
                "8901234567",
                "6789",
                "0123459789",
                "0123459789",
                "8901234567",
                "6789",
                "0123459789"
            ]
        );
        let chunk_id = result
            .column_by_name("chunk_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            chunk_id,
            vec!["0_0", "0_1", "0_2", "1_0", "2_0", "2_1", "2_2", "3_0"]
        );

        Ok(())
    }
}
