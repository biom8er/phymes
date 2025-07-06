use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, Float32Array, Int32Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use serde::{Deserialize, Serialize};

use crate::session::common_traits::MappableTrait;

use super::arrow_table::{ArrowTable, ArrowTableTrait};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum ArrowTablePublish {
    /// Push a new vector of record batches onto the table
    Extend { table_name: String },
    /// Push a new vector of record batches onto the table
    /// after joining the chunks along the named column
    ExtendChunks {
        table_name: String,
        col_name: String,
    },
    /// Replace the existing vector of record batches with a new one
    Replace { table_name: String },
    /// Replace only the last record batch
    ReplaceLast { table_name: String },
    /// No updates
    #[default]
    None,
    /// Custom update function
    Custom(String),
}

impl ArrowTablePublish {
    pub fn get_name(&self) -> String {
        match self {
            Self::Extend { table_name: tn } => format!("extend-{tn}"),
            Self::ExtendChunks {
                table_name: tn,
                col_name: cn,
            } => format!("extend-chunks-{tn}-{cn}"),
            Self::Replace { table_name: tn } => format!("replace-{tn}"),
            Self::ReplaceLast { table_name: tn } => format!("replace-last-{tn}"),
            Self::None => "none".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    pub fn get_table_name(&self) -> &str {
        match self {
            Self::Extend { table_name: tn } => tn,
            Self::ExtendChunks {
                table_name: tn,
                col_name: _cn,
            } => tn,
            Self::Replace { table_name: tn } => tn,
            Self::ReplaceLast { table_name: tn } => tn,
            Self::None => "",
            Self::Custom(_name) => "",
        }
    }
}

/// Update an arrow table with record batches coming from a new table
pub trait ArrowTableUpdateTrait: ArrowTableTrait {
    fn get_record_batches_mut(&mut self) -> &mut Vec<RecordBatch>;
    fn update_table(&mut self, new: Vec<RecordBatch>, update: ArrowTablePublish) -> Result<()>;
}

impl ArrowTableUpdateTrait for ArrowTable {
    fn get_record_batches_mut(&mut self) -> &mut Vec<RecordBatch> {
        &mut self.record_batches
    }
    fn update_table(&mut self, new: Vec<RecordBatch>, update: ArrowTablePublish) -> Result<()> {
        match update {
            ArrowTablePublish::Extend { table_name: tn } => {
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
            ArrowTablePublish::ExtendChunks {
                table_name: tn,
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
            ArrowTablePublish::Replace { table_name: tn } => {
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
            ArrowTablePublish::ReplaceLast { table_name: tn } => {
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
            ArrowTablePublish::None => Ok(()),
            ArrowTablePublish::Custom(_) => Ok(()),
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
            DataType::Int32 => {
                let array = column.as_any().downcast_ref::<Int32Array>().unwrap();
                array.value(0).to_string()
            }
            DataType::Float32 => {
                let array = column.as_any().downcast_ref::<Float32Array>().unwrap();
                array.value(0).to_string()
            }
            _ => return Err(anyhow!("Unsupported type")),
        };
        first_row.push(value);
    }
    Ok(first_row)
}

/// Create a new record batch from the first row
///   BUT replace the streamed chunks row
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
            DataType::Int32 => {
                let values = vec![first_row[i].parse::<i32>().unwrap()];
                Arc::new(Int32Array::from(values)) as ArrayRef
            }
            DataType::Float32 => {
                let values = vec![first_row[i].parse::<f32>().unwrap()];
                Arc::new(Float32Array::from(values)) as ArrayRef
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

    use crate::table::arrow_table::test_table::{make_test_table, make_test_table_chat};

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
    fn test_update_table_wrong_table_name() -> Result<()> {
        let mut old = make_test_table("test_table", 4, 0, 3)?;
        let new = make_test_table("test_table", 1, 0, 1)?;
        match old.update_table(
            new.clone().get_record_batches_own(),
            ArrowTablePublish::Extend {
                table_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        match old.update_table(
            new.clone().get_record_batches_own(),
            ArrowTablePublish::Replace {
                table_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        match old.update_table(
            new.clone().get_record_batches_own(),
            ArrowTablePublish::ReplaceLast {
                table_name: "missing".to_string(),
            },
        ) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between table name test_table and update table target missing."
            ),
        }
        match old.update_table(
            new.clone().get_record_batches_own(),
            ArrowTablePublish::ExtendChunks {
                table_name: "missing".to_string(),
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
    fn test_extend_update() -> Result<()> {
        let mut old = make_test_table("test_table", 4, 0, 3)?;
        let new = make_test_table("test_table", 1, 0, 1)?;
        old.update_table(
            new.get_record_batches_own(),
            ArrowTablePublish::Extend {
                table_name: "test_table".to_string(),
            },
        )?;
        assert_eq!(old.count_rows(), 13);
        Ok(())
    }

    #[test]
    fn test_replace_update() -> Result<()> {
        let mut old = make_test_table("test_table", 4, 0, 3)?;
        let new = make_test_table("test_table", 1, 0, 1)?;
        old.update_table(
            new.get_record_batches_own(),
            ArrowTablePublish::Replace {
                table_name: "test_table".to_string(),
            },
        )?;
        assert_eq!(old.count_rows(), 1);
        Ok(())
    }

    #[test]
    fn test_none_update() -> Result<()> {
        let mut old = make_test_table("test_table", 4, 0, 3)?;
        let new = make_test_table("test_table", 1, 0, 1)?;
        old.update_table(new.get_record_batches_own(), ArrowTablePublish::None)?;
        assert_eq!(old.count_rows(), 12);
        Ok(())
    }

    #[test]
    fn test_extend_chunks_update() -> Result<()> {
        let mut old = make_test_table_chat("messages")?;
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
        old.update_table(
            vec![new_1, new_2],
            ArrowTablePublish::ExtendChunks {
                table_name: "messages".to_string(),
                col_name: "content".to_string(),
            },
        )?;
        assert_eq!(old.count_rows(), 5);
        assert_eq!(
            old.get_column_as_str_vec("role"),
            ["user", "assistant", "user", "assistant", "assistant"]
        );
        assert_eq!(
            old.get_column_as_str_vec("content"),
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
}
