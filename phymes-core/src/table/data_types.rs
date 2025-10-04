use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow::datatypes::{DataType, Field};

/// Helper function to convert an arrow [DataType] to a [String]
pub fn from_data_type_to_str(data_type: &DataType) -> String {
    match data_type {
        DataType::FixedSizeList(f, s) => {
            format!("FixedSizeList-{}-{}", f.data_type(), s)
        }
        DataType::List(f) => {
            format!("List-{}", f.data_type())
        }
        _ => data_type.to_string(),
    }
}

/// Helper function to convert a [String] to an arrow [DataType]
pub fn from_str_to_data_type(data_type: &str) -> Result<DataType> {
    let data_type = match data_type {
        s if s == DataType::UInt8.to_string() => DataType::UInt8,
        s if s == DataType::UInt16.to_string() => DataType::UInt16,
        s if s == DataType::UInt32.to_string() => DataType::UInt32,
        s if s == DataType::Int64.to_string() => DataType::Int64,
        s if s == DataType::Float32.to_string() => DataType::Float32,
        s if s == DataType::Float64.to_string() => DataType::Float64,
        s if s == DataType::Utf8.to_string() => DataType::Utf8,
        s if s == DataType::Null.to_string() => DataType::Null,
        s if s == DataType::Boolean.to_string() => DataType::Boolean,
        s if s.contains("FixedSizeList-UInt8-") => {
            let size = data_type
                .split("FixedSizeList-UInt8-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::UInt8, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-UInt32-") => {
            let size = data_type
                .split("FixedSizeList-UInt32-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::UInt32, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Int64-") => {
            let size = data_type
                .split("FixedSizeList-Int64-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Int64, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Float32-") => {
            let size = data_type
                .split("FixedSizeList-Float32-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Float64-") => {
            let size = data_type
                .split("FixedSizeList-Float64-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float64, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Utf8-") => {
            let size = data_type
                .split("FixedSizeList-Utf8-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(Arc::new(Field::new_list_field(DataType::Utf8, false)), size)
        }
        s if s.contains("List-UInt8") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::UInt8, false)))
        }
        s if s.contains("List-UInt32") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)))
        }
        s if s.contains("List-Int64") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false)))
        }
        s if s.contains("List-Float32") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)))
        }
        s if s.contains("List-Float64") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, false)))
        }
        s if s.contains("List-Utf8") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)))
        }
        _ => return Err(anyhow!("Unrecognized data type {data_type}")),
    };
    Ok(data_type)
}