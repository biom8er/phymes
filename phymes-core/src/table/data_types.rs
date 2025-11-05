use std::{str::FromStr, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    compute::kernels::cast_utils::Parser,
    datatypes::{
        DataType, Field, Float32Type, Float64Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    },
};
use serde_json::Value;

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
        _ => {
            return Err(anyhow!(
                "Unrecognized data type {data_type} available options are {}, {}, {}, {}, {}, {}, {}, {}, {}, and FixedSizeList- or List- with primitive types.",
                DataType::UInt8,
                DataType::UInt16,
                DataType::UInt32,
                DataType::Int64,
                DataType::Float32,
                DataType::Float64,
                DataType::Utf8,
                DataType::Null,
                DataType::Boolean
            ));
        }
    };
    Ok(data_type)
}

/// Helper function to parse a [String] into a [Value] based on the [DataType]
///
/// # Notes
/// * Nested types (i.e., `List` and `FixedSizeList`) must be serialized using [serde_json]
///   for the parsing and deserialization to work as expected!
pub fn parse_str_to_data_type(data: &str, data_type: &DataType) -> Result<Value> {
    let parsed = match data_type {
        DataType::UInt8 => match UInt8Type::parse(data) {
            Some(p) => Value::from(p),
            None => return Err(anyhow!("Cannot parse {data} to data type {data_type}.")),
        },
        DataType::UInt16 => match UInt16Type::parse(data) {
            Some(p) => Value::from(p),
            None => return Err(anyhow!("Cannot parse {data} to data type {data_type}.")),
        },
        DataType::UInt32 => match UInt32Type::parse(data) {
            Some(p) => Value::from(p),
            None => return Err(anyhow!("Cannot parse {data} to data type {data_type}.")),
        },
        DataType::Int64 => match Int64Type::parse(data) {
            Some(p) => Value::from(p),
            None => return Err(anyhow!("Cannot parse {data} to data type {data_type}.")),
        },
        DataType::Float32 => match Float32Type::parse(data) {
            Some(p) => Value::from(p),
            None => return Err(anyhow!("Cannot parse {data} to data type {data_type}.")),
        },
        DataType::Float64 => match Float64Type::parse(data) {
            Some(p) => Value::from(p),
            None => return Err(anyhow!("Cannot parse {data} to data type {data_type}.")),
        },
        DataType::Utf8 => Value::String(data.to_string()),
        DataType::Null => Value::Null,
        DataType::Boolean => Value::Bool(FromStr::from_str(data)?),
        DataType::List(_) | DataType::FixedSizeList(_, _) => serde_json::from_str::<Value>(data)?,
        _ => {
            return Err(anyhow!(
                "Unsupported data type {data_type} for String parsing. Supported data types are {}, {}, {}, {}, {}, {}, {}, {}, {}, and FixedSizeList- or List- with primitive types.",
                DataType::UInt8,
                DataType::UInt16,
                DataType::UInt32,
                DataType::Int64,
                DataType::Float32,
                DataType::Float64,
                DataType::Utf8,
                DataType::Null,
                DataType::Boolean
            ));
        }
    };
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_str_to_data_type() -> Result<()> {
        let value = parse_str_to_data_type("1", &DataType::UInt8)?;
        assert_eq!(value.as_u64().unwrap(), 1);
        let value = parse_str_to_data_type("1", &DataType::UInt16)?;
        assert_eq!(value.as_u64().unwrap(), 1);
        let value = parse_str_to_data_type("1", &DataType::UInt32)?;
        assert_eq!(value.as_u64().unwrap(), 1);
        let value = parse_str_to_data_type("1", &DataType::Int64)?;
        assert_eq!(value.as_i64().unwrap(), 1);
        let value = parse_str_to_data_type("1", &DataType::Float32)?;
        assert_eq!(value.as_f64().unwrap(), 1.0);
        let value = parse_str_to_data_type("1", &DataType::Float64)?;
        assert_eq!(value.as_f64().unwrap(), 1.0);
        let value = parse_str_to_data_type("1", &DataType::Utf8)?;
        assert_eq!(value.as_str().unwrap(), "1");
        let value = parse_str_to_data_type("true", &DataType::Boolean)?;
        assert!(value.as_bool().unwrap());
        let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
        let value = parse_str_to_data_type("[\"1\",\"2\",\"3\"]", &list_data_type)?;
        assert_eq!(value.as_array().unwrap(), &["1", "2", "3"]);
        let value = parse_str_to_data_type("[1,2,3]", &list_data_type)?;
        assert_eq!(value.as_array().unwrap(), &[1, 2, 3]);
        Ok(())
    }
}
