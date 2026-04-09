use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

pub fn btree_schema<K: Into<DataType>, V: Into<DataType>>(key_type: K, value_type: V) -> Schema {
    Schema::new(vec![
        Field::new("node_id", DataType::UInt64, false),
        Field::new("is_leaf", DataType::Boolean, false),
        Field::new(
            "keys",
            DataType::List(Arc::new(Field::new("item", key_type.into(), false))),
            false,
        ),
        Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", value_type.into(), true))),
            true,
        ),
        Field::new(
            "children",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            true,
        ),
        Field::new("next_leaf", DataType::UInt64, true),
    ])
}

pub fn hash_index_schema<K: Into<DataType>, V: Into<DataType>>(
    key_type: K,
    value_type: V,
) -> Schema {
    Schema::new(vec![
        Field::new("bucket_id", DataType::UInt64, false),
        Field::new("hash", DataType::UInt64, false),
        Field::new("key", key_type.into(), false),
        Field::new("value", value_type.into(), false),
    ])
}

pub fn gist_schema<P: Into<DataType>, T: Into<DataType>>(
    predicate_type: P,
    tuple_type: T,
) -> Schema {
    Schema::new(vec![
        Field::new("node_id", DataType::UInt64, false),
        Field::new("is_leaf", DataType::Boolean, false),
        Field::new("predicate", predicate_type.into(), false),
        Field::new("child_id", DataType::UInt64, true),
        Field::new("tuple", tuple_type.into(), true),
    ])
}

pub fn spgist_schema<K: Into<DataType> + Clone, V: Into<DataType>>(
    key_type: K,
    value_type: V,
) -> Schema {
    Schema::new(vec![
        Field::new("node_id", DataType::UInt64, false),
        Field::new("is_leaf", DataType::Boolean, false),
        Field::new("label", key_type.clone().into(), true),
        Field::new(
            "child_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            true,
        ),
        Field::new("key", key_type.into(), true),
        Field::new("value", value_type.into(), true),
    ])
}

pub fn gin_schema<K: Into<DataType>>(key_type: K) -> Schema {
    Schema::new(vec![
        Field::new("key", key_type.into(), false),
        Field::new(
            "posting_list",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            false,
        ),
    ])
}

pub fn brin_schema<S: Into<DataType>>(summary_type: S) -> Schema {
    Schema::new(vec![
        Field::new("block_start", DataType::UInt64, false),
        Field::new("block_end", DataType::UInt64, false),
        Field::new("summary", summary_type.into(), false),
    ])
}
