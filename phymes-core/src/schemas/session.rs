use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{
        ArrayRef, Int64Array, Int64Builder, ListBuilder, RecordBatch, StringArray, StringBuilder,
        UInt8Array, UInt32Array,
    },
    datatypes::{DataType, Field, Fields},
};

pub(crate) fn create_session_subject_schemas_fields() -> Fields {
    let field_names = ["session_name", "subject_name", "column_name", "type_name"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_subject_schemas_batch(
    session_names: Vec<String>,
    subject_names: Vec<String>,
    cols_names: Vec<String>,
    type_names: Vec<String>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let cols_names: ArrayRef = Arc::new(StringArray::from(cols_names));
    let type_names: ArrayRef = Arc::new(StringArray::from(type_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("subject_name", subject_names),
        ("column_name", cols_names),
        ("type_name", type_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_subject_stores_fields() -> Fields {
    let field_names = ["session_name", 
        "subject_name", 
        "location",
        "bucket",
        "metadata",
        ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_subject_stores_batch(
    session_names: Vec<String>,
    subject_names: Vec<String>,
    cols_names: Vec<String>,
    type_names: Vec<String>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let cols_names: ArrayRef = Arc::new(StringArray::from(cols_names));
    let type_names: ArrayRef = Arc::new(StringArray::from(type_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("subject_name", subject_names),
        ("column_name", cols_names),
        ("type_name", type_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_fields() -> Fields {
    let field_names = [
        "session_name",
        "task_name",
        "processor_name",
        "runtime_env_name",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_tasks_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
    processor_names: Vec<String>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
        ("processor_name", processor_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_processors_fields() -> Fields {
    let field_names = [
        "session_name",
        "processor_name",
        "processor_type",
        "publication_subscription_name",
        "publication_subscription_table_name",
        "subscribe_type",
        "update_type",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["is_subscription"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt8, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

#[allow(clippy::too_many_arguments)]
pub fn create_session_processors_batch(
    session_names: Vec<String>,
    processor_names: Vec<String>,
    processor_types: Vec<String>,
    pub_sub_name: Vec<String>,
    pub_sub_table_names: Vec<String>,
    subscribe_types: Vec<String>,
    update_types: Vec<String>,
    is_sub: Vec<u8>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
    let pub_sub_name: ArrayRef = Arc::new(StringArray::from(pub_sub_name));
    let pub_sub_table_names: ArrayRef = Arc::new(StringArray::from(pub_sub_table_names));
    let subscribe_types: ArrayRef = Arc::new(StringArray::from(subscribe_types));
    let update_types: ArrayRef = Arc::new(StringArray::from(update_types));
    let is_sub: ArrayRef = Arc::new(UInt8Array::from(is_sub));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("processor_name", processor_names),
        ("processor_type", processor_types),
        ("publication_subscription_name", pub_sub_name),
        ("publication_subscription_table_name", pub_sub_table_names),
        ("subscribe_type", subscribe_types),
        ("update_type", update_types),
        ("is_subscription", is_sub),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_runtime_envs_fields() -> Fields {
    let field_names = ["session_name", "runtime_env_name", 
        // "object_store_backend", 
        // "object_store_bucket", 
        "object_store_config", 
        "subject_folder_partitioning", 
        "subject_file_partitioning"
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["max_memory", "max_time", "max_steps", "max_tasks"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_session_runtime_envs_batch(
    session_names: Vec<String>,
    runtime_env_names: Vec<String>,
    object_store_backend: Vec<String>,
    object_store_bucket: Vec<String>,
    object_store_config: Vec<String>,
    subject_folder_partitioning: Vec<String>,
    subject_file_partitioning: Vec<String>,
    max_memory: Vec<u32>,
    max_time: Vec<u32>,
    max_steps: Vec<u32>,
    max_tasks: Vec<u32>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let runtime_env_names: ArrayRef = Arc::new(StringArray::from(runtime_env_names));
    let object_store_backend: ArrayRef = Arc::new(StringArray::from(object_store_backend));
    let object_store_bucket: ArrayRef = Arc::new(StringArray::from(object_store_bucket));
    let object_store_config: ArrayRef = Arc::new(StringArray::from(object_store_config));
    let subject_folder_partitioning: ArrayRef = Arc::new(StringArray::from(subject_folder_partitioning));
    let subject_file_partitioning: ArrayRef = Arc::new(StringArray::from(subject_file_partitioning));
    let max_memory: ArrayRef = Arc::new(UInt32Array::from(max_memory));
    let max_time: ArrayRef = Arc::new(UInt32Array::from(max_time));
    let max_steps: ArrayRef = Arc::new(UInt32Array::from(max_steps));
    let max_tasks: ArrayRef = Arc::new(UInt32Array::from(max_tasks));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("runtime_env_name", runtime_env_names),
        ("object_store_backend", object_store_backend),
        ("object_store_bucket", object_store_bucket),
        ("object_store_config", object_store_config),
        ("subject_folder_partitioning", subject_folder_partitioning),
        ("subject_file_partitioning", subject_file_partitioning),
        ("max_memory", max_memory),
        ("max_time", max_time),
        ("max_steps", max_steps),
        ("max_tasks", max_tasks),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_run_log_fields() -> Fields {
    let field_names = ["session_name", "task_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["superstep", "timestamp"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_session_tasks_run_log_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
    supersteps: Vec<i64>,
    timestamps: Vec<i64>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let supersteps: ArrayRef = Arc::new(Int64Array::from(supersteps));
    let timestamps: ArrayRef = Arc::new(Int64Array::from(timestamps));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
        ("superstep", supersteps),
        ("timestamp", timestamps),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_check_fields() -> Fields {
    let field_names = ["session_name", "task_name"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_tasks_check_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_subscribe_fields() -> Fields {
    let field_names = [
        "session_name",
        "task_name",
        "processor_name",
        "processor_type",
        "subscription_name",
        "subscription_table_name",
        // "subscribe_type",
        // "update_type",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_tasks_subscribe_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
    processor_names: Vec<String>,
    processor_types: Vec<String>,
    subscription_names: Vec<String>,
    subscription_table_name: Vec<String>,
    // subscribe_types: Vec<String>,
    // update_types: Vec<String>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
    let subscription_names: ArrayRef = Arc::new(StringArray::from(subscription_names));
    let subscription_table_name: ArrayRef = Arc::new(StringArray::from(subscription_table_name));
    // let subscribe_types: ArrayRef = Arc::new(StringArray::from(subscribe_types));
    // let update_types: ArrayRef = Arc::new(StringArray::from(update_types));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
        ("processor_name", processor_names),
        ("processor_type", processor_types),
        ("subscription_name", subscription_names),
        ("subscription_table_name", subscription_table_name),
        // ("subscribe_type", subscribe_types),
        // ("update_type", update_types),
    ])?;
    Ok(batch)
}

// DM: Only the subjects that are subscribed to should be here
pub(crate) fn create_session_tasks_subscribe_aggregate_fields() -> Fields {
    let field_names = [
        "session_name",
        "task_name",
        "processor_name",
        "processor_type",
        "subscribe_type-Last",
        "update_type-Last",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
    let field_names = ["subscription_name-List", "subscription_table_name-List"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>(),
    );
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false)));
    let field_names = ["superstep-List", "superstep-Max-List"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

#[allow(clippy::too_many_arguments)]
pub fn create_session_tasks_subscribe_aggregate_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
    processor_names: Vec<String>,
    processor_types: Vec<String>,
    subscribe_types: Vec<String>,
    update_types: Vec<String>,
    subscription_names: Vec<Vec<String>>,
    subscription_table_names: Vec<Vec<String>>,
    supersteps: Vec<Vec<i64>>,
    superstep_lasts: Vec<Vec<i64>>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
    let subscribe_types: ArrayRef = Arc::new(StringArray::from(subscribe_types));
    let update_types: ArrayRef = Arc::new(StringArray::from(update_types));
    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in subscription_names.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let subscription_names: ArrayRef = Arc::new(list_builder.finish());
    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in subscription_table_names.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let subscription_table_names: ArrayRef = Arc::new(list_builder.finish());

    let value_builder = Int64Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Int64, false));
    for values in supersteps.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let supersteps: ArrayRef = Arc::new(list_builder.finish());
    let value_builder = Int64Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Int64, false));
    for values in superstep_lasts.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let superstep_lasts: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
        ("processor_name", processor_names),
        ("processor_type", processor_types),
        ("subscribe_type-Last", subscribe_types),
        ("update_type-Last", update_types),
        ("subscription_name-List", subscription_names),
        ("subscription_table_name-List", subscription_table_names),
        ("superstep-List", supersteps),
        ("superstep-Max-List", superstep_lasts),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_publish_fields() -> Fields {
    let field_names = [
        "session_name",
        "task_name",
        "processor_name",
        "processor_type",
        "publication_names",
        "publication_table_names",
    ];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub(crate) fn create_session_tasks_publish_aggregate_fields() -> Fields {
    let field_names = ["session_name", "task_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
    let field_names = ["publication_names", "publication_table_names"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_session_tasks_publish_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
    processor_names: Vec<String>,
    processor_types: Vec<String>,
    publication_names: Vec<String>,
    publication_table_names: Vec<String>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
    let publication_names: ArrayRef = Arc::new(StringArray::from(publication_names));
    let publication_table_names: ArrayRef = Arc::new(StringArray::from(publication_table_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
        ("processor_name", processor_names),
        ("processor_type", processor_types),
        ("publication_names", publication_names),
        ("publication_table_names", publication_table_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_subscribe_publish_fields() -> Fields {
    let field_names = [
        "session_name",
        "task_name",
        "processor_name",
        "processor_type",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
    let field_names = [
        "subscription_names",
        "subscription_table_names",
        "publication_names",
        "publication_table_names",
    ];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>(),
    );
    // let list_data_type = DataType::List(
    //     Arc::new(Field::new_list_field(DataType::UInt8, false))
    // );
    // let field_names = ["is_subscription_updated"];
    // fields_vec.extend(
    //     field_names
    //         .iter()
    //         .map(|f| Field::new(*f, list_data_type.clone(), false))
    //         .collect::<Vec<_>>(),
    // );
    Fields::from(fields_vec)
}

#[allow(clippy::too_many_arguments)]
pub fn create_session_tasks_subscribe_publish_batch(
    session_names: Vec<String>,
    task_names: Vec<String>,
    processor_names: Vec<String>,
    processor_types: Vec<String>,
    subscription_names: Vec<Vec<String>>,
    subscription_table_names: Vec<Vec<String>>,
    publication_names: Vec<Vec<String>>,
    publication_table_names: Vec<Vec<String>>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in subscription_names.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let subscription_names: ArrayRef = Arc::new(list_builder.finish());
    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in subscription_table_names.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let subscription_table_names: ArrayRef = Arc::new(list_builder.finish());

    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in publication_names.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let publication_names: ArrayRef = Arc::new(list_builder.finish());
    let value_builder = StringBuilder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Utf8, false));
    for values in publication_table_names.into_iter() {
        for value in values.into_iter() {
            list_builder.values().append_value(&value);
        }
        list_builder.append(true);
    }
    let publication_table_names: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("task_name", task_names),
        ("processor_name", processor_names),
        ("processor_type", processor_types),
        ("subscription_names", subscription_names),
        ("subscription_table_names", subscription_table_names),
        ("publication_names", publication_names),
        ("publication_table_names", publication_table_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_supersteps_fields() -> Fields {
    let field_names = ["session_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["superstep"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_session_supersteps_batch(
    session_names: Vec<String>,
    supersteps: Vec<u32>,
) -> Result<RecordBatch> {
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let supersteps: ArrayRef = Arc::new(UInt32Array::from(supersteps));
    let batch = RecordBatch::try_from_iter(vec![
        ("session_name", session_names),
        ("superstep", supersteps),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_superstep_max_fields() -> Fields {
    let field_names = ["session_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["superstep-Max"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}
