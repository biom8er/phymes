use anyhow::Result;
use arrow::{
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};

use crate::{Table, TableBuilder};

/// Todo
/// - initialize an `object_store_metadata` subject for each subject backed by object storage
/// - initialize an `index_` subject for each subject that converts non-gpu compatible types to gpu compatible types
/// - add a `constraints` and `failed_constraints` subjects per session to track single/multi-table constraints and record any constraint violations

/// Convert a possible nested Json-like structure into a single [RecordBatch]
pub trait JsonSchemaTrait {
    fn to_record_batch(self, publisher: &str) -> Result<RecordBatch>;
}

/// Materialize the [Schema] for the object
pub trait AvailableSchemaTrait {
    fn to_schema(&self) -> SchemaRef;
}

/// Materialize the [Table] or [TableBuilder] for building the table for the object
pub trait AvailableSubjectsTrait: AvailableSchemaTrait {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Table>;
    fn to_table_builder(&self, name: Option<&str>) -> TableBuilder;
}

