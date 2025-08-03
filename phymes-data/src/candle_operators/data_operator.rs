use anyhow::Result;
use arrow::{
    array::RecordBatch,
    datatypes::{Field, SchemaRef},
};
use candle_core::Device;
use std::fmt::Debug;

/// Data operators and other tools that utilize tensor services
pub trait DataOperatorTrait: Send + Sync + Debug {
    /// Short name for the DataOperator, such as 'AddRows'.
    /// Like [`get_name`](DataOperator::get_name) but can be called without an instance.
    fn get_static_name() -> &'static str
    where
        Self: Sized,
    {
        let full_name = std::any::type_name::<Self>();
        let maybe_start_idx = full_name.rfind(':');
        match maybe_start_idx {
            Some(start_idx) => &full_name[start_idx + 1..],
            None => "UNKNOWN",
        }
    }

    /// The user defined name of the DataOperator
    fn get_name() -> String
    where
        Self: Sized,
    {
        Self::get_static_name().to_string()
    }

    /// Create a new instance of the data operator
    /// with the given keyword arguments.
    ///
    /// # Arguments
    /// * `lhs_pk` - Primary Key for the LHS table
    /// * `lhs_fk` - Foreign Key for the LHS table
    /// * `lhs_values` - Values column(s) for the LHS table
    ///                  Either a string or a JSON list of strings
    /// * `kwargs` - Optional JSON string with keyword arguments
    fn new(
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_values: &str,
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        rhs_values: Option<&str>,
        kwargs: Option<&str>,
    ) -> Self
    where
        Self: Sized;

    /// Run the data operator in the backward direction
    // fn backward(&self, data: &[RecordBatch]) -> Result<(RecordBatch, Option<RecordBatch>)>;
    /// Run the data operator in the forward direction
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch>;

    /// Get the mandatory fields that are expected to be found in the LHS input schema
    fn get_schema_lhs_input(
        &self,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef>;

    /// Get the mandatory fields that are expected to be found in the RHS input schema
    fn get_schema_rhs_input(
        &self,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef>;

    /// Get the mandatory fields that are expected to be found in the output schema
    fn get_schema_output(
        &self,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef>;

    /// Check the expected mandatory fields for the LHS input
    fn check_schema_lhs_input(&self, other: SchemaRef) -> Result<Option<bool>>;

    /// Check the expected mandatory fields for the RHS input
    fn check_schema_rhs_input(&self, other: SchemaRef) -> Result<Option<bool>>;

    /// Check the expected mandatory fields for the output
    fn check_schema_output(&self, other: SchemaRef) -> Result<Option<bool>>;

    /// The description to use for the operation
    fn get_description() -> String
    where
        Self: Sized;

    /// The description to use for the operation
    fn get_json_tool_schema() -> String
    where
        Self: Sized;
}
