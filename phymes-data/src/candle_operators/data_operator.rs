use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use candle_core::Device;
use std::{fmt::Debug, sync::Arc};

/// Helper function to create an error message that
/// can be sent back to a function-calling agent
pub fn make_error_record_batch(error: &str) -> RecordBatch {
    let error: ArrayRef = Arc::new(StringArray::from(vec![error]));
    RecordBatch::try_from_iter(vec![("error", error)]).unwrap()
}

/// Data operators and other tools that utilize tensor services
pub trait DataOperatorTrait: Send + Sync + Debug {
    /// Short name for the DataOperator, such as 'AddRows'.
    /// Like `get_name` but can be called without an instance.
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
    ///   Either a string or a JSON list of strings
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

    /// The description to use for the operation
    fn get_description() -> String
    where
        Self: Sized;

    /// The description to use for the operation
    fn get_json_tool_schema() -> String
    where
        Self: Sized;
}
