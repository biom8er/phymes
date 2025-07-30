use arrow::{array::RecordBatch, datatypes::{Field, SchemaRef}};
use anyhow::Result;
use candle_core::Device;
use phymes_core::session::common_traits::MappableTrait;
use std::fmt::Debug;

/// Data operators and other tools that utilize tensor services
pub trait DataOperatorTrait: MappableTrait + Send + Sync + Debug {
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

    /// Create a new instance of the data operator
    /// with the given keyword arguments.
    /// 
    /// # Arguments
    /// * `kwargs` - Optional JSON string with keyword arguments
    fn new(kwargs: Option<&str>) -> Self where Self: Sized;

    /// Run the data operator in the backward direction
    // fn backward(&self, data: &[RecordBatch]) -> Result<(RecordBatch, Option<RecordBatch>)>;
    /// Run the data operator in the forward direction
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, 
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        lhs_args: &[RecordBatch], 
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        rhs_value: Option<&str>,
        rhs_args: Option<&[RecordBatch]>,
        device: &Device
    ) -> Result<RecordBatch>;

    /// Get the mandatory fields that are expected to be found in the LHS input schema
    fn get_schema_lhs_input(
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> where Self: Sized;    

    /// Get the mandatory fields that are expected to be found in the RHS input schema
    fn get_schema_rhs_input(
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> where Self: Sized;

    /// Get the mandatory fields that are expected to be found in the output schema
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn get_schema_output(
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> where Self: Sized;

    /// Check the expected mandatory fields for the LHS input
    #[allow(unused_variables)]
    fn check_schema_lhs_input(
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> where Self: Sized;    

    /// Check the expected mandatory fields for the RHS input
    #[allow(unused_variables)]
    fn check_schema_rhs_input(
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> where Self: Sized;    

    /// Check the expected mandatory fields for the output
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn check_schema_output(
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> where Self: Sized;    

    /// The description to use for the operation
    fn get_description() -> &str where Self: Sized;    

    /// The description to use for the operation
    fn get_json_tool_schema() -> String where Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compilation test to ensure that the `DataOperator::get_name()` method can
    /// be called from a trait object.
    #[allow(dead_code)]
    fn use_data_operator_name_as_trait_object(data_op: &dyn DataOperatorTrait) {
        let _ = data_op.get_name();
    }
}