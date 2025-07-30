use arrow::{array::RecordBatch, datatypes::{Field, SchemaRef}};
use anyhow::Result;
use phymes_core::session::common_traits::MappableTrait;
use std::fmt::Debug;

/// Data operators and other tools that utilize tensor services
pub trait DataOperatorTrait: MappableTrait + Send + Sync + Debug {
    /// Create a new instance of the data operator
    /// with the given keyword arguments.
    /// 
    /// # Arguments
    /// * `kwargs` - Optional JSON string with keyword arguments
    fn new(kwargs: Option<&str>) -> Self;

    /// Run the data operator in the backward direction
    // fn backward(&self, data: &[RecordBatch]) -> Result<(RecordBatch, Option<RecordBatch>)>;
    /// Run the data operator in the forward direction
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        lhs_args: &[RecordBatch], 
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        rhs_value: Option<&str>,
        rhs_args: Option<&[RecordBatch]>
    ) -> Result<RecordBatch>;

    /// Get the mandatory fields that are expected to be found in the LHS input schema
    fn get_schema_lhs_input(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef>;    

    /// Get the mandatory fields that are expected to be found in the RHS input schema
    fn get_schema_rhs_input(
        &self,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef>;

    /// Get the mandatory fields that are expected to be found in the output schema
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn get_schema_output(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef>;

    /// Check the expected mandatory fields for the LHS input
    #[allow(unused_variables)]
    fn check_schema_lhs_input(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>>;    

    /// Check the expected mandatory fields for the RHS input
    #[allow(unused_variables)]
    fn check_schema_rhs_input(
        &self,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>>;    

    /// Check the expected mandatory fields for the output
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn check_schema_output(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>>;    

    /// The description to use for the operation
    fn get_description(&self) -> &str;    

    /// The description to use for the operation
    fn get_json_tool_schema(&self) -> String;

}