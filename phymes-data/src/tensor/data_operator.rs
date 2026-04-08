use anyhow::Result;
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::MappableTrait;
use std::fmt::Debug;

use crate::DataConfig;

/// Traits for methods that can be called as tools
pub trait ToolTrait: Debug + Default {
    /// The description to use for the operation
    fn get_description(&self) -> String;

    /// The description to use for the operation
    fn to_json_tool_schema(&self) -> String;
}

/// Data operators and other tools that utilize tensor services
pub trait DataOperatorTrait: MappableTrait + Send + Sync + Debug {
    /// Create a new instance of the data operator
    /// with the given [DataConfig].
    ///
    /// # Arguments
    /// * `config` - [DataConfig] struct describing the input parameters
    ///   including an optional `ops_kwargs` JSON string with keyword arguments
    fn new(config: &DataConfig) -> Result<Self>
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
}
