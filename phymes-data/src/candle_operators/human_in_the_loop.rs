use arrow::record_batch::RecordBatch;

use super::data_operator::DataOperatorTrait;
use anyhow::Result;
use candle_core::Device;
use phymes_core::{schemas::{chat_completion, message_history::{create_messages_record_batch, create_timestamp_micros}, types}, session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};
use std::collections::HashMap;

/// Redirect a tool call to the user for intervention
#[derive(Debug)]
pub struct HumanInTheLoop;

impl DataOperatorTrait for HumanInTheLoop {
    fn get_static_name() -> &'static str {
        "human-in-the-loop"
    }
    fn new(
        _lhs_pk: &str,
        _lhs_fk: &str,
        _lhs_values: &str,
        _rhs_pk: Option<&str>,
        _rhs_fk: Option<&str>,
        _rhs_values: Option<&str>,
        _kwargs: Option<&str>,
    ) -> Self {
        HumanInTheLoop {}
    }
    fn get_description() -> String {
        "The response to the user.".to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_args".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("Format lhs_arg value as JSON according to the schema {\"content\": \"`RESPONSE`\"} where `RESPONSE` is where you put your response for the user".to_string()),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_name(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec!["lhs_args".to_string()]),
            },
        };
        let tool = chat_completion::Tool {
            r#type: chat_completion::ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        let content = ArrowTable::get_builder()
            .with_record_batches(lhs_args.to_vec())?
            .with_name("")
            .build()?
            .get_column_as_vec_str("content")
            .first()
            .unwrap()
            .to_string();
        create_messages_record_batch(vec!["assistant".to_string()], vec![content], vec![create_timestamp_micros()])
    }
}