use arrow::record_batch::RecordBatch;

use super::data_operator::DataOperatorTrait;
use anyhow::Result;
use candle_core::Device;
use phymes_core::schemas::{chat_completion, message_history::{create_messages_record_batch, create_timestamp_micros}, types};
use std::collections::HashMap;

/// Redirect a tool call to the user for intervention
#[derive(Debug)]
pub struct HumanInTheLoop {
    lhs_values: String
}

impl DataOperatorTrait for HumanInTheLoop {
    fn get_static_name() -> &'static str {
        "human-in-the-loop"
    }
    fn new(
        _lhs_pk: &str,
        _lhs_fk: &str,
        lhs_values: &str,
        _rhs_pk: Option<&str>,
        _rhs_fk: Option<&str>,
        _rhs_values: Option<&str>,
        _kwargs: Option<&str>,
    ) -> Self {
        HumanInTheLoop { lhs_values: lhs_values.to_string()}
    }
    fn get_description() -> String {
        "1. Ask a question to clarify the query from the user if information is missing or if you are uncertain about your response; or 2. provide the answer to the user if you are certain about your response.".to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The content for the user".to_string()),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_name(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec!["lhs_values".to_string()]),
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
        _lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        create_messages_record_batch(vec!["assistant".to_string()], vec![self.lhs_values.to_string()], vec![create_timestamp_micros()])
    }
}