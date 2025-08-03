use arrow::record_batch::RecordBatch;

use super::data_operator::DataOperatorTrait;
use anyhow::Result;
use candle_core::Device;
use phymes_ml::openai_asset::{chat_completion, types};
use std::collections::HashMap;

/// Compute the relative similarity between two [RecordBatch]es where each [RecordBatch] represents a list of vector embeddings
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
        HumanInTheLoop
    }
    fn get_description() -> String {
        "Ask a question to clarify the user's query, ask a questionn to get additional information that the user did not provide, confirm a choice of tool, confirm arguments for a tool before answering the user's query or calling a tool, or provide the answer to the user's query.".to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_args".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The question or answer for the user. Format lhs_arg value as JSON according to the schema {\"role\": \"assistant\", \"content\": \"`RESPONSE`\"} where `RESPONSE` is where you put your question or answer for the user".to_string()),
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
        Ok(lhs_args.first().unwrap().clone())
    }
}
