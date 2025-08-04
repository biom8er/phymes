use arrow::{array::{ArrayRef, StringArray}, record_batch::RecordBatch};
use tracing::{event, Level};

use super::data_operator::DataOperatorTrait;
use anyhow::Result;
use candle_core::Device;
use phymes_ml::{candle_chat::message_history::create_timestamp, openai_asset::{chat_completion, types}};
use std::{collections::HashMap, sync::Arc};

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
        prepare_hitl_record_batch(&self.lhs_values)
    }
}

/// Wrap the lhs_values into a record batch according to the messages schema
fn prepare_hitl_record_batch(content: &str) -> Result<RecordBatch> {
    let role_arr: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
    let content_arr: ArrayRef = Arc::new(StringArray::from(vec![content.to_string()]));
    let timestamp_arr: ArrayRef = Arc::new(StringArray::from(vec![create_timestamp()]));
    let batch = RecordBatch::try_from_iter(vec![
        ("role", role_arr),
        ("content", content_arr),
        ("timestamp", timestamp_arr),
    ])?;
    event!(Level::DEBUG, "Messages joined: {:?}.", &batch);
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use phymes_core::{session::common_traits::{BuildableTrait, BuilderTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}};

    use super::*;

    #[test]
    fn test_prepare_hitl_record_batch() {
        let result = prepare_hitl_record_batch("").unwrap();
        let result_table = ArrowTable::get_builder()
            .with_record_batches(vec![result]).unwrap()
            .with_name("")
            .build().unwrap();
        assert_eq!(result_table.get_column_as_vec_str("role"), &["assistant"]);
        assert_eq!(result_table.get_column_as_vec_str("content"), &[""]);
    }
}