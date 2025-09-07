use arrow::record_batch::RecordBatch;

use crate::candle_operators::data_operator::make_error_record_batch;

use super::data_operator::DataOperatorTrait;
use anyhow::Result;
use candle_core::Device;
use phymes_core::{
    schemas::{
        chat_completion,
        available_subjects::create_timestamp_micros,
        chat::create_chat_record_batch,
        types,
    },
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::table::{Table, TableBuilderTrait, TableTrait},
};
use std::collections::HashMap;

/// Redirect a tool call to the user for intervention
#[derive(Debug)]
pub struct HumanInTheLoop;

impl MappableTrait for HumanInTheLoop {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for HumanInTheLoop {
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
                description: Some("Format lhs_args value according to the schema {\"content\": \"`RESPONSE`\"} where `RESPONSE` is where you put your response for the user".to_string()),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_static_name().to_string(),
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
        match create_hitl_record_batch(lhs_args) {
            Ok(batch) => Ok(batch),
            Err(err) => Ok(make_error_record_batch(err.to_string().as_str())),
        }
    }
}

fn create_hitl_record_batch(lhs_args: &[RecordBatch]) -> Result<RecordBatch> {
    let content = Table::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?
        .get_column_as_vec_str("content")
        .first()
        .unwrap()
        .to_string();
    create_chat_record_batch(
        vec!["assistant".to_string()],
        vec![content],
        vec![create_timestamp_micros()],
    )
}
