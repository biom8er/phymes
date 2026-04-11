use arrow::record_batch::RecordBatch;
use phymes_diagnostics::create_timestamp_micros;

use crate::{DataConfig, DataOperatorTrait, ToolTrait};

use anyhow::Result;
use candle_core::Device;
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType,
    create_chat_record_batch,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Redirect a tool call to the user for intervention
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct HumanInTheLoop;

impl MappableTrait for HumanInTheLoop {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for HumanInTheLoop {
    fn get_description(&self) -> String {
        "The response to the user.".to_string()
    }
    fn to_json_tool_schema(&self) -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_args".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("Format lhs_args value according to the schema {\"content\": \"`RESPONSE`\"} where `RESPONSE` is where you put your response for the user".to_string()),
                ..Default::default()
            }),
        );
        let function = Function {
            name: Self::get_static_name().to_string(),
            description: Some(self.get_description()),
            parameters: FunctionParameters {
                schema_type: JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec!["lhs_args".to_string()]),
            },
        };
        let tool = Tool {
            r#type: ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

impl DataOperatorTrait for HumanInTheLoop {
    fn new(_config: &DataConfig) -> Result<Self> {
        Ok(HumanInTheLoop {})
    }
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device,
    ) -> Result<RecordBatch> {
        create_hitl_record_batch(lhs_args)
    }
}

fn create_hitl_record_batch(lhs_args: &[RecordBatch]) -> Result<RecordBatch> {
    let content = Subject::get_builder()
        .with_name("create_hitl_record_batch")
        .with_record_batches(lhs_args.to_vec())?
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
