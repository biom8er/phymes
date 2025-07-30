use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray}, datatypes::{DataType, Field, Schema, SchemaRef}, record_batch::RecordBatch
};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use phymes_core::session::common_traits::MappableTrait;
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;
use super::data_operator::DataOperatorTrait;
use phymes_ai::openai_asset::{chat_completion, types};

/// Compute the relative similarity between two [RecordBatch]es where each [RecordBatch] represents a list of vector embeddings
#[derive(Debug)]
pub struct HumanInTheLoop;

impl MappableTrait for HumanInTheLoop {
    fn get_name(&self) -> &str {
        "human-in-the-loop"
    }
}

impl DataOperatorTrait for HumanInTheLoop {
    fn new(_kwargs: Option<&str>) -> Self {
        HumanInTheLoop
    }
    fn get_description(&self) -> &str {
        "Ask a question to clarify the user's query, ask a questionn to get additional information that the user did not provide, confirm a choice of tool, confirm arguments for a tool before answering the user's query or calling a tool, or provide the answer to the user's query."
    }
    fn get_schema_lhs_input(
        &self,
        _lhs_pk: &str,
        _lhs_fk: &str,
        _lhs_value: &str,
        _list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {        
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let mut fields = vec![role, content];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn get_schema_rhs_input(
        &self,
        _rhs_pk: &str,
        _rhs_fk: &str,
        _rhs_values: &str,
        _list_size: Option<usize>,
        _other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        None
    }
    fn get_schema_output(
        &self,
        _lhs_pk: &str,
        _lhs_fk: &str,
        _lhs_value: &str,
        _rhs_pk: &str,
        _rhs_fk: &str,
        _rhs_values: &str,
        _list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let mut fields = vec![role, content];
        if let Some(other) = other {
            fields.extend(other);
        }
        Some(Arc::new(Schema::new(fields)))
    }
    fn check_schema_lhs_input(
        &self,
        _lhs_pk: &str,
        _lhs_fk: &str,
        _lhs_value: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> {
        if other.column_with_name("role").is_none() {
            return Err(anyhow!("LHS input is missing column for role."));
        }
        if other.column_with_name("content").is_none() {
            return Err(anyhow!("LHS input is missing column for content."));
        }
        Ok(Some(true))
    }
    fn check_schema_rhs_input(
        &self,
        _rhs_pk: &str,
        _rhs_fk: &str,
        _rhs_values: &str,
        _other: SchemaRef,
    ) -> Result<Option<bool>> {
        Ok(None)
    }
    fn check_schema_output(
        &self,
        _lhs_pk: &str,
        _lhs_fk: &str,
        _lhs_value: &str,
        _rhs_pk: &str,
        _rhs_fk: &str,
        _rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> {    
        if other.column_with_name("role").is_none() {
            return Err(anyhow!("Output is missing column for role."));
        }
        if other.column_with_name("content").is_none() {
            return Err(anyhow!("Output is missing column for content."));
        }
        Ok(Some(true))
    }
    fn get_json_tool_schema(&self) -> String {        
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
            name: self.get_name().to_string(),
            description: Some(self.get_description().to_string()),
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
    fn forward(&self, 
        _lhs_pk: &str,
        _lhs_fk: &str,
        _lhs_value: &str,
        lhs_args: &[RecordBatch], 
        _rhs_pk: Option<&str>,
        _rhs_fk: Option<&str>,
        _rhs_value: Option<&str>,
        _rhs_args: Option<&[RecordBatch]>,
        _device: &Device
    ) -> Result<RecordBatch> {
        Ok(lhs_args.first().unwrap().clone())
    } 
}