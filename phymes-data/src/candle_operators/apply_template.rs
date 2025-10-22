use std::collections::HashMap;

use anyhow::Result;
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    schemas::{available_subjects::create_values_record_batch, chat_completion, types},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::{DataFormat, table_script::TableScript, table_trait::{Table, TableBuilderTrait, TableTrait}},
};
use serde_json::{json, Value};
use tracing::instrument;

use crate::{candle_data::{data_config::DataConfig, summary_processor::table_and_data_format_to_record_batch}, candle_operators::data_operator::DataOperatorTrait};

/// Inject a table into a string template
#[derive(Debug)]
pub struct ApplyTemplate {
    doc_template: String,
    doc_name: String,
    table_expression: String,
    doc_input: Value,
    format: DataFormat,
}

impl MappableTrait for ApplyTemplate {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for ApplyTemplate {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        apply_template(
            lhs_args,
            &self.doc_template,
            &self.doc_name,
            &self.table_expression,
            &self.doc_input,
            &self.format,
            device,
        )
    }
    fn new(config: &DataConfig) -> Self {
        let doc_template = config.doc_template.clone().unwrap_or_default();
        let doc_name = config.doc_name.clone().unwrap_or_default();
        let table_expression = config.table_expression.clone().unwrap_or_default();
        let doc_input = if let Some(doc_input) = config.doc_input.as_ref() {
            serde_json::from_str::<Value>(doc_input).unwrap_or_default()
        } else {
            Value::default()
        };
        let format = config.format.clone().unwrap_or_default();

        // Make the object
        ApplyTemplate {
            doc_template,
            doc_name,
            table_expression,
            doc_input,
            format,
        }
    }
    fn get_description() -> String {
        "Inject a table into a string template."
            .to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "op_kwargs".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "template, table_expression, and input_template in the form of a JSON object".to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_static_name().to_string(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "op_kwargs".to_string(),
                    ]),
            },
        };
        let tool = chat_completion::Tool {
            r#type: chat_completion::ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

/// Inject [RecordBatch]es into a [String] template
///
/// # Notes
/// Equivalent using Minijinja2 would be the following: 
/// {%- for row in TABLENAME %}
///     {{- 'COL1 value' + row.COL1 + '\\n' }}
///     {%- if OTHERCOLS %}
///         {{- 'COL2 value' + row.COL2 + '\\n' }}
///     {%- endif %}///     
/// {%- endfor %}
/// 
/// Where `table_expression` is TABLENAME
///   `input_template` is e.g., {OTHERCOLS: true}
///   and column names for the table are COL1 and COL2
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `doc_template` - Minijinja [String] template
/// * `doc_name` - The name of the resulting document
/// * `table_expression` - The expression for the table within the minijinja template
/// * `doc_input` - A JSON Value representing the input for the template beyond the table_expression
///   where the table_expression will be inserted into to complete the input for the template
/// * `doc_extension` - The document extension e.g., .py, .html, .md, .txt, etc.
/// * `device` - The compute device
#[instrument(skip(
    lhs_args,
    doc_template,
    doc_name,
    table_expression,
    doc_input,
    format,
    _device
))]
pub fn apply_template(
    lhs_args: &[RecordBatch],
    doc_template: &str,
    doc_name: &str,
    table_expression: &str,
    doc_input: &Value,
    format: &DataFormat,
    _device: &Device,
) -> Result<RecordBatch> {
    // Convert the RecordBatches into a json objct
    let lhs_json_object = Table::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?
        .to_json_object()?;

    // Complete the input
    let input = if let Some(input_object) = doc_input.as_object() {
        let mut input_object = input_object.to_owned();
        let _ =  input_object.insert(table_expression.to_string(), lhs_json_object.into());
        serde_json::to_value(input_object)?
    } else {
        json!({table_expression.to_string(): lhs_json_object})
    };

    // Apply the template
    let document = TableScript::new_from_template(doc_template.to_string())
        .apply_template(&input)?;

    // Wrap into a table
    let batch = create_values_record_batch(vec![String::new()], vec![String::new()], vec![String::new()], vec![document])?;
    let table = Table::get_builder()
        .with_name(doc_name)
        .with_record_batches(vec![batch])?
        .build()?;

    // Convert to the desired format
    table_and_data_format_to_record_batch(&table, &format)
}

#[cfg(test)]
mod tests {
    use phymes_core::{session::common_traits::device, table::table_trait::test_table::make_test_table_chat};

    use super::*;

    #[test]
    fn test_apply_template() -> Result<()> {
        // Make the test record batches
        let test_table = make_test_table_chat("messages")?;

        let template = r#"""{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n{{- messages[0]['content'] }}\n    {%- else %}\n{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- '\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}\n    {%- for tool in tools %}\n{{- '\\n' }}\n{{- tool | tojson }}\n    {%- endfor %}\n    {{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}\n{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == 'assistant' %}\n{{- '<|im_start|>' + message.role }}\n{%- if message.content %}\n    {{- '\\n' + message.content }}\n{%- endif %}\n{%- for tool_call in message.tool_calls %}\n    {%- if tool_call.function is defined %}\n{%- set tool_call = tool_call.function %}\n    {%- endif %}\n    {{- '\\n<tool_call>\\n{\"name\": \"' }}\n    {{- tool_call.name }}\n    {{- '\", \"arguments\": ' }}\n    {{- tool_call.arguments | tojson }}\n    {{- '}\\n</tool_call>' }}\n{%- endfor %}\n{{- '<|im_end|>\\n' }}\n    {%- elif message.role == 'tool' %}\n{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n    {{- '<|im_start|>user' }}\n{%- endif %}\n{{- '\\n<tool_response>\\n' }}\n{{- message.content }}\n{{- '\\n</tool_response>' }}\n{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}\n    {{- '<|im_end|>\\n' }}\n{%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""#;
        let table_expression = "messages";
        let input_template = serde_json::json!({
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "add_generation_prompt": true,
        });

        // Make the device
        let device = device(false)?;

        let result = apply_template(
            test_table.get_record_batches(),
            template,
            "viz",
            table_expression,
            &input_template,
            &DataFormat::Html,
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("filename");
        assert_eq!(lhs_text, ["viz"]);
        let lhs_text = result_table.get_column_as_vec_str("extension");
        assert_eq!(lhs_text, ["html"]);
        let lhs_text = result_table.get_column_as_vec_str("metadata");
        assert_eq!(lhs_text, ["assistant"]);
        let lhs_text = result_table.get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .map(|bytes| String::from_utf8(bytes).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lhs_text, [
            "\"\"\\n\\n<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\""
        ]);

        Ok(())
    }
}
