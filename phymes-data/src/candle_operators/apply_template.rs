use std::collections::HashMap;

use anyhow::Result;
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    schemas::{available_subjects::create_documents_batch, chat_completion, types},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::{table_script::TableScript, table_trait::{Table, TableBuilderTrait, TableTrait}},
};
use serde_json::{json, Value};
use tracing::{event, instrument, Level};

use crate::{candle_data::data_config::DataConfig, candle_operators::data_operator::{make_error_record_batch, DataOperatorTrait}};

/// Inject a table into a string template
#[derive(Debug)]
pub struct ApplyTemplate {
    template: String,
    table_expression: String,
    input_template: Value,
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
        match apply_template(
            lhs_args,
            &self.template,
            &self.table_expression,
            &self.input_template,
            device,
        ) {
            Ok(batch) => Ok(batch),
            Err(err) => {
                event!(Level::ERROR, "{err}");
                Ok(make_error_record_batch(err.to_string().as_str()))
            },
        }
    }
    fn new(config: &DataConfig) -> Self {
        let template = config.template.clone().unwrap_or_default();
        let table_expression = config.table_expression.clone().unwrap_or_default();
        let input_template = config.input_template.clone().unwrap_or(serde_json::Value::default());

        // Make the object
        ApplyTemplate {
            template,
            table_expression,
            input_template,
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
/// * `template` - Minijinja [String] template
/// * `table_expression` - The expression for the table within the minijinja template
/// * `input_template` - A JSON Value representing the input for the template beyond the table_expression
///   where the table_expression will be inserted into to complete the input for the template
/// * `device` - The compute device
#[instrument(skip(
    lhs_args,
    template,
    table_expression,
    input_template,
    _device
))]
pub fn apply_template(
    lhs_args: &[RecordBatch],
    template: &str,
    table_expression: &str,
    input_template: &Value,
    _device: &Device,
) -> Result<RecordBatch> {
    // Convert the RecordBatches into a json objct
    let lhs_json_object = Table::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?
        .to_json_object()?;

    // Complete the input
    let input = if let Some(input_object) = input_template.as_object() {
        let mut input_object = input_object.to_owned();
        let _ =  input_object.insert(table_expression.to_string(), lhs_json_object.into());
        serde_json::to_value(input_object)?
    } else {
        json!({table_expression.to_string(): lhs_json_object})
    };

    // Apply the template
    let document = TableScript::new_from_template(template.to_string())
        .apply_template(&input)?;

    let batch = create_documents_batch(vec![table_expression.to_string()], vec![table_expression.to_string()], vec![document])?;
    Ok(batch)
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
            table_expression,
            &input_template,
            &device,
        )?;

        // Check the results
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let lhs_text = result_table.get_column_as_vec_str("chunk_id");
        assert_eq!(lhs_text, ["messages"]);
        let lhs_text = result_table.get_column_as_vec_str("document_id");
        assert_eq!(lhs_text, ["messages"]);
        let lhs_text = result_table.get_column_as_vec_str("text");
        assert_eq!(lhs_text, [
            "\"\"\\n\\n<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\""
        ]);

        Ok(())
    }
}
