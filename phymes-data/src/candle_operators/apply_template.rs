use std::collections::HashMap;

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, Function, FunctionParameters, JSONSchemaDefine,
    JSONSchemaType, MappableTrait, Table, TableBuilderTrait, TableScript, TableTrait, Tool,
    ToolType, create_mermaid_content_template_batch, create_values_record_batch,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::instrument;

use crate::{
    AvailableJinja2Templates, ToolTrait,
    candle_data::{DataConfig, table_and_data_format_to_record_batch},
    candle_operators::DataOperatorTrait,
};

/// Inject a table into a string template
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ApplyTemplate {
    doc_template: AvailableJinja2Templates,
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

impl ToolTrait for ApplyTemplate {
    fn get_description(&self) -> String {
        "Inject a table into a string template.".to_string()
    }
    fn to_json_tool_schema(&self) -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "op_kwargs".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "template, table_expression, and input_template in the form of a JSON object"
                        .to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = Function {
            name: Self::get_static_name().to_string(),
            description: Some(self.get_description()),
            parameters: FunctionParameters {
                schema_type: JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec!["lhs_name".to_string(), "op_kwargs".to_string()]),
            },
        };
        let tool = Tool {
            r#type: ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
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
    fn new(config: &DataConfig) -> Result<Self> {
        let doc_template = config.doc_template.clone().ok_or(anyhow!(
            "Missing `doc_template` for `{}`.",
            Self::get_static_name()
        ))?;
        let doc_name = config.doc_name.clone().ok_or(anyhow!(
            "Missing `doc_name` for `{}`.",
            Self::get_static_name()
        ))?;
        let table_expression = config.table_expression.clone().ok_or(anyhow!(
            "Missing `table_expression` for `{}`.",
            Self::get_static_name()
        ))?;
        let doc_input = if let Some(doc_input) = config.doc_input.as_ref() {
            serde_json::from_str::<Value>(doc_input)?
        } else {
            return Err(anyhow!(
                "Missing `doc_input` for `{}`.",
                Self::get_static_name()
            ));
        };
        let format = config.format.ok_or(anyhow!(
            "Missing `format` for `{}`.",
            Self::get_static_name()
        ))?;

        // Make the object
        Ok(ApplyTemplate {
            doc_template,
            doc_name,
            table_expression,
            doc_input,
            format,
        })
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
/// * `rhs_args` - Optional Slice of [RecordBatch]es used to generate the template
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
    doc_template: &AvailableJinja2Templates,
    doc_name: &str,
    table_expression: &str,
    doc_input: &Value,
    format: &DataFormat,
    _device: &Device,
) -> Result<RecordBatch> {
    // Convert the RecordBatches into a json object
    let lhs_json_object = Table::get_builder()
        .with_name("apply_template")
        .with_record_batches(lhs_args.to_vec())?
        .build()?
        .to_json_object()?;

    // Complete the input
    let input = if let Some(input_object) = doc_input.as_object() {
        let mut input_object = input_object.to_owned();
        let _ = input_object.insert(table_expression.to_string(), lhs_json_object.into());
        serde_json::to_value(input_object)?
    } else {
        json!({table_expression.to_string(): lhs_json_object})
    };

    // Apply the template
    let document =
        TableScript::new_from_template(doc_template.to_template()).apply_template(&input)?;

    // Wrap into a table
    let batch = match format {
        DataFormat::None => create_mermaid_content_template_batch(vec![document])?,
        _ => create_values_record_batch(
            vec![String::new()],
            vec![String::new()],
            vec![String::new()],
            vec![document],
        )?,
    };
    let table = Table::get_builder()
        .with_name(doc_name)
        .with_record_batches(vec![batch])?
        .build()?;

    // Convert to the desired format
    table_and_data_format_to_record_batch(&table, format, Some("content"))
}

#[cfg(test)]
mod tests {
    use phymes_core::{device, test_table::make_test_table_chat};

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
        let jinja2_template = AvailableJinja2Templates::Custom(template.to_string());

        // Make the device
        let device = device(false)?;

        let result = apply_template(
            test_table.get_record_batches(),
            &jinja2_template,
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
        let lhs_text = result_table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .map(|bytes| String::from_utf8(bytes).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            lhs_text,
            [
                "\"\"\\n\\n<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\""
            ]
        );

        Ok(())
    }
}
