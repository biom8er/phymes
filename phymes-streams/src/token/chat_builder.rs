use anyhow::Result;
use arrow::array::{Int64Array, StringArray};
use futures::StreamExt;
use phymes_core::{
    RecordBatchReceiverStream, SendableRecordBatchStream, Subject, SubjectBuilder, SubjectTrait,
};
use phymes_data::SubjectScript;
use phymes_diagnostics::create_timestamp_micros;
use phymes_schemas::{
    ChatCompletionMessage, Content, MessageRole, Tool, ToolCall, create_chat_record_batch,
};
use tracing::{Level, event};

pub trait ChatTraitExt: Sized {
    /// Apply a template to build the message
    fn to_chat_prompt(
        self,
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        add_generation_prompt: bool,
        tools: Option<Vec<Tool>>,
    ) -> Result<String>;

    fn to_openai_messages(self) -> Vec<ChatCompletionMessage>;
}

impl ChatTraitExt for Subject {
    fn to_chat_prompt(
        self,
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        add_generation_prompt: bool,
        tools: Option<Vec<Tool>>,
    ) -> Result<String> {
        // // Trim all white spaces from the prompt
        // let template = chat_template
        //     .lines()
        //     .map(|line| line.trim())
        //     .collect::<Vec<&str>>()
        //     .join("");

        // Prepare the chat template inputs
        let chat_template_inputs = serde_json::json!({
            "messages": self.to_json_object()?,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "add_generation_prompt": add_generation_prompt,
            "tools": tools,
        });

        SubjectScript::new_from_template(chat_template.to_string())
            .apply_template(&chat_template_inputs)
    }

    fn to_openai_messages(self) -> Vec<ChatCompletionMessage> {
        let role_vec = self.get_column_as_vec_str("role");
        let content_vec = self.get_column_as_vec_str("content");
        let mut messages = Vec::with_capacity(role_vec.len());
        for i in 0..role_vec.len() {
            #[allow(clippy::if_same_then_else)]
            let role = if *role_vec.get(i).unwrap() == "user" {
                MessageRole::user
            } else if *role_vec.get(i).unwrap() == "system" {
                MessageRole::system
            // DM: change back to "tool" and remove other "tool" after upgrading
            //  to Qwen 3 series
            } else if *role_vec.get(i).unwrap() == "tool_call" {
                MessageRole::tool
            } else if *role_vec.get(i).unwrap() == "assistant" {
                MessageRole::assistant
            } else if *role_vec.get(i).unwrap() == "tool" {
                MessageRole::function
            } else if *role_vec.get(i).unwrap() == "function" {
                MessageRole::function
            } else {
                MessageRole::user
            };
            let message = match role {
                MessageRole::tool => {
                    let tools: Vec<ToolCall> =
                        serde_json::from_str(content_vec.get(i).unwrap()).unwrap();
                    ChatCompletionMessage {
                        role,
                        content: Content::Text(content_vec.get(i).unwrap().to_string()),
                        name: None,
                        tool_calls: Some(tools),
                        tool_call_id: None, // DM: not handling the tool_call_id yet
                    }
                }
                MessageRole::function => ChatCompletionMessage {
                    role,
                    content: Content::Text(content_vec.get(i).unwrap().to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None, // DM: not handling the tool_call_id yet
                },
                _ => ChatCompletionMessage {
                    role,
                    content: Content::Text(content_vec.get(i).unwrap().to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            };
            messages.push(message);
        }
        messages
    }
}

pub trait ChatBuilderTraitExt: Sized {
    /// Insert the system template to the chat history
    fn insert_system_template_str(self, system_prompt: &str) -> Result<Self>;

    /// Append the user query to the chat history
    fn append_new_user_query_str(self, content: &str, role: &str) -> Result<Self>;

    /// Stream print the chat response to the console and update the chat history
    #[allow(async_fn_in_trait)]
    async fn append_chat_response_sendable_record_batch_stream(
        self,
        stream: &mut SendableRecordBatchStream,
        capacity: usize,
    ) -> Result<(Self, SendableRecordBatchStream)>;
}

impl ChatBuilderTraitExt for SubjectBuilder {
    fn insert_system_template_str(mut self, system_prompt: &str) -> Result<Self> {
        // Fill in the system template

        // Add the system content to the history (should be the first record batch)
        let batch = create_chat_record_batch(
            vec!["system".to_string()],
            vec![system_prompt.to_string()],
            vec![create_timestamp_micros()],
        )?;
        match self.record_batches {
            Some(ref mut batches) => {
                batches.insert(0, batch);
                Ok(self)
            }
            None => {
                self.schema = Some(batch.schema());
                self.record_batches = Some(vec![batch]);
                Ok(self)
            }
        }
    }

    fn append_new_user_query_str(mut self, content: &str, role: &str) -> Result<Self> {
        let batch = create_chat_record_batch(
            vec![role.to_string()],
            vec![content.to_string()],
            vec![create_timestamp_micros()],
        )?;
        match self.record_batches {
            Some(ref mut batches) => {
                batches.push(batch);
                Ok(self)
            }
            None => {
                event!(
                    Level::DEBUG,
                    "Could not append new user query batch to missing chat history!"
                );
                self.schema = Some(batch.schema());
                self.record_batches = Some(vec![batch]);
                Ok(self)
            }
        }
    }

    async fn append_chat_response_sendable_record_batch_stream(
        mut self,
        stream: &mut SendableRecordBatchStream,
        capacity: usize,
    ) -> Result<(Self, SendableRecordBatchStream)> {
        // stream the chat response
        let mut builder =
            RecordBatchReceiverStream::builder(self.schema.clone().unwrap(), capacity);
        let mut content = Vec::<String>::new();
        let mut role = String::new();
        let mut timestamp: i64 = 0;
        while let Some(result) = stream.next().await {
            match result {
                Ok(batch) => {
                    let batch_copy = batch.clone();

                    // Extract out the content
                    let content_string = batch_copy
                        .column_by_name("content")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .iter()
                        .map(|s| s.unwrap_or(""))
                        .collect::<Vec<_>>()
                        .first()
                        .unwrap()
                        .to_string();
                    content.push(content_string);

                    // Extract out the role
                    if role.is_empty() {
                        role = batch_copy
                            .column_by_name("role")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or(""))
                            .collect::<Vec<_>>()
                            .first()
                            .unwrap()
                            .to_string();
                    }

                    // Extract out the timestamp
                    timestamp = batch_copy
                        .column_by_name("timestamp")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .iter()
                        .map(|s| s.unwrap_or_default())
                        .collect::<Vec<_>>()
                        .first()
                        .unwrap()
                        .to_owned();

                    // Forward the stream
                    let tx_1 = builder.tx();
                    builder.spawn(async move {
                        tx_1.send(Ok(batch)).await.unwrap();
                        Ok(())
                    });
                }
                Err(_e) => unreachable!(),
            }
        }

        // update the chat history
        let content_string: String = content.join("");
        let batch = create_chat_record_batch(
            vec![role.to_string()],
            vec![content_string.to_string()],
            vec![timestamp],
        )?;
        match self.record_batches {
            Some(ref mut batches) => {
                batches.push(batch);
                Ok((self, builder.build()))
            }
            None => {
                event!(
                    Level::DEBUG,
                    "Could not append chat response batch to missing chat history!"
                );
                self.schema = Some(batch.schema());
                self.record_batches = Some(vec![batch]);
                Ok((self, builder.build()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{BuilderTrait, test_subject::make_test_subject_chat};

    use phymes_schemas::Tool;

    use super::*;

    #[test]
    fn test_to_chat_prompt_no_tool_no_docs() -> Result<()> {
        let test_table = make_test_subject_chat("messages")?;

        let chat_template = r#"""{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n{{- messages[0]['content'] }}\n    {%- else %}\n{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- '\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}\n    {%- for tool in tools %}\n{{- '\\n' }}\n{{- tool | tojson }}\n    {%- endfor %}\n    {{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}\n{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == 'assistant' %}\n{{- '<|im_start|>' + message.role }}\n{%- if message.content %}\n    {{- '\\n' + message.content }}\n{%- endif %}\n{%- for tool_call in message.tool_calls %}\n    {%- if tool_call.function is defined %}\n{%- set tool_call = tool_call.function %}\n    {%- endif %}\n    {{- '\\n<tool_call>\\n{\"name\": \"' }}\n    {{- tool_call.name }}\n    {{- '\", \"arguments\": ' }}\n    {{- tool_call.arguments | tojson }}\n    {{- '}\\n</tool_call>' }}\n{%- endfor %}\n{{- '<|im_end|>\\n' }}\n    {%- elif message.role == 'tool' %}\n{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n    {{- '<|im_start|>user' }}\n{%- endif %}\n{{- '\\n<tool_response>\\n' }}\n{{- message.content }}\n{{- '\\n</tool_response>' }}\n{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}\n    {{- '<|im_end|>\\n' }}\n{%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""#;

        let prompt =
            test_table.to_chat_prompt(chat_template, Some("[BOS]"), Some("[EOS]"), true, None)?;

        assert_eq!(
            prompt,
            "\"\"\\n\\n<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\""
        );

        Ok(())
    }

    #[test]
    fn test_to_chat_prompt_with_tools() -> Result<()> {
        let test_table = make_test_subject_chat("messages")?;

        let tools_string = r#"[{"type": "function", "function": {"name": "get_current_weather","description": "Get the current weather","parameters": {"type": "object","properties": {"location": {"type": "string","description": "The city and state, e.g. San Francisco, CA"},"format": {"type": "string", "enum_values": ["celsius", "fahrenheit"], "description": "The temperature unit to use. Infer this from the users location."}}, "required": ["location", "format"]}}}]"#.to_string();
        let test_tools: Vec<Tool> = serde_json::from_str(&tools_string).unwrap();

        let chat_template = r#"""{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n{{- messages[0]['content'] }}\n    {%- else %}\n{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- '\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}\n    {%- for tool in tools %}\n{{- '\\n' }}\n{{- tool | tojson }}\n    {%- endfor %}\n    {{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}\n{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == 'assistant' %}\n{{- '<|im_start|>' + message.role }}\n{%- if message.content %}\n    {{- '\\n' + message.content }}\n{%- endif %}\n{%- for tool_call in message.tool_calls %}\n    {%- if tool_call.function is defined %}\n{%- set tool_call = tool_call.function %}\n    {%- endif %}\n    {{- '\\n<tool_call>\\n{\"name\": \"' }}\n    {{- tool_call.name }}\n    {{- '\", \"arguments\": ' }}\n    {{- tool_call.arguments | tojson }}\n    {{- '}\\n</tool_call>' }}\n{%- endfor %}\n{{- '<|im_end|>\\n' }}\n    {%- elif message.role == 'tool' %}\n{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n    {{- '<|im_start|>user' }}\n{%- endif %}\n{{- '\\n<tool_response>\\n' }}\n{{- message.content }}\n{{- '\\n</tool_response>' }}\n{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}\n    {{- '<|im_end|>\\n' }}\n{%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""#;

        let prompt = test_table.to_chat_prompt(
            chat_template,
            Some("[BOS]"),
            Some("[EOS]"),
            true,
            Some(test_tools),
        )?;

        assert_eq!(
            prompt,
            "\"\"\\n<|im_start|>system\\n\\n\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\\n\\n\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n\\n\\n\\n{\"function\":{\"description\":\"Get the current weather\",\"name\":\"get_current_weather\",\"parameters\":{\"properties\":{\"format\":{\"description\":\"The temperature unit to use. Infer this from the users location.\",\"enum_values\":[\"celsius\",\"fahrenheit\"],\"type\":\"string\"},\"location\":{\"description\":\"The city and state, e.g. San Francisco, CA\",\"type\":\"string\"}},\"required\":[\"location\",\"format\"],\"type\":\"object\"}},\"type\":\"function\"}\\n\\n\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\""
        );

        Ok(())
    }

    #[test]
    fn test_to_openai_messages() -> Result<()> {
        //let tools_call_string = "[{\"type\": \"function\", \"function\": {\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Boston, MA\"}}}]";
        let tool_call_str = r#"[{"id":"fc_12345xyz","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\":\"San Francisco, CA\",\"format\":\"celsius\"}"}}]"#;
        let test_table = SubjectBuilder::new()
            .with_name("messages")
            .insert_system_template_str("Hello from system.")?
            .append_new_user_query_str("Hello from user.", "user")?
            .append_new_user_query_str("Hello from assistant.", "assistant")?
            .append_new_user_query_str(tool_call_str, "tool_call")?
            .append_new_user_query_str("30", "function")?
            .build()?;

        let messages = test_table.to_openai_messages();

        assert_eq!(messages.first().unwrap().role, MessageRole::system);
        assert_eq!(
            messages.first().unwrap().content,
            Content::Text("Hello from system.".to_string())
        );
        assert!(messages.first().unwrap().name.is_none());
        assert!(messages.first().unwrap().tool_calls.is_none());
        assert!(messages.first().unwrap().tool_call_id.is_none());

        assert_eq!(messages.get(1).unwrap().role, MessageRole::user);
        assert_eq!(
            messages.get(1).unwrap().content,
            Content::Text("Hello from user.".to_string())
        );
        assert!(messages.get(1).unwrap().name.is_none());
        assert!(messages.get(1).unwrap().tool_calls.is_none());
        assert!(messages.get(1).unwrap().tool_call_id.is_none());

        assert_eq!(messages.get(2).unwrap().role, MessageRole::assistant);
        assert_eq!(
            messages.get(2).unwrap().content,
            Content::Text("Hello from assistant.".to_string())
        );
        assert!(messages.get(2).unwrap().name.is_none());
        assert!(messages.get(2).unwrap().tool_calls.is_none());
        assert!(messages.get(2).unwrap().tool_call_id.is_none());

        let tool_call =
            serde_json::to_string(messages.get(3).unwrap().tool_calls.as_ref().unwrap())?;
        assert_eq!(messages.get(3).unwrap().role, MessageRole::tool);
        assert_eq!(
            messages.get(3).unwrap().content,
            Content::Text(tool_call_str.to_string())
        );
        assert!(messages.get(3).unwrap().name.is_none());
        assert_eq!(tool_call.as_str(), tool_call_str);
        assert!(messages.get(3).unwrap().tool_call_id.is_none());

        assert_eq!(messages.get(4).unwrap().role, MessageRole::function);
        assert_eq!(
            messages.get(4).unwrap().content,
            Content::Text("30".to_string())
        );
        assert!(messages.get(4).unwrap().name.is_none());
        assert!(messages.get(4).unwrap().tool_calls.is_none());
        assert!(messages.get(4).unwrap().tool_call_id.is_none());

        Ok(())
    }
}
