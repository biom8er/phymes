use phymes_core::table::{
    arrow_script::ArrowTableScript,
    arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
    stream::SendableRecordBatchStream,
    stream_adapter::RecordBatchReceiverStream,
};

use arrow::{
    array::{ArrayRef, StringArray},
    record_batch::RecordBatch,
};
use chrono::{DateTime, Utc};
use anyhow::{Result, anyhow};
use futures::StreamExt;
use std::sync::Arc;
use tracing::{Level, event};

use crate::openai_asset::chat_completion::{
    self, ChatCompletionMessage, Content, MessageRole, ToolCall,
};

/// Generate a timestamp that can be added to the message table
/// Same as in phymes-app/src/ui/messaging_state.rs
pub fn create_timestamp() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.format("%a %b %e %T %Y").to_string()
}

pub trait MessageHistoryTraitExt: Sized {
    /// Apply a template to build the message
    fn to_chat_prompt(
        self,
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        add_generation_prompt: bool,
        tools: Option<Vec<chat_completion::Tool>>,
    ) -> Result<String>;

    fn to_openai_messages(self) -> Vec<ChatCompletionMessage>;
}

impl MessageHistoryTraitExt for ArrowTable {
    fn to_chat_prompt(
        self,
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        add_generation_prompt: bool,
        tools: Option<Vec<chat_completion::Tool>>,
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

        ArrowTableScript::new_from_template(chat_template.to_string())
            .apply_template(&chat_template_inputs)
    }

    fn to_openai_messages(self) -> Vec<ChatCompletionMessage> {
        let role_vec = self.get_column_as_str_vec("role");
        let content_vec = self.get_column_as_str_vec("content");
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

pub trait MessageHistoryBuilderTraitExt: Sized {
    /// Insert the system template to the chat history
    fn insert_system_template_str(self, system_prompt: &str) -> Result<Self>;

    /// Append the user query to the chat history
    fn append_new_user_query_str(self, content: &str, role: &str) -> Result<Self>;

    /// Append the user query to the chat history
    fn append_new_user_query(self, user_query: RecordBatch) -> Result<Self>;

    /// Stream print the chat response to the console and update the chat history
    #[allow(async_fn_in_trait)]
    async fn append_chat_response_sendable_record_batch_stream(
        self,
        stream: &mut SendableRecordBatchStream,
        capacity: usize,
    ) -> Result<(Self, SendableRecordBatchStream)>;
}

impl MessageHistoryBuilderTraitExt for ArrowTableBuilder {
    fn insert_system_template_str(mut self, system_prompt: &str) -> Result<Self> {
        // Fill in the system template

        // Add the system content to the history (should be the first record batch)
        let role: ArrayRef = Arc::new(StringArray::from(vec!["system"]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![system_prompt]));
        let timestamp: ArrayRef = Arc::new(StringArray::from(vec![create_timestamp()]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content), ("timestamp", timestamp)])?;
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
        let role: ArrayRef = Arc::new(StringArray::from(vec![role]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![content]));
        let timestamp: ArrayRef = Arc::new(StringArray::from(vec![create_timestamp()]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content), ("timestamp", timestamp)])?;
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

    fn append_new_user_query(mut self, user_query: RecordBatch) -> Result<Self> {
        match self.record_batches {
            Some(ref mut batches) => {
                if !self.schema.clone().unwrap().eq(&user_query.schema()) {
                    return Err(anyhow!("Mismatch between schema and batches!"));
                }
                batches.push(user_query);
                Ok(self)
            }
            None => {
                event!(
                    Level::DEBUG,
                    "Could not append new user query to missing chat history!"
                );
                self.schema = Some(user_query.schema());
                self.record_batches = Some(vec![user_query]);
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
        let mut timestamp = String::new();
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
                    if timestamp.is_empty() {
                        timestamp = batch_copy
                            .column_by_name("timestamp")
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
        let role: ArrayRef = Arc::new(StringArray::from(vec![role]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![content_string]));
        let timestamp: ArrayRef = Arc::new(StringArray::from(vec![timestamp]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content), ("timestamp", timestamp)])?;
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

mod test_message_history {
    use super::*;
    use anyhow::anyhow;
    use arrow::datatypes::SchemaRef;
    use futures::Stream;
    use parking_lot::Mutex;
    use phymes_core::{
        metrics::{ArrowTaskMetricsSet, HashMap},
        session::{
            common_traits::{BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap},
            runtime_env::RuntimeEnv,
        },
        table::{
            arrow_table_publish::ArrowTablePublish,
            arrow_table_subscribe::ArrowTableSubscribe,
            stream::{RecordBatchStream, SendableRecordBatchStream},
        },
        task::{
            arrow_message::{
                ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
                ArrowOutgoingMessageTrait,
            },
            arrow_processor::ArrowProcessorTrait,
        },
    };
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, ready},
    };

    #[allow(dead_code)]
    #[derive(Default, Debug)]
    pub struct CandleChatMockProcessor {
        name: String,
        publications: Vec<ArrowTablePublish>,
        subscriptions: Vec<ArrowTableSubscribe>,
        forward: Vec<String>,
    }

    impl MappableTrait for CandleChatMockProcessor {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl ArrowProcessorTrait for CandleChatMockProcessor {
        fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait> {
            Arc::new(Self {
                name: name.to_string(),
                publications: vec![ArrowTablePublish::None],
                subscriptions: vec![ArrowTableSubscribe::None],
                forward: Vec::new(),
            })
        }

        fn get_publications(&self) -> &[ArrowTablePublish] {
            &self.publications
        }

        fn get_subscriptions(&self) -> &[ArrowTableSubscribe] {
            &self.subscriptions
        }

        fn get_forward_subscriptions(&self) -> &[String] {
            self.forward.as_slice()
        }

        fn process(
            &self,
            mut message: OutgoingMessageMap,
            _metrics: ArrowTaskMetricsSet,
            _runtime_env: Arc<Mutex<RuntimeEnv>>,
        ) -> Result<OutgoingMessageMap> {
            // Create the stream response
            let input = match message.remove("messages") {
                Some(i) => i,
                None => return Err(anyhow!("Message not provided.")),
            };
            let tools = message.remove("tools").map(|i| i.get_message_own());

            // TODO: check the state for tools and documents...

            // Generate the stream
            let out = Box::pin(ChatProcessorMockStream {
                schema: input.get_message().schema(),
                input: input.get_message_own(),
                tools,
                found_tools: false,
                sample: 0,
                sample_len: 10,
            });

            // Prepare the outbox
            let mut outbox = HashMap::<String, ArrowOutgoingMessage>::new();
            let out_m = ArrowOutgoingMessage::get_builder()
                .with_name("messages")
                .with_publisher(self.get_name())
                .with_subject("messages")
                .with_update(&ArrowTablePublish::None)
                .with_message(out)
                .build()?;
            let _ = outbox.insert(out_m.get_name().to_string(), out_m);
            Ok(outbox)
        }
    }

    #[allow(dead_code)]
    struct ChatProcessorMockStream {
        /// Output schema after the projection
        schema: SchemaRef,
        /// The input task to process.
        input: SendableRecordBatchStream,
        /// Mock tool call
        tools: Option<SendableRecordBatchStream>,
        found_tools: bool,
        /// Parameters for running chat inference
        sample: usize,
        sample_len: usize,
    }

    impl Stream for ChatProcessorMockStream {
        type Item = Result<RecordBatch>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if self.sample == 0 {
                // Collect the chat history
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(self.input.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
                let messages = ArrowTableBuilder::new()
                    .with_name("messages")
                    .with_record_batches(batches)?
                    .build()?;

                // Collect the tools
                let tools = match self.tools {
                    Some(ref mut tools) => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) = ready!(tools.poll_next_unpin(cx)) {
                            batches.push(batch);
                        }
                        let tool_table = ArrowTableBuilder::new()
                            .with_name("messages")
                            .with_record_batches(batches)?
                            .build()?;
                        let tool_vec: Vec<chat_completion::Tool> = tool_table
                            .get_column_as_str_vec("tool")
                            .iter()
                            .map(|s| {
                                let tool: chat_completion::Tool = serde_json::from_str(s).unwrap();
                                tool
                            })
                            .collect::<Vec<_>>();
                        Some(tool_vec)
                    }
                    None => None,
                };
                if tools.is_some() {
                    self.found_tools = true;
                }

                // ... and then to prompt
                let chat_template = r#"""{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n{{- messages[0]['content'] }}\n    {%- else %}\n{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- '\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}\n    {%- for tool in tools %}\n{{- '\\n' }}\n{{- tool | tojson }}\n    {%- endfor %}\n    {{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}\n{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == 'assistant' %}\n{{- '<|im_start|>' + message.role }}\n{%- if message.content %}\n    {{- '\\n' + message.content }}\n{%- endif %}\n{%- for tool_call in message.tool_calls %}\n    {%- if tool_call.function is defined %}\n{%- set tool_call = tool_call.function %}\n    {%- endif %}\n    {{- '\\n<tool_call>\\n{\"name\": \"' }}\n    {{- tool_call.name }}\n    {{- '\", \"arguments\": ' }}\n    {{- tool_call.arguments | tojson }}\n    {{- '}\\n</tool_call>' }}\n{%- endfor %}\n{{- '<|im_end|>\\n' }}\n    {%- elif message.role == 'tool' %}\n{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n    {{- '<|im_start|>user' }}\n{%- endif %}\n{{- '\\n<tool_response>\\n' }}\n{{- message.content }}\n{{- '\\n</tool_response>' }}\n{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}\n    {{- '<|im_end|>\\n' }}\n{%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""#;
                let prompt = messages.to_chat_prompt(
                    chat_template,
                    Some("[BOS]"),
                    Some("[EOS]"),
                    true,
                    tools,
                )?;

                // mock generationg of next token
                let role: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
                let content: ArrayRef = Arc::new(StringArray::from(vec![prompt]));
                let timestamp: ArrayRef = Arc::new(StringArray::from(vec![create_timestamp()]));
                let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content), ("timestamp", timestamp)])?;

                // record the poll
                self.sample += 1;
                Poll::Ready(Some(Ok(batch)))
            } else if self.sample < self.sample_len {
                // mock generationg of next token
                let response = match self.found_tools {
                    true => format!("Function{}", self.sample),
                    false => format!("Response{}", self.sample),
                };
                let content: ArrayRef = Arc::new(StringArray::from(vec![response]));
                let role: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
                let timestamp: ArrayRef = Arc::new(StringArray::from(vec![create_timestamp()]));
                let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content), ("timestamp", timestamp)])?;

                // record the poll
                self.sample += 1;
                Poll::Ready(Some(Ok(batch)))
            } else {
                Poll::Ready(None)
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            // Same number of record batches
            self.input.size_hint()
        }
    }

    impl RecordBatchStream for ChatProcessorMockStream {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::openai_asset::chat_completion::Tool;
    use futures::TryStreamExt;
    use parking_lot::Mutex;
    use phymes_core::{
        metrics::{ArrowTaskMetricsSet, HashMap},
        session::{
            common_traits::{BuildableTrait, BuilderTrait},
            runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        },
        table::{
            arrow_table::test_table::{make_test_table_chat, make_test_table_tool},
            arrow_table_publish::ArrowTablePublish,
        },
        task::{
            arrow_message::{
                ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
                ArrowOutgoingMessageTrait,
            },
            arrow_processor::ArrowProcessorTrait,
        },
    };

    use super::*;

    #[test]
    fn test_to_chat_prompt_no_tool_no_docs() -> Result<()> {
        let test_table = make_test_table_chat("messages")?;

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
        let test_table = make_test_table_chat("messages")?;

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
        let test_table = ArrowTableBuilder::new()
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

    #[tokio::test]
    async fn test_message_builder_no_tool_no_doc() -> Result<()> {
        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name("messages")
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the message
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            "messages".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("messages")
                .with_publisher("")
                .with_subject("messages")
                .with_update(&ArrowTablePublish::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        let chat_processor: Arc<dyn ArrowProcessorTrait> =
            test_message_history::CandleChatMockProcessor::new_arc("ChatBot");
        let mut stream = chat_processor.process(
            message,
            ArrowTaskMetricsSet::new(),
            Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt"))),
        )?;

        // Update the chat history with the response
        let (message_builder, stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove("messages").unwrap().get_message_own(),
                10,
            )
            .await?;
        let messages = message_builder.clone().build()?;

        let messages_content = messages.get_column_as_str_vec("content");
        assert_eq!(
            messages_content,
            &[
                "You are a helpful assistant.",
                "Write a function to count prime numbers up to N.",
                "\"\"\\n\\n<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n\\n\\n<|im_start|>user\\nWrite a function to count prime numbers up to N.<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"Response1Response2Response3Response4Response5Response6Response7Response8Response9"
            ]
        );

        // Check that the forwarded stream also matches
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let messages = ArrowTableBuilder::new()
            .with_name("")
            .with_record_batches(batches)?
            .build()?;
        let messages_content = messages.get_column_as_str_vec("content");
        assert_eq!(
            messages_content,
            &[
                "\"\"\\n\\n<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n\\n\\n<|im_start|>user\\nWrite a function to count prime numbers up to N.<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"",
                "Response1",
                "Response2",
                "Response3",
                "Response4",
                "Response5",
                "Response6",
                "Response7",
                "Response8",
                "Response9"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_message_builder_with_tool() -> Result<()> {
        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name("messages")
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the message
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            "messages".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("messages")
                .with_publisher("")
                .with_subject("messages")
                .with_update(&ArrowTablePublish::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            "tools".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("tools")
                .with_publisher("")
                .with_subject("tools")
                .with_update(&ArrowTablePublish::None)
                .with_message(make_test_table_tool("tools")?.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        let chat_processor = test_message_history::CandleChatMockProcessor::new_arc("ChatBot");
        let mut stream = chat_processor.process(
            message,
            ArrowTaskMetricsSet::new(),
            Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt"))),
        )?;

        // Update the chat history with the response
        let (message_builder, stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove("messages").unwrap().get_message_own(),
                10,
            )
            .await?;
        let messages = message_builder.clone().build()?;

        let messages_content = messages.get_column_as_str_vec("content");
        assert_eq!(
            messages_content,
            &[
                "You are a helpful assistant.",
                "Write a function to count prime numbers up to N.",
                "\"\"\\n<|im_start|>system\\n\\n\\nYou are a helpful assistant.\\n\\n\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n\\n\\n\\n{\"function\":{\"description\":\"description1\",\"name\":\"tool1\",\"parameters\":{\"properties\":{\"parameter1\":{\"description\":\"Param1 description\",\"type\":\"string\"},\"parameter2\":{\"description\":\"An Enum.\",\"enum_values\":[\"A\",\"B\"],\"type\":\"string\"}},\"required\":[\"parameter1\",\"parameter2\"],\"type\":\"object\"}},\"type\":\"function\"}\\n\\n\\n\\n{\"function\":{\"description\":\"Open ended response with no specific tool selected\",\"name\":\"no_tool\",\"parameters\":{\"properties\":{\"content\":{\"description\":\"The response content\",\"type\":\"string\"}},\"required\":[\"content\"],\"type\":\"object\"}},\"type\":\"function\"}\\n\\n\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n\\n\\n\\n\\n\\n\\n<|im_start|>user\\nWrite a function to count prime numbers up to N.<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"Function1Function2Function3Function4Function5Function6Function7Function8Function9"
            ]
        );

        // Check that the forwarded stream also matches
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let messages = ArrowTableBuilder::new()
            .with_name("")
            .with_record_batches(batches)?
            .build()?;
        let messages_content = messages.get_column_as_str_vec("content");
        assert_eq!(
            messages_content,
            &[
                "\"\"\\n<|im_start|>system\\n\\n\\nYou are a helpful assistant.\\n\\n\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n\\n\\n\\n{\"function\":{\"description\":\"description1\",\"name\":\"tool1\",\"parameters\":{\"properties\":{\"parameter1\":{\"description\":\"Param1 description\",\"type\":\"string\"},\"parameter2\":{\"description\":\"An Enum.\",\"enum_values\":[\"A\",\"B\"],\"type\":\"string\"}},\"required\":[\"parameter1\",\"parameter2\"],\"type\":\"object\"}},\"type\":\"function\"}\\n\\n\\n\\n{\"function\":{\"description\":\"Open ended response with no specific tool selected\",\"name\":\"no_tool\",\"parameters\":{\"properties\":{\"content\":{\"description\":\"The response content\",\"type\":\"string\"}},\"required\":[\"content\"],\"type\":\"object\"}},\"type\":\"function\"}\\n\\n\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n\\n\\n\\n\\n\\n\\n<|im_start|>user\\nWrite a function to count prime numbers up to N.<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"",
                "Function1",
                "Function2",
                "Function3",
                "Function4",
                "Function5",
                "Function6",
                "Function7",
                "Function8",
                "Function9"
            ]
        );

        Ok(())
    }
}
