/// Integration tests for chat_builder
mod test_messages {
    use crate::ProcessorTrait;
    use anyhow::{Result, anyhow};
    use arrow::{array::RecordBatch, datatypes::SchemaRef};
    use futures::{Stream, StreamExt};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RecordBatchStream, RuntimeEnv,
        SendableRecordBatchStream, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_diagnostics::{DiagnosticBuilder, HashMap, create_timestamp_micros};
    use phymes_message::{
        MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
        SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
        SendableRecordBatchStreamMessageMap,
    };
    use phymes_schemas::{Tool, create_chat_record_batch};
    use phymes_streams::ChatTraitExt;
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, ready},
    };

    #[allow(dead_code)]
    #[derive(Debug)]
    pub struct CandleChatMockProcessor {
        name: String,
        r#type: String,
    }

    impl MappableTrait for CandleChatMockProcessor {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl ProcessorTrait for CandleChatMockProcessor {
        fn new(name: &str, r#type: &str) -> Self {
            Self {
                name: name.to_string(),
                r#type: r#type.to_string(),
            }
        }

        fn get_type(&self) -> &str {
            &self.r#type
        }

        fn line_and_file(&self) -> (u32, String) {
            (line!(), file!().to_string())
        }

        fn process(
            &self,
            mut message: SendableRecordBatchStreamMessageMap,
            _diagnostic_builder: Option<&DiagnosticBuilder>,
            _runtime_env: Arc<RuntimeEnv>,
        ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
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
            let mut outbox = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
            let out_m = SendableRecordBatchStreamMessage::get_builder()
                .with_name("messages")
                .with_message(out);
            let _ = outbox.insert("messages".to_string(), out_m);
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
                let messages = SubjectBuilder::new()
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
                        let tool_table = SubjectBuilder::new()
                            .with_name("messages")
                            .with_record_batches(batches)?
                            .build()?;
                        let tool_vec: Vec<Tool> = tool_table
                            .get_column_as_vec_str("tool")
                            .iter()
                            .map(|s| {
                                let tool: Tool = serde_json::from_str(s).unwrap();
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
                let batch = create_chat_record_batch(
                    vec!["assistant".to_string()],
                    vec![prompt],
                    vec![create_timestamp_micros()],
                )?;

                // record the poll
                self.sample += 1;
                Poll::Ready(Some(Ok(batch)))
            } else if self.sample < self.sample_len {
                // mock generationg of next token
                let response = match self.found_tools {
                    true => format!("Function{}", self.sample),
                    false => format!("Response{}", self.sample),
                };
                let batch = create_chat_record_batch(
                    vec!["assistant".to_string()],
                    vec![response],
                    vec![create_timestamp_micros()],
                )?;

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
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::array::RecordBatch;
    use futures::TryStreamExt;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, RuntimeEnv, SubjectBuilder, SubjectBuilderTrait,
        SubjectTrait, test_subject,
    };
    use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, HashMap};
    use phymes_event::Publication;
    use phymes_message::{MessageBuilderTrait, SendableRecordBatchStreamMessage};
    use phymes_streams::ChatBuilderTraitExt;

    use super::*;
    use crate::ProcessorTrait;

    #[tokio::test]
    async fn test_message_builder_no_tool_no_doc() -> Result<()> {
        // Make the system prompt and add the user query
        let message_builder = SubjectBuilder::new()
            .with_name("messages")
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the message
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            "messages".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("messages")
                .with_publisher("")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        let chat_processor = test_messages::CandleChatMockProcessor::new("ChatBot", "");
        let mut stream = chat_processor.process(
            message,
            Some(&DiagnosticBuilder::new(&Diagnostics::new())),
            Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?),
        )?;

        // Update the chat history with the response
        let (message_builder, stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove("messages").unwrap().message.take().unwrap(),
                10,
            )
            .await?;
        let messages = message_builder.clone().build()?;

        let messages_content = messages.get_column_as_vec_str("content");
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
        let messages = SubjectBuilder::new()
            .with_name("")
            .with_record_batches(batches)?
            .build()?;
        let messages_content = messages.get_column_as_vec_str("content");
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
        let message_builder = SubjectBuilder::new()
            .with_name("messages")
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the message
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            "messages".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("messages")
                .with_publisher("")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            "tools".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("tools")
                .with_publisher("")
                .with_subject("tools")
                .with_update(&Publication::None)
                .with_message(
                    test_subject::make_test_subject_tool("tools")?.to_record_batch_stream(),
                )
                .build()?,
        );

        // Build the chat task
        let chat_processor = test_messages::CandleChatMockProcessor::new("ChatBot", "");
        let mut stream = chat_processor.process(
            message,
            Some(&DiagnosticBuilder::new(&Diagnostics::new())),
            Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?),
        )?;

        // Update the chat history with the response
        let (message_builder, stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove("messages").unwrap().message.take().unwrap(),
                10,
            )
            .await?;
        let messages = message_builder.clone().build()?;

        let messages_content = messages.get_column_as_vec_str("content");
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
        let messages = SubjectBuilder::new()
            .with_name("")
            .with_record_batches(batches)?
            .build()?;
        let messages_content = messages.get_column_as_vec_str("content");
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
