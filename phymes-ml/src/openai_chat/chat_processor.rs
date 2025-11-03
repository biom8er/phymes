use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuildableTrait,
    BuilderTrait, ChatCompletionRequest, ChatCompletionResponse, ChatTraitExt, FinishReason,
    MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, PubSubTrait,
    RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageMap, StateMap, SubscribeTrait, Table, TableBuilderTrait,
    TablePublish, TableSubscribe, TableTrait, Tool, ToolChoiceType, create_chat_record_batch, remove_message_by_subject
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
    create_timestamp_micros,
};
use reqwest::{Client, header::CONTENT_TYPE};
use tracing::{Level, event};

use crate::{candle_chat::CandleChatConfig, openai_asset::OpenAIRequestState};

#[derive(Debug)]
pub struct OpenAIChatProcessor {
    name: String,
    publications: Vec<TablePublish>,
    subscriptions: Vec<TableSubscribe>,
    subscribe_policy: Box<dyn SubscribeTrait>,
}

impl MappableTrait for OpenAIChatProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for OpenAIChatProcessor {
    fn get_publications(&self) -> Vec<&TablePublish> {
        self.publications.iter().collect()
    }

    fn get_subscriptions(&self) -> Vec<&TableSubscribe> {
        self.subscriptions.iter().collect()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe_policy
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ProcessorTrait for OpenAIChatProcessor {
    fn new(
        name: &str,
        publications: &[TablePublish],
        subscriptions: &[TableSubscribe],
        subscribe_policy: Box<dyn SubscribeTrait>,
    ) -> Arc<dyn ProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe_policy,
        })
    }

    fn new_arc(name: &str) -> Arc<dyn ProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: vec![TablePublish::None],
            subscriptions: vec![TableSubscribe::None],
            subscribe_policy: AvailableTableSubscribePolicies::default().build(),
        })
    }

    fn get_subscribe_policy(&self) -> &dyn SubscribeTrait {
        self.subscribe_policy.as_ref()
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Trace the inbox
        let trace = if let Some(diagnostic_builder) = diagnostic_builder {
            let trace_builder = diagnostic_builder.clone().to_child(self.get_name())?;
            let trace = trace_builder
                .clone()
                .messages(line!(), file!(), self.get_name());
            trace.enter(&message.values().collect::<Vec<_>>());
            Some((trace, trace_builder))
        } else {
            None
        };

        // Extract out the messages, tools, and config
        let messages = match remove_message_by_subject(self.subscriptions.first().unwrap().get_table_name(), &mut message) {
            Some(i) => i.get_message_own(),
            None => {
                return Err(anyhow!(
                    "Messages not provided for {}. Available messages are {:?}",
                    self.get_name(),
                    message.keys()
                ));
            }
        };
        let tools = remove_message_by_subject(self.subscriptions.get(1).unwrap().get_table_name(), &mut message)
            .map(|i| i.get_message_own());
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => {
                return Err(anyhow!(
                    "Config not provided for {}. Available messages are {:?}",
                    self.get_name(),
                    message.keys()
                ));
            }
        };

        // Run the chat stream
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(OpenAIChatStream::new(
            messages,
            tools,
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
            .make_name()?
            .build()?;
        let _ = message.insert(out_m.get_name().to_string(), out_m);

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&message.values().collect::<Vec<_>>());
        }
        Ok(message)
    }
}

pub struct OpenAIChatStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
    /// Optional tools to add to the message
    tools_stream: Option<SendableRecordBatchStream>,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    _runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<CandleChatConfig>,
    /// State of the OpenAI API request
    state: OpenAIRequestState,
}

impl OpenAIChatStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        tools_stream: Option<SendableRecordBatchStream>,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            message_stream,
            tools_stream,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            state: OpenAIRequestState::NotStarted,
        })
    }

    /// Initialize the config for text generation inference
    fn init_config(&mut self, config_table: Table) -> Result<()> {
        if self.config.is_none() {
            let config = CandleChatConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }

    /// Create the request
    fn make_request(&self, messages: Table, tools: Option<Vec<Tool>>) -> ChatCompletionRequest {
        // Convert messages to openAI schema
        let messages_openai = messages.to_openai_messages();

        // Create the request
        let mut req = ChatCompletionRequest::new(
            self.config
                .as_ref()
                .unwrap()
                .openai_asset
                .as_ref()
                .unwrap()
                .get_repository()
                .to_string(),
            messages_openai,
        )
        .max_tokens(self.config.as_ref().unwrap().max_tokens.try_into().unwrap())
        .frequency_penalty(self.config.as_ref().unwrap().frequency_penalty.into())
        .presence_penalty(self.config.as_ref().unwrap().repeat_penalty.into())
        .seed(self.config.as_ref().unwrap().seed.try_into().unwrap())
        .temperature(self.config.as_ref().unwrap().temperature);
        // .top_p(self.config.as_ref().unwrap().top_p.unwrap());

        // Tool arguments
        if let Some(tools) = tools {
            req = req.tools(tools).tool_choice(ToolChoiceType::Required);
        }
        req
    }
}

impl Stream for OpenAIChatStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Iterate through each state until the API request is completed
        match &mut self.state {
            OpenAIRequestState::NotStarted => {
                // Collect the chat history
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
                let messages = Table::get_builder()
                    .with_name("messages")
                    .with_record_batches(batches)?
                    .build()?;

                // Collect the tools
                let tools = match self.tools_stream {
                    Some(ref mut tools) => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) = ready!(tools.poll_next_unpin(cx)) {
                            batches.push(batch);
                        }
                        let tool_table = Table::get_builder()
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

                // initialize the config
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
                let config_table = Table::get_builder()
                    .with_name("config")
                    .with_record_batches(batches)?
                    .build()?;
                self.init_config(config_table)?;

                // make the request
                let fut = Client::new()
                    .post(
                        self.config
                            .as_ref()
                            .unwrap()
                            .openai_asset
                            .unwrap()
                            .get_api_url(self.config.as_ref().unwrap().api_url.clone()),
                    )
                    .bearer_auth(
                        self.config
                            .as_ref()
                            .unwrap()
                            .openai_asset
                            .unwrap()
                            .get_api_key(),
                    )
                    .header(CONTENT_TYPE, "application/json")
                    .json(&self.make_request(messages, tools))
                    .send();
                self.state = OpenAIRequestState::Connecting(Box::pin(fut));
                self.poll_next(cx)
            }
            OpenAIRequestState::Connecting(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(response) => {
                    let fut = response.text();
                    self.state = OpenAIRequestState::ToText(Box::pin(fut));
                    self.poll_next(cx)
                }
                Err(err) => {
                    self.state = OpenAIRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(err.to_string()))))
                }
            },
            OpenAIRequestState::ToText(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(text) => {
                    // Initialize the metrics
                    let baseline_metrics =
                        if let Some(diagnostic_builder) = &self.diagnostic_builder {
                            Some(
                                diagnostic_builder
                                    .clone()
                                    .to_child("OpenAIChatStream")?
                                    .baseline_metrics(line!(), file!(), "poll_next"),
                            )
                        } else {
                            None
                        };
                    let _timer = baseline_metrics
                        .as_ref()
                        .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                    // Parse the response
                    let result = serde_json::from_str::<ChatCompletionResponse>(&text).unwrap();
                    let content = match result.choices[0].finish_reason {
                        None => result.choices[0].message.content.to_owned(),
                        Some(FinishReason::stop) => result.choices[0].message.content.to_owned(),
                        Some(FinishReason::length) => result.choices[0].message.content.to_owned(),
                        Some(FinishReason::tool_calls) => Some(
                            serde_json::to_string(
                                result.choices[0].message.tool_calls.as_ref().unwrap(),
                            )
                            .unwrap(),
                        ),
                        Some(FinishReason::content_filter) => {
                            result.choices[0].message.content.to_owned()
                        }
                        Some(FinishReason::null) => result.choices[0].message.content.to_owned(),
                    };

                    // Handle the returned content
                    let content = match content {
                        Some(s) => s,
                        _ => "".to_string(),
                    };
                    event!(
                        Level::INFO,
                        "Generated the next token {}.",
                        content.as_str()
                    );

                    // Wrap into a record batch
                    let batch = create_chat_record_batch(
                        vec!["assistant".to_string()],
                        vec![content.to_string()],
                        vec![create_timestamp_micros()],
                    )?;
                    self.state = OpenAIRequestState::Done;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    if let Some(baseline_metrics) = &baseline_metrics {
                        baseline_metrics.record_poll(poll)
                    } else {
                        poll
                    }
                }
                Err(err) => {
                    self.state = OpenAIRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(err.to_string()))))
                }
            },
            OpenAIRequestState::Done => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl RecordBatchStream for OpenAIChatStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use phymes_core::{ChatBuilderTraitExt, TableBuilder};
    #[allow(unused_imports)]
    use phymes_diagnostics::{DiagnosticBuilder, Diagnostics, HashMap, SpanBuilder};

    #[cfg(not(feature = "candle"))]
    #[tokio::test]
    async fn test_openai_chat_processor() -> Result<()> {
        use phymes_core::RuntimeEnvTrait;

        use crate::AvailableOpenAIAssets;

        let name = "OpenAIChatProcessor";
        let messages = "messages";

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the chat processor config
        let candle_chat_config = CandleChatConfig {
            max_tokens: 1000,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            api_url: Some("http://0.0.0.0:8000/v1".to_string()),
            openai_asset: Some(AvailableOpenAIAssets::MetaLlamaV3p2_1B),
            ..Default::default()
        };
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config)?;
        let candle_chat_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublish::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            candle_chat_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(candle_chat_config_table.get_name())
                .with_publisher("")
                .with_subject(candle_chat_config_table.get_name())
                .with_update(&TablePublish::None)
                .with_message(candle_chat_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        let chat_processor = OpenAIChatProcessor::new(
            name,
            &[TablePublish::ExtendChunks {
                table_name: messages.to_string(),
                col_name: "content".to_string(),
            }],
            &[
                TableSubscribe::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscribe::None,
                TableSubscribe::AlwaysFullTable {
                    table_name: candle_chat_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = chat_processor.process(
            message,
            Some(&diagnostic_builder),
            Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt"))),
        )?;

        // Update the chat history with the response
        let (message_builder, _stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove(messages).unwrap().get_message_own(),
                1000,
            )
            .await?;
        let messages = message_builder.clone().build()?;
        let json_data = messages.to_json_object()?;
        for row in &json_data {
            if row["role"] != "system" {
                println!("{}: {}", row["role"], row["content"])
            }
        }

        // Expected
        // "**Counting Prime Numbers Up to N**\n=====================================\n\nHere is a Python function that counts prime numbers up to a given number `N`:\n\n```python\ndef count_prime_numbers(n):\n    \"\"\"\n    Returns the count of prime numbers up to n.\n\n    Args:\n        n (int): The upper limit (exclusive) for counting prime numbers.\n\n    Returns:\n        int: The count of prime numbers up to n.\n    \"\"\"\n    def is_prime(num):\n        \"\"\"\n        Checks if a number is prime.\n\n        Args:\n            num (int): The number to check.\n\n        Returns:\n            bool: True if the number is prime, False otherwise.\n        \"\"\"\n        if num < 2:\n            return False\n        for i in range(2, int(num ** 0.5) + 1):\n            if num % i == 0:\n                return False\n        return True\n\n    count = 0\n    for i in range(2, n):\n        if is_prime(i):\n            count += 1\n    return count\n```\n\n**Example Use Cases**\n---------------------\n\n```python\n# Count prime numbers up to 20\nprint(count_prime_numbers(20))  # Output: 8\n\n# Count prime numbers up to 50\nprint(count_prime_numbers(50))  # Output: 15\n```\n\nThis function works by defining a helper function `is_prime` that checks whether a given number is prime or not. It then uses a simple loop to iterate from 2 to `n-1`, and increments the count each time it finds a prime number. The final count is returned by the main function `count_prime_numbers`."

        assert_eq!(json_data.first().unwrap().get("role").unwrap(), "system");
        assert_eq!(
            json_data.first().unwrap().get("content").unwrap(),
            "You are a helpful assistant."
        );
        assert_eq!(json_data.get(1).unwrap().get("role").unwrap(), "user");
        assert_eq!(
            json_data.get(1).unwrap().get("content").unwrap(),
            "Write a function to count prime numbers up to N."
        );
        assert_eq!(json_data.get(2).unwrap().get("role").unwrap(), "assistant");
        assert!(json_data.get(2).unwrap().get("content").is_some());

        Ok(())
    }
}
