use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use super::chat_completion::{
    ChatCompletionRequest, ChatCompletionResponse, FinishReason, Tool, ToolChoiceType,
};
use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
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
use reqwest::{Client, header::CONTENT_TYPE};
use tracing::{Level, event};

use crate::{
    candle_chat::{chat_config::CandleChatConfig, message_history::{create_timestamp, MessageHistoryTraitExt}},
    openai_asset::OpenAIRequestState,
};

#[derive(Default, Debug)]
pub struct OpenAIChatProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl OpenAIChatProcessor {
    pub fn new_with_pub_sub_for(
        name: &str,
        publications: &[ArrowTablePublish],
        subscriptions: &[ArrowTableSubscribe],
        forward: &[&str],
    ) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            forward: forward.iter().map(|s| s.to_string()).collect(),
        })
    }
}

impl MappableTrait for OpenAIChatProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for OpenAIChatProcessor {
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
        metrics: ArrowTaskMetricsSet,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract out the messages, documents, tools, and config
        let messages = match message.remove(self.subscriptions.first().unwrap().get_table_name()) {
            Some(i) => i.get_message_own(),
            None => return Err(anyhow!("Messages not provided for {}.", self.get_name())),
        };
        let tools = message
            .remove(self.subscriptions.get(1).unwrap().get_table_name())
            .map(|i| i.get_message_own());
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the chat stream
        let out = Box::pin(OpenAIChatStream::new(
            messages,
            tools,
            config,
            Arc::clone(&runtime_env),
            BaselineMetrics::new(&metrics.clone(), self.get_name()),
        )?);
        let out_m = ArrowOutgoingMessage::get_builder()
            .with_name(self.publications.first().unwrap().get_table_name())
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
            .build()?;
        let _ = message.insert(out_m.get_name().to_string(), out_m);
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
    baseline_metrics: BaselineMetrics,
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
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        // Default schema
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let timestamp = Field::new("timestamp", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content, timestamp]));

        Ok(Self {
            schema,
            message_stream,
            tools_stream,
            baseline_metrics,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            state: OpenAIRequestState::NotStarted,
        })
    }

    /// Initialize the config for text generation inference
    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
        if self.config.is_none() {
            let config: CandleChatConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
            self.config.replace(config);
        }
        Ok(())
    }

    /// Create the request
    fn make_request(
        &self,
        messages: ArrowTable,
        tools: Option<Vec<Tool>>,
    ) -> ChatCompletionRequest {
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
            req = req.tools(tools).tool_choice(ToolChoiceType::Auto);
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
                let messages = ArrowTable::get_builder()
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
                        let tool_table = ArrowTable::get_builder()
                            .with_name("messages")
                            .with_record_batches(batches)?
                            .build()?;
                        let tool_vec: Vec<Tool> = tool_table
                            .get_column_as_str_vec("tool")
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
                let config_table = ArrowTable::get_builder()
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
                    let metrics = self.baseline_metrics.clone();
                    let _timer = metrics.elapsed_compute().timer();

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

                    // Wrap into a record batch
                    let role_arr: ArrayRef =
                        Arc::new(StringArray::from(vec!["assistant".to_string()]));
                    let content_arr: ArrayRef = Arc::new(StringArray::from(vec![content]));
                    let timestamp_arr: ArrayRef = Arc::new(StringArray::from(vec![create_timestamp()]));
                    let batch = RecordBatch::try_from_iter(vec![("role", role_arr), ("content", content_arr), ("timestamp", timestamp_arr)])?;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    self.state = OpenAIRequestState::Done;
                    metrics.record_poll(poll)
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
    use crate::candle_chat::message_history::MessageHistoryBuilderTraitExt;
    #[allow(unused_imports)]
    use phymes_core::{
        metrics::HashMap, session::runtime_env::RuntimeEnvTrait,
        table::arrow_table::ArrowTableBuilder,
    };

    #[cfg(not(feature = "candle"))]
    #[tokio::test]
    async fn test_openai_chat_processor() -> Result<()> {
        let name = "OpenAIChatProcessor";
        let messages = "messages";

        // Metrics to compute time and rows
        let metrics = ArrowTaskMetricsSet::new();

        // State for the chat processor config
        let candle_chat_config = CandleChatConfig {
            max_tokens: 1000,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            api_url: Some("http://0.0.0.0:8000/v1".to_string()),
            openai_asset: Some(
                crate::openai_asset::openai_which::WhichOpenAIAsset::MetaLlamaV3p2_1B,
            ),
            ..Default::default()
        };
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config)?;
        let candle_chat_config_table = ArrowTableBuilder::new()
            .with_name(name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name(messages)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the current message state
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&ArrowTablePublish::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            candle_chat_config_table.get_name().to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(candle_chat_config_table.get_name())
                .with_publisher("")
                .with_subject(candle_chat_config_table.get_name())
                .with_update(&ArrowTablePublish::None)
                .with_message(candle_chat_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        let chat_processor = OpenAIChatProcessor::new_with_pub_sub_for(
            name,
            &[ArrowTablePublish::ExtendChunks {
                table_name: messages.to_string(),
                col_name: "content".to_string(),
            }],
            &[
                ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                ArrowTableSubscribe::None,
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: candle_chat_config_table.get_name().to_string(),
                },
            ],
            &[],
        );
        let mut stream = chat_processor.process(
            message,
            metrics.clone(),
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

        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 1);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);

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
