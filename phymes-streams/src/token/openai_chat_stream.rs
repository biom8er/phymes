use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{FutureExt, Stream, StreamExt};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait, create_timestamp_micros,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::CandleChatConfig;
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, ChatCompletionRequest, ChatCompletionResponse,
    FinishReason, Tool, ToolChoiceType, create_chat_record_batch,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    Subject, SubjectBuilderTrait, SubjectTrait,
};
use reqwest::{Client, header::CONTENT_TYPE};
use tracing::{Level, event};

use crate::{ChatTraitExt, HTTPClientRequestState};

pub struct OpenAIChatStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The messages and optional tools
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    _runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<CandleChatConfig>,
    /// State of the OpenAI API request
    state: HTTPClientRequestState,
}

impl OpenAIChatStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            messages,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            state: HTTPClientRequestState::NotStarted,
        })
    }

    /// Initialize the config for text generation inference
    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = CandleChatConfig::from_subject(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }

    /// Create the request
    fn make_request(&self, messages: Subject, tools: Option<Vec<Tool>>) -> ChatCompletionRequest {
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
            HTTPClientRequestState::NotStarted => {
                // Initialize the config
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
                let config_table = Subject::get_builder()
                    .with_name("config")
                    .with_record_batches(batches)?
                    .build()?;
                self.init_config(config_table)?;

                // Collect the chat history
                let messages_table = self.config.as_ref().unwrap().messages.clone();
                let mut message_stream = if let Some(s) =
                    remove_message_by_subject(&messages_table, &mut self.messages)
                {
                    s.get_message_own()
                } else {
                    return Poll::Ready(Some(Err(anyhow!(
                        "Message history subject was not found in the message stream. Available messages are {:?}",
                        self.messages.keys()
                    ))));
                };
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(message_stream.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
                let messages = Subject::get_builder()
                    .with_name("messages")
                    .with_record_batches(batches)?
                    .build()?;

                // Collect the tools
                let tools_table_name = self
                    .config
                    .as_ref()
                    .unwrap()
                    .tools
                    .as_ref()
                    .map(|tools_table_name| tools_table_name.to_string());
                let tools = if let Some(tools_table_name) = tools_table_name {
                    if let Some(s) =
                        remove_message_by_subject(&tools_table_name, &mut self.messages)
                    {
                        let mut tools_stream = s.get_message_own();
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) = ready!(tools_stream.poll_next_unpin(cx)) {
                            batches.push(batch);
                        }
                        let tool_table = Subject::get_builder()
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
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Make the request
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
                self.state = HTTPClientRequestState::Connecting(Box::pin(fut));
                self.poll_next(cx)
            }
            HTTPClientRequestState::Connecting(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(response) => {
                    let fut = response.text();
                    self.state = HTTPClientRequestState::ToText(Box::pin(fut));
                    self.poll_next(cx)
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(err.to_string()))))
                }
            },
            HTTPClientRequestState::ToText(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
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
                    self.state = HTTPClientRequestState::Done;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    if let Some(baseline_metrics) = &baseline_metrics {
                        baseline_metrics.record_poll(poll)
                    } else {
                        poll
                    }
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(err.to_string()))))
                }
            },
            HTTPClientRequestState::ToBytes(_fut) => Poll::Ready(None),
            HTTPClientRequestState::Ready(_batches) => Poll::Ready(None),
            HTTPClientRequestState::Done => Poll::Ready(None),
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
