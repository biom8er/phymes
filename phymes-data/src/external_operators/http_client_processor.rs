use std::{
    fmt::Write,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
    time::Duration,
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use bytes::Bytes;
use futures::{FutureExt, Stream, StreamExt};
use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, MappableTrait,
    MessageBuilderTrait, MessageTrait, ProcessorTrait, RecordBatchStream, RuntimeEnv,
    SendableRecordBatchStream, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, Table, TableBuilder, TableBuilderTrait, TableTrait,
    create_attachments_batch, create_bytes_fields, create_chat_record_batch, create_values_fields,
    remove_message_by_subject,
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, create_timestamp_micros,
};
use reqwest::{
    Client, Response,
    header::{CONTENT_TYPE, USER_AGENT},
};
use serde_json::{Map, Value};

use crate::{
    DataConfigTrait,
    external_operators::http_client_config::{
        HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType,
    },
};

/// The state of the HTTP Client API request
///
/// # Notes
/// * We need to capture each stage of the request so that the connection
///   is not dropped during repeated polling of the stream.
pub enum HTTPClientRequestState {
    NotStarted,
    Connecting(Pin<Box<dyn Future<Output = Result<Response, reqwest::Error>> + Send + 'static>>),
    ToText(Pin<Box<dyn Future<Output = Result<String, reqwest::Error>> + Send + 'static>>),
    ToBytes(Pin<Box<dyn Future<Output = Result<Bytes, reqwest::Error>> + Send + 'static>>),
    Ready(Vec<RecordBatch>),
    Done,
}

/// Error reporting method for Reqwest error
pub(crate) fn error_report(mut err: &(dyn std::error::Error + 'static)) -> String {
    let mut s = format!("{err}");
    while let Some(src) = err.source() {
        let _ = write!(s, "\n\nCaused by: {src}");
        err = src;
    }
    s
}

#[derive(Debug)]
pub struct HTTPClientRequestProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for HTTPClientRequestProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for HTTPClientRequestProcessor {
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
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the stream
        let out = Box::pin(HTTPClientRequestStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.cloned(),
        )?);

        // Prepare the message builder
        let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let builder = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.get_name())
            .with_message(out);
        let _ = builder_map.insert(self.get_name().to_string(), builder);

        Ok(builder_map)
    }
}

pub struct HTTPClientRequestStream {
    /// Output schema
    schema: SchemaRef,
    /// The messages containing the lhs and rhs
    /// which we cannot determine until we intialize the config
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    _runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<HTTPClientConfig>,
    /// State of the OpenAI API request
    state: HTTPClientRequestState,
    /// The polled record batches from the input
    record_batches: Option<RecordBatch>,
    /// The record batches or url from the config
    json_str: Option<String>,
    /// Optional copy of the query string which is needed for downloading PDFs and other data assets
    url: Option<String>,
    /// Optional copy of the contenty type string which is needed for downloading PDFs and other data assets
    content_type: Option<String>,
}

impl HTTPClientRequestStream {
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
            record_batches: None,
            json_str: None,
            url: None,
            content_type: None,
        })
    }
}

impl Stream for HTTPClientRequestStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Iterate through each state until the API request is completed
        match &mut self.state {
            HTTPClientRequestState::NotStarted => {
                // Initialize the config
                if self.config.is_none() {
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    let config_table = TableBuilder::new()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    if config_table
                        .get_schema()
                        .fields()
                        .contains(&create_values_fields())
                    {
                        let config_json = config_table.get_column_as_vec_str("values").join("");
                        let config = serde_json::from_str::<HTTPClientConfig>(&config_json)?;
                        self.config.replace(config);
                    } else if config_table
                        .get_schema()
                        .fields()
                        .contains(&create_bytes_fields())
                    {
                        let config_json = config_table
                            .get_column_as_vec_nested_primitive::<u8>("bytes")?
                            .into_iter()
                            .map(|b| String::from_utf8(b).unwrap())
                            .collect::<Vec<_>>()
                            .join("");
                        let config = serde_json::from_str::<HTTPClientConfig>(&config_json)?;
                        self.config.replace(config);
                    } else {
                        let config = HTTPClientConfig::from_table(&config_table)?;
                        self.config.replace(config);
                    }
                }

                // Collect the request data
                if self.record_batches.is_none()
                    && self.json_str.is_none()
                    && let Some(subject_name) = self.config.as_ref().unwrap().subject_name.clone()
                {
                    match remove_message_by_subject(&subject_name, &mut self.messages) {
                        // Poll the next batches in a streaming fashion
                        Some(mut fut) => {
                            if let Some(Ok(batch)) =
                                ready!(fut.get_message_mut().poll_next_unpin(cx))
                            {
                                self.record_batches.replace(batch);
                            }
                            self.messages.insert(fut.get_name().to_string(), fut);
                        }
                        // Extract the data from the config
                        None => {
                            if let Some(json) = self.config.as_mut().unwrap().json.take() {
                                self.json_str.replace(json.to_string());
                            } else {
                                self.state = HTTPClientRequestState::Done;
                                return Poll::Ready(Some(Err(anyhow!(
                                    "Subject `{subject_name}` was not found in the messages. The available message subjects are `{:?}`",
                                    self.messages.keys()
                                ))));
                            }
                        }
                    }
                } else if self.record_batches.is_none()
                    && self.json_str.is_none()
                    && let Some(json) = self.config.as_mut().unwrap().json.take()
                {
                    // Extract the data from the config
                    self.json_str.replace(json.to_string());
                }

                // The poll ends when there are no more batches
                if self.record_batches.is_none() && self.json_str.is_none() {
                    self.state = HTTPClientRequestState::Done;
                    return Poll::Ready(None);
                }

                // Create HTTP client with timeout
                let client = Client::builder()
                    .timeout(Duration::from_secs(
                        self.config.as_ref().unwrap().timeout.try_into()?,
                    ))
                    .build()?;

                // Make the request
                // DM: A future optimization maybe to treat each row as a parallel API request
                let fut = match self.config.as_ref().unwrap().request_type {
                    HTTPClientRequestType::Get => {
                        // Prioritize the message data over the config when building the url
                        let query_url = if let Some(batches) = self.record_batches.take() {
                            let messages = Table::get_builder()
                                .with_name("messages")
                                .with_record_batches(vec![batches])?
                                .build()?;

                            // Join the `content` fields together for the case of multiple rows
                            let query_str = messages.get_column_as_vec_str("content").join("");

                            Some(query_str)
                        } else {
                            self.json_str.clone()
                        };
                        let url = self.config.as_ref().unwrap().url(query_url.as_deref());

                        // Save the URL when downloading data
                        if self.config.as_ref().unwrap().request_schema
                            == HTTPClientRequestSchemas::Attachments
                        {
                            self.url.replace(url.to_owned());
                        }

                        // Make the request
                        let mut client = client.get(url);
                        if let Ok(token) = self.config.as_ref().unwrap().api_key() {
                            client = client.bearer_auth(token);
                        }
                        client.header(USER_AGENT, self.config.as_ref().unwrap().user_agent_type.clone().ok_or(anyhow!("User Agent type (header value) needs to be specified for GET requests."))?)
                            .send()
                    }
                    HTTPClientRequestType::Post => {
                        // Prioritize the message data over the config when building the JSON body and url
                        let (json_data, url) = if let Some(batches) = self.record_batches.take() {
                            // Extract the table as a JSON object
                            let messages = Table::get_builder()
                                .with_name("messages")
                                .with_record_batches(vec![batches])?
                                .build()?;
                            let mut json_object = messages.to_json_object()?;

                            // DM: currently, only the last row is used similar to configs...
                            let json_data = json_object.pop().unwrap();

                            // Build the url
                            let url = if let Some(json_str) =
                                self.config.as_ref().unwrap().json.as_ref()
                            {
                                self.config.as_ref().unwrap().url(Some(json_str))
                            } else {
                                self.config.as_ref().unwrap().url(None)
                            };

                            (json_data, url)
                        } else if let Some(json_str) = self.json_str.as_ref() {
                            let json_data = serde_json::from_str::<Map<String, Value>>(json_str)?;
                            let url = self.config.as_ref().unwrap().base_url.clone();
                            (json_data, url)
                        } else {
                            self.state = HTTPClientRequestState::Done;
                            return Poll::Ready(Some(Err(anyhow!(
                                "POST json data was not found in the messages nor in the config."
                            ))));
                        };

                        // Save the URL when downloading data
                        if self.config.as_ref().unwrap().request_schema
                            == HTTPClientRequestSchemas::Attachments
                        {
                            self.json_str.replace(url.to_owned());
                        }

                        // Make the request
                        let mut client = client.post(url);
                        if let Ok(token) = self.config.as_ref().unwrap().api_key() {
                            client = client.bearer_auth(token)
                        }
                        if let Some(user_agent) =
                            self.config.as_ref().unwrap().user_agent_type.as_ref()
                        {
                            client = client.header(USER_AGENT, user_agent.to_string());
                        }
                        client
                            .header(
                                CONTENT_TYPE,
                                self.config.as_ref().unwrap().content_type.clone().ok_or(
                                    anyhow!(
                                        "Content type needs to be specified for POST requests."
                                    ),
                                )?,
                            )
                            .json(&json_data)
                            .send()
                    }
                    _ => {
                        self.state = HTTPClientRequestState::Done;
                        return Poll::Ready(Some(Err(anyhow!(
                            "Request type {} is not supported yet.",
                            self.config.as_ref().unwrap().request_type
                        ))));
                    }
                };

                // Update the request state and poll next
                self.state = HTTPClientRequestState::Connecting(Box::pin(fut));
                self.poll_next(cx)
            }
            HTTPClientRequestState::Connecting(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(response) => {
                    // Determine the content type and parse accordingly
                    let content_type = response
                        .headers()
                        .get(CONTENT_TYPE)
                        .and_then(|ct| ct.to_str().ok())
                        .unwrap_or("");
                    self.content_type.replace(content_type.to_owned());

                    match self.config.as_ref().unwrap().request_schema {
                        HTTPClientRequestSchemas::Messages => {
                            let text = response.text();
                            self.state = HTTPClientRequestState::ToText(Box::pin(text));
                            self.poll_next(cx)
                        }
                        HTTPClientRequestSchemas::Attachments => {
                            let bytes = response.bytes();
                            self.state = HTTPClientRequestState::ToBytes(Box::pin(bytes));
                            self.poll_next(cx)
                        }
                        _ => {
                            let bytes = response.bytes();
                            self.state = HTTPClientRequestState::ToBytes(Box::pin(bytes));
                            self.poll_next(cx)
                        }
                    }
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
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
                                    .to_child("HTTPClientRequestStream")?
                                    .baseline_metrics(line!(), file!(), "poll_next"),
                            )
                        } else {
                            None
                        };
                    let _timer = baseline_metrics
                        .as_ref()
                        .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                    // Parse the response
                    let batch = match self.config.as_ref().unwrap().request_schema {
                        HTTPClientRequestSchemas::Messages => create_chat_record_batch(
                            vec!["tool".to_string()],
                            vec![text],
                            vec![create_timestamp_micros()],
                        )?,
                        _ => {
                            self.state = HTTPClientRequestState::Done;
                            return Poll::Ready(Some(Err(anyhow!(
                                "Request schema {} is not supported yet.",
                                self.config.as_ref().unwrap().request_schema
                            ))));
                        }
                    };

                    // Ready to poll the batches
                    self.state = HTTPClientRequestState::Ready(vec![batch]);
                    self.poll_next(cx)
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
                }
            },
            HTTPClientRequestState::ToBytes(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(bytes) => {
                    // Initialize the metrics
                    let baseline_metrics =
                        if let Some(diagnostic_builder) = &self.diagnostic_builder {
                            Some(
                                diagnostic_builder
                                    .clone()
                                    .to_child("HTTPClientRequestStream")?
                                    .baseline_metrics(
                                        line!(),
                                        file!(),
                                        "poll_next.HTTPClientRequestState::ToBytes",
                                    ),
                            )
                        } else {
                            None
                        };
                    let _timer = baseline_metrics
                        .as_ref()
                        .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                    // Determine the filename
                    let filename = if let Some(url) = self.url.take() {
                        url.split("/").last().unwrap_or_default().to_string()
                    } else {
                        String::new()
                    };

                    // Parse the response
                    let batch = match self.config.as_ref().unwrap().request_schema {
                        HTTPClientRequestSchemas::Attachments => create_attachments_batch(
                            vec![filename],
                            vec![self.content_type.take().unwrap_or_default()],
                            vec![bytes.to_vec()],
                            vec!["tool".to_string()],
                            vec![create_timestamp_micros()],
                        )?,
                        _ => {
                            self.state = HTTPClientRequestState::Done;
                            return Poll::Ready(Some(Err(anyhow!(
                                "Request schema {} is not supported yet.",
                                self.config.as_ref().unwrap().request_schema
                            ))));
                        }
                    };

                    // Ready to poll the batches
                    self.state = HTTPClientRequestState::Ready(vec![batch]);
                    self.poll_next(cx)
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
                }
            },
            HTTPClientRequestState::Ready(batches) => {
                // Ready the next poll
                if let Some(batch) = batches.pop() {
                    // Initialize the metrics
                    let baseline_metrics =
                        if let Some(diagnostic_builder) = &self.diagnostic_builder {
                            Some(
                                diagnostic_builder
                                    .clone()
                                    .to_child("HTTPClientRequestStream")?
                                    .baseline_metrics(
                                        line!(),
                                        file!(),
                                        "poll_next.HTTPClientRequestState::Ready",
                                    ),
                            )
                        } else {
                            None
                        };
                    let _timer = baseline_metrics
                        .as_ref()
                        .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                    // Record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    if let Some(baseline_metrics) = &baseline_metrics {
                        baseline_metrics.record_poll(poll)
                    } else {
                        poll
                    }
                // Or reset the state to poll the next batch
                } else {
                    self.state = HTTPClientRequestState::NotStarted;
                    self.poll_next(cx)
                }
            }
            HTTPClientRequestState::Done => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for HTTPClientRequestStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use crate::{extract_pdf, filter_pdf, load_pdf_document};

    use super::*;
    use futures::TryStreamExt;
    use phymes_core::{
        ChatBuilderTraitExt, RuntimeEnvTrait, TableBuilder, TablePublication, open_alex,
        semantic_scholar,
    };
    use phymes_diagnostics::{DiagnosticBuilder, Diagnostics, HashMap, SpanBuilder};

    #[tokio::test]
    async fn test_http_client_processor_open_alex_get_message_from_message() -> Result<()> {
        // Case 1: GET from messages
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // OpenAlex request filters
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "publication_year".to_string(),
            Value::String("2020".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Works,
            ..Default::default()
        };

        // Config for the HTTP Processor
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(&open_alex_request.to_get_query()?, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        assert!(snippet.contains("https://openalex.org/W3038568908"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_open_alex_get_message_from_config() -> Result<()> {
        // Case 1: GET from messages
        let name = "HTTPClientRequestProcessor";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // OpenAlex request filters
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "publication_year".to_string(),
            Value::String("2020".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Works,
            ..Default::default()
        };

        // Config for the HTTP Processor
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        assert!(snippet.contains("https://openalex.org/W3038568908"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_open_alex_get_blob_from_message() -> Result<()> {
        // Case 1: GET from messages
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // OpenAlex request filters
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "publication_year".to_string(),
            Value::String("2020".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Works,
            ..Default::default()
        };

        // Config for the HTTP Processor
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(&open_alex_request.to_get_query()?, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("filename");
        assert_eq!(
            result,
            ["works?page=1&per-page=1&filter=publication_year:\"2020\""]
        );
        let result = table.get_column_as_vec_str("extension");
        assert_eq!(result, ["application/json"]);
        let result = table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let snippet = String::from_utf8(result)?;
        assert!(snippet.contains("https://openalex.org/W3038568908"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_e_utils_e_search() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build ESearch query
        let mesh_term = "Diabetes Mellitus";
        let year_from = 2020;
        let year_to = 2023;
        let journal_filter = Some("Lancet");
        let mut query = format!("{mesh_term}[MeSH Terms]");
        if let Some(journal) = journal_filter {
            query.push_str(&format!(" AND \"{journal}\"[Journal]"));
        }

        let esearch_url = format!(
            "db=pubmed&term={}&retmode=json&retmax=5&mindate={}&maxdate={}",
            urlencoding::encode(&query),
            year_from,
            year_to
        );

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?".to_string(),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(&esearch_url, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content").join("");
        assert!(result.contains("\"idlist\":["));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_e_utils_e_fetch() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build EFetch query
        let ids = ["37997144", "37997132", "37997130", "37997120", "37997092"].join(",");
        let efetch_url = format!("db=pubmed&id={ids}&retmode=xml");

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?".to_string(),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(&efetch_url, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("filename");
        assert_eq!(
            result,
            ["efetch.fcgi?db=pubmed&id=37997144,37997132,37997130,37997120,37997092&retmode=xml"]
        );
        let result = table.get_column_as_vec_str("extension");
        assert_eq!(result, ["text/xml; charset=UTF-8"]);
        let result = table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let snippet = String::from_utf8(result)?;
        assert!(snippet.contains("MedlineCitation"));
        assert!(snippet.contains("!DOCTYPE PubmedArticleSet PUBLIC"));
        assert!(snippet.contains("https://dtd.nlm.nih.gov/ncbi/pubmed/"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_pdf_download() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build pathname for download
        let id = "2508.18700";
        let download_url = format!("pdf/{id}");

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://arxiv.org/".to_string(),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(&download_url, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;
        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(result, ["tool"]);
        let filenames = table.get_column_as_vec_str("filename");
        assert_eq!(filenames, ["2508.18700"]);
        let result = table.get_column_as_vec_str("extension");
        assert_eq!(result, ["application/pdf"]);

        // Check the PDF
        let result = table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let pdf = filter_pdf(load_pdf_document(&result)?);
        let docs = [(filenames.first().unwrap().to_string(), pdf)];
        let pdf_batch = extract_pdf(&docs)?;
        let table = TableBuilder::new()
            .with_record_batches(vec![pdf_batch])?
            .with_name("")
            .build()?;
        let result = table.get_column_as_vec_str("chunk_id");
        assert_eq!(result, ["2508.18700_1", "2508.18700_2", "2508.18700_3"]);
        let result = table.get_column_as_vec_str("document_id");
        assert_eq!(result, ["2508.18700", "2508.18700", "2508.18700"]);
        let result = table.get_column_as_vec_str("text");
        let snippet = result.first().unwrap().to_string();
        assert_eq!(
            snippet[..100],
            *"Taming the One-Epoch Phenomenon in Online Recommendation System by Two-stage Contrastive ID Pre-trai"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_semantic_scholar() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 30,
            request_type: HTTPClientRequestType::Post,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            content_type: Some("application/json".to_string()),
            base_url: "https://api.semanticscholar.org/recommendations/v1/papers/?".to_string(),
            json: Some("fields=title,url,authors&limit=3".to_string()),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the request body
        let req_body = semantic_scholar::RecommendationsRequest {
            positive_papers: Some(vec!["649def34f8be52c8b66281af98ae884c09aef38b".to_string()]),
            negative_papers: Some(vec!["ArXiv:1805.02262".to_string()]),
        };
        let req_body_json = serde_json::to_vec(&req_body)?;
        let req_body_table = TableBuilder::new()
            .with_name(messages)
            .with_json(&req_body_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(req_body_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        let snippet = result.first().unwrap().to_string();
        assert!(snippet.contains("{\"paperId\":"));

        Ok(())
    }

    #[tokio::test]
    #[ignore = "for generating data to test OpenAlex parsers"]
    async fn test_http_client_processor_open_alex_test_data() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Author
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("has_orcid".to_string(), Value::String("true".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Authors,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Author: {snippet}");

        // Institution
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("country_code".to_string(), Value::String("us".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Institutions,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Institution: {snippet}");

        // Topic
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "display_name.search".to_string(),
            Value::String("artificial+intelligence".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Topics,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Topic: {snippet}");

        // Award
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "funder.id".to_string(),
            Value::String("F4320306076".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Awards,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Award: {snippet}");

        // Funder
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("country_code".to_string(), Value::String("us".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Funders,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Funder: {snippet}");

        // Publisher
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "display_name.search".to_string(),
            Value::String("elsevier".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Publishers,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Publisher: {snippet}");

        // Source
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("has_issn".to_string(), Value::String("true".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Sources,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Source: {snippet}");

        Ok(())
    }
}
