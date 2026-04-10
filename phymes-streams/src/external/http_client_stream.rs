use std::{
    collections::VecDeque,
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
use phymes_subject::{
    BuilderTrait, MappableTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait, create_timestamp_micros,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, create_attachments_batch, create_chat_record_batch,
};
use reqwest::{
    Client, Response,
    header::{CONTENT_TYPE, USER_AGENT},
};
use serde_json::{Map, Value};

use crate::{HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType};

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
    record_batches: Option<VecDeque<Map<String, Value>>>,
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
                    let config_table = SubjectBuilder::new()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    let config = HTTPClientConfig::from_subject(&config_table)?;
                    self.config.replace(config);
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
                                let json_object = SubjectBuilder::default()
                                    .with_name("")
                                    .with_record_batches(vec![batch])?
                                    .build()?
                                    .to_json_object()?;
                                self.record_batches.replace(json_object.into());
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
                let fut = match self.config.as_ref().unwrap().request_type {
                    HTTPClientRequestType::Get => {
                        // Prioritize the message data over the config when building the url
                        let query_url = if let Some(mut batch) = self.record_batches.take() {
                            if let Some(row) = batch.pop_front() {
                                let query_str = row.get("content").ok_or(anyhow!("Missing key `content` to build query string from RecordBatches in HTTPClientRequestStream."))?;
                                let query_str = query_str.as_str().ok_or(anyhow!("Unable to build string from key `content` from RecordBatches in HTTPClientRequestStream."))?.to_string();
                                if !batch.is_empty() {
                                    self.record_batches.replace(batch);
                                }
                                Some(query_str)
                            } else {
                                self.state = HTTPClientRequestState::Done;
                                return Poll::Ready(None);
                            }
                        } else if let Some(json_str) = self.json_str.take() {
                            Some(json_str)
                        } else {
                            self.state = HTTPClientRequestState::Done;
                            return Poll::Ready(None);
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
                        let (json_data, url) = if let Some(mut batch) = self.record_batches.take() {
                            let json_data = if let Some(row) = batch.pop_front() {
                                if !batch.is_empty() {
                                    self.record_batches.replace(batch);
                                }
                                row
                            } else {
                                self.state = HTTPClientRequestState::Done;
                                return Poll::Ready(None);
                            };

                            // Build the url
                            let url = if let Some(json_str) =
                                self.config.as_ref().unwrap().json.as_ref()
                            {
                                self.config.as_ref().unwrap().url(Some(json_str))
                            } else {
                                self.config.as_ref().unwrap().url(None)
                            };

                            (json_data, url)
                        } else if let Some(json_str) = self.json_str.take() {
                            let json_data = serde_json::from_str::<Map<String, Value>>(&json_str)?;
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
                            self.url.replace(url.to_owned());
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
                    // self.state = HTTPClientRequestState::Done;
                    self.state = HTTPClientRequestState::NotStarted;
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
                        // url.split("/").last().unwrap_or_default().to_string()
                        url
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
                    // self.state = HTTPClientRequestState::Done;
                    self.state = HTTPClientRequestState::NotStarted;
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
