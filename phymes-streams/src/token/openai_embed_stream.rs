use anyhow::{Result, anyhow};
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use futures::{FutureExt, Stream, StreamExt};
use phymes_subject::{
    BuildableTrait, BuilderTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::{CandleEmbedConfig, convert_embedding_vector_to_record_batch};
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, EmbeddingRequest, EmbeddingResponse, EncodingFormat,
};
use reqwest::{Client, header::CONTENT_TYPE};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use crate::HTTPClientRequestState;

pub struct OpenAIEmbedStream {
    /// Output schema (embeddings)
    schema: SchemaRef,
    /// The documents (or queries) to parse
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for embed inference
    documents_stream: Option<SendableRecordBatchStream>,
    /// Parameters for embed inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    _runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for embed inference
    config: Option<CandleEmbedConfig>,
    /// The input documents
    documents: Option<Subject>,
    /// State of the OpenAI API request
    state: HTTPClientRequestState,
    /// sample number
    sample: usize,
}

impl OpenAIEmbedStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::DocumentEmbeddings.to_schema(),
            messages,
            documents_stream: None,
            config_stream,
            diagnostic_builder,
            _runtime_env: runtime_env,
            config: None,
            documents: None,
            state: HTTPClientRequestState::NotStarted,
            sample: 0,
        })
    }

    /// Initialize the config for text embedding inference
    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = CandleEmbedConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }

    /// Create the request
    fn make_request(&self, documents: Vec<String>) -> EmbeddingRequest {
        // Determine the input type
        // DM: NVIDIA embedding models have the `input_type` parameter
        //  which is not OpenAI compatible, so the NVIDIA team offers
        //  a workaround by appending the input_type to the model name
        // see <https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html>
        let model = format!(
            "{}-{}",
            self.config
                .as_ref()
                .unwrap()
                .openai_asset
                .as_ref()
                .unwrap()
                .get_repository(),
            self.config.as_ref().unwrap().input_type,
        );
        let mut req = EmbeddingRequest::new(model, documents);

        // Specify the dimensions
        if self.config.as_ref().unwrap().dimensions.is_some() {
            req.dimensions = Some(self.config.as_ref().unwrap().dimensions.unwrap());
        }

        // Specify the encodings
        if self.config.as_ref().unwrap().encoding_format.as_str() == "base64" {
            req.encoding_format = Some(EncodingFormat::Base64);
        } else {
            req.encoding_format = Some(EncodingFormat::Float);
        }

        req
    }
}

impl Stream for OpenAIEmbedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Embed each stream of record batches whereby each
        // record batch row is a query
        if self.sample == 0 {
            // Iterate through each state until the API request is completed
            match &mut self.state {
                HTTPClientRequestState::NotStarted => {
                    // Initialize the config
                    if self.config.is_none() {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                            batches.push(batch);
                        }
                        let config_table = Subject::get_builder()
                            .with_name("config")
                            .with_record_batches(batches)?
                            .build()?;
                        self.init_config(config_table)?;
                    }

                    // Collect the next batch of queries
                    if self.documents_stream.is_none() {
                        let docs_table = self.config.as_ref().unwrap().documents.clone();
                        if let Some(s) = remove_message_by_subject(&docs_table, &mut self.messages)
                        {
                            self.documents_stream.replace(s.get_message_own())
                        } else {
                            return Poll::Ready(Some(Err(anyhow!(
                                "Documents and queries subject was not found in the message stream. Available messages are {:?}",
                                self.messages.keys()
                            ))));
                        };
                    }
                    let batch =
                        match ready!(self.documents_stream.as_mut().unwrap().poll_next_unpin(cx)) {
                            Some(Ok(batch)) => batch,
                            _ => return Poll::Ready(None),
                        };

                    // Convert to a list of queries
                    let table = SubjectBuilder::new()
                        .with_name("queries")
                        .with_record_batches(vec![batch])?
                        .build()?;
                    let input: Vec<String> = table
                        .get_column_as_vec_str("text")
                        .into_iter()
                        .map(|s| s.to_owned())
                        .collect();
                    self.documents = Some(table);

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
                        .json(&self.make_request(input))
                        .send();
                    self.state = HTTPClientRequestState::Connecting(Box::pin(fut));
                    self.poll_next(cx)
                }
                HTTPClientRequestState::Connecting(fut) => {
                    match ready!(fut.as_mut().poll_unpin(cx)) {
                        Ok(response) => {
                            let fut = response.text();
                            self.state = HTTPClientRequestState::ToText(Box::pin(fut));
                            self.poll_next(cx)
                        }
                        Err(err) => {
                            self.state = HTTPClientRequestState::Done;
                            Poll::Ready(Some(Err(anyhow!(err.to_string()))))
                        }
                    }
                }
                HTTPClientRequestState::ToText(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                    Ok(text) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("OpenAIEmbedStream")?
                                        .baseline_metrics(line!(), file!(), "poll_next"),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Parse the response
                        let result = serde_json::from_str::<EmbeddingResponse>(&text).unwrap();
                        let mut embedding_data: Vec<Vec<f32>> = Vec::new();
                        for embedding in result.data.into_iter() {
                            embedding_data.push(embedding.embedding);
                        }

                        // Wrap into a record batch
                        let batch = convert_embedding_vector_to_record_batch(
                            embedding_data,
                            self.documents.take().unwrap().get_record_batches_own(),
                        )
                        .unwrap();

                        // Record the schema
                        self.schema = batch.schema();
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
                HTTPClientRequestState::Done => {
                    // Increase the sample count
                    self.sample += 1;
                    self.state = HTTPClientRequestState::NotStarted;
                    self.poll_next(cx)
                }
            }
        } else {
            // Iterate through each state until the API request is completed
            match &mut self.state {
                HTTPClientRequestState::NotStarted => {
                    // Collect the next batch of queries
                    let batch =
                        match ready!(self.documents_stream.as_mut().unwrap().poll_next_unpin(cx)) {
                            Some(Ok(batch)) => batch,
                            _ => return Poll::Ready(None),
                        };

                    // Convert to a list of queries
                    let table = SubjectBuilder::new()
                        .with_name("queries")
                        .with_record_batches(vec![batch])?
                        .build()?;
                    let input: Vec<String> = table
                        .get_column_as_vec_str("text")
                        .into_iter()
                        .map(|s| s.to_owned())
                        .collect();
                    self.documents = Some(table);

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
                        .json(&self.make_request(input))
                        .send();
                    self.state = HTTPClientRequestState::Connecting(Box::pin(fut));
                    self.poll_next(cx)
                }
                HTTPClientRequestState::Connecting(fut) => {
                    match ready!(fut.as_mut().poll_unpin(cx)) {
                        Ok(response) => {
                            let fut = response.text();
                            self.state = HTTPClientRequestState::ToText(Box::pin(fut));
                            self.poll_next(cx)
                        }
                        Err(err) => {
                            self.state = HTTPClientRequestState::Done;
                            Poll::Ready(Some(Err(anyhow!(err.to_string()))))
                        }
                    }
                }
                HTTPClientRequestState::ToText(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                    Ok(text) => {
                        // Initialize the metrics
                        let baseline_metrics =
                            if let Some(diagnostic_builder) = &self.diagnostic_builder {
                                Some(
                                    diagnostic_builder
                                        .clone()
                                        .to_child("OpenAIEmbedStream")?
                                        .baseline_metrics(line!(), file!(), "poll_next"),
                                )
                            } else {
                                None
                            };
                        let _timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                        // Parse the response
                        let result = serde_json::from_str::<EmbeddingResponse>(&text).unwrap();
                        let mut embedding_data: Vec<Vec<f32>> = Vec::new();
                        for embedding in result.data.into_iter() {
                            embedding_data.push(embedding.embedding);
                        }

                        // Wrap into a record batch
                        let batch = convert_embedding_vector_to_record_batch(
                            embedding_data,
                            self.documents.take().unwrap().get_record_batches_own(),
                        )
                        .unwrap();

                        // Record the schema
                        self.schema = batch.schema();
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
                HTTPClientRequestState::Done => {
                    // Increase the sample count
                    self.sample += 1;
                    self.state = HTTPClientRequestState::NotStarted;
                    self.poll_next(cx)
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for OpenAIEmbedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
