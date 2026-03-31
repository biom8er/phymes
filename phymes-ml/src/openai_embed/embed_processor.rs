use crate::{CandleEmbedConfig, candle_embed::convert_embedding_vector_to_record_batch};

use phymes_data::{DataConfigTrait, HTTPClientRequestState};
use reqwest::{Client, header::CONTENT_TYPE};

use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, EmbeddingRequest,
    EmbeddingResponse, EncodingFormat, MappableTrait, MessageBuilderTrait, MessageTrait,
    ProcessorTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilder,
    SendableRecordBatchStreamMessageBuilderMap, SendableRecordBatchStreamMessageMap, Subject,
    SubjectBuilder, SubjectBuilderTrait, SubjectTrait, remove_message_by_subject,
};
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use anyhow::{Result, anyhow};
use futures::{FutureExt, Stream, StreamExt};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event};

#[derive(Debug)]
pub struct OpenAIEmbedProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for OpenAIEmbedProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for OpenAIEmbedProcessor {
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
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Re-index the messages by the subject name which needs to be unique at this stage
        let message = message
            .into_iter()
            .map(|(_k, v)| (v.get_subject().to_string(), v))
            .collect::<HashMap<_, _>>();

        // Run the embed stream
        let out = Box::pin(OpenAIEmbedStream::new(
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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
    #[allow(unused_imports)]
    use futures::TryStreamExt;
    #[allow(unused_imports)]
    use phymes_core::Publication;

    #[allow(unused_imports)]
    use super::*;

    #[cfg(not(feature = "candle"))]
    #[tokio::test]
    async fn test_openai_embed_processor() -> Result<()> {
        use phymes_diagnostics::{Diagnostics, SpanBuilder};

        use crate::AvailableOpenAIAssets;

        let config = CandleEmbedConfig {
            documents: "text".to_string(),
            input_type: "passage".to_string(),
            api_url: Some("http://0.0.0.0:8001/v1".to_string()),
            openai_asset: Some(AvailableOpenAIAssets::NvidiaLlamaV3p2NvEmbedQA1BV2),
            ..Default::default()
        };

        // Make the config
        let config_table = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config.clone())?, 1)?
            .build()?;

        // Make the runtime
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Case 1: streaming query
        // Make the query input stream
        let query_vec = vec![
            "How much protein should a female eat.",
            "Summit define",
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        ];
        let text: ArrayRef = Arc::new(StringArray::from(query_vec));
        let batch = RecordBatch::try_from_iter(vec![("text", text)])?;
        let document_table = SubjectBuilder::new()
            .with_name("text")
            .with_record_batches(vec![batch])?
            .build()?;
        let document_message = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher("")
            .with_subject("text")
            .with_update(&Publication::None)
            .with_message(document_table.to_record_batch_stream())
            .make_name()?
            .build()?;
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(document_message.get_name().to_string(), document_message);

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make and run the embeddings stream
        let embed_stream = OpenAIEmbedStream::new(
            messages,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder),
        )?;
        let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
        assert_eq!(embeddings.len(), 1);

        // Expected data
        let embeddings_test: Vec<Vec<f32>> = vec![
            vec![-0.0199745, -0.03612664, 0.015255524],
            vec![0.013020273, 0.012949716, 0.015651252],
            vec![-0.016750623, 0.017388858, -0.007890748],
            vec![0.038596537, 0.00942193, 0.011650219],
        ];
        let embeddings_vec = embeddings
            .first()
            .unwrap()
            .column_by_name("embeddings")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .iter()
            .map(|s| {
                s.unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|f| f.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            embeddings_vec.first().unwrap()[0..3],
            embeddings_test.first().unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(1).unwrap()[0..3],
            embeddings_test.get(1).unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(2).unwrap()[0..3],
            embeddings_test.get(2).unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(3).unwrap()[0..3],
            embeddings_test.get(3).unwrap()[0..3]
        );

        // Case 2: streaming query with multiple batches
        // Make the query input stream
        let query_vec1 = vec!["How much protein should a female eat.", "Summit define"];
        let query_vec2 = vec![
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        ];
        let embeddings1: ArrayRef = Arc::new(StringArray::from(query_vec1));
        let embeddings2: ArrayRef = Arc::new(StringArray::from(query_vec2));
        let batch1 = RecordBatch::try_from_iter(vec![("text", embeddings1)])?;
        let batch2 = RecordBatch::try_from_iter(vec![("text", embeddings2)])?;
        let document_table = SubjectBuilder::new()
            .with_name("text")
            .with_record_batches(vec![batch1, batch2])?
            .build()?;
        let document_message = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher("")
            .with_subject("text")
            .with_update(&Publication::None)
            .with_message(document_table.to_record_batch_stream())
            .make_name()?
            .build()?;
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(document_message.get_name().to_string(), document_message);

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make and run the embeddings stream
        let embed_stream = OpenAIEmbedStream::new(
            messages,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder),
        )?;
        let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
        assert_eq!(embeddings.len(), 2);
        let embeddings_vec = embeddings
            .first()
            .unwrap()
            .column_by_name("embeddings")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .iter()
            .map(|s| {
                s.unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|f| f.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Expected data
        let embeddings_test: Vec<Vec<f32>> = vec![
            vec![-0.019996276, -0.036101155, 0.015297651],
            vec![0.013022803, 0.012991025, 0.015626218],
        ];
        assert_eq!(
            embeddings_vec.first().unwrap()[0..3],
            embeddings_test.first().unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(1).unwrap()[0..3],
            embeddings_test.get(1).unwrap()[0..3]
        );
        Ok(())
    }
}
