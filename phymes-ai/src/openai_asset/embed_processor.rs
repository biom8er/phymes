use crate::{
    candle_embed::{
        embed_config::CandleEmbedConfig, embed_processor::convert_embedding_vector_to_record_batch,
    },
    openai_asset::OpenAIRequestState,
};

use super::embedding::{EmbeddingRequest, EmbeddingResponse, EncodingFormat};
use reqwest::{Client, header::CONTENT_TYPE};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
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

use arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event};

#[derive(Default, Debug)]
pub struct OpenAIEmbedProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl OpenAIEmbedProcessor {
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

impl MappableTrait for OpenAIEmbedProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for OpenAIEmbedProcessor {
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

        // Extract out the documents and config
        let documents = match message.remove(self.subscriptions.first().unwrap().get_table_name()) {
            Some(i) => i.get_message_own(),
            None => return Err(anyhow!("Documents not provided for {}.", self.get_name())),
        };
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(OpenAIEmbedStream::new(
            documents,
            config,
            Arc::clone(&runtime_env),
            BaselineMetrics::new(&metrics, self.get_name()),
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

pub struct OpenAIEmbedStream {
    /// Output schema (embeddings)
    schema: SchemaRef,
    /// The input task to process.
    document_stream: SendableRecordBatchStream,
    /// Parameters for embed inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    _runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Parameters for embed inference
    config: Option<CandleEmbedConfig>,
    /// The input documents
    documents: Option<ArrowTable>,
    /// State of the OpenAI API request
    state: OpenAIRequestState,
    /// sample number
    sample: usize,
}

impl OpenAIEmbedStream {
    pub fn new(
        document_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        // Initialize with an empty schema
        // since it is not so straight forward to know the size of the vector embeddings beforehand
        // i.e., it is defined as the "hidden_size" in the model_config.json
        let schema = Arc::new(Schema::empty());
        Ok(Self {
            schema,
            document_stream,
            config_stream,
            baseline_metrics,
            _runtime_env: runtime_env,
            config: None,
            documents: None,
            state: OpenAIRequestState::NotStarted,
            sample: 0,
        })
    }

    /// Initialize the config for text embedding inference
    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
        if self.config.is_none() {
            let config: CandleEmbedConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
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
                OpenAIRequestState::NotStarted => {
                    // Initialize the config
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    let config_table = ArrowTableBuilder::new()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    self.init_config(config_table)?;

                    // Collect the next batch of queries
                    let batch = match ready!(self.document_stream.poll_next_unpin(cx)) {
                        Some(Ok(batch)) => batch,
                        _ => return Poll::Ready(None),
                    };

                    // Convert to a list of queries
                    let table = ArrowTableBuilder::new()
                        .with_name("queries")
                        .with_record_batches(vec![batch])?
                        .build()?;
                    let input: Vec<String> = table
                        .get_column_as_str_vec("text")
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
                OpenAIRequestState::Done => {
                    // Increase the sample count
                    self.sample += 1;
                    self.state = OpenAIRequestState::NotStarted;
                    self.poll_next(cx)
                }
            }
        } else {
            // Iterate through each state until the API request is completed
            match &mut self.state {
                OpenAIRequestState::NotStarted => {
                    // Collect the next batch of queries
                    let batch = match ready!(self.document_stream.poll_next_unpin(cx)) {
                        Some(Ok(batch)) => batch,
                        _ => return Poll::Ready(None),
                    };

                    // Convert to a list of queries
                    let table = ArrowTableBuilder::new()
                        .with_name("queries")
                        .with_record_batches(vec![batch])?
                        .build()?;
                    let input: Vec<String> = table
                        .get_column_as_str_vec("text")
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
                OpenAIRequestState::Done => {
                    // Increase the sample count
                    self.sample += 1;
                    self.state = OpenAIRequestState::NotStarted;
                    self.poll_next(cx)
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.document_stream.size_hint()
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
    use super::*;

    #[cfg(not(feature = "candle"))]
    #[tokio::test]
    async fn test_openai_embed_processor() -> Result<()> {
        let config = CandleEmbedConfig {
            input_type: "passage".to_string(),
            api_url: Some("http://0.0.0.0:8001/v1".to_string()),
            openai_asset: Some(
                crate::openai_asset::openai_which::WhichOpenAIAsset::NvidiaLlamaV3p2NvEmbedQA1BV2,
            ),
            ..Default::default()
        };

        // Make the config
        let config_table = ArrowTable::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config.clone())?, 1)?
            .build()?;

        // Make the runtime
        let runtime_env = Arc::new(Mutex::new(RuntimeEnv::default()));

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
        let document_table = ArrowTableBuilder::new()
            .with_name("text")
            .with_record_batches(vec![batch])?
            .build()?;

        // Make the metrics
        let metrics = ArrowTaskMetricsSet::new();
        let baseline_metrics = BaselineMetrics::new(&metrics, "candle_embed_processor");

        // Make and run the embeddings stream
        let embed_stream = OpenAIEmbedStream::new(
            document_table.to_record_batch_stream(),
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
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
        let document_table = ArrowTableBuilder::new()
            .with_name("text")
            .with_record_batches(vec![batch1, batch2])?
            .build()?;

        // Make the metrics
        let metrics = ArrowTaskMetricsSet::new();
        let baseline_metrics = BaselineMetrics::new(&metrics, "candle_embed_processor");

        // Make and run the embeddings stream
        let embed_stream = OpenAIEmbedStream::new(
            document_table.to_record_batch_stream(),
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
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
