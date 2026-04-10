use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use candle_core::{DType, Tensor};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use phymes_subject::{
    BuilderTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, Subject,
    SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::{DataConfigTrait, device};
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::{
    CandleEmbedConfig, TokenStreamTrait, TokenWrapper, convert_embedding_tensor_to_record_batch,
    process_prompt_embed,
};
use phymes_schemas::{AvailableSchemaTrait, AvailableSubjects};
use tracing::instrument;

pub struct CandleEmbedStream {
    /// Output schema (embeddings)
    schema: SchemaRef,
    /// The documents (or queries) to parse
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for embed inference
    documents_stream: Option<SendableRecordBatchStream>,
    /// Parameters for embed inference
    config_stream: SendableRecordBatchStream,
    /// The runtime environment
    _runtime_env: Arc<RuntimeEnv>,
    /// The candle asset needed for inference
    // DM: In a single thread environment, there is minimal to no penalty of using a mutex here
    // DM: in a mult-thread environment, we prevent copying the model assets each time we use it
    token_service: Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for embed inference
    config: Option<CandleEmbedConfig>,
    /// sample number
    sample: usize,
    /// sample number + prompt_tokens.len()
    index: usize,
}

impl CandleEmbedStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        token_service: Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::DocumentEmbeddings.to_schema(),
            messages,
            documents_stream: None,
            config_stream,
            diagnostic_builder,
            _runtime_env: runtime_env,
            token_service,
            config: None,
            sample: 0,
            index: 0,
        })
    }

    #[instrument(skip(self))]
    fn init_token_service(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            if self.token_service.lock().is_none() {
                let device = device(config.cpu)?;
                let mut asset = config.candle_asset.unwrap().build(
                    config.weights_config_file.clone(),
                    config.tokenizer_file.clone(),
                    config.weights_file.clone(),
                    config.tokenizer_config_file.clone(),
                    DType::F32,
                    device,
                )?;

                // DM: the eos_token_id is provided in the config
                //  which is model family dependent and captured currently
                //  when loading the model assets
                if asset.tokenizer_config.eos_token_id.is_none() {
                    // asset.tokenizer_config.eos_token_id = Some(151643);
                    asset.tokenizer_config.eos_token_id = Some(0);
                }

                // Concurrent embeddings can hold onto the lock simultaneous
                let _ = self.token_service.lock().replace(Box::new(asset));
            }
        } else {
            return Err(anyhow!(
                "The config for embeddings processor needs to be initialized before trying to initialize the token service."
            ));
        }
        Ok(())
    }

    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = CandleEmbedConfig::from_subject(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }

    #[instrument(skip(self, tokens, masks))]
    fn batch_embed(&mut self, tokens: &[Vec<u32>], masks: &[Vec<u32>]) -> Result<Tensor> {
        let logits = self.token_service.lock().as_mut().unwrap().forward(
            &TokenWrapper::D2(tokens.to_vec()),
            0,
            Some(&TokenWrapper::D2(masks.to_vec())),
            false,
        )?;

        // Extract the last hidden states as embeddings since inputs are padded left.
        let (_, seq_len, _) = logits.dims3()?;
        let embedding = logits
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?
            .to_dtype(DType::F32)?;
        Ok(embedding)
    }
}

impl Stream for CandleEmbedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Embed each stream of record batches whereby each
        // record batch row is a query
        if self.sample == 0 {
            // Initialize the metrics
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("CandleEmbedStream")?
                        .baseline_metrics(line!(), file!(), "poll_next"),
                )
            } else {
                None
            };
            let _timer = baseline_metrics
                .as_ref()
                .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

            // initialize the config
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = SubjectBuilder::new()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;

            // Collect the next batch of queries
            if self.documents_stream.is_none() {
                let docs_table = self.config.as_ref().unwrap().documents.clone();
                if let Some(s) = remove_message_by_subject(&docs_table, &mut self.messages) {
                    self.documents_stream.replace(s.get_message_own())
                } else {
                    return Poll::Ready(Some(Err(anyhow!(
                        "Documents and queries subject was not found in the message stream. Available messages are {:?}",
                        self.messages.keys()
                    ))));
                };
            }
            let batch = match ready!(self.documents_stream.as_mut().unwrap().poll_next_unpin(cx)) {
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

            // Tokenize the queries
            self.init_token_service()?;
            let mut tokenizer = self
                .token_service
                .lock()
                .as_ref()
                .unwrap()
                .get_tokenizer()
                .clone();
            let tokenizer_config = self
                .token_service
                .lock()
                .as_ref()
                .unwrap()
                .get_tokenizer_config()
                .clone();
            let (tokens, masks) = process_prompt_embed(
                &input,
                &mut tokenizer,
                tokenizer_config.eos_token_id.unwrap(),
                tokenizer_config.eos_token.unwrap().as_str(),
                tokenizer_config.model_max_length,
            )?;

            // Embed the query
            let embedding = self.batch_embed(&tokens, &masks)?;
            let batch = convert_embedding_tensor_to_record_batch(
                embedding,
                table.get_record_batches_own(),
            )?;

            // Record the schema
            self.schema = batch.schema();

            // increment the sample
            self.sample += 1;
            self.index += batch.num_rows();

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            if let Some(baseline_metrics) = &baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
        } else {
            // Keep embedding the remaining streams
            // Initialize the metrics
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("CandleEmbedStream")?
                        .baseline_metrics(line!(), file!(), "poll_next"),
                )
            } else {
                None
            };
            let _timer = baseline_metrics
                .as_ref()
                .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

            // Collect the next batch of queries
            let batch = match ready!(self.documents_stream.as_mut().unwrap().poll_next_unpin(cx)) {
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

            // Tokenize the queries
            let mut tokenizer = self
                .token_service
                .lock()
                .as_ref()
                .unwrap()
                .get_tokenizer()
                .clone();
            let tokenizer_config = self
                .token_service
                .lock()
                .as_ref()
                .unwrap()
                .get_tokenizer_config()
                .clone();
            let (tokens, masks) = process_prompt_embed(
                &input,
                &mut tokenizer,
                tokenizer_config.eos_token_id.unwrap(),
                tokenizer_config.eos_token.unwrap().as_str(),
                tokenizer_config.model_max_length,
            )?;

            // Embed the query
            let embedding = self.batch_embed(&tokens, &masks).unwrap();
            let batch =
                convert_embedding_tensor_to_record_batch(embedding, table.get_record_batches_own())
                    .unwrap();

            // increment the sample
            self.sample += 1;
            self.index += batch.num_rows();

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            if let Some(baseline_metrics) = &baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for CandleEmbedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
