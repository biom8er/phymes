use crate::candle_assets::device::device;

use candle_core::{DType, Tensor};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics},
    session::{
        common_traits::{
            BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap, TokenWrapper,
        },
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
    array::{ArrayData, ArrayRef, FixedSizeListArray, StringArray},
    buffer::Buffer,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};

use anyhow::{Error, Result, anyhow};
use futures::{Stream, StreamExt};
use parking_lot::{Mutex, lock_api::RwLock};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event};

use super::embed_config::CandleEmbedConfig;

#[derive(Default, Debug)]
pub struct CandleEmbedProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl CandleEmbedProcessor {
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

impl MappableTrait for CandleEmbedProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for CandleEmbedProcessor {
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
        let out = Box::pin(CandleEmbedStream::new(
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

pub struct CandleEmbedStream {
    /// Output schema (embeddings)
    schema: SchemaRef,
    /// The input task to process.
    document_stream: SendableRecordBatchStream,
    /// Parameters for embed inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Parameters for embed inference
    config: Option<CandleEmbedConfig>,
    /// sample number
    sample: usize,
    /// sample number + prompt_tokens.len()
    index: usize,
}

impl CandleEmbedStream {
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
            runtime_env,
            config: None,
            sample: 0,
            index: 0,
        })
    }

    fn init_token_service(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            if self.runtime_env.try_lock().unwrap().token_service.is_none() {
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
                    asset.tokenizer_config.eos_token_id = Some(151643);
                }

                // Concurrent embeddings can hold onto the lock simultaneous
                let _ = self
                    .runtime_env
                    .try_lock()
                    .unwrap()
                    .token_service
                    .replace(Arc::new(RwLock::new(asset)));
            }
        } else {
            return Err(anyhow!(
                "The config for embeddings processor needs to be initialized before trying to initialize the token service."
            ));
        }
        Ok(())
    }

    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
        if self.config.is_none() {
            let config: CandleEmbedConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
            self.config.replace(config);
        }
        Ok(())
    }

    fn batch_embed(&mut self, tokens: &[Vec<u32>], masks: &[Vec<u32>]) -> Result<Tensor> {
        let logits = self
            .runtime_env
            .try_lock()
            .unwrap()
            .token_service
            .as_mut()
            .unwrap()
            .try_write()
            .unwrap()
            .forward(
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
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // initialize the config
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

            // Tokenize the queries
            self.init_token_service()?;
            let mut tokenizer = self
                .runtime_env
                .try_lock()
                .unwrap()
                .token_service
                .as_ref()
                .unwrap()
                .try_read()
                .unwrap()
                .get_tokenizer()
                .clone();
            let tokenizer_config = self
                .runtime_env
                .try_lock()
                .unwrap()
                .token_service
                .as_ref()
                .unwrap()
                .try_read()
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

            // Record the schema
            self.schema = batch.schema();

            // increment the sample
            self.sample += 1;
            self.index += batch.num_rows();

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
        } else {
            // Keep embedding the remaining streams
            // Initialize the metrics
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

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

            // Tokenize the queries
            let mut tokenizer = self
                .runtime_env
                .try_lock()
                .unwrap()
                .token_service
                .as_ref()
                .unwrap()
                .try_read()
                .unwrap()
                .get_tokenizer()
                .clone();
            let tokenizer_config = self
                .runtime_env
                .try_lock()
                .unwrap()
                .token_service
                .as_ref()
                .unwrap()
                .try_read()
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
            metrics.record_poll(poll)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.document_stream.size_hint()
    }
}

impl RecordBatchStream for CandleEmbedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

pub fn convert_embedding_vector_to_record_batch(
    embedding_vec: Vec<Vec<f32>>,
    other: Vec<RecordBatch>,
) -> Result<RecordBatch> {
    let embedding_len = embedding_vec.first().unwrap().len();
    let n_embedding = embedding_vec.len();
    assert_eq!(
        n_embedding,
        other
            .iter()
            .map(|batch| batch.num_rows())
            .collect::<Vec<_>>()
            .iter()
            .sum::<usize>()
    );
    let total_len = embedding_len * n_embedding;

    // Wrap into a record batch
    let value_data = ArrayData::builder(DataType::Float32)
        .len(total_len)
        .add_buffer(Buffer::from_slice_ref(
            embedding_vec.into_iter().flatten().collect::<Vec<_>>(),
        ))
        .build()
        .unwrap();
    let list_data_type = DataType::FixedSizeList(
        Arc::new(Field::new_list_field(DataType::Float32, false)),
        embedding_len.try_into().unwrap(),
    );
    let list_data = ArrayData::builder(list_data_type.clone())
        .len(n_embedding)
        .add_child_data(value_data.clone())
        .build()
        .unwrap();
    let embeddings: ArrayRef = Arc::new(FixedSizeListArray::from(list_data));

    // Extract out all of the other columns
    let mut batch_vec = Vec::new();
    let columns: Vec<String> = other
        .first()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            if (field.name() != "text") & (field.data_type() == &DataType::Utf8) {
                Some(field.name().clone())
            } else {
                None
            }
        })
        .collect();
    for column in columns.iter() {
        let array_vec = other
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name(column)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let array_ref: ArrayRef = Arc::new(StringArray::from(array_vec));
        batch_vec.push((column, array_ref));
    }

    let embeddings_name = "embeddings".to_string();
    batch_vec.push((&embeddings_name, embeddings));
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Ok(batch)
}

pub fn convert_embedding_tensor_to_record_batch(
    embedding: Tensor,
    other: Vec<RecordBatch>,
) -> Result<RecordBatch> {
    // Convert the tensor to a vec of vecs
    let embedding_vec: Vec<Vec<f32>> = embedding.to_vec2()?;
    convert_embedding_vector_to_record_batch(embedding_vec, other)
}

type ProcessPromptEmbedOutput = Vec<Vec<u32>>;
/**
Return the prompt tokens as Tensors optionally shortening to the maximum input length

# Arguments

* `prompts` - `Vec<String>` of the chat prompt generated by `create_prompt_embed`
* `tokenizer` - `Tokenizer` to use for generating the tokens
* `eos_token_id` - `u32` id of the end of sentence token used for padding
* `eos_token` - `&str` literal of the end of sentence token used for padding
* `max_seq_length` - `Optional<usize>` of the maximum input length. Note that this feature is currently not implemented
* `device` - `Device` from Candle used to create the Tensors

# Returns

* `tokens` - `Tensor` of tokens of dimension prompt_length by # of prompts
  Note that the tokens are currently returned directly due to memory borrowing issues
  during the token generation process using `Tokenizer` for batch embeddings
* `masks` - `Tensor` of tokens of dimension prompt_length by # of prompts

*/
pub fn process_prompt_embed(
    prompts: &[String],
    tokenizer: &mut Tokenizer,
    eos_token_id: u32,
    eos_token: &str,
    _max_seq_length: Option<usize>,
) -> anyhow::Result<(ProcessPromptEmbedOutput, ProcessPromptEmbedOutput)> {
    let padding = PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        pad_to_multiple_of: None,
        pad_id: eos_token_id,
        pad_type_id: 0,
        pad_token: String::from(eos_token),
    };

    tokenizer.with_padding(Some(padding));
    let encoded = tokenizer
        .encode_batch(prompts.to_vec(), true)
        .map_err(Error::msg)?;
    let tokens: Vec<Vec<u32>> = encoded.iter().map(|x| x.get_ids().to_vec()).collect();
    let mask: Vec<Vec<u32>> = encoded
        .iter()
        .map(|x| x.get_attention_mask().to_vec())
        .collect();

    // let tokens: Vec<u32> = encoded.iter().flat_map(|x| x.get_ids().into_iter().map(|x| *x)).collect();
    // let mask: Vec<u32> = encoded.iter().flat_map(|x| x.get_attention_mask().into_iter().map(|x| *x)).collect();
    // dbg!(&mask);

    // let (tokens, mask) = match max_seq_length {
    //     Some(msl) => {
    //         if tokens.len()  > msl - 10 {
    //             let to_remove = tokens.len() + 10 - msl;
    //             if to_remove > tokens.len() {
    //                 return Err(anyhow!("The prompt size is {}, the maximum allowable sequence length is {}, and the remove length is {} which is greater than the prompt size!",
    //                 tokens.len(), msl, to_remove));
    //             }
    //             (tokens[tokens.len().saturating_sub(to_remove)..].to_vec(),
    //             mask[mask.len().saturating_sub(to_remove)..].to_vec())
    //         } else {
    //             (tokens, mask)
    //         }
    //     },
    //     None => (tokens, mask)
    // };

    Ok((tokens, mask))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float32Array, StringArray};
    use candle_core::Device;
    use futures::TryStreamExt;

    use crate::candle_assets::candle_which::{load_model_asset_path, load_tokenizer};

    use super::*;

    #[test]
    fn test_process_prompt_embed() {
        let prompts: Vec<String> = vec!["As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.<|endoftext|>}".to_string(),
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.<|endoftext|>}".to_string()];
        let eos_token: &str = "<|endoftext|>";
        let eos_token_id: u32 = 151643;

        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "tokenizer.json".to_string();
        let revision = "main".to_string();
        let max_seq_length: Option<usize> = Some(2048);
        let mut tokenizer =
            load_tokenizer(load_model_asset_path(&path, &repo, &filename, &revision))
                .expect("Tokenizer failed to load!");

        if let Ok((tokens, masks)) = process_prompt_embed(
            &prompts,
            &mut tokenizer,
            eos_token_id,
            eos_token,
            max_seq_length,
        ) {
            let tokens_expected: Vec<u32> = vec![101, 2004, 1037];
            let masks_expected: Vec<u32> = vec![1, 1, 1];
            assert_eq!(&tokens[0][0..3], tokens_expected.as_slice());
            assert_eq!(&masks[0][0..3], masks_expected.as_slice());
        }
    }

    #[test]
    fn test_convert_embedding_tensor_to_record_batch() -> Result<()> {
        // Create the embeddings tensor
        let embeddings_vec: Vec<Vec<f32>> = vec![
            vec![0., 1., 2., 3., 4., 5., 6., 7.],
            vec![8., 9., 10., 11., 12., 13., 14., 15.],
            vec![16., 17., 18., 19., 20., 21., 22., 23.],
        ];
        let indices_vec = vec!["1", "2", "3"];
        let indices_ref: ArrayRef = Arc::new(StringArray::from(indices_vec.clone()));
        let batch = RecordBatch::try_from_iter(vec![("index", indices_ref)])?;

        let tensor = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((3, 8))?;
        assert_eq!(tensor.to_vec2::<f32>()?, embeddings_vec);

        // Convert to a record batch
        let batch = convert_embedding_tensor_to_record_batch(tensor, vec![batch])?;
        let indices = batch
            .column_by_name("index")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(indices, indices_vec);
        let embeddings = batch
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
        assert_eq!(embeddings, embeddings_vec);

        Ok(())
    }

    #[cfg(all(not(target_family = "wasm"), feature = "candle"))]
    #[tokio::test]
    async fn test_candle_embed_stream_nowasm() -> Result<()> {
        let config = CandleEmbedConfig {
            // WASM testing
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                crate::candle_assets::candle_which::WhichCandleAsset::QwenV2_1p5bEmbed,
            ),
            ..Default::default()
        };

        // Make the config
        let config_table = ArrowTable::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config.clone())?, 1)?
            .build()?;

        // Make the runtime
        let mut asset = config.candle_asset.unwrap().build(
            config.weights_config_file.clone(),
            config.tokenizer_file.clone(),
            config.weights_file.clone(),
            config.tokenizer_config_file.clone(),
            DType::F32,
            device(config.cpu)?,
        )?;
        asset.tokenizer_config.eos_token_id = Some(151643);
        let runtime_env = RuntimeEnv {
            token_service: Some(Arc::new(RwLock::new(asset))),
            tensor_service: None,
            name: "asset".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Case 1: streaming query
        // Make the query input stream
        let query_vec = vec![
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat.",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define",
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
        let embed_stream = CandleEmbedStream::new(
            document_table.to_record_batch_stream(),
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
        )?;

        // DM: Skip actually running the tests as they take too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "wsl-gpu"
        )) {
            let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
            assert_eq!(embeddings.len(), 1);

            // Expected data
            let _embeddings_test: Vec<Vec<f32>> = vec![
                vec![-3.0385482, 7.2247167, 3.2304974],
                vec![2.4326377, 1.8344411, -0.7329114],
                vec![-2.7296476, 6.784784, 3.0706217],
                vec![-4.392374, 2.938572, -4.162841],
            ];
            let _embeddings_vec = embeddings
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

            // DM: the results also dependent upon the system the model is ran
            // assert_eq!(
            //     embeddings_vec.first().unwrap()[0..3],
            //     embeddings_test.first().unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(1).unwrap()[0..3],
            //     embeddings_test.get(1).unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(2).unwrap()[0..3],
            //     embeddings_test.get(2).unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(3).unwrap()[0..3],
            //     embeddings_test.get(3).unwrap()[0..3]
            // );

            // Case 2: streaming query with multiple batches
            // Make the query input stream
            let query_vec1 = vec![
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat.",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define",
            ];
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
            let embed_stream = CandleEmbedStream::new(
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
            let _embeddings_test: Vec<Vec<f32>> = vec![
                vec![-3.29946, 7.5989823, 3.311682],
                vec![2.2690444, 2.090072, -0.8259398],
            ];
            assert_eq!(embeddings_vec.first().unwrap().len(), 1536); // hidden size in config.json

            // DM: the results also dependent upon the system the model is ran
            // assert_eq!(
            //     embeddings_vec.first().unwrap()[0..3],
            //     embeddings_test.first().unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(1).unwrap()[0..3],
            //     embeddings_test.get(1).unwrap()[0..3]
            // );
        }
        Ok(())
    }

    #[ignore = "QuantBERT embedding model is not yet supported for WASM."]
    #[tokio::test(flavor = "current_thread")]
    async fn test_candle_embed_stream_wasm() -> Result<()> {
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

        // Make the config
        let config = CandleEmbedConfig {
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-MiniLM-L6-v2.Q4_K_M.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                crate::candle_assets::candle_which::WhichCandleAsset::QuantizedBertEmbed,
            ),
            ..Default::default()
        };
        let config_table = ArrowTable::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the metrics
        let metrics = ArrowTaskMetricsSet::new();
        let baseline_metrics = BaselineMetrics::new(&metrics, "candle_embed_processor");

        // Make the runtime
        let mut asset = config.candle_asset.unwrap().build(
            config.weights_config_file.clone(),
            config.tokenizer_file.clone(),
            config.weights_file.clone(),
            config.tokenizer_config_file.clone(),
            DType::F32,
            device(config.cpu)?,
        )?;
        asset.tokenizer_config.eos_token_id = Some(151643);
        let runtime_env = RuntimeEnv {
            token_service: Some(Arc::new(RwLock::new(asset))),
            tensor_service: None,
            name: "asset".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Make and run the embeddings stream
        let embed_stream = CandleEmbedStream::new(
            document_table.to_record_batch_stream(),
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
        )?;
        let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
        assert_eq!(embeddings.len(), 1);

        // Expected data
        let _embeddings_test: Vec<Vec<f32>> = vec![
            vec![-3.2244308, 7.4192524, 2.9019766],
            vec![2.163365, 1.8837537, -0.18565525],
            vec![-3.260014, 6.5834556, 2.9206438],
            vec![-5.446545, 2.0517492, -4.0273705],
        ];
        let _embeddings_vec = embeddings
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

        // DM: the results also dependent upon the system the model is ran
        // assert_eq!(
        //     embeddings_vec.first().unwrap()[0..3],
        //     embeddings_test.first().unwrap()[0..3]
        // );
        // assert_eq!(
        //     embeddings_vec.get(1).unwrap()[0..3],
        //     embeddings_test.get(1).unwrap()[0..3]
        // );
        // assert_eq!(
        //     embeddings_vec.get(2).unwrap()[0..3],
        //     embeddings_test.get(2).unwrap()[0..3]
        // );
        // assert_eq!(
        //     embeddings_vec.get(3).unwrap()[0..3],
        //     embeddings_test.get(3).unwrap()[0..3]
        // );
        Ok(())
    }
}
