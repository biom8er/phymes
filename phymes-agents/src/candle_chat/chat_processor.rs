use crate::candle_assets::{device::device, token_output_stream::TokenOutputStream};

use candle_core::DType;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

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
    array::{ArrayRef, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use futures::{Stream, StreamExt};
use parking_lot::{Mutex, RwLock};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event};

use super::{chat_config::CandleChatConfig, message_history::MessageHistoryTraitExt};
use crate::openai_asset::chat_completion::Tool;

#[derive(Default, Debug)]
pub struct CandleChatProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl CandleChatProcessor {
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

impl MappableTrait for CandleChatProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for CandleChatProcessor {
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
        let out = Box::pin(CandleChatStream::new(
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

pub struct CandleChatStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
    /// Optional tools to add to the message
    tools_stream: Option<SendableRecordBatchStream>,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Parameters for chat inference
    config: Option<CandleChatConfig>,
    /// Enables streaming token outputs for candle assets
    tos: Option<TokenOutputStream>,
    /// Logits to tokens sampler for candle assets
    logits_processor: Option<LogitsProcessor>,
    /// The number of tokens to sample after the prompt
    ///
    /// Inference will be invoked until `to_sample` > `sample`
    to_sample: usize,
    /// Sample number
    sample: usize,
    /// The index position for candle inference
    ///
    /// Transformer input index = Sample number + prompt_tokens.len()
    index: usize,
}

impl CandleChatStream {
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
        let schema = Arc::new(Schema::new(vec![role, content]));

        Ok(Self {
            schema,
            message_stream,
            tools_stream,
            baseline_metrics,
            config_stream,
            runtime_env,
            tos: None,
            logits_processor: None,
            config: None,
            to_sample: 0,
            sample: 0,
            index: 0,
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

    /// Initialize the token service for text generation inference
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

                let _ = self
                    .runtime_env
                    .try_lock()
                    .unwrap()
                    .token_service
                    .replace(Arc::new(RwLock::new(asset)));
            }
        } else {
            return Err(anyhow!(
                "The config for chat processor needs to be initialized before trying to initialize the token service."
            ));
        }
        Ok(())
    }

    /// Initialize the logits processor for text generation inference
    fn init_logits_processor(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            if self.logits_processor.is_none() {
                let logits_processor = process_logits_sampler(
                    config.temperature,
                    config.seed,
                    config.top_k,
                    config.top_p,
                );
                self.logits_processor.replace(logits_processor);
            }
        } else {
            return Err(anyhow!(
                "The config for chat processor needs to be initialized before trying to initialize the logits processor."
            ));
        }
        Ok(())
    }

    /// Stream the text generation inference
    ///
    /// # Notes
    ///
    /// Only for `CandleAssets`
    fn stream_candle_tgi(&mut self, prompt_tokens: &Option<Vec<u32>>) -> Result<Option<String>> {
        let next_token =
            match prompt_tokens {
                None => match self.tos.as_mut().unwrap().tokens().last() {
                    Some(t) => {
                        let logits = self
                            .runtime_env
                            .try_lock()
                            .unwrap()
                            .token_service
                            .as_mut()
                            .unwrap()
                            .try_write()
                            .unwrap()
                            .forward(&TokenWrapper::D1(vec![*t]), self.index, None, true)?;
                        let logits = logits.squeeze(0)?;
                        let logits =
                            if self.config.as_ref().unwrap().repeat_penalty == 1. {
                                logits
                            } else {
                                let start_at =
                                    self.tos.as_mut().unwrap().tokens().len().saturating_sub(
                                        self.config.as_ref().unwrap().repeat_last_n,
                                    );
                                candle_transformers::utils::apply_repeat_penalty(
                                    &logits,
                                    self.config.as_ref().unwrap().repeat_penalty,
                                    &self.tos.as_mut().unwrap().tokens()[start_at..],
                                )?
                            };
                        self.logits_processor.as_mut().unwrap().sample(&logits)?
                    }
                    None => return Err(anyhow!("Missing prompt and processed tokens")),
                },
                Some(p) => {
                    if !self.config.as_ref().unwrap().split_prompt {
                        let logits = self
                            .runtime_env
                            .try_lock()
                            .unwrap()
                            .token_service
                            .as_mut()
                            .unwrap()
                            .try_write()
                            .unwrap()
                            .forward(&TokenWrapper::D1(p.to_vec()), 0, None, true)?;
                        let logits = logits.squeeze(0)?;
                        self.logits_processor.as_mut().unwrap().sample(&logits)?
                    } else {
                        let mut next_token = 0;
                        for (pos, token) in p.iter().enumerate() {
                            let logits = self
                                .runtime_env
                                .try_lock()
                                .unwrap()
                                .token_service
                                .as_mut()
                                .unwrap()
                                .try_write()
                                .unwrap()
                                .forward(&TokenWrapper::D1(vec![*token]), pos, None, true)?;
                            let logits = logits.squeeze(0)?;
                            next_token = self.logits_processor.as_mut().unwrap().sample(&logits)?
                        }
                        next_token
                    }
                }
            };
        let text = self.tos.as_mut().unwrap().next_token(next_token)?;
        Ok(text)
    }
}

impl Stream for CandleChatStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Case 1: inference over the prompt
        if self.to_sample == 0 {
            // Initialize the metrics
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // Collect the chat history
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let messages = ArrowTableBuilder::new()
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
                    let tool_table = ArrowTableBuilder::new()
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
            let config_table = ArrowTableBuilder::new()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;

            // initialize the logits processor and candle token service
            self.init_logits_processor()?;
            self.init_token_service()?;

            // Convert to a prompt
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
            let prompt = messages.to_chat_prompt(
                tokenizer_config.chat_template.or_else(||
                    Some(r#"""{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n{{- messages[0]['content'] }}\n    {%- else %}\n{{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- '\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}\n    {%- for tool in tools %}\n{{- '\\n' }}\n{{- tool | tojson }}\n    {%- endfor %}\n    {{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}\n{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == 'assistant' %}\n{{- '<|im_start|>' + message.role }}\n{%- if message.content %}\n    {{- '\\n' + message.content }}\n{%- endif %}\n{%- for tool_call in message.tool_calls %}\n    {%- if tool_call.function is defined %}\n{%- set tool_call = tool_call.function %}\n    {%- endif %}\n    {{- '\\n<tool_call>\\n{\"name\": \"' }}\n    {{- tool_call.name }}\n    {{- '\", \"arguments\": ' }}\n    {{- tool_call.arguments | tojson }}\n    {{- '}\\n</tool_call>' }}\n{%- endfor %}\n{{- '<|im_end|>\\n' }}\n    {%- elif message.role == 'tool' %}\n{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n    {{- '<|im_start|>user' }}\n{%- endif %}\n{{- '\\n<tool_response>\\n' }}\n{{- message.content }}\n{{- '\\n</tool_response>' }}\n{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}\n    {{- '<|im_end|>\\n' }}\n{%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""#.to_string())
                    ).unwrap().as_str(),
                tokenizer_config.bos_token.as_deref(),
                tokenizer_config.eos_token.as_deref(),
                true,
                tools,
            )?;
            event!(Level::DEBUG, "Chat Processor Prompt: {}.", prompt.as_str());

            // Create the prompt tokens
            let model_max_length = self
                .runtime_env
                .try_lock()
                .unwrap()
                .token_service
                .as_ref()
                .unwrap()
                .try_read()
                .unwrap()
                .get_tokenizer_config()
                .model_max_length;
            let (prompt_tokens, to_sample, tos) = process_prompt_chat(
                prompt,
                self.runtime_env
                    .try_lock()
                    .unwrap()
                    .token_service
                    .as_ref()
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_tokenizer(),
                self.config.as_ref().unwrap().max_tokens,
                model_max_length,
            )?;
            self.to_sample = to_sample;
            self.tos = Some(tos);

            // Inference to generate the next token
            // This can be handled directly as a null in RecordBatch
            let index = prompt_tokens.len();
            let content = self.stream_candle_tgi(&Some(prompt_tokens));

            // initialize the index
            self.index = index;

            // Handle the returned content
            let content = match content {
                Ok(Some(s)) => s,
                _ => "".to_string(),
            };
            // println!("Chat Processor content: {}", content.as_str());

            // Wrap into a record batch
            let role_arr: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
            let content_arr: ArrayRef = Arc::new(StringArray::from(vec![content]));
            let batch =
                RecordBatch::try_from_iter(vec![("role", role_arr), ("content", content_arr)])?;

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
        } else if self.sample < self.to_sample {
            // Initialize the metrics
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // Inference to generate the next token
            // This can be handled directly as a null in RecordBatch
            let content = match self.stream_candle_tgi(&None) {
                Ok(Some(s)) => s,
                _ => "".to_string(),
            };
            // println!("Chat Processor Next string: {}", content.as_str());

            // Increment the sample count after the prompt inference
            self.sample += 1;
            self.index += 1;

            // Check for EOS token
            let eos_token = *self
                .tos
                .as_mut()
                .unwrap()
                .tokenizer()
                .get_vocab(true)
                .get(
                    self.runtime_env
                        .try_lock()
                        .unwrap()
                        .token_service
                        .as_ref()
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_tokenizer_config()
                        .eos_token
                        .as_ref()
                        .unwrap()
                        .as_str(),
                )
                .unwrap();
            if let Some(token) = self.tos.as_mut().unwrap().tokens().last()
                && *token == eos_token
            {
                self.sample = self.to_sample;
            }

            // Wrap into a record batch
            let role_arr: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
            let content_arr: ArrayRef = Arc::new(StringArray::from(vec![content]));
            let batch =
                RecordBatch::try_from_iter(vec![("role", role_arr), ("content", content_arr)])?;

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
        } else if self.sample == self.to_sample {
            // Increment the sample count
            self.sample += 1;

            // Flush out any remaining tokens
            if let Ok(Some(rest)) = self
                .tos
                .as_mut()
                .unwrap()
                .decode_rest()
                .map_err(candle_core::Error::msg)
            {
                // Wrap into a record batch
                let role_arr: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
                let content_arr: ArrayRef = Arc::new(StringArray::from(vec![rest]));
                let batch =
                    RecordBatch::try_from_iter(vec![("role", role_arr), ("content", content_arr)])?;

                // record the poll
                Poll::Ready(Some(Ok(batch)))
            } else {
                Poll::Ready(None)
            }
        } else {
            Poll::Ready(None)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.to_sample))
    }
}

impl RecordBatchStream for CandleChatStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

pub fn process_logits_sampler(
    temperature: f64,
    seed: u64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> LogitsProcessor {
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}

/**
Return the prompt tokens optionally shortening to the maximum input length

# Arguments

* `prompt` - String of the chat prompt generated by `create_prompt_chat`
* `tokenizer` - Tokenizer to use for generating the tokens
* `sample_len` - usize of the number of samples for chat generation
* `max_seq_length` - Optional usize of the maximum input length

# Returns

* `prompt_tokens` - `Vec<usize>` of the number of samples for chat generation
* `to_sample` - `usize` of the number of samples for chat generation
* `tos` - `TokenOutputStream` for use in `stream_chat`

*/
fn process_prompt_chat(
    prompt: String,
    tokenizer: &Tokenizer,
    sample_len: usize,
    max_seq_length: Option<usize>,
) -> anyhow::Result<(Vec<u32>, usize, TokenOutputStream)> {
    let tos = TokenOutputStream::new(tokenizer.clone());

    let tokens = tos
        .tokenizer()
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?;
    let to_sample = sample_len.saturating_sub(1);

    let prompt_tokens = tokens.get_ids();

    let prompt_tokens = match max_seq_length {
        Some(msl) => {
            if prompt_tokens.len() + to_sample > msl - 10 {
                let to_remove = prompt_tokens.len() + to_sample + 10 - msl;
                if to_remove > prompt_tokens.len() {
                    return Err(anyhow!(
                        "The prompt size is {}, the sample size is {}, the maximum allowable sequence length is {}, and the remove length is {} which is greater than the prompt size!",
                        prompt_tokens.len(),
                        to_sample,
                        msl,
                        to_remove
                    ));
                }
                prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
            } else {
                prompt_tokens.to_vec()
            }
        }
        None => prompt_tokens.to_vec(),
    };

    Ok((prompt_tokens, to_sample, tos))
}

#[cfg(test)]
mod tests {
    use phymes_core::{metrics::HashMap, session::runtime_env::RuntimeEnvTrait};

    use crate::{
        candle_assets::candle_which::{load_model_asset_path, load_tokenizer},
        candle_chat::message_history::MessageHistoryBuilderTraitExt,
    };

    use super::*;

    #[test]
    fn process_prompt_chat_test() {
        let prompt = "<|im_start|>system\nYou are an expert AI assistant. You are given a question, a set of possible functions/tools, and a set of possible documents. \nBased on the question, you may need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions/tools can be used, point it out. \nIf the given question lacks the parameters required by the function/tool, also point it out.\n\nYou have access to the following tools:\n<tools>[{\"name\":\"multiply\",\"description\":\"A function that multiplies two numbers\",\"parameters\":{\"properties\":{\"x\":{\"description\":\"The first number to multiply\",\"type\":\"number\"},\"y\":{\"description\":\"The second number to multiply\",\"type\":\"number\"}},\"required\":[\"x\",\"y\"],\"type\":\"object\"}}]</tools>\n\nGiven the context information and not prior knowledge, answer the question and provide citations from the documents.\nIf none of the documents are required to answer the question, point it out. \n\nYou have access to the following documents:\n<documents>[{\"title\":\"Title\",\"text\":\"The super informative document\"}]</documents><|im_end|>\n<|im_start|>user\nMultiply 2 by 2.<|im_end|>\n<|im_start|>assistant\n2 x 2 = 4<|im_end|>\n<|im_start|>assistant\n".to_string();

        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "tokenizer.json".to_string();
        let revision = "main".to_string();
        let max_seq_length: Option<usize> = Some(2048);

        let tokenizer = load_tokenizer(load_model_asset_path(&path, &repo, &filename, &revision))
            .expect("Tokenizer failed to load!");
        if let Ok((tokens, _, _)) =
            process_prompt_chat(prompt.clone(), &tokenizer, 1000, max_seq_length)
        {
            let tokens_expected: Vec<u32> = vec![
                101, 1026, 1064, 10047, 1035, 2707, 1064, 1028, 2291, 2017, 2024, 2019, 6739, 9932,
                3353, 1012, 2017, 2024, 2445, 1037, 3160, 1010, 1037, 2275, 1997, 2825, 4972, 1013,
                5906, 1010, 1998, 1037, 2275, 1997, 2825, 5491, 1012, 2241, 2006, 1996, 3160, 1010,
                2017, 2089, 2342, 2000, 2191, 2028, 2030, 2062, 3853, 1013, 6994, 4455, 2000, 6162,
                1996, 3800, 1012, 2065, 3904, 1997, 1996, 4972, 1013, 5906, 2064, 2022, 2109, 1010,
                2391, 2009, 2041, 1012, 2065, 1996, 2445, 3160, 14087, 1996, 11709, 3223, 2011,
                1996, 3853, 1013, 6994, 1010, 2036, 2391, 2009, 2041, 1012, 2017, 2031, 3229, 2000,
                1996, 2206, 5906, 1024, 1026, 5906, 1028, 1031, 1063, 1000, 2171, 1000, 1024, 1000,
                4800, 22086, 1000, 1010, 1000, 6412, 1000, 1024, 1000, 1037, 3853, 2008, 4800,
                24759, 3111, 2048, 102,
            ];
            assert_eq!(tokens, tokens_expected);
        }

        let max_seq_length: Option<usize> = Some(1128);
        let tokenizer = load_tokenizer(load_model_asset_path(&path, &repo, &filename, &revision))
            .expect("Tokenizer failed to load!");
        if let Ok((tokens, _, _)) =
            process_prompt_chat(prompt.clone(), &tokenizer, 1000, max_seq_length)
        {
            let tokens_expected: Vec<u32> =
                vec![1000, 1037, 3853, 2008, 4800, 24759, 3111, 2048, 102];
            assert_eq!(tokens, tokens_expected);
        }

        let max_seq_length: Option<usize> = Some(50);
        let tokenizer = load_tokenizer(load_model_asset_path(&path, &repo, &filename, &revision))
            .expect("Tokenizer failed to load!");
        let error = process_prompt_chat(prompt.clone(), &tokenizer, 1000, max_seq_length);
        assert!(error.is_err());
    }

    #[tokio::test]
    async fn test_candle_chat_processor() -> Result<(), Box<dyn std::error::Error>> {
        let name = "CandleChatProcessor";
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
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                crate::candle_assets::candle_which::WhichCandleAsset::SmolLM2_135MChat,
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
        let chat_processor = CandleChatProcessor::new_with_pub_sub_for(
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

        // DM: Skip actually running the tests as they take too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
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
            // "\nimport math\n\ndef count_primes(n):\n    \"\"\"\n    Finds all prime numbers up to n and counts them.\n    \n    Args:\n        n (int): The upper limit of the range to find primes in.\n\n    Returns:\n        int: The total number of prime numbers found.\n    \"\"\"\n\n    # Initialize a boolean array that indicates the primality of each number\n    is_prime = [True for _ in range(n + 1)]\n\n    # Set initial values based on small numbers and even numbers\n    i, p = 2, 3\n    while i * i <= n:\n        if is_prime[i]:\n            j = (i * i)\n            while j <= n:\n                is_prime[j] = False\n                j += i\n        i += 1\n\n    # Count the number of primality\n    return sum(1 for num in range(2, n + 1) if is_prime[num])"

            assert!(metrics.clone_inner().output_rows().unwrap() >= 10);
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
        }

        Ok(())
    }
}
