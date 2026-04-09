use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use candle_core::DType;
use candle_transformers::generation::LogitsProcessor;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    BuilderTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, Subject,
    SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::{DataConfigTrait, device};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait, create_timestamp_micros,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::{
    CandleChatConfig, TokenOutputStream, TokenStreamTrait, TokenWrapper, process_logits_sampler,
    process_prompt_chat,
};
use phymes_schemas::{AvailableSchemaTrait, AvailableSubjects, Tool, create_chat_record_batch};
use tracing::{Level, event, instrument};

use crate::ChatTraitExt;

pub struct CandleChatStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The messages and optional tools
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The runtime environment
    _runtime_env: Arc<RuntimeEnv>,
    /// The candle asset needed for inference
    // DM: In a single thread environment, there is minimal to no penalty of using a mutex here
    // DM: in a mult-thread environment, we prevent copying the model assets each time we use it
    token_service: Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
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
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        token_service: Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            messages,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            token_service,
            tos: None,
            logits_processor: None,
            config: None,
            to_sample: 0,
            sample: 0,
            index: 0,
        })
    }

    /// Initialize the config for text generation inference
    #[instrument(skip(self))]
    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = CandleChatConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }

    /// Initialize the token service for text generation inference
    #[instrument(skip(self))]
    fn init_token_service(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            // Update the runtime if needed
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
                    asset.tokenizer_config.eos_token_id = Some(151643);
                }

                let _ = self.token_service.lock().replace(Box::new(asset));
            }
        } else {
            return Err(anyhow!(
                "The config for chat processor needs to be initialized before trying to initialize the token service."
            ));
        }
        Ok(())
    }

    /// Initialize the logits processor for text generation inference
    #[instrument(skip(self))]
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
    #[instrument(skip(self, prompt_tokens))]
    fn stream_candle_tgi(&mut self, prompt_tokens: &Option<Vec<u32>>) -> Result<Option<String>> {
        let next_token =
            match prompt_tokens {
                None => match self.tos.as_mut().unwrap().tokens().last() {
                    Some(t) => {
                        let logits = self.token_service.lock().as_mut().unwrap().forward(
                            &TokenWrapper::D1(vec![*t]),
                            self.index,
                            None,
                            true,
                        )?;
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
                        let logits = self.token_service.lock().as_mut().unwrap().forward(
                            &TokenWrapper::D1(p.to_vec()),
                            0,
                            None,
                            true,
                        )?;
                        let logits = logits.squeeze(0)?;
                        self.logits_processor.as_mut().unwrap().sample(&logits)?
                    } else {
                        let mut next_token = 0;
                        for (pos, token) in p.iter().enumerate() {
                            let logits = self.token_service.lock().as_mut().unwrap().forward(
                                &TokenWrapper::D1(vec![*token]),
                                pos,
                                None,
                                true,
                            )?;
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
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("CandleChatStream")?
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
                if batch.num_rows() > 0 {
                    batches.push(batch);
                }
            }
            let messages = SubjectBuilder::new()
                .with_name("Message History for CandleChatStream")
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
                if let Some(s) = remove_message_by_subject(&tools_table_name, &mut self.messages) {
                    let mut tools_stream = s.get_message_own();
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(tools_stream.poll_next_unpin(cx)) {
                        if batch.num_rows() > 0 {
                            batches.push(batch);
                        }
                    }
                    if let Ok(subject_builder) = SubjectBuilder::new()
                        .with_name("Tools for CandleChatStream")
                        .with_record_batches(batches)
                    {
                        let tool_table = subject_builder.build()?;
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
                }
            } else {
                None
            };

            // initialize the logits processor and candle token service
            self.init_logits_processor()?;
            self.init_token_service()?;
            event!(Level::INFO, "Initialized the chat token service.");

            // Convert to a prompt
            let tokenizer_config = self
                .token_service
                .lock()
                .as_ref()
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
            event!(Level::INFO, "Chat Processor Prompt: {}.", prompt.as_str());

            // Create the prompt tokens
            let model_max_length = tokenizer_config.model_max_length;
            let (prompt_tokens, to_sample, tos) = process_prompt_chat(
                prompt,
                self.token_service.lock().as_ref().unwrap().get_tokenizer(),
                self.config.as_ref().unwrap().max_tokens,
                model_max_length,
            )?;
            self.to_sample = to_sample;
            self.tos = Some(tos);
            event!(Level::INFO, "Processed the chat prompt.");

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
            event!(
                Level::INFO,
                "Generated the first token {}.",
                content.as_str()
            );

            // Wrap into a record batch
            let batch = create_chat_record_batch(
                vec!["assistant".to_string()],
                vec![content.to_string()],
                vec![create_timestamp_micros()],
            )?;

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            if let Some(baseline_metrics) = &baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
        } else if self.sample < self.to_sample {
            // Initialize the metrics
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("CandleChatStream")?
                        .baseline_metrics(line!(), file!(), "poll_next"),
                )
            } else {
                None
            };
            let _timer = baseline_metrics
                .as_ref()
                .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

            // Inference to generate the next token
            // This can be handled directly as a null in RecordBatch
            let content = match self.stream_candle_tgi(&None) {
                Ok(Some(s)) => s,
                _ => "".to_string(),
            };
            event!(
                Level::INFO,
                "Generated the next token {}.",
                content.as_str()
            );

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
                    self.token_service
                        .lock()
                        .as_ref()
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
            let batch = create_chat_record_batch(
                vec!["assistant".to_string()],
                vec![content.to_string()],
                vec![create_timestamp_micros()],
            )?;

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            if let Some(baseline_metrics) = &baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
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
                let batch = create_chat_record_batch(
                    vec!["assistant".to_string()],
                    vec![rest.to_string()],
                    vec![create_timestamp_micros()],
                )?;

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
