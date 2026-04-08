use anyhow::Result;
use candle_core::Tensor;
use parking_lot::Mutex;
use phymes_data::TensorProcessorTrait;
use phymes_processor::ProcessorTrait;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};
use tokenizers::Tokenizer;

/// Tokens representations in different dimensions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenWrapper {
    /// Text generation input
    D1(Vec<u32>),
    /// Embedding generation input
    D2(Vec<Vec<u32>>),
}

/// Tokenizer configurations and templates
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: Option<usize>,
    pub chat_template: Option<String>, // Jinja2 template is provided in tokenizer_config.json
    pub eos_token: Option<String>, // can be inferred from vocab.json and config.json and provided in tokenizer_config.json
    pub eos_token_id: Option<u32>, // provided in config.json
    pub bos_token: Option<String>,
    pub bos_token_id: Option<u32>, // provided in config.json
                                   // pub completion_template: Option<String>, // provided in tokenizer_config.json
                                   // pub tokenizer_class: Option<String>, // provided in tokenizer_config.json
}

/// For services that process tokens
pub trait TokenStreamTrait: TensorProcessorTrait + Send + Sync + Debug {
    /// Backward: propogation of the error signal for updating the tensor weights
    //fn backward(&mut self) -> Result<Self::T>;
    /// Forward: inference using the tensor weights on the specified device
    fn forward(
        &mut self,
        input: &TokenWrapper,
        index_position: usize,
        attention_mask: Option<&TokenWrapper>,
        include_lm_head: bool,
    ) -> Result<Tensor>;

    fn get_tokenizer(&self) -> &Tokenizer;

    fn get_tokenizer_config(&self) -> &TokenizerConfig;
}

/// [ProcessorTrait] extension to running and caching of [TokenStreamTrait] objects
pub trait TokenStreamTraitExt: ProcessorTrait {
    /// Access the token service
    fn token_service(&self) -> &Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>;
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
/// From <https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs>
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn tokens(&self) -> &Vec<u32> {
        &self.tokens
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> candle_core::Result<String> {
        match self.tokenizer.decode(tokens, true) {
            std::result::Result::Ok(str) => candle_core::Result::Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> candle_core::Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            candle_core::Result::Ok(Some(text.1.to_string()))
        } else {
            candle_core::Result::Ok(None)
        }
    }

    pub fn decode_rest(&self) -> candle_core::Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            candle_core::Result::Ok(Some(text.1.to_string()))
        } else {
            candle_core::Result::Ok(None)
        }
    }

    pub fn decode_all(&self) -> candle_core::Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
