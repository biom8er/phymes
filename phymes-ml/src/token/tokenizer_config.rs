use serde::{Deserialize, Serialize};

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