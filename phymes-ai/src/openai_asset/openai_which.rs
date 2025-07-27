use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tracing::{Level, event};

/// General dependencies
use std::env;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum WhichOpenAIAsset {
    #[value(name = "o3")]
    O3,
    #[value(name = "o3-mini")]
    O3Mini,
    #[value(name = "gpt-4o-mini")]
    GPT4OMini,
    #[value(name = "text-embedding-3-large")]
    TextEmbedding3Large,
    #[value(name = "text-embedding-3-small")]
    TextEmbedding3Small,
    #[value(name = "Llama-3.2-1b-instruct")]
    MetaLlamaV3p2_1B,
    #[value(name = "Llama-3.2-3b-instruct")]
    MetaLlamaV3p2_3B,
    #[value(name = "Llama-3.2-nv-embedqa-1b-v2")]
    NvidiaLlamaV3p2NvEmbedQA1BV2,
}

impl Default for WhichOpenAIAsset {
    fn default() -> Self {
        Self::GPT4OMini
    }
}

impl WhichOpenAIAsset {
    /// The name of the asset
    pub fn get_name(&self) -> &str {
        match self {
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::GPT4OMini => "gpt-4o-mini",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::MetaLlamaV3p2_1B => "Llama-3.2-1B-Instruct",
            Self::MetaLlamaV3p2_3B => "Llama-3.2-3B-Instruct",
            Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => "NVIDIA Retrieval QA Llama 3.2 1B Embedding v2",
        }
    }

    /// The name of the asset
    pub fn get_repository(&self) -> &str {
        match self {
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::GPT4OMini => "gpt-4o-mini",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::MetaLlamaV3p2_1B => "meta/llama-3.2-1b-instruct",
            Self::MetaLlamaV3p2_3B => "meta/llama-3.2-3b-instruct",
            Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => "nvidia/llama-3.2-nv-embedqa-1b-v2",
        }
    }
    /// The name of the asset
    pub fn get_latest(&self) -> &str {
        match self {
            Self::O3 => "latest",
            Self::O3Mini => "latest",
            Self::GPT4OMini => "latest",
            Self::TextEmbedding3Large => "latest",
            Self::TextEmbedding3Small => "latest",
            Self::MetaLlamaV3p2_1B => "1.8.6",
            Self::MetaLlamaV3p2_3B => "latest",
            Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => "latest",
        }
    }

    /// The base URL to access the asset
    pub fn get_endpoint(&self) -> &str {
        match self {
            Self::O3
            | Self::O3Mini
            | Self::GPT4OMini
            | Self::TextEmbedding3Large
            | Self::TextEmbedding3Small => "https://api.openai.com/v1",
            Self::MetaLlamaV3p2_1B
            | Self::MetaLlamaV3p2_3B
            | Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => "https://integrate.api.nvidia.com/v1",
        }
    }

    /// The request post extension
    pub fn get_endpoint_env_var(&self) -> &str {
        match self {
            Self::O3
            | Self::O3Mini
            | Self::GPT4OMini
            | Self::MetaLlamaV3p2_1B
            | Self::MetaLlamaV3p2_3B => "CHAT_API_URL",
            Self::TextEmbedding3Large
            | Self::TextEmbedding3Small
            | Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => "EMBED_API_URL",
        }
    }

    /// The request post extension
    pub fn get_post(&self) -> &str {
        match self {
            Self::O3
            | Self::O3Mini
            | Self::GPT4OMini
            | Self::MetaLlamaV3p2_1B
            | Self::MetaLlamaV3p2_3B => "chat/completions",
            Self::TextEmbedding3Large
            | Self::TextEmbedding3Small
            | Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => "embeddings",
        }
    }

    /// The endpoint post url
    pub fn get_api_url(&self, endpoint: Option<String>) -> String {
        // Check the environmental variables first
        let endpoint = if let Ok(endpoint) = std::env::var(self.get_endpoint_env_var()) {
            endpoint
        } else {
            // and then the configs next
            endpoint.unwrap_or(self.get_endpoint().to_string())
        };

        // Create the URL
        let url = format!("{}/{}", endpoint, self.get_post());

        event!(Level::INFO, "OpenAI API endpoint url: {}", url.as_str());
        url
    }

    /// The user api key
    pub fn get_api_key(&self) -> String {
        match env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_e) => match env::var("NGC_API_KEY") {
                Ok(key) => key,
                Err(e) => panic!("{}", e.to_string()),
            },
        }
    }

    pub fn is_chat(&self) -> bool {
        match self {
            Self::O3
            | Self::O3Mini
            | Self::GPT4OMini
            | Self::MetaLlamaV3p2_1B
            | Self::MetaLlamaV3p2_3B => true,
            Self::TextEmbedding3Large
            | Self::TextEmbedding3Small
            | Self::NvidiaLlamaV3p2NvEmbedQA1BV2 => false,
        }
    }
}
