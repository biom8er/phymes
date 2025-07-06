use anyhow::Result;
use clap::ValueEnum;
use phymes_core::session::common_traits::TokenizerConfig;
use serde::{Deserialize, Serialize};
/// General dependencies
use std::fs::File;

/// Candle dependencies
use candle_core::{DType, Device, quantized::gguf_file};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use candle_transformers::models::bert::{BertModel as Bert, Config as BertConfig};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use candle_transformers::quantized_var_builder::VarBuilder as QuantVarBuilder;
use tokenizers::Tokenizer;

/// All supported models
use crate::candle_models::{
    //quantized_nomic::NomicBertModel as QuantizedNomic,
    nomic::{
        NomicBertModel as Nomic, NomicConfig,
        tei_backend_core::{ModelType, Pool},
    },
    quantized_bert::{BertModel as QuantizedBert, Config as QuantizerdBertConfig},
    quantized_qwen2::ModelWeights as QuantizedQwen2,
};

/// Crates
use super::candle_asset::CandleAsset;

/// The model weights objects that store
/// the actual tensors needed for inference
pub enum CandleModelWeights {
    QuantizedQwen2(QuantizedQwen2),
    QuantizedLlama(QuantizedLlama),
    //QuantizedNomic(QuantizedNomic),
    QuantizedBert(QuantizedBert),
    Bert(Bert),
    Nomic(Nomic),
}

impl std::fmt::Debug for CandleModelWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QuantizedQwen2(_) => write!(f, "Quantized-Qwen2"),
            Self::QuantizedLlama(_) => write!(f, "Quantized-Llama"),
            Self::QuantizedBert(_) => write!(f, "Quantized-Bert"),
            Self::Bert(_) => write!(f, "Bert"),
            Self::Nomic(_) => write!(f, "Nomic"),
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum WhichCandleAsset {
    #[value(name = "Qwen-v2.5-0.5b-chat")]
    QwenV2p5_0p5bChat,
    #[value(name = "Qwen-v2.5-1.5b-chat")]
    QwenV2p5_1p5bChat,
    #[value(name = "Qwen-v2.5-3b-chat")]
    QwenV2p5_3bChat,
    #[value(name = "Qwen-v2.5-7b-chat")]
    QwenV2p5_7bChat,
    #[value(name = "Qwen-v2.5-14b-chat")]
    QwenV2p5_14bChat,
    #[value(name = "Qwen-v2.5-72b-chat")]
    QwenV2p5_72bChat,
    #[value(name = "Qwen-v2-1.5b-embed")]
    QwenV2_1p5bEmbed,
    #[value(name = "Qwen-v2-7b-embed")]
    QwenV2_7bEmbed,
    #[value(name = "Qwen-v2.5-0.5b-coder")]
    QwenV2p5_0p5bCoder,
    #[value(name = "Qwen-v2.5-1.5b-coder")]
    QwenV2p5_1p5bCoder,
    #[value(name = "Qwen-v2.5-3b-coder")]
    QwenV2p5_3bCoder,
    #[value(name = "Qwen-v2.5-7b-coder")]
    QwenV2p5_7bCoder,
    #[value(name = "Qwen-v2.5-14b-coder")]
    QwenV2p5_14bCoder,
    #[value(name = "Qwen-v2.5-72b-coder")]
    QwenV2p5_72bCoder,
    #[value(name = "Quantized-llama-v3.2-7b-chat")]
    LlamaV3p2_7bChat,
    #[value(name = "Quantized-llama-v3.2-13b-chat")]
    LlamaV3p2_13bChat,
    #[value(name = "Quantized-llama-v3.2-70b-chat")]
    LlamaV3p2_70bChat,
    #[value(name = "SmoLM2-135M-chat")]
    SmolLM2_135MChat,
    #[value(name = "SmoLM2-360M-chat")]
    SmolLM2_360MChat,
    #[value(name = "SmoLM2-1.7B-chat")]
    SmolLM2_1p7BChat,
    #[value(name = "BERT-embed")]
    BertEmbed,
    #[value(name = "Quantized-BERT-embed")]
    QuantizedBertEmbed,
    #[value(name = "Nomic-embed")]
    NomicEmbed,
}

impl Default for WhichCandleAsset {
    fn default() -> Self {
        Self::QwenV2p5_1p5bChat
    }
}

impl WhichCandleAsset {
    /// The name of the asset
    pub fn get_name(&self) -> &str {
        match self {
            Self::QwenV2p5_0p5bChat => "Qwen2.5-0.5B-Instruct",
            Self::QwenV2p5_1p5bChat => "Qwen2.5-1.5B-Instruct",
            Self::QwenV2p5_3bChat => "Qwen2.5-3B-Instruct",
            Self::QwenV2p5_7bChat => "Qwen2.5-7B-Instruct",
            Self::QwenV2p5_14bChat => "Qwen2.5-14B-Instruct",
            Self::QwenV2p5_72bChat => "Qwen2.5-72B-Instruct",
            Self::QwenV2_1p5bEmbed => "gte-Qwen2-1.5B-instruct",
            Self::QwenV2_7bEmbed => "gte-Qwen2-7B-instruct",
            Self::QwenV2p5_0p5bCoder => "Qwen2.5-Coder-0.5B-Instruct",
            Self::QwenV2p5_1p5bCoder => "Qwen2.5-Coder-1.5B-Instruct",
            Self::QwenV2p5_3bCoder => "Qwen2.5-Coder-3B-Instruct",
            Self::QwenV2p5_7bCoder => "Qwen2.5-Coder-7B-Instruct",
            Self::QwenV2p5_14bCoder => "Qwen2.5-Coder-14B-Instruct",
            Self::QwenV2p5_72bCoder => "Qwen2.5-Coder-72B-Instruct",
            Self::LlamaV3p2_7bChat => todo!(),
            Self::LlamaV3p2_13bChat => todo!(),
            Self::LlamaV3p2_70bChat => todo!(),
            Self::SmolLM2_135MChat => "SmolLM2-135M-Instruct",
            Self::SmolLM2_360MChat => todo!(),
            Self::SmolLM2_1p7BChat => todo!(),
            Self::BertEmbed => "all-MiniLM-L6-v2",
            Self::QuantizedBertEmbed => "all-MiniLM-L6-v2",
            Self::NomicEmbed => todo!(),
        }
    }

    /// The base URL to access the asset
    pub fn get_endpoint(&self) -> &str {
        match self {
            Self::QwenV2p5_0p5bChat
            | Self::QwenV2p5_1p5bChat
            | Self::QwenV2p5_3bChat
            | Self::QwenV2p5_7bChat
            | Self::QwenV2p5_14bChat
            | Self::QwenV2p5_72bChat
            | Self::QwenV2p5_0p5bCoder
            | Self::QwenV2p5_1p5bCoder
            | Self::QwenV2p5_3bCoder
            | Self::QwenV2p5_7bCoder
            | Self::QwenV2p5_14bCoder
            | Self::QwenV2p5_72bCoder
            | Self::LlamaV3p2_7bChat
            | Self::LlamaV3p2_13bChat
            | Self::LlamaV3p2_70bChat
            | Self::SmolLM2_135MChat
            | Self::SmolLM2_360MChat
            | Self::SmolLM2_1p7BChat
            | Self::QwenV2_1p5bEmbed
            | Self::QwenV2_7bEmbed
            | Self::BertEmbed
            | Self::QuantizedBertEmbed
            | Self::NomicEmbed => "https://huggingface.co/",
        }
    }

    /// The repository where the tokenizer and other config assets can be found
    pub fn get_repo_tokenizer(&self) -> &str {
        match self {
            Self::QwenV2p5_0p5bChat => "Qwen/Qwen2.5-0.5B-Instruct",
            Self::QwenV2p5_1p5bChat => "Qwen/Qwen2.5-1.5B-Instruct",
            Self::QwenV2p5_3bChat => "Qwen/Qwen2.5-3B-Instruct",
            Self::QwenV2p5_7bChat => "Qwen/Qwen2.5-7B-Instruct",
            Self::QwenV2p5_14bChat => "Qwen/Qwen2.5-14B-Instruct",
            Self::QwenV2p5_72bChat => "Qwen/Qwen2.5-72B-Instruct",
            Self::QwenV2_1p5bEmbed => "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            Self::QwenV2_7bEmbed => "Alibaba-NLP/gte-Qwen2-7B-instruct",
            Self::QwenV2p5_0p5bCoder => "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            Self::QwenV2p5_1p5bCoder => "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            Self::QwenV2p5_3bCoder => "Qwen/Qwen2.5-Coder-3B-Instruct",
            Self::QwenV2p5_7bCoder => "Qwen/Qwen2.5-Coder-7B-Instruct",
            Self::QwenV2p5_14bCoder => "Qwen/Qwen2.5-Coder-14B-Instruct",
            Self::QwenV2p5_72bCoder => "Qwen/Qwen2.5-Coder-72B-Instruct",
            Self::LlamaV3p2_7bChat => todo!(),
            Self::LlamaV3p2_13bChat => todo!(),
            Self::LlamaV3p2_70bChat => todo!(),
            Self::SmolLM2_135MChat => "HuggingFaceTB/SmolLM2-135M-Instruct", //https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
            Self::SmolLM2_360MChat => todo!(), //https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
            Self::SmolLM2_1p7BChat => todo!(), //https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
            Self::BertEmbed => "sentence-transformers/all-MiniLM-L6-v2",
            Self::QuantizedBertEmbed => "sentence-transformers/all-MiniLM-L6-v2",
            Self::NomicEmbed => todo!(),
        }
    }

    /// The repository where the model weights can be found
    pub fn get_repo_gguf(&self) -> &str {
        match self {
            Self::QwenV2p5_0p5bChat => "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            Self::QwenV2p5_1p5bChat => "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            Self::QwenV2p5_3bChat => "Qwen/Qwen2.5-3B-Instruct-GGUF",
            Self::QwenV2p5_7bChat => "Qwen/Qwen2.5-7B-Instruct-GGUF",
            Self::QwenV2p5_14bChat => "Qwen/Qwen2.5-14B-Instruct-GGUF",
            Self::QwenV2p5_72bChat => "Qwen/Qwen2.5-72B-Instruct-GGUF",
            Self::QwenV2_1p5bEmbed => "tensorblock/gte-Qwen2-1.5B-instruct-GGUF",
            Self::QwenV2_7bEmbed => "tensorblock/gte-Qwen2-7B-instruct-GGUF",
            Self::QwenV2p5_0p5bCoder => "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            Self::QwenV2p5_1p5bCoder => "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            Self::QwenV2p5_3bCoder => "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
            Self::QwenV2p5_7bCoder => "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            Self::QwenV2p5_14bCoder => "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
            Self::QwenV2p5_72bCoder => "Qwen/Qwen2.5-Coder-72B-Instruct-GGUF",
            Self::LlamaV3p2_7bChat => todo!(),
            Self::LlamaV3p2_13bChat => todo!(),
            Self::LlamaV3p2_70bChat => todo!(),
            Self::SmolLM2_135MChat => "Segilmez06/SmolLM2-135M-Instruct-Q4_K_M-GGUF",
            Self::SmolLM2_360MChat => todo!(),
            Self::SmolLM2_1p7BChat => todo!(),
            Self::BertEmbed => "sentence-transformers/all-MiniLM-L6-v2",
            Self::QuantizedBertEmbed => "Jarbas/all-MiniLM-L6-v2-Q4_K_M-GGUF",
            Self::NomicEmbed => todo!(),
        }
    }

    /// The name of the weights file or weights catalog name
    pub fn get_filename(&self) -> &str {
        match self {
            Self::QwenV2p5_0p5bChat => "qwen2.5-0.5b-instruct-q4_0.gguf",
            Self::QwenV2p5_1p5bChat => "qwen2.5-1.5b-instruct-q4_0.gguf",
            Self::QwenV2p5_3bChat => "qwen2.5-3b-instruct-q4_0.gguf",
            Self::QwenV2p5_7bChat => "qwen2.5-7b-instruct-q4_0.gguf",
            Self::QwenV2p5_14bChat => "qwen2.5-14b-instruct-q4_0.gguf",
            Self::QwenV2p5_72bChat => "qwen2.5-72b-instruct-q4_0.gguf",
            Self::QwenV2_1p5bEmbed => "gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
            Self::QwenV2_7bEmbed => "gte-Qwen2-7B-instruct-Q4_K_M.gguf",
            Self::QwenV2p5_0p5bCoder => "qwen2.5-0.5b-coder-instruct-q4_0.gguf",
            Self::QwenV2p5_1p5bCoder => "qwen2.5-1.5b-coder-instruct-q4_0.gguf",
            Self::QwenV2p5_3bCoder => "qwen2.5-3b-coder-instruct-q4_0.gguf",
            Self::QwenV2p5_7bCoder => "qwen2.5-7b-coder-instruct-q4_0.gguf",
            Self::QwenV2p5_14bCoder => "qwen2.5-14b-coder-instruct-q4_0.gguf",
            Self::QwenV2p5_72bCoder => "qwen2.5-72b-coder-instruct-q4_0.gguf",
            Self::LlamaV3p2_7bChat => todo!(),
            Self::LlamaV3p2_13bChat => todo!(),
            Self::LlamaV3p2_70bChat => todo!(),
            Self::SmolLM2_135MChat => "smollm2-135m-instruct-q4_k_m.gguf",
            Self::SmolLM2_360MChat => todo!(),
            Self::SmolLM2_1p7BChat => todo!(),
            Self::BertEmbed => "model.safetensors",
            Self::QuantizedBertEmbed => "all-minilm-l6-v2-q4_k_m.gguf",
            Self::NomicEmbed => todo!(),
        }
    }

    /// The version of the asset
    pub fn get_revision(&self) -> &str {
        match self {
            Self::QwenV2p5_0p5bChat => "main",
            Self::QwenV2p5_1p5bChat => "main",
            Self::QwenV2p5_3bChat => "main",
            Self::QwenV2p5_7bChat => "main",
            Self::QwenV2p5_14bChat => "main",
            Self::QwenV2p5_72bChat => "main",
            Self::QwenV2_1p5bEmbed => "main",
            Self::QwenV2_7bEmbed => "main",
            Self::QwenV2p5_0p5bCoder => "main",
            Self::QwenV2p5_1p5bCoder => "main",
            Self::QwenV2p5_3bCoder => "main",
            Self::QwenV2p5_7bCoder => "main",
            Self::QwenV2p5_14bCoder => "main",
            Self::QwenV2p5_72bCoder => "main",
            Self::LlamaV3p2_7bChat => todo!(),
            Self::LlamaV3p2_13bChat => todo!(),
            Self::LlamaV3p2_70bChat => todo!(),
            Self::SmolLM2_135MChat => "main",
            Self::SmolLM2_360MChat => todo!(),
            Self::SmolLM2_1p7BChat => todo!(),
            Self::BertEmbed => "main",
            Self::QuantizedBertEmbed => "main",
            Self::NomicEmbed => todo!(),
        }
    }

    pub fn load_tokenizer(&self, path: Option<String>) -> Result<Tokenizer> {
        load_tokenizer(load_model_asset_path(
            &path,
            self.get_repo_tokenizer(),
            "tokenizer.json",
            self.get_revision(),
        ))
    }

    pub fn load_tokenizer_config(&self, path: Option<String>) -> Result<TokenizerConfig> {
        load_config(load_model_asset_path(
            &path,
            self.get_repo_tokenizer(),
            "tokenizer_config.json",
            self.get_revision(),
        ))
    }

    pub fn is_chat(&self) -> bool {
        match self {
            Self::QwenV2p5_0p5bChat
            | Self::QwenV2p5_1p5bChat
            | Self::QwenV2p5_3bChat
            | Self::QwenV2p5_7bChat
            | Self::QwenV2p5_14bChat
            | Self::QwenV2p5_72bChat
            | Self::QwenV2p5_0p5bCoder
            | Self::QwenV2p5_1p5bCoder
            | Self::QwenV2p5_3bCoder
            | Self::QwenV2p5_7bCoder
            | Self::QwenV2p5_14bCoder
            | Self::QwenV2p5_72bCoder
            | Self::LlamaV3p2_7bChat
            | Self::LlamaV3p2_13bChat
            | Self::LlamaV3p2_70bChat
            | Self::SmolLM2_135MChat
            | Self::SmolLM2_360MChat
            | Self::SmolLM2_1p7BChat => true,
            Self::QwenV2_1p5bEmbed
            | Self::QwenV2_7bEmbed
            | Self::BertEmbed
            | Self::QuantizedBertEmbed
            | Self::NomicEmbed => false,
        }
    }

    pub fn load_model_weights(
        &self,
        model_weights_file: Option<String>,
        model_weights_config_file: Option<String>,
        dtype: DType,
        device: &Device,
    ) -> Result<CandleModelWeights> {
        match self {
            Self::QwenV2p5_0p5bChat
            | Self::QwenV2p5_1p5bChat
            | Self::QwenV2p5_3bChat
            | Self::QwenV2p5_7bChat
            | Self::QwenV2p5_14bChat
            | Self::QwenV2p5_72bChat
            | Self::QwenV2p5_0p5bCoder
            | Self::QwenV2p5_1p5bCoder
            | Self::QwenV2p5_3bCoder
            | Self::QwenV2p5_7bCoder
            | Self::QwenV2p5_14bCoder
            | Self::QwenV2p5_72bCoder
            | Self::QwenV2_1p5bEmbed
            | Self::QwenV2_7bEmbed => {
                let (content, mut file) = load_model_gguf(load_model_asset_path(
                    &model_weights_file,
                    self.get_repo_gguf(),
                    self.get_filename(),
                    self.get_revision(),
                ))?;
                let model_weights = QuantizedQwen2::from_gguf(content, &mut file, device)?;
                Ok(CandleModelWeights::QuantizedQwen2(model_weights))
            }
            Self::LlamaV3p2_7bChat
            | Self::LlamaV3p2_13bChat
            | Self::LlamaV3p2_70bChat
            | Self::SmolLM2_135MChat
            | Self::SmolLM2_360MChat
            | Self::SmolLM2_1p7BChat => {
                let (content, mut file) = load_model_gguf(load_model_asset_path(
                    &model_weights_file,
                    self.get_repo_gguf(),
                    self.get_filename(),
                    self.get_revision(),
                ))?;
                let model_weights = QuantizedLlama::from_gguf(content, &mut file, device)?;
                Ok(CandleModelWeights::QuantizedLlama(model_weights))
            }
            Self::BertEmbed => {
                let vb = load_model_varbuilder(
                    load_model_asset_path(
                        &model_weights_file,
                        self.get_repo_gguf(),
                        self.get_filename(),
                        self.get_revision(),
                    ),
                    dtype,
                    device,
                )?;
                let model_config: BertConfig = load_config(load_model_asset_path(
                    &model_weights_config_file,
                    self.get_repo_tokenizer(),
                    "config.json",
                    self.get_revision(),
                ))?;
                let model_weights = Bert::load(vb, &model_config)?;
                Ok(CandleModelWeights::Bert(model_weights))
            }
            Self::NomicEmbed => {
                let vb = load_model_varbuilder(
                    load_model_asset_path(
                        &model_weights_file,
                        self.get_repo_gguf(),
                        self.get_filename(),
                        self.get_revision(),
                    ),
                    dtype,
                    device,
                )?;
                let model_config: NomicConfig = load_config(load_model_asset_path(
                    &model_weights_config_file,
                    self.get_repo_tokenizer(),
                    "config.json",
                    self.get_revision(),
                ))?;
                let model_weights =
                    Nomic::load(vb, &model_config, ModelType::Embedding(Pool::Mean))?;
                Ok(CandleModelWeights::Nomic(model_weights))
            }
            Self::QuantizedBertEmbed => {
                let vb = QuantVarBuilder::from_gguf(
                    load_model_asset_path(
                        &model_weights_file,
                        self.get_repo_gguf(),
                        self.get_filename(),
                        self.get_revision(),
                    )?,
                    device,
                )?;
                let model_config: QuantizerdBertConfig = load_config(load_model_asset_path(
                    &model_weights_config_file,
                    self.get_repo_tokenizer(),
                    "config.json",
                    self.get_revision(),
                ))?;
                let model_weights = QuantizedBert::load(vb, &model_config)?;
                Ok(CandleModelWeights::QuantizedBert(model_weights))
            }
        }
    }

    pub fn build(
        &self,
        config_file: Option<String>,
        tokenizer_file: Option<String>,
        weights_file: Option<String>,
        tokenizer_config_file: Option<String>,
        dtype: DType,
        device: Device,
    ) -> Result<CandleAsset> {
        let tokenizer = self.load_tokenizer(tokenizer_file)?;
        let tokenizer_config: TokenizerConfig =
            self.load_tokenizer_config(tokenizer_config_file)?;
        let model_weights = self.load_model_weights(weights_file, config_file, dtype, &device)?;
        Ok(CandleAsset {
            device,
            model_weights,
            tokenizer,
            tokenizer_config,
            dtype,
        })
    }
}

/**
Returns a path to the model asset including tokenizer, config, and model files

# Arguments

* `path` - An `Option<String>`
* `repo` - string slice that holds the name of the repository
* `filename` - string slice that holds the name of the file (e.g.,
  tokenizer.json, config.json, model.safetensor, model.safetensors.index.json,
  model.bin, model.ggml, model.gguf, etc.)
* `revision` - string slice that holds the name of the repository revision (default = main)

*/
#[allow(unused_variables)]
pub(crate) fn load_model_asset_path(
    path: &Option<String>,
    repo: &str,
    filename: &str,
    revision: &str,
) -> Result<std::path::PathBuf> {
    let asset_path = match &path {
        Some(config) => std::path::PathBuf::from(config),
        #[cfg(feature = "hf_hub")]
        None => {
            let api = candle_hf_hub::api::sync::Api::new()?;
            api.repo(candle_hf_hub::Repo::with_revision(
                repo.to_string(),
                candle_hf_hub::RepoType::Model,
                revision.to_string(),
            ))
            .get(filename)?
        }
        #[cfg(not(feature = "hf_hub"))]
        None => panic!("Asset path needs to be provided!"),
    };
    Ok(asset_path)
}

pub(crate) fn load_tokenizer(path: Result<std::path::PathBuf>) -> Result<Tokenizer> {
    match path {
        Ok(asset_path) => Tokenizer::from_file(asset_path).map_err(anyhow::Error::msg),
        Err(err) => panic!("Asset path needs to be provided! {err:?}"),
    }
}

/// load any json config file including
/// model.json, prompt.json, ...
fn load_config<T: for<'a> serde::Deserialize<'a>>(path: Result<std::path::PathBuf>) -> Result<T> {
    match path {
        Ok(asset_path) => {
            let config_str: String = std::fs::read_to_string(asset_path).expect("File not found.");
            let config: T = serde_json::from_str(&config_str)?; //.map_err(anyhow::Error::msg);
            Ok(config)
        }
        Err(err) => panic!("Asset path needs to be provided! {err:?}"),
    }
}

fn load_model_varbuilder(
    path: Result<std::path::PathBuf>,
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::var_builder::VarBuilderArgs<'_, Box<dyn SimpleBackend>>> {
    match path {
        Ok(asset_path) => {
            if asset_path.extension().unwrap() == "bin" {
                let vb = VarBuilder::from_pth(&asset_path, dtype, device)?;
                Ok(vb)
            } else if asset_path.extension().unwrap() == "safetensors" {
                let model_files = vec![asset_path];
                let vb =
                    unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, device)? };
                Ok(vb)
            } else if asset_path.extension().unwrap() == "safetensors.index.json" {
                // Parse file
                let index_file_string: String = std::fs::read_to_string(&asset_path)?;
                let json: serde_json::Value = serde_json::from_str(&index_file_string)?;
                let weight_map = match json.get("weight_map") {
                    None => {
                        panic!("no weight map in {asset_path:?}");
                    }
                    Some(serde_json::Value::Object(map)) => map,
                    Some(_) => {
                        panic!("weight map in {asset_path:?} is not a map");
                    }
                };
                let mut safetensors_files = std::collections::HashSet::new();
                for value in weight_map.values() {
                    if let Some(file) = value.as_str() {
                        safetensors_files.insert(file.to_string());
                    }
                }
                // Collect paths
                #[allow(unused_assignments)]
                let mut model_files = Vec::new();
                model_files = safetensors_files
                    .iter()
                    .map(|n| asset_path.join(n))
                    .collect();
                let vb =
                    unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, device)? };
                Ok(vb)
            } else {
                panic!(
                    "Extension {:?} not recognized for VarBuilder.",
                    asset_path.extension().unwrap()
                )
            }
        }
        Err(err) => panic!("Asset path needs to be provided! {err:?}"),
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes}B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

fn load_model_gguf(path: Result<std::path::PathBuf>) -> Result<(gguf_file::Content, File)> {
    match path {
        Ok(asset_path) => {
            if asset_path.extension().unwrap() == "gguf" {
                let start = std::time::Instant::now();
                let mut file = std::fs::File::open(&asset_path)?;
                let model =
                    gguf_file::Content::read(&mut file).map_err(|e| e.with_path(asset_path))?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensor_infos.iter() {
                    let elem_count = tensor.shape.elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
                }
                println!(
                    "loaded {:?} tensors ({}) in {:.2}s",
                    model.tensor_infos.len(),
                    &format_size(total_size_in_bytes),
                    start.elapsed().as_secs_f32(),
                );
                Ok((model, file))
            } else {
                panic!(
                    "Extension {:?} not recognized for VarBuilder.",
                    asset_path.extension().unwrap()
                )
            }
        }
        Err(err) => panic!("Asset path needs to be provided! {err:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_model_asset_path_test() {
        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "tokenizer.json".to_string();
        let revision = "main".to_string();

        let path_buff = load_model_asset_path(&path, &repo, &filename, &revision)
            .expect("Failed to run load_model_asset_path");
        assert_eq!(
            path_buff,
            std::path::PathBuf::from(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            ))
        );
    }

    #[test]
    fn load_tokenizer_test() {
        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "tokenizer.json".to_string();
        let revision = "main".to_string();

        let _ = load_tokenizer(load_model_asset_path(&path, &repo, &filename, &revision));
    }

    #[cfg(not(target_family = "wasm"))]
    use candle_transformers::models::bert::BertModel;
    use candle_transformers::models::bert::{Config, HiddenAct};

    #[test]
    fn load_config_test() {
        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "config.json".to_string();
        let revision = "main".to_string();

        let config: Config = load_config(load_model_asset_path(&path, &repo, &filename, &revision))
            .expect("testing...");
        assert_eq!(config.hidden_act, HiddenAct::Gelu);
    }

    #[test]
    fn load_tokenizer_config_test() {
        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "HuggingFaceTB/SmolLM2-135M-Instruct".to_string();
        let filename = "tokenizer_config.json".to_string();
        let revision = "main".to_string();

        let config: TokenizerConfig =
            load_config(load_model_asset_path(&path, &repo, &filename, &revision))
                .expect("testing...");
        assert_eq!(config.bos_token.unwrap(), "<|im_start|>".to_string());
        assert_eq!(config.eos_token.unwrap(), "<|im_end|>".to_string());
        assert_eq!(config.model_max_length.unwrap(), 8192);
        assert_eq!(config.chat_template.unwrap(), "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}".to_string());
    }

    use candle_core::DType;

    #[test]
    fn load_model_varbuilder_test() {
        let path_bin: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        #[cfg(not(target_family = "wasm"))]
        let path_safetensor: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/model.safetensors",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename_bin = "pytorch_model.bin".to_string();
        let revision = "main".to_string();
        let device = crate::candle_assets::device::device(true).expect("CPU should always work...");

        let _vb_pyt = load_model_varbuilder(
            load_model_asset_path(&path_bin, &repo, &filename_bin, &revision),
            DType::F32,
            &device,
        )
        .expect("testing...");
        #[cfg(not(target_family = "wasm"))]
        let filename_safetensor = "model.safetensors".to_string();
        #[cfg(not(target_family = "wasm"))]
        let _vb_safetensor = load_model_varbuilder(
            load_model_asset_path(&path_safetensor, &repo, &filename_safetensor, &revision),
            DType::F32,
            &device,
        )
        .expect("testing...");
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn load_config_varbuilder_test() {
        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "config.json".to_string();
        let revision = "main".to_string();
        let config: Config = load_config(load_model_asset_path(&path, &repo, &filename, &revision))
            .expect("testing...");

        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/model.safetensors",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let filename = "model.safetensors".to_string();
        let revision = "main".to_string();
        let device = crate::candle_assets::device::device(true).expect("CPU should always work...");

        let vb = load_model_varbuilder(
            load_model_asset_path(&path, &repo, &filename, &revision),
            DType::F32,
            &device,
        )
        .expect("testing...");
        let _model = BertModel::load(vb, &config).expect("testing...");
    }

    #[test]
    fn load_model_gguf_test() {
        use candle_transformers::models::quantized_llama::ModelWeights;
        let path: Option<String> = Some(format!(
            "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
            std::env::var("HOME").unwrap_or("".to_string())
        ));
        let repo = "Smol/smollm2-135m-instruct-q4_k_m.gguf".to_string();
        let filename = "smollm2-135m-instruct-q4_k_m.gguf".to_string();
        let revision = "main".to_string();
        let device = crate::candle_assets::device::device(true).expect("CPU should always work...");

        let (content, mut file) =
            load_model_gguf(load_model_asset_path(&path, &repo, &filename, &revision))
                .expect("testing...");
        let _model = ModelWeights::from_gguf(content, &mut file, &device).expect("testing...");
    }
}
