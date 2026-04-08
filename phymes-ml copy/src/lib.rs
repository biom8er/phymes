mod candle_assets;
mod candle_chat;
mod candle_embed;
mod candle_models;
mod openai_asset;
#[cfg(feature = "api")]
mod openai_chat;
#[cfg(feature = "api")]
mod openai_embed;

pub use candle_assets::{
    AvailableCandleAssets, CandleAsset, CandleModelWeights, TokenOutputStream, TokenStreamTrait,
    TokenStreamTraitExt, TokenWrapper, TokenizerConfig, load_model_asset_path, load_tokenizer,
};
pub use candle_chat::{
    CandleChatConfig, CandleChatProcessor, MessageParserProcessor, ToolCallConfig,
    ToolCallProcessor, bench_chat_processor, extract_tool_calls_str, process_logits_sampler,
    process_prompt_chat, ChatBuilderTraitExt, ChatTraitExt,
};
pub use candle_embed::{CandleEmbedConfig, CandleEmbedProcessor};
pub use candle_models::{QuantizedBert, QuantizedQwen2, QuantizerdBertConfig};
pub use openai_asset::AvailableOpenAIAssets;
#[cfg(feature = "api")]
pub use openai_chat::OpenAIChatProcessor;
#[cfg(feature = "api")]
pub use openai_embed::OpenAIEmbedProcessor;
