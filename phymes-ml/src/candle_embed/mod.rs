mod embed_config;
mod candle_embed_processor;

pub use embed_config::CandleEmbedConfig;
pub use candle_embed_processor::CandleEmbedProcessor;
#[allow(unused_imports)]
pub use candle_embed_processor::convert_embedding_vector_to_record_batch;
