mod embed_config;
mod embed_processor;

pub use embed_config::CandleEmbedConfig;
pub use embed_processor::CandleEmbedProcessor;
#[allow(unused_imports)]
pub use embed_processor::convert_embedding_vector_to_record_batch;
