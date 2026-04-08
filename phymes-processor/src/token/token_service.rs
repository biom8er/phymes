use anyhow::Result;
use candle_core::Tensor;
use parking_lot::Mutex;
use phymes_processor::ProcessorTrait;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

/// [ProcessorTrait] extension to running and caching of [TokenStreamTrait] objects
pub trait TokenStreamTraitExt: ProcessorTrait {
    /// Access the token service
    fn token_service(&self) -> &Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>;
}