use parking_lot::Mutex;
use phymes_ml::TokenStreamTrait;
use std::sync::Arc;

use crate::ProcessorTrait;

/// [ProcessorTrait] extension to running and caching of [TokenStreamTrait] objects
pub trait TokenStreamTraitExt: ProcessorTrait {
    /// Access the token service
    fn token_service(&self) -> &Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>;
}
