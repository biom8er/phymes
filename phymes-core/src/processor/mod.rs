mod processor_builder;
mod processor_echo;
mod processor_trait;
pub use processor_builder::ProcessorBuilder;
pub use processor_echo::ProcessorEcho;
pub use processor_trait::{ProcessorTrait, test_processor};

use phymes_diagnostics::HashMap;
use std::sync::Arc;

/// Processor HashMap with Arc-based abstraction
pub type ProcessorMap = HashMap<String, Arc<dyn ProcessorTrait>>;
