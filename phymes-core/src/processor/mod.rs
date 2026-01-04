mod processor_builder;
mod processor_echo;
mod processor_trait;
mod processor_plan;
mod processor_plan_builder;
pub use processor_builder::ProcessorBuilder;
pub use processor_echo::ProcessorEcho;
pub use processor_trait::{ProcessorTrait, test_processor};
pub use processor_plan::ProcessorPlan;
pub use processor_plan_builder::ProcessorPlanBuilder;

use phymes_diagnostics::HashMap;
use std::sync::Arc;

/// Processor HashMap with Arc-based abstraction
pub type ProcessorMap = HashMap<String, Arc<dyn ProcessorTrait>>;
