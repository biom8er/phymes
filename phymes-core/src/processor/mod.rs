mod processor_trait;
mod processor_builder;
mod processor_echo;
pub use processor_trait::{ProcessorTrait, test_processor};
pub use processor_builder::ProcessorBuilder;
pub use processor_echo::ProcessorEcho;