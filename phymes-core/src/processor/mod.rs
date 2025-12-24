mod processor_builder;
mod processor_echo;
mod processor_trait;
pub use processor_builder::ProcessorBuilder;
pub use processor_echo::ProcessorEcho;
pub use processor_trait::{ProcessorTrait, test_processor};
