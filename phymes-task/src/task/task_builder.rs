use std::sync::Arc;

use anyhow::Result;
use phymes_subject::BuilderTrait;
use phymes_processor::ProcessorTrait;

use crate::Task;

pub trait TaskBuilderTrait: BuilderTrait {
    fn with_processor(self, processor: Vec<Arc<dyn ProcessorTrait>>) -> Self;
}

/// Builder for [Task]s
#[derive(Default)]
pub struct TaskBuilder {
    /// Task name
    pub name: Option<String>,
    /// Function that implements the logic
    pub processor: Option<Vec<Arc<dyn ProcessorTrait>>>,
}

impl BuilderTrait for TaskBuilder {
    type T = Task;
    fn new() -> Self {
        Self {
            name: None,
            processor: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            processor: self.processor.unwrap(),
        })
    }
}

impl TaskBuilderTrait for TaskBuilder {
    fn with_processor(mut self, processor: Vec<Arc<dyn ProcessorTrait>>) -> Self {
        self.processor = Some(processor);
        self
    }
}
