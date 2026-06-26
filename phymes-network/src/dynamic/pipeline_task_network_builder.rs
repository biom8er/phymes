use std::collections::VecDeque;

use anyhow::anyhow;
use phymes_subject::BuilderTrait;

use crate::{DynamicTaskNetworkBuilder, NetworkBuilder};

/// Template Dynamic or Static Processor Chain Network Builder
#[derive(Debug, Clone)]
pub struct PipelineTaskNetworkBuilder {
    /// Network name
    pub network_name: Option<String>,
    /// Dynamic or Static tasks
    pub tasks: Option<VecDeque<DynamicTaskNetworkBuilder>>,
}

impl Default for PipelineTaskNetworkBuilder {
    fn default() -> Self {
        Self { network_name: None, tasks: None }
    }
}

impl PipelineTaskNetworkBuilder {
    /// Add [DynamicTaskNetworkBuilder]s
    pub fn with_tasks(mut self, tasks: &[DynamicTaskNetworkBuilder]) -> Self {
        self.tasks = Some(tasks.into_iter().map(|t| t.clone()).collect::<VecDeque<_>>());
        self
    }
}

impl BuilderTrait for PipelineTaskNetworkBuilder {
    type T = NetworkBuilder;

    fn new() -> Self {
        Self::default()
    }

    fn with_name(mut self, name: &str) -> Self {
        self.network_name = Some(name.to_string());
        self
    }

    fn build(mut self) -> anyhow::Result<Self::T> {
        if let Some(mut tasks) = self.tasks.take() {
            let mut network_builder = tasks.pop_front().unwrap().build_dynamic();
            while let Some(task) = tasks.pop_front() {
                network_builder = network_builder.extend(task.build_dynamic()).unwrap();
            }
            Ok(network_builder)
        } else {
            Err(anyhow!("Please add tasks before building the Processor Chain Network Builder."))
        }
    }
}