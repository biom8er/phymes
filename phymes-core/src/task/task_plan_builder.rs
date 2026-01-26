use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::TaskPlan;

/// Builder for [TaskPlan]s
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TaskPlanBuilder {
    pub name: Option<String>,
    pub runtime_env_name: Option<String>,
    pub processor_names: Option<Vec<String>>,
}

impl TaskPlanBuilder {
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    pub fn with_runtime_env_name(mut self, runtime_env_name: &str) -> Self {
        self.runtime_env_name = Some(runtime_env_name.to_string());
        self
    }
    pub fn with_processor_names(mut self, processor_names: &[&str]) -> Self {
        self.processor_names = Some(
            processor_names
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        );
        self
    }
    pub fn build(mut self) -> Result<TaskPlan> {
        if let Some(name) = self.name.as_ref() {
            if self.runtime_env_name.as_ref().is_none() {
                return Err(anyhow!("Missing runtime_env_name for task {name}",));
            } else if self.processor_names.as_ref().is_none() {
                return Err(anyhow!("Missing processor_names for task {name}",));
            }
        } else {
            return Err(anyhow!("Missing task name"));
        }

        let task_plan = TaskPlan {
            task_name: self.name.take().unwrap(),
            runtime_env_name: self.runtime_env_name.take().unwrap(),
            processor_names: self.processor_names.take().unwrap(),
        };
        Ok(task_plan)
    }
}
