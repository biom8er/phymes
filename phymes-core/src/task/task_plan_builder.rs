use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::TaskPlan;

/// Builder for [TaskPlan]s
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TaskPlanBuilder {
    pub task_name: Option<String>,
    pub runtime_env_name: Option<String>,
    pub processor_names: Option<Vec<String>>,
}

impl TaskPlanBuilder {
    pub fn build(mut self) -> Result<TaskPlan> {
        if self.task_name.is_none() {
            return Err(anyhow!("Missing task name"));
        } else if self.runtime_env_name.as_ref().is_none() {
            return Err(anyhow!(
                "Missing runtime_env_name for task {}",
                self.task_name.as_ref().unwrap()
            ));
        } else if self.processor_names.as_ref().is_none() {
            return Err(anyhow!(
                "Missing processor_names for task {}",
                self.task_name.as_ref().unwrap()
            ));
        }

        let task_plan = TaskPlan {
            task_name: self.task_name.take().unwrap(),
            runtime_env_name: self.runtime_env_name.take().unwrap(),
            processor_names: self.processor_names.take().unwrap(),
        };
        Ok(task_plan)
    }
}