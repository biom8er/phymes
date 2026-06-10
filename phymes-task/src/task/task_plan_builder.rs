use anyhow::{Result, anyhow};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

use crate::TaskPlan;

/// Builder for [TaskPlan]s
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TaskPlanBuilder {
    pub name: Option<String>,
    pub processor_names: Option<Vec<String>>,
}

impl TaskPlanBuilder {
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
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
            if self.processor_names.as_ref().is_none() {
                return Err(anyhow!("Missing processor_names for task {name}",));
            }
        } else {
            return Err(anyhow!("Missing task name"));
        }

        let task_plan = TaskPlan {
            task_name: self.name.take().unwrap(),
            processor_names: self.processor_names.take().unwrap(),
        };
        Ok(task_plan)
    }

    /// Extend the current [TaskPlanBuilder] with the Processor Names from another
    pub fn extend(mut self, other: TaskPlanBuilder) -> Result<Self> {
        let other_processors = if let Some(processors) = self.processor_names.as_ref() {
            if let Some(other) = other.processor_names {
                let names = processors.iter().collect::<HashSet<_>>();
                other
                    .into_iter()
                    .filter(|t| !names.contains(t))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if let Some(processors) = self.processor_names.as_mut() {
            processors.extend(other_processors);
        } else if !other_processors.is_empty() {
            self.processor_names.replace(other_processors);
        }

        Ok(self)
    }

    /// Break the [TaskPlanBuilder] into multiple individual [TaskPlanBuilder]s each with only a single Processor
    pub fn individualize(self) -> Result<Vec<Self>> {
        let task_name = self.name.ok_or(anyhow!(
            "Add a `name` before trying to individualize the TaskPlanBuilder."
        ))?;
        let tasks = self
            .processor_names
            .ok_or(anyhow!(
                "Add the `processor_name`s before trying to individualize the TaskPlanBuilder."
            ))?
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let name = format!("{task_name}_{i}");
                TaskPlanBuilder::default()
                    .with_name(&name)
                    .with_processor_names(&[p.as_str()])
            })
            .collect::<Vec<_>>();
        Ok(tasks)
    }
}
