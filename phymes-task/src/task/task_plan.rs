use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

/// The plan for the tasks
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct TaskPlan {
    /// The name of the task
    pub task_name: String,
    /// The name of processors that process the messages stream
    pub processor_names: Vec<String>,
}

impl TaskPlan {   

    /// Extend the current [TaskPlan] with the Processor Names from another
    pub fn extend(self, other: TaskPlan) -> Self {
        let names = self.processor_names
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<_>>();
        let other_processors = other.processor_names
            .into_iter()
            .filter(|t| !names.contains(t))
            .collect::<Vec<_>>();
        let processors = self.processor_names
            .into_iter()
            .chain(other_processors)
            .collect::<Vec<_>>();
        TaskPlan { task_name: self.task_name, processor_names: processors }
    }

    /// Break the [TaskPlan] into multiple individual [TaskPlan]s each with only a single Processor
    pub fn individualize(self) -> Vec<Self> {
        self.processor_names.into_iter()
            .enumerate()
            .map(|(i, p)| {
                let name = format!("{}_{i}", self.task_name);
                TaskPlan { task_name: name, processor_names: vec![p] }
            })
            .collect::<Vec<_>>()
    }
}
