use serde::{Deserialize, Serialize};

/// The plan for the tasks
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct TaskPlan {
    /// The name of the task
    pub task_name: String,
    /// The name of processors that process the messages stream
    pub processor_names: Vec<String>,
}
