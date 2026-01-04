use serde::{Deserialize, Serialize};

/// The plan for the tasks
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TaskPlan {
    /// The name of the task
    pub task_name: String,
    /// The runtime environment name for the task
    pub runtime_env_name: String,
    /// The name of processors that process the messages stream
    pub processor_names: Vec<String>,
}