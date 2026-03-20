mod task_builder;
mod task_plan;
mod task_plan_builder;
mod task_trait;

pub use task_builder::{TaskBuilder, TaskBuilderTrait};
pub use task_plan::TaskPlan;
pub use task_plan_builder::TaskPlanBuilder;
pub use task_trait::{Task, TaskTrait, test_task};

use phymes_diagnostics::HashMap;
use std::sync::Arc;
/// Task HashMap
pub type TaskMap = HashMap<String, Arc<Task>>;
