mod publish_subscribe;
mod task_builder;
mod task_trait;
mod test_exec;
mod task_plan;
mod task_plan_builder;

pub use publish_subscribe::PublishAndSubscribeTrait;
pub use task_builder::{TaskBuilder, TaskBuilderTrait};
pub use task_trait::{RunnableTrait, Task, TaskTrait, test_task};
pub use task_plan::TaskPlan;
pub use task_plan_builder::TaskPlanBuilder;

#[allow(unused_imports)]
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub(crate) use test_exec::{BlockingExec, assert_strong_count_converges_to_zero};
#[allow(unused_imports)]
pub(crate) use test_exec::{MockExec, PanicExecWrapper, SendableRecordBatchExecTrait};

use phymes_diagnostics::HashMap;
use std::sync::Arc;
/// Task HashMap
pub type TaskMap = HashMap<String, Arc<Task>>;
