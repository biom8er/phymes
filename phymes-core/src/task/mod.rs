mod publish_subscribe;
mod task_builder;
mod task_trait;
mod test_exec;

pub use publish_subscribe::PublishAndSubscribeTrait;
pub use task_builder::{TaskBuilder, TaskBuilderTrait};
pub use task_trait::{RunnableTrait, Task, TaskTrait, test_task};

#[allow(unused_imports)]
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub(crate) use test_exec::{BlockingExec, assert_strong_count_converges_to_zero};
#[allow(unused_imports)]
pub(crate) use test_exec::{MockExec, PanicExecWrapper, SendableRecordBatchExecTrait};

use std::sync::Arc;
use phymes_diagnostics::HashMap;
/// Task HashMap
pub type TaskMap = HashMap<String, Arc<Task>>;