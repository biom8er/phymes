mod message;
mod processor;
mod task_trait;
mod coalesce;
mod limit;
mod publish_subscribe;
mod test_exec;

pub use message::{MessageTrait, IPCMessage, SendableRecordBatchStreamMessage, MessageBuilderTrait, IPCMessageBuilder, SendableRecordBatchStreamMessageBuilder};
pub use processor::{ProcessorTrait, ProcessorEcho, ProcessorBuilder, test_processor};
pub use task_trait::{TaskTrait, Task, TaskBuilderTrait, TaskBuilder, test_task};
pub use publish_subscribe::PubSubTrait;

#[allow(unused_imports)]
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub(crate) use test_exec::{BlockingExec, assert_strong_count_converges_to_zero};
#[allow(unused_imports)]
pub(crate) use test_exec::{SendableRecordBatchExecTrait, MockExec, PanicExecWrapper};
