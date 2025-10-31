mod coalesce;
mod limit;
mod message;
mod processor;
mod publish_subscribe;
mod task_trait;
mod test_exec;

pub use message::{
    IPCMessage, IPCMessageBuilder, MessageBuilderTrait, MessageTrait, remove_message_by_subject,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilder,
};
pub use processor::{ProcessorBuilder, ProcessorEcho, ProcessorTrait, test_processor};
pub use publish_subscribe::PubSubTrait;
pub use task_trait::{Task, TaskBuilder, TaskBuilderTrait, TaskTrait, test_task};

#[allow(unused_imports)]
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub(crate) use test_exec::{BlockingExec, assert_strong_count_converges_to_zero};
#[allow(unused_imports)]
pub(crate) use test_exec::{MockExec, PanicExecWrapper, SendableRecordBatchExecTrait};
