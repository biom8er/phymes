mod stream_trait;
mod stream_adapter;
mod test_exec;

pub use stream_trait::{
    IPCRecordBatchStream, RecordBatchStream, SendableIPCRecordBatchStream,
    SendableRecordBatchStream,
};
pub use stream_adapter::{
    EmptyRecordBatchStream, RecordBatchReceiverStream, RecordBatchReceiverStreamBuilder,
    RecordBatchStreamAdapter,
};
#[allow(unused_imports)]
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub(crate) use test_exec::{BlockingExec, assert_strong_count_converges_to_zero};
#[allow(unused_imports)]
pub(crate) use test_exec::{MockExec, PanicExecWrapper, SendableRecordBatchExecTrait};
