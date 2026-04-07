mod message_builder;
mod message_trait;

pub use message_builder::{
    IPCMessageBuilder, MessageBuilderTrait, SendableRecordBatchStreamMessageBuilder, make_random_id,
};
pub use message_trait::{
    IPCMessage, MessageTrait, SendableRecordBatchStreamMessage, remove_message_by_subject,
};

/// Types
use phymes_diagnostics::HashMap;
pub type IPCMessageMap = HashMap<String, IPCMessage>;
pub type SendableRecordBatchStreamMessageMap = HashMap<String, SendableRecordBatchStreamMessage>;
pub type SendableRecordBatchStreamMessageBuilderMap =
    HashMap<String, SendableRecordBatchStreamMessageBuilder>;
