mod message_builder;
mod message_trait;
mod session_interface_message;

pub use message_builder::{
    IPCMessageBuilder, MessageBuilderTrait, SendableRecordBatchStreamMessageBuilder,
};
pub use message_trait::{
    IPCMessage, MessageTrait, SendableRecordBatchStreamMessage, remove_message_by_subject,
};
pub use session_interface_message::{
    SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait,
    SessionInterfaceMessageTrait,
};

/// Types
use phymes_diagnostics::HashMap;
pub type IPCMessageMap = HashMap<String, IPCMessage>;
pub type SendableRecordBatchStreamMessageMap = HashMap<String, SendableRecordBatchStreamMessage>;
pub type SendableRecordBatchStreamMessageBuilderMap = HashMap<String, SendableRecordBatchStreamMessageBuilder>;
