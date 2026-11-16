use anyhow::Result;
use phymes_message::IPCMessageMap;

/// Return the pre-compiled task subscriptions and publications as messages
pub trait PreCompiledNetworkTrait {
    fn as_task_messages(&self) -> Result<Vec<IPCMessageMap>>;
}