use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static ACTIVE_SESSION_NAME: GlobalSignal<String> = Signal::global(|| String::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentActiveSessionState {
    pub name: String,
}

pub async fn sync_current_active_session_state(
    mut rx: UnboundedReceiver<SyncCurrentActiveSessionState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*ACTIVE_SESSION_NAME.write()).clear();
        (*ACTIVE_SESSION_NAME.write()).push_str(updated_state.name.as_str());
    }
}
