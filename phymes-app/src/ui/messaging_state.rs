// Dioxus imports
use dioxus::prelude::*;

// General imports
use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

/// Generate a timestamp that can be added to the message table
/// Same as in phymes-agents/src/candle_chat/message_history.rs
pub fn create_timestamp() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.format("%a %b %e %T %Y").to_string()
}

// Current message state
#[allow(clippy::redundant_closure)]
pub static ROLE: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static CONTENT: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static INDEX: GlobalSignal<Vec<u32>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static TIMESTAMP: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentMessageState {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

pub async fn sync_current_message_state(mut rx: UnboundedReceiver<SyncCurrentMessageState>) {
    while let Some(updated_message_state) = rx.next().await {
        (*ROLE.write()).push(updated_message_state.role);
        (*CONTENT.write()).push(updated_message_state.content);
        if INDEX.len() == 0 {
            (*INDEX.write()).push(0);
        } else {
            let mut index: u32 = *INDEX.last().unwrap();
            index += 1;
            (*INDEX.write()).push(index);
        }
        (*TIMESTAMP.write()).push(updated_message_state.timestamp);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentMessageContentState {
    pub content: String,
    pub replace_last: bool,
}

pub async fn sync_current_message_content_state(
    mut rx: UnboundedReceiver<SyncCurrentMessageContentState>,
) {
    while let Some(updated_message_state) = rx.next().await {
        let mut tmp: String = (*CONTENT.write().pop().unwrap()).to_string();
        if updated_message_state.replace_last {
            (*CONTENT.write()).push(updated_message_state.content);
        } else {
            tmp.push_str(&updated_message_state.content);
            (*CONTENT.write()).push(tmp);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClearCurrentMessageState {}

pub async fn clear_current_message_state(mut _rx: UnboundedReceiver<ClearCurrentMessageState>) {
    (*ROLE.write()).clear();
    (*CONTENT.write()).clear();
    (*INDEX.write()).clear();
    (*TIMESTAMP.write()).clear();
}
