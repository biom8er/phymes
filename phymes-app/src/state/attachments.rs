// Dioxus imports
use dioxus::prelude::*;

// General imports
use futures::StreamExt;
use serde::{Deserialize, Serialize};

// Current message state
#[allow(clippy::redundant_closure)]
pub static ATTACHMENTS_ROLE: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static ATTACHMENTS_CONTENT: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static ATTACHMENTS_INDEX: GlobalSignal<Vec<u32>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static ATTACHMENTS_TIMESTAMP: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static ATTACHMENTS_FILENAME: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static ATTACHMENTS_EXTENSION: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentAttachmentsState {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub filename: String,
    pub extension: String,
}

pub async fn sync_current_attachments_state(mut rx: UnboundedReceiver<SyncCurrentAttachmentsState>) {
    while let Some(updated_state) = rx.next().await {
        (*ATTACHMENTS_ROLE.write()).push(updated_state.role);
        (*ATTACHMENTS_CONTENT.write()).push(updated_state.content);
        if ATTACHMENTS_INDEX.len() == 0 {
            (*ATTACHMENTS_INDEX.write()).push(0);
        } else {
            let mut index: u32 = *ATTACHMENTS_INDEX.last().unwrap();
            index += 1;
            (*ATTACHMENTS_INDEX.write()).push(index);
        }
        (*ATTACHMENTS_TIMESTAMP.write()).push(updated_state.timestamp);
        (*ATTACHMENTS_FILENAME.write()).push(updated_state.filename);
        (*ATTACHMENTS_EXTENSION.write()).push(updated_state.extension);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClearCurrentAttachmentsState {}

pub async fn clear_current_attachments_state(mut _rx: UnboundedReceiver<ClearCurrentAttachmentsState>) {
    (*ATTACHMENTS_ROLE.write()).clear();
    (*ATTACHMENTS_CONTENT.write()).clear();
    (*ATTACHMENTS_INDEX.write()).clear();
    (*ATTACHMENTS_TIMESTAMP.write()).clear();
    (*ATTACHMENTS_FILENAME.write()).clear();
    (*ATTACHMENTS_EXTENSION.write()).clear();
}