use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static TASK_TASK_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static TASK_PROCESSOR_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static TASK_SUBJECT_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static TASK_PUB_OR_SUB: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

/// Task information
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SyncCurrentTaskInfoState {
    pub task_task_name: String,
    pub task_processor_name: String,
    pub task_subject_name: String,
    pub task_pub_or_sub: String,
}

pub async fn sync_current_task_info_state(mut rx: UnboundedReceiver<SyncCurrentTaskInfoState>) {
    while let Some(updated_state) = rx.next().await {
        (*TASK_TASK_NAMES.write()).push(updated_state.task_task_name);
        (*TASK_PROCESSOR_NAMES.write()).push(updated_state.task_processor_name);
        (*TASK_SUBJECT_NAMES.write()).push(updated_state.task_subject_name);
        (*TASK_PUB_OR_SUB.write()).push(updated_state.task_pub_or_sub);
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearTaskInfoState {}

pub async fn clear_task_info_state(mut _rx: UnboundedReceiver<ClearTaskInfoState>) {
    (*TASK_TASK_NAMES.write()).clear();
    (*TASK_PROCESSOR_NAMES.write()).clear();
    (*TASK_SUBJECT_NAMES.write()).clear();
    (*TASK_PUB_OR_SUB.write()).clear();
}
