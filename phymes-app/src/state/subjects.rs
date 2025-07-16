use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_COLUMNS: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_TYPES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_ROWS: GlobalSignal<Vec<usize>> = Signal::global(|| Vec::new());

/// Subject infoormation
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SyncCurrentSubjectInfoState {
    pub subject_schema_name: String,
    pub subject_schema_column: String,
    pub subject_schema_type: String,
    pub subject_schema_row: usize,
}

pub async fn sync_current_subject_info_state(
    mut rx: UnboundedReceiver<SyncCurrentSubjectInfoState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*SUBJECT_SCHEMA_NAMES.write()).push(updated_state.subject_schema_name);
        (*SUBJECT_SCHEMA_COLUMNS.write()).push(updated_state.subject_schema_column);
        (*SUBJECT_SCHEMA_TYPES.write()).push(updated_state.subject_schema_type);
        (*SUBJECT_SCHEMA_ROWS.write()).push(updated_state.subject_schema_row);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearSubjectInfoState {}

pub async fn clear_subject_info_state(mut _rx: UnboundedReceiver<ClearSubjectInfoState>) {
    (*SUBJECT_SCHEMA_NAMES.write()).clear();
    (*SUBJECT_SCHEMA_COLUMNS.write()).clear();
    (*SUBJECT_SCHEMA_TYPES.write()).clear();
    (*SUBJECT_SCHEMA_ROWS.write()).clear();
}
