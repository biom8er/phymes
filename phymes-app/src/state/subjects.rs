use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_COLUMNS: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static SUBJECT_SCHEMA_TYPES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SyncCurrentSubjectSchemaState {
    pub subject_schema_name: String,
    pub subject_schema_column: String,
    pub subject_schema_type: String,
}

pub async fn sync_current_subject_schema_state(
    mut rx: UnboundedReceiver<SyncCurrentSubjectSchemaState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*SUBJECT_SCHEMA_NAMES.write()).push(updated_state.subject_schema_name);
        (*SUBJECT_SCHEMA_COLUMNS.write()).push(updated_state.subject_schema_column);
        (*SUBJECT_SCHEMA_TYPES.write()).push(updated_state.subject_schema_type);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearSubjectSchemaState {}

pub async fn clear_subject_schema_state(mut _rx: UnboundedReceiver<ClearSubjectSchemaState>) {
    (*SUBJECT_SCHEMA_NAMES.write()).clear();
    (*SUBJECT_SCHEMA_COLUMNS.write()).clear();
    (*SUBJECT_SCHEMA_TYPES.write()).clear();
}

#[allow(clippy::redundant_closure)]
pub static SUBJECT_NAMES: GlobalSignal<Vec<usize>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static SUBJECT_NUM_ROWS: GlobalSignal<Vec<usize>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SyncCurrentSubjectNumRowsState {
    pub subject_name: String,
    pub subject_num_row: usize,
}

pub async fn sync_current_subject_num_rows_state(
    mut rx: UnboundedReceiver<SyncCurrentSubjectNumRowsState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*SUBJECT_NAMES.write()).push(updated_state.subject_name);
        (*SUBJECT_NUM_ROWS.write()).push(updated_state.subject_num_row);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearSubjectNumRowsState {}

pub async fn clear_subject_num_rows_state(mut _rx: UnboundedReceiver<ClearSubjectNumRowsState>) {
    (*SUBJECT_NAMES.write()).clear();
    (*SUBJECT_NUM_ROWS.write()).clear();
}