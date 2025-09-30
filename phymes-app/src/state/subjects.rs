use dioxus::prelude::*;
use futures::StreamExt;
use phymes_core::session::message::SessionInterfaceMessage;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static ACTIVE_SUBJECT_NAME: GlobalSignal<String> = Signal::global(|| String::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentActiveSubjectState {
    pub name: String,
}

pub async fn sync_current_active_subject_state(
    mut rx: UnboundedReceiver<SyncCurrentActiveSubjectState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*ACTIVE_SUBJECT_NAME.write()).clear();
        (*ACTIVE_SUBJECT_NAME.write()).push_str(updated_state.name.as_str());
    }
}

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
pub static SUBJECT_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
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

#[allow(clippy::redundant_closure)]
pub static FILES_UPLOADED: GlobalSignal<Vec<SessionInterfaceMessage>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static FILENAMES_UPLOADED: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static EXTENSIONS_UPLOADED: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SyncFilesUploadedState {
    pub file: SessionInterfaceMessage, 
    pub filename: String,
    pub extension: String,
}

pub async fn sync_current_files_uploaded_state(
    mut rx: UnboundedReceiver<SyncFilesUploadedState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*FILES_UPLOADED.write()).push(updated_state.file);
        (*FILENAMES_UPLOADED.write()).push(updated_state.filename);
        (*EXTENSIONS_UPLOADED.write()).push(updated_state.extension);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ClearFilesUploadedState {}

pub async fn clear_files_uploaded_state(mut _rx: UnboundedReceiver<ClearFilesUploadedState>) {
    (*FILES_UPLOADED.write()).clear();
    (*FILENAMES_UPLOADED.write()).clear();
    (*EXTENSIONS_UPLOADED.write()).clear();
}

#[allow(clippy::redundant_closure)]
pub static FILES_DOWNLOADED: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static FILENAMES_DOWNLOADED: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static EXTENSIONS_DOWNLOADED: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SyncFilesDownloadedState { 
    pub file: String,
    pub filename: String,
    pub extension: String,
}

pub async fn sync_current_files_downloaded_state(
    mut rx: UnboundedReceiver<SyncFilesDownloadedState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*FILES_DOWNLOADED.write_unchecked()).push(updated_state.file);
        (*FILENAMES_DOWNLOADED.write_unchecked()).push(updated_state.filename);
        (*EXTENSIONS_DOWNLOADED.write_unchecked()).push(updated_state.extension);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ClearFilesDownloadedState {}

pub async fn clear_files_downloaded_state(mut _rx: UnboundedReceiver<ClearFilesDownloadedState>) {
    (*FILES_DOWNLOADED.write_unchecked()).clear();
    (*FILENAMES_DOWNLOADED.write_unchecked()).clear();
    (*EXTENSIONS_DOWNLOADED.write_unchecked()).clear();
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SyncCurrentAttachments {
    pub filename: String, 
    pub bytes: Vec<u8>, 
    pub extension: String, 
    pub metadata: String,
    pub timestamp: i64,
}

pub const SUBJECT_SCHEMA_HEADERS: [&str; 2] = ["Column", "Type"];

/// File download
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct DownloadSubject {
    pub download: String,
    pub href: String,
}

/// Chunk a document
///
/// # Arguments
///
/// * `contents` - A string
/// * `chunk_size` - The number of chars (each char is 4 bytes)
///
/// # Returns
///
/// * vector of chunks
#[allow(dead_code)]
pub fn chunk_document(mut doc: String, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    while doc.len() > chunk_size {
        let (s1, s2) = doc.split_at(chunk_size);
        chunks.push(s1.to_string());
        doc = s2.to_string();
    }
    chunks.push(doc);
    chunks
}

pub fn get_subject_schema_col_type_by_subject_name(
    active_subject: &str,
    subject_schema_names: &[&str],
    subject_schema_columns: &[&str],
    subject_schema_types: &[&str],
) -> (Vec<String>, Vec<String>) {
    let indices = subject_schema_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    let columns = subject_schema_columns
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let types = subject_schema_types
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    (columns, types)
}

pub fn get_subject_num_rows_by_subject_name(
    active_subject: &str,
    subject_names: &[&str],
    subject_num_rows: &[&usize],
) -> Vec<usize> {
    let indices = subject_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    subject_num_rows
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_owned().to_owned())
        .collect::<Vec<_>>()
}