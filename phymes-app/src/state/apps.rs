#[cfg(feature = "serverless")]
use std::sync::Arc;

use dioxus::prelude::*;
use futures::StreamExt;
#[cfg(feature = "serverless")]
use phymes_core::{BuildableTrait, BuilderTrait, ObjectStorageBackend, RuntimeEnv, RuntimeEnvBuilderTrait, make_store};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

#[cfg(feature = "serverless")]
pub static RUNTIME_ENV: std::sync::LazyLock<Arc<RuntimeEnv>> = std::sync::LazyLock::new(|| RuntimeEnv::get_builder()
    .with_name("serverless_rt")
    .with_object_store(make_store(&ObjectStorageBackend::InMemory, None, None).unwrap())
    .build_arc()
    .unwrap());

pub static ACTIVE_SESSION_NAME: GlobalSignal<String> = Signal::global(String::new);

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

#[cfg(feature = "mermaid_js")]
#[derive(Debug, Deserialize, Serialize)]
pub struct MermaidJsObject {
    pub svg: Option<String>,
    pub error: Option<String>,
}

/// Filter in mermaid diagrams by session name
pub fn filter_in_mermaid_diagrams_by_session_name(
    active_session_context_names: &str,
    builder_session_context_names: &[&str],
    builder_flowchart_diagram: &[&str],
    builder_er_diagram: &[&str],
    builder_timestamp: &[i64],
) -> (Vec<String>, Vec<String>, Vec<String>, Vec<i64>) {
    let indices = builder_session_context_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_session_context_names)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    let session_context_name = builder_session_context_names
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let flowchart_diagram = builder_flowchart_diagram
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let er_diagram = builder_er_diagram
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let timestamp = builder_timestamp
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_owned())
        .collect::<Vec<_>>();
    (
        session_context_name,
        flowchart_diagram,
        er_diagram,
        timestamp,
    )
}

/// Filter out mermaid diagrams by session name
pub fn filter_out_mermaid_diagrams_by_session_name(
    active_session_context_names: &str,
    builder_session_context_names: &[&str],
    builder_flowchart_diagram: &[&str],
    builder_er_diagram: &[&str],
    builder_timestamp: &[i64],
) -> (Vec<String>, Vec<String>, Vec<String>, Vec<i64>) {
    let indices = builder_session_context_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s != active_session_context_names)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    let session_context_name = builder_session_context_names
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let flowchart_diagram = builder_flowchart_diagram
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let er_diagram = builder_er_diagram
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let timestamp = builder_timestamp
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_owned())
        .collect::<Vec<_>>();
    (
        session_context_name,
        flowchart_diagram,
        er_diagram,
        timestamp,
    )
}

/// Get a non duplicated list of sorted subject names
pub fn get_non_duplicated_sorted_subjects(subjects: &[&str]) -> Vec<String> {
    let subjects_set = subjects
        .iter()
        .map(|s| s.to_string())
        .collect::<HashSet<_>>();
    let mut subjects_vec = subjects_set.into_iter().collect::<Vec<_>>();
    subjects_vec.sort();
    subjects_vec
}
