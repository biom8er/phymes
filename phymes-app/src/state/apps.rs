use dioxus::prelude::*;
use futures::StreamExt;
use phymes_core::metrics::HashSet;
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

#[allow(clippy::redundant_closure)]
pub static SESSION_FLOWCHART_DIAGRAM: GlobalSignal<String> = Signal::global(|| String::new());
#[allow(clippy::redundant_closure)]
pub static SESSION_ER_DIAGRAM: GlobalSignal<String> = Signal::global(|| String::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentSessionMermaidJSState {
    pub flowchart_diagram: Option<String>,
    pub er_diagram: Option<String>,
}

pub async fn sync_current_session_mermaid_state(
    mut rx: UnboundedReceiver<SyncCurrentSessionMermaidJSState>,
) {
    while let Some(updated_state) = rx.next().await {
        if let Some(flowchart) = updated_state.flowchart_diagram {
            (*SESSION_FLOWCHART_DIAGRAM.write()).clear();
            (*SESSION_FLOWCHART_DIAGRAM.write()).push_str(flowchart.as_str());
        }
        if let Some(erdiagram) = updated_state.er_diagram {
            (*SESSION_ER_DIAGRAM.write()).clear();
            (*SESSION_ER_DIAGRAM.write()).push_str(erdiagram.as_str());
        }
    }
}

#[cfg(feature = "mermaid_js")]
#[derive(Debug, Deserialize, Serialize)]
pub struct MermaidJsObject {
    pub svg: Option<String>,
    pub error: Option<String>,
}

#[allow(clippy::redundant_closure)]
pub static IS_FLOWCHART_SHOWN: GlobalSignal<bool> = Signal::global(|| true);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncIsFlowchartShownState {
    pub is_shown: bool,
}

pub async fn sync_is_flowchart_shown_state(
    mut rx: UnboundedReceiver<SyncIsFlowchartShownState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*IS_FLOWCHART_SHOWN.write()) = updated_state.is_shown;
    }
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
    (session_context_name, flowchart_diagram, er_diagram, timestamp)
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