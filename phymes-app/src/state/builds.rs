use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static MERMAID_SESSION_CONTEXT_NAME: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static MERMAID_FLOWCHART_DIAGRAM: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static MERMAID_ER_DIAGRAM: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static MERMAID_TIMESTAMP: GlobalSignal<Vec<i64>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentMermaidState {
    pub session_context_name: String,
    pub flowchart_diagram: String,
    pub er_diagram: String,
    pub timestamp: i64,
}

pub async fn sync_current_mermaid_state(
    mut rx: UnboundedReceiver<SyncCurrentMermaidState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*MERMAID_SESSION_CONTEXT_NAME.write()).push(updated_state.session_context_name);
        (*MERMAID_FLOWCHART_DIAGRAM.write()).push(updated_state.flowchart_diagram);
        (*MERMAID_ER_DIAGRAM.write()).push(updated_state.er_diagram);
        (*MERMAID_TIMESTAMP.write()).push(updated_state.timestamp);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearCurrentMermaidState {}

pub async fn clear_current_mermaid_state(mut _rx: UnboundedReceiver<ClearCurrentMermaidState>) {
    (*MERMAID_SESSION_CONTEXT_NAME.write()).clear();
    (*MERMAID_FLOWCHART_DIAGRAM.write()).clear();
    (*MERMAID_ER_DIAGRAM.write()).clear();
    (*MERMAID_TIMESTAMP.write()).clear();
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
    (session_context_name, flowchart_diagram, er_diagram, timestamp)
}