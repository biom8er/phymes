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