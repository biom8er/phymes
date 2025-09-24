use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static BUILDER_SESSION_CONTEXT_NAME: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static BUILDER_FLOWCHART_DIAGRAM: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static BUILDER_ER_DIAGRAM: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static BUILDER_TIMESTAMP: GlobalSignal<Vec<i64>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentBuilderState {
    pub session_context_name: String,
    pub flowchart_diagram: String,
    pub er_diagram: String,
    pub timestamp: i64,
}

pub async fn sync_current_builder_state(
    mut rx: UnboundedReceiver<SyncCurrentBuilderState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*BUILDER_SESSION_CONTEXT_NAME.write()).push(updated_state.session_context_name);
        (*BUILDER_FLOWCHART_DIAGRAM.write()).push(updated_state.flowchart_diagram);
        (*BUILDER_ER_DIAGRAM.write()).push(updated_state.er_diagram);
        (*BUILDER_TIMESTAMP.write()).push(updated_state.timestamp);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearBuilderState {}

pub async fn clear_builder_schema_state(mut _rx: UnboundedReceiver<ClearBuilderState>) {
    (*BUILDER_SESSION_CONTEXT_NAME.write()).clear();
    (*BUILDER_FLOWCHART_DIAGRAM.write()).clear();
    (*BUILDER_ER_DIAGRAM.write()).clear();
    (*BUILDER_TIMESTAMP.write()).clear();
}