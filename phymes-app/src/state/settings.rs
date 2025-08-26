use dioxus::prelude::*;
use futures::StreamExt;
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
pub static SESSION_MERMAID_FLOWCHART: GlobalSignal<String> = Signal::global(|| String::new());
#[allow(clippy::redundant_closure)]
pub static SESSION_MERMAID_ERDIAGRAM: GlobalSignal<String> = Signal::global(|| String::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentSessionMermaidJSState {
    pub flowchart: Option<String>,
    pub erdiagram: Option<String>,
}

pub async fn sync_current_session_mermaid_state(
    mut rx: UnboundedReceiver<SyncCurrentSessionMermaidJSState>,
) {
    while let Some(updated_state) = rx.next().await {
        if let Some(flowchart) = updated_state.flowchart {
            (*SESSION_MERMAID_FLOWCHART.write()).clear();
            (*SESSION_MERMAID_FLOWCHART.write()).push_str(flowchart.as_str());
        }
        if let Some(erdiagram) = updated_state.erdiagram {
            (*SESSION_MERMAID_ERDIAGRAM.write()).clear();
            (*SESSION_MERMAID_ERDIAGRAM.write()).push_str(erdiagram.as_str());
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
