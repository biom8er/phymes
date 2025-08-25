use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static MERMAID_PROCESSOR_TRACES: GlobalSignal<String> = Signal::global(|| String::new());
#[allow(clippy::redundant_closure)]
pub static MERMAID_ELAPSED_COMPUTE: GlobalSignal<String> = Signal::global(|| String::new());
#[allow(clippy::redundant_closure)]
pub static MERMAID_OUTPUT_ROWS: GlobalSignal<String> = Signal::global(|| String::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCurrentMetricsMermaidJSState {
    pub processor_traces: String,
    pub elapsed_compute: String,
    pub output_rows: String,
}

pub async fn sync_current_metrics_mermaid_state(
    mut rx: UnboundedReceiver<SyncCurrentMetricsMermaidJSState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*MERMAID_PROCESSOR_TRACES.write()).clear();
        (*MERMAID_PROCESSOR_TRACES.write()).push_str(updated_state.processor_traces.as_str());
        (*MERMAID_ELAPSED_COMPUTE.write()).clear();
        (*MERMAID_ELAPSED_COMPUTE.write()).push_str(updated_state.elapsed_compute.as_str());
        (*MERMAID_OUTPUT_ROWS.write()).clear();
        (*MERMAID_OUTPUT_ROWS.write()).push_str(updated_state.output_rows.as_str());
    }
}

#[allow(clippy::redundant_closure)]
pub static ACTIVE_METRIC: GlobalSignal<String> = Signal::global(|| String::new());

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SyncCurrentActiveMetricState {
    pub name: String,
}

pub async fn sync_current_active_metric_state(
    mut rx: UnboundedReceiver<SyncCurrentActiveMetricState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*ACTIVE_METRIC.write()).clear();
        (*ACTIVE_METRIC.write()).push_str(updated_state.name.as_str());
    }
}
