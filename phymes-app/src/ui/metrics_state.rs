use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(clippy::redundant_closure)]
pub static METRIC_TASK_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static METRIC_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());
#[allow(clippy::redundant_closure)]
pub static METRIC_VALUES: GlobalSignal<Vec<u64>> = Signal::global(|| Vec::new());

/// Task information
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SyncCurrentMetricsInfoState {
    pub metric_task_name: String,
    pub metric_name: String,
    pub metric_value: u64,
}

pub async fn sync_current_metrics_info_state(
    mut rx: UnboundedReceiver<SyncCurrentMetricsInfoState>,
) {
    while let Some(updated_state) = rx.next().await {
        (*METRIC_TASK_NAMES.write()).push(updated_state.metric_task_name);
        (*METRIC_NAMES.write()).push(updated_state.metric_name);
        (*METRIC_VALUES.write()).push(updated_state.metric_value);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearMetricsInfoState {}

pub async fn clear_metrics_info_state(mut _rx: UnboundedReceiver<ClearMetricsInfoState>) {
    (*METRIC_TASK_NAMES.write()).clear();
    (*METRIC_NAMES.write()).clear();
    (*METRIC_VALUES.write()).clear();
}
