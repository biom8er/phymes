use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{diagnostics::JSONObjectTrait, events::Event, metrics::Metric, traces::Trace};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum DiagnosticsType {
    /// Traces
    Trace,
    /// Events
    Event,
    /// Metrics
    #[default]
    Metric,
}

/// The available diagnostics
#[derive(Debug, Clone)]
pub enum AvailableDiagnostics {
    /// Traces
    Trace(Trace),
    /// Events
    Event(Event),
    /// Metrics
    Metric(Metric),
}

impl AvailableDiagnostics {
    pub fn diagnostic_type(&self) -> DiagnosticsType {
        match self {
            Self::Event(_) => DiagnosticsType::Event,
            Self::Metric(_) => DiagnosticsType::Metric,
            Self::Trace(_) => DiagnosticsType::Trace,
        }
    }
}

impl JSONObjectTrait for AvailableDiagnostics {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        match self {
            Self::Event(event) => event.to_json_object(),
            Self::Metric(metric) => metric.to_json_object(),
            Self::Trace(trace) => trace.to_json_object(),
        }
    }
}
