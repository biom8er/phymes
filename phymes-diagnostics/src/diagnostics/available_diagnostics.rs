use crate::{events::Event, metrics::Metric, traces::Trace};

/// The event type
#[derive(Debug, Clone)]
pub enum AvailableDiagnostics {
    /// Traces
    Trace(Trace),
    /// Events
    Event(Event),
    /// Metrics
    Metric(Metric)
}