use serde::{Deserialize, Serialize};

use crate::{events::Event, metrics::Metric};

/// The event type
#[derive(Debug, Clone)]
pub enum AvailableDiagnostics {
    /// Traces
    Trace,
    /// Events
    Event(Event),
    /// Metrics
    Metric(Metric)
}