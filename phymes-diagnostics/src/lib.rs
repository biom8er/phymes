use std::sync::Arc;

use parking_lot::Mutex;

use crate::{events::EventType, traces::{CurrentContext, Span}};

pub mod metrics;
pub mod traces;
pub mod events;

/// The diagnostic tools
#[derive(Debug, Clone)]
pub struct DiagnosticSpan {
    /// The current context that the diagnostic was recorded
    current_context: CurrentContext,
    /// The span to which the diagnostic belongs
    span: Span,
    /// The event of the diagnostic including all diagnostic information
    /// whether a trace, event, or metric
    event: EventType,
    /// A unique ID for the diagnostic
    id: u64,
}

#[derive(Default, Debug, Clone)]
pub struct DiagnosticSet {
    diagnostics: Vec<Arc<DiagnosticSpan>>,
}

impl DiagnosticSet {
    /// Create a new container of diagnostics
    pub fn new() -> Self {
        DiagnosticSet::default()
    }

    /// Add the specified metric
    pub fn push(&mut self, diagnostic: Arc<DiagnosticSpan>) {
        self.diagnostics.push(metric)
    }

    /// Returns an iterator across all metrics
    pub fn iter(&self) -> impl Iterator<Item = &Arc<DiagnosticSpan>> {
        self.diagnostics.iter()
    }
}

pub struct Diagnostics {
    inner: Arc<Mutex<DiagnosticSet>>,
}

impl Diagnostics {
    /// Create a new empty shared metrics set
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(DiagnosticSet::new())),
        }
    }

    /// Add the specified metric to the underlying metric set
    pub fn register(&self, metric: Arc<Metric>) {
        self.inner.lock().push(metric)
    }

    /// Return a clone of the inner [`MetricsSet`]
    pub fn clone_inner(&self) -> MetricsSet {
        let guard = self.inner.lock();
        (*guard).clone()
    }

    /// Clear the metrics
    pub fn clear(&mut self) {
        self.inner.try_lock().unwrap().metrics.clear();
    }
}