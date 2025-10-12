use std::sync::Arc;

pub mod builder;
pub mod diagnostic_set;
pub mod available_diagnostics;
pub mod label;

use parking_lot::Mutex;

use crate::diagnostics::diagnostic_set::{DiagnosticSet, DiagnosticSpan};

#[derive(Default, Debug, Clone)]
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

    /// Add the specified [DiagnosticSpan] to the underlying [DiagnosticSet]
    pub fn register(&self, metric: Arc<DiagnosticSpan>) {
        self.inner.lock().push(metric)
    }

    /// Return a clone of the inner [DiagnosticSet]
    pub fn clone_inner(&self) -> DiagnosticSet {
        let guard = self.inner.lock();
        (*guard).clone()
    }

    /// Clear the [DiagnosticSet]
    pub fn clear(&mut self) {
        self.inner.try_lock().unwrap().diagnostics.clear();
    }
}