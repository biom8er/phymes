use std::sync::Arc;

mod available_diagnostics;
mod builder;
mod diagnostic_set;
mod label;

use parking_lot::Mutex;

pub use available_diagnostics::{AvailableDiagnostics, DiagnosticsType};
pub use builder::{DiagnosticBuilder, DiagnosticBuilderTrait};
pub use diagnostic_set::{DiagnosticSet, DiagnosticSpan, JSONObjectTrait};
pub use label::Label;

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
    pub fn register(&self, diagnostic: Arc<DiagnosticSpan>) {
        self.inner.lock().push(diagnostic)
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
