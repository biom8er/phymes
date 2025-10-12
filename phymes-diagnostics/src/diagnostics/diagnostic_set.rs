use std::sync::Arc;
use anyhow::Result;

use crate::{diagnostics::{available_diagnostics::AvailableDiagnostics, label::Label}, traces::create_random_id};
pub use crate::traces::{CurrentContext, Span};

/// The diagnostic tools
#[derive(Debug, Clone)]
pub struct DiagnosticSpan {
    /// The event of the diagnostic including all diagnostic information
    /// whether a trace, event, or metric
    diagnostic: AvailableDiagnostics,
    /// The current context that the diagnostic was recorded
    current_context: CurrentContext,
    /// The span to which the diagnostic belongs
    span: Span,
    /// A unique ID for the diagnostic
    id: u64,
    /// arbitrary name=value pairs identifying this metric
    labels: Vec<Label>,
}

impl DiagnosticSpan {
    /// Create a new diagnostic span instantiating the [CurrentContext] and generating a unique `id`
    pub fn new(diagnostic: &AvailableDiagnostics, span: &Span, function: &str, labels: &[Label]) -> Result<Self> {
        Ok(Self { 
            diagnostic: diagnostic.to_owned(), 
            current_context: CurrentContext::new(function), 
            span: span.to_owned(), 
            id: create_random_id()?,
            labels: labels.to_owned(),
        })
    }
}

#[derive(Default, Debug, Clone)]
pub struct DiagnosticSet {
    pub(crate) diagnostics: Vec<Arc<DiagnosticSpan>>,
}

impl DiagnosticSet {
    /// Create a new container of diagnostics
    pub fn new() -> Self {
        DiagnosticSet::default()
    }

    /// Add the specified metric
    pub fn push(&mut self, diagnostic: Arc<DiagnosticSpan>) {
        self.diagnostics.push(diagnostic)
    }

    /// Returns an iterator across all metrics
    pub fn iter(&self) -> impl Iterator<Item = &Arc<DiagnosticSpan>> {
        self.diagnostics.iter()
    }
}