//! Builder for creating diagnostics

use std::{borrow::Cow, sync::Arc};
use anyhow::{anyhow, Result};

use crate::{diagnostics::{available_diagnostics::AvailableDiagnostics, label::Label, DiagnosticSpan, Diagnostics}, traces::{Span, SpanBuilder}};

/// Trait for diagnostic builders (traces, events, and metrics) to extend
pub trait DiagnosticBuilderTrait {
    /// Create a new `DiagnosticBuilder` that will register the result of `build()` with the `diagnostics`
    fn new(diagnostics: &Diagnostics) -> Self;

    /// Add a label to the metric being constructed
    fn with_label(self, label: Label) -> Self;

    /// Add a label to the metric being constructed
    fn with_new_label(
        self,
        name: impl Into<Cow<'static, str>>,
        value: impl Into<Cow<'static, str>>,
    ) -> Self;

    /// Add the span
    fn with_span(self, span: &Span) -> Self;

    /// Move the current span to parent and add in the child span
    fn to_child(self, span_name: &str) -> Result<Self> where Self: Sized;

    /// Consume self and create a [DiagnosticSpan] of the specified value
    /// registered with the [Diagnostics]
    fn build(self, diagnostic: &AvailableDiagnostics, function: &str);
}

/// Structure for constructing diagnostics including traces, events, and metrics
#[derive(Clone, Debug)]
pub struct DiagnosticBuilder {
    /// Location that the metric created by this builder will be added do
    pub(crate) diagnostics: Diagnostics,
    pub(crate) span: Option<Span>,
    pub(crate) labels: Vec<Label>,
}

impl DiagnosticBuilderTrait for DiagnosticBuilder {
    fn new(diagnostics: &Diagnostics) -> Self {
        Self {
            diagnostics: diagnostics.clone(),
            span: None,
            labels: vec![],
        }
    }

    fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    fn with_new_label(
        self,
        name: impl Into<Cow<'static, str>>,
        value: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.with_label(Label::new(name.into(), value.into()))
    }

    fn with_span(mut self, span: &Span) -> Self {
        self.span = Some(span.to_owned());
        self
    }

    fn to_child(mut self, span_name: &str) -> Result<Self> {
        let span = if let Some(s) = self.span {
            SpanBuilder::default().with_parent_span(s.span())
                .with_span(span_name)
                .build()?
        } else {
            return Err(anyhow!("Provide a `span` before attempting to create a child diagnostic builder!"));
        };
        self.span = Some(span);
        Ok(self)
    }

    fn build(self, diagnostic: &AvailableDiagnostics, function: &str) {
        let Self {
            diagnostics,
            span,
            labels,
        } = self;
        let diagnostic_span = Arc::new(DiagnosticSpan::new(diagnostic, &span.unwrap(), function, &labels).unwrap());
        diagnostics.register(diagnostic_span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_builder_to_child() -> Result<()> {
        // Initialize the diagnostics and span
        let diagnostics = Diagnostics::new();
        let span = SpanBuilder::default().with_parent("parent_span").with_span("span_name").build()?;

        // Test error when trying to create a child without a parent
        let builder_err = DiagnosticBuilder::new(&diagnostics).to_child("child_span");
        assert!(builder_err.is_err());

        // Test for the expected span
        let builder = DiagnosticBuilder::new(&diagnostics).with_span(&span).to_child("child_span")?;
        assert_ne!(builder.span.as_ref().unwrap(), &span);
        assert_eq!(builder.span.as_ref().unwrap().parent().0.as_ref().unwrap(), span.span().0);
        assert_eq!(builder.span.as_ref().unwrap().parent().1.as_ref().unwrap(), span.span().1);

        Ok(())
    }
}