use std::{sync::Arc, thread::ThreadId};
use anyhow::Result;
use serde_json::{Map, Value};

use crate::{diagnostics::{available_diagnostics::{AvailableDiagnostics, DiagnosticsType}, label::Label}, traces::create_random_id};
pub use crate::traces::{CurrentContext, Span};

/// Trait to convert a complex data structure into a `Vec<Map<String, Value>>`
/// 
/// # Notes
/// In the future, it would be better to implement custom serde Serializers
pub trait JSONObjectTrait {
    /// Convert to a JSON object
    fn to_json_object(&self) -> Vec<Map<String, Value>>;
}

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
    pub fn new(diagnostic: &AvailableDiagnostics, span: &Span, line: u32, file: &str, function: &str, labels: &[Label]) -> Result<Self> {
        Ok(Self { 
            diagnostic: diagnostic.to_owned(), 
            current_context: CurrentContext::new(function, line, file), 
            span: span.to_owned(), 
            id: create_random_id(),
            labels: labels.to_owned(),
        })
    }

    /// Return a reference to the diagnostic
    pub fn diagnostic(&self) -> &AvailableDiagnostics {
        &self.diagnostic
    }

    /// Return a reference to the diagnostic
    pub fn diagnostic_mut(&mut self) -> &mut AvailableDiagnostics {
        &mut self.diagnostic
    }
}

impl JSONObjectTrait for DiagnosticSpan {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        // Start with the ID and Labels
        let mut map = Map::new();
        map.insert("id".to_string(), self.id.into());
        let labels = self.labels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(";");
        map.insert("labels".to_string(), labels.into());

        // Convert the span and current context
        map.extend(self.span.to_json_object().pop().unwrap());
        map.extend(self.current_context.to_json_object().pop().unwrap());

        // Iterate over diagnostics
        let mut object = Vec::new();
        for mut item in self.diagnostic.to_json_object() {
            item.extend(map.clone());
            object.push(item);
        }
        object
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

    /// Filter by the [DiagnosticType]
    pub fn filter_by_diagnostic_type(&self, diagnostic_type: DiagnosticsType) -> Self {
        let diagnostics = self.diagnostics.iter()
            .filter_map(|d| 
                if d.diagnostic.diagnostic_type() == diagnostic_type {
                    Some(d.clone())
                } else {
                    None
                }
            )
            .collect::<Vec<_>>();
        Self { diagnostics }
    }

    /// Return a columnar representation of the [DiagnosticSet]
    pub fn to_columns(&self) -> (
        Vec<AvailableDiagnostics>,
        Vec<String>,
        Vec<u64>,
        Vec<String>,
        Vec<u64>,
        Vec<u32>,
        Vec<String>,
        Vec<ThreadId>,
        Vec<String>,
        Vec<i64>,
        Vec<String>,
        Vec<u64>,
    ) {
        // Diagnostics
        let mut diagnostic_vec = Vec::<AvailableDiagnostics>::new();

        // Span columns
        let mut parent_names_vec = Vec::<String>::new();
        let mut parent_ids_vec = Vec::<u64>::new();
        let mut span_names_vec = Vec::<String>::new();
        let mut span_ids_vec = Vec::<u64>::new();

        // Context columns
        let mut line_vec = Vec::<u32>::new();
        let mut file_vec = Vec::<String>::new();
        let mut thread_vec = Vec::<ThreadId>::new();
        let mut function_vec = Vec::<String>::new();
        let mut timestamp_vec = Vec::<i64>::new();

        // Labels and IDs
        let mut labels_vec = Vec::<String>::new();
        let mut ids_vec = Vec::<u64>::new();

        // Interate through the set
        for diagnostic_span in self.diagnostics.iter() {
            diagnostic_vec.push(diagnostic_span.diagnostic.to_owned());
            let parent_span = diagnostic_span.span.parent();
            let span = diagnostic_span.span.span();
            parent_names_vec.push(parent_span.0.clone().unwrap_or_default());
            parent_ids_vec.push(parent_span.1.clone().unwrap_or_default());
            span_names_vec.push(span.0.to_owned());
            span_ids_vec.push(span.1.to_owned());
            line_vec.push(diagnostic_span.current_context.line().to_owned());
            file_vec.push(diagnostic_span.current_context.file().to_owned());
            thread_vec.push(diagnostic_span.current_context.thread().to_owned());
            function_vec.push(diagnostic_span.current_context.function().to_owned());
            timestamp_vec.push(diagnostic_span.current_context.timestamp().to_owned());
            let labels = diagnostic_span.labels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(";");
            labels_vec.push(labels);
            ids_vec.push(diagnostic_span.id.to_owned());
        }

        (diagnostic_vec, parent_names_vec, parent_ids_vec, span_names_vec, span_ids_vec, line_vec, file_vec, thread_vec, function_vec, timestamp_vec, labels_vec, ids_vec)
    }
}

impl JSONObjectTrait for DiagnosticSet {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        self.diagnostics.iter()
            .flat_map(|d| d.to_json_object())
            .collect::<Vec<_>>()
    }
}