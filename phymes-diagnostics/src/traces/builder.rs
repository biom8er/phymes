use crate::{diagnostics::{AvailableDiagnostics, DiagnosticBuilder, DiagnosticBuilderTrait}, traces::{tracer::TraceRecord, Trace}};

/// Trait extension constructing traces
pub trait TraceBuilderTrait: DiagnosticBuilderTrait {
    /// Consume self and create a new record for recording message traces
    fn messages(self, function: &str) -> TraceRecord;
}

impl TraceBuilderTrait for DiagnosticBuilder {
    fn messages(self, function: &str) -> TraceRecord {
        let record = TraceRecord::new();
        let diagnostic = AvailableDiagnostics::Trace(Trace::Messages(record.clone()));
        self.build(&diagnostic, function);
        record
    }
}