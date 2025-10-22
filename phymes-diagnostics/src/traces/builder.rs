use crate::{
    diagnostics::{AvailableDiagnostics, DiagnosticBuilder, DiagnosticBuilderTrait},
    traces::{Trace, tracer::TraceRecord},
};

/// Trait extension constructing traces
pub trait TraceBuilderTrait: DiagnosticBuilderTrait {
    /// Consume self and create a new record for recording message traces
    fn messages(self, line: u32, file: &str, function: &str) -> TraceRecord;
}

impl TraceBuilderTrait for DiagnosticBuilder {
    fn messages(self, line: u32, file: &str, function: &str) -> TraceRecord {
        let record = TraceRecord::new();
        let diagnostic = AvailableDiagnostics::Trace(Trace::Messages(record.clone()));
        self.build(&diagnostic, line, file, function);
        record
    }
}
