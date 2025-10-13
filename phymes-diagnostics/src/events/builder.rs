use crate::{diagnostics::{AvailableDiagnostics, DiagnosticBuilder, DiagnosticBuilderTrait}, events::{available_events::Event, event_record::EventRecord}};

/// Trait extension constructing event records
pub trait EventBuilderTrait: DiagnosticBuilderTrait {
    /// Consume self and create a new event record at the Trace level
    fn trace(self, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Debug level
    fn debug(self, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Info level
    fn info(self, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Warn level
    fn warn(self, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Error level
    fn error(self, function: &str) -> EventRecord;
}

impl EventBuilderTrait for DiagnosticBuilder {
    fn trace(self, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Trace(record.clone()));
        self.build(&diagnostic, function);
        record
    }
    
    fn debug(self, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Debug(record.clone()));
        self.build(&diagnostic, function);
        record
    }
    
    fn info(self, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Info(record.clone()));
        self.build(&diagnostic, function);
        record
    }
    
    fn warn(self, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Warn(record.clone()));
        self.build(&diagnostic, function);
        record
    }
    
    fn error(self, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Error(record.clone()));
        self.build(&diagnostic, function);
        record
    }
}