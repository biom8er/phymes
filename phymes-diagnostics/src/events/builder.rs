use crate::{diagnostics::{AvailableDiagnostics, DiagnosticBuilder, DiagnosticBuilderTrait}, events::{available_events::Event, event_record::EventRecord}};

/// Trait extension constructing event records
pub trait EventBuilderTrait: DiagnosticBuilderTrait {
    /// Consume self and create a new event record at the Trace level
    fn trace(self, line: u32, file: &str, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Debug level
    fn debug(self, line: u32, file: &str, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Info level
    fn info(self, line: u32, file: &str, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Warn level
    fn warn(self, line: u32, file: &str, function: &str) -> EventRecord;
    /// Consume self and create a new event record at the Error level
    fn error(self, line: u32, file: &str, function: &str) -> EventRecord;
}

impl EventBuilderTrait for DiagnosticBuilder {
    fn trace(self, line: u32, file: &str, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Trace(record.clone()));
        self.build(&diagnostic, line, file, function);
        record
    }
    
    fn debug(self, line: u32, file: &str, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Debug(record.clone()));
        self.build(&diagnostic, line, file, function);
        record
    }
    
    fn info(self, line: u32, file: &str, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Info(record.clone()));
        self.build(&diagnostic, line, file, function);
        record
    }
    
    fn warn(self, line: u32, file: &str, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Warn(record.clone()));
        self.build(&diagnostic, line, file, function);
        record
    }
    
    fn error(self, line: u32, file: &str, function: &str) -> EventRecord {
        let record = EventRecord::new();
        let diagnostic = AvailableDiagnostics::Event(Event::Error(record.clone()));
        self.build(&diagnostic, line, file, function);
        record
    }
}