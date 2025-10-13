use std::fmt::Display;

use crate::events::event_record::EventRecord;

/// Events to add context to enable building a comprehensive timeline of what happened, when it happened, and why it happened
#[derive( Debug, Clone)]
pub enum Event {
    /// For extremely fine-grained information, useful for tracing program execution.
    Trace(EventRecord),
    /// For detailed debugging information, typically used during development.
    Debug(EventRecord),
    /// For general informational messages about the program's operation.
    Info(EventRecord),
    /// For warnings about potential issues that are not immediately fatal.
    Warn(EventRecord),
    /// For critical errors that cause the program to fail or behave incorrectly.
    Error(EventRecord),
}

impl Display for Event {
    /// Prints the value of this event
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Trace(_event) => write!(f, "Trace"),
            Self::Debug(_event) => write!(f, "Debug"),
            Self::Info(_event) => write!(f, "Info"),
            Self::Warn(_event) => write!(f, "Warn"),
            Self::Error(_event) => write!(f, "Error"),
        }
    }
}