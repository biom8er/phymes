use std::fmt::Display;

use serde_json::{Map, Value};

use crate::{diagnostics::JSONObjectTrait, events::event_record::EventRecord};

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

impl Event {
    pub fn value(&self) -> Map<String, Value> {
        match self {
            Self::Trace(event) => event.value().unwrap_or_default(),
            Self::Debug(event) => event.value().unwrap_or_default(),
            Self::Info(event) => event.value().unwrap_or_default(),
            Self::Warn(event) => event.value().unwrap_or_default(),
            Self::Error(event) => event.value().unwrap_or_default(),
        }
    }
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

impl JSONObjectTrait for Event {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        let mut object = Vec::new();
        for (k, v) in self.value() {
            let mut map = Map::new();
            map.insert("event_level".to_string(), self.to_string().into());
            map.insert("record_name".to_string(), k.into());
            map.insert("record_value".to_string(), v);
            object.push(map);
        }
        object
    }
}