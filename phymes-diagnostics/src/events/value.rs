use std::{fmt::Display, sync::Arc};

use parking_lot::Mutex;
use serde_json::{Map, Value};

/// A JSON structured event record
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// value of the metric gauge
    value: Arc<Mutex<Option<Map<String, Value>>>>,
}

impl EventRecord {
    /// Create a new event with no values
    pub fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }

    /// Inserts a key/value pair into the record
    pub fn insert(&self, k: &str, v: &Value) {
        let mut value = match self.value.lock().take() {
            Some(value) => value,
            None => Map::new(),
        };
        let _ = value.insert(k.to_string(), v.to_owned());
        *self.value.lock() = Some(value);
    }

    /// Sets the record with a JSON object
    pub fn set(&self, value: &Map<String, Value>) {
        *self.value.lock() = Some(value.to_owned());
    }

    /// Return the event value
    pub fn value(&self) -> Option<Map<String, Value>> {
        self.value.lock().clone()
    }
}

impl Display for EventRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.value() {
            None => write!(f, "NONE"),
            Some(v) => {
                write!(f, "{v:?}")
            }
        }
    }
}

/// Events used for tracking execution history and context
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
    /// Prints the value of this metric
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