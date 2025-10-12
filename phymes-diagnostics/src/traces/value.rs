use std::{fmt::Display, sync::Arc};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A tracer often a [SendableRecordBatchStreamMessage]
/// 
/// [SendableRecordBatchStreamMessage]: phymes_core::tasks::messages::SendableRecordBatchStreamMessage
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct Tracer {
    /// The name of the message
    message_name: String,
    /// The name of the subject of the message
    subject_name: String,
}

impl Tracer {
    pub fn new(message_name: &str, subject_name: &str) -> Self {
        Self {
            message_name: message_name.to_string(),
            subject_name: subject_name.to_string(),
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct TracerEvents {
    entered: Vec<Tracer>,
    exited: Vec<Tracer>,
}

impl TracerEvents {
    pub fn enter(&mut self, tracer: &Tracer) {
        self.entered.push(tracer.to_owned());
    }
    pub fn exit(&mut self, tracer: &Tracer) {
        self.entered.push(tracer.to_owned());
    }
}

/// A JSON structured event record
#[derive(Debug, Clone)]
pub struct TraceRecord {
    /// value of the metric gauge
    value: Arc<Mutex<Option<>>>,
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