use std::{fmt::Display, sync::Arc};

use parking_lot::Mutex;
use serde_json::{Map, Value};

/// A JSON structured event record
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// value of the metric gauge
    value: Arc<Mutex<Option<Map<String, Value>>>>,
}

impl Default for EventRecord {
    fn default() -> Self {
        Self::new()
    }
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
