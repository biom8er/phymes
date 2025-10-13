use std::{fmt::Display, sync::Arc};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Convert to a Tracer
pub trait Traceable {
    fn to_trace(&self) -> Tracer;
}

/// A tracer often a [SendableRecordBatchStreamMessage]
/// 
/// [SendableRecordBatchStreamMessage]: phymes_core::tasks::messages::SendableRecordBatchStreamMessage
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct Tracer {
    /// The name of the message
    pub message_name: String,
    /// The name of the subject of the message
    pub subject_name: String,
}

impl Tracer {
    pub fn new(message_name: &str, subject_name: &str) -> Self {
        Self {
            message_name: message_name.to_string(),
            subject_name: subject_name.to_string(),
        }
    }
}

impl Display for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "message:{};subject:{}", self.message_name, self.subject_name)
    }
}

/// A JSON structured event record
#[derive(Debug, Clone)]
pub struct TraceRecord {
    entered: Arc<Mutex<Option<Vec<Tracer>>>>,
    exited: Arc<Mutex<Option<Vec<Tracer>>>>,
}

impl TraceRecord {
    /// Create a new trace record with no values
    pub fn new() -> Self {
        Self {
            entered: Arc::new(Mutex::new(None)),
            exited: Arc::new(Mutex::new(None)),
        }
    }

    /// Enter the trace span
    pub fn enter<T: Traceable>(&self, tracers: &[T]) {
        *self.entered.lock() = Some(tracers.iter().map(|t| t.to_trace()).collect::<Vec<_>>());
    }

    /// Exit the trace span
    pub fn exit<T: Traceable>(&self, tracers: &[T]) {
        *self.exited.lock() = Some(tracers.iter().map(|t| t.to_trace()).collect::<Vec<_>>());
    }

    /// Return the event value
    pub fn entered(&self) -> Option<Vec<Tracer>> {
        self.entered.lock().clone()
    }

    /// Return the event value
    pub fn exited(&self) -> Option<Vec<Tracer>> {
        self.exited.lock().clone()
    }
}

impl Display for TraceRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        todo!()
    }
}