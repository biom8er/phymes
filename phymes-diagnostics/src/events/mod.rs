use std::{default, fmt::Display};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::metrics::{Label, MetricValue};

/// Event Levels (from highest to lowest priority):
/// error!: For critical errors that cause the program to fail or behave incorrectly.
/// warn!: For warnings about potential issues that are not immediately fatal.
/// info!: For general informational messages about the program's operation.
/// debug!: For detailed debugging information, typically used during development.
/// trace!: For extremely fine-grained information, useful for tracing program execution.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub enum EventLevel {
    Trace,
    #[default]
    Debug,
    Info,
    Warn,
    Error,
}

impl Display for EventLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventLevel::Trace => write!(f, "Trace"),
            EventLevel::Debug => write!(f, "Debug"),
            EventLevel::Info => write!(f, "Info"),
            EventLevel::Warn => write!(f, "Warn"),
            EventLevel::Error => write!(f, "Error"),
        }
    }
}

/// The event type
#[derive(Debug, Clone)]
pub enum EventType {
    /// The start of a message trace
    Enter(TraceEvent),
    /// The end of a message trace
    Exit(TraceEvent),
    /// General events
    Log(LogEvent),
    /// Metrics
    Measurement(MeasurementEvent),
}

/// A trace event
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct TraceEvent {
    /// The name of the message
    message_name: String,
    /// The name of the subject of the message
    subject_name: String,
    /// The event level (always Trace)
    level: EventLevel,
}

impl TraceEvent {
    pub fn new(message_name: &str, subject_name: &str) -> Self {
        Self {
            message_name: message_name.to_string(),
            subject_name: subject_name.to_string(),
            level: EventLevel::Trace
        }
    }
}

/// A log event
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct LogEvent {
    /// The event level
    level: EventLevel,
    /// A JSON structured value for the event
    value: Value,
}

impl LogEvent {
    pub fn new(level: &EventLevel, value: &Value) -> Self {
        Self {
            level: level.to_owned(),
            value: value.to_owned(),
        }
    }
}

/// A measurement event
#[derive(Debug, Clone)]
pub struct MeasurementEvent {
    /// The event level (always Info)
    level: EventLevel,
    /// The value of the metric
    value: MetricValue,
    /// arbitrary name=value pairs identifying this metric
    labels: Vec<Label>,
}

impl MeasurementEvent {
    pub fn new(value: &MetricValue, labels: &[Label]) -> Self {
        Self {
            level: EventLevel::Info,
            value: value.to_owned(),
            labels: labels.to_owned(),
        }
    }
}