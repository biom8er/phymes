use std::fmt::Display;

use crate::traces::tracer::TraceRecord;

/// Traces to track the flow of subject messages through tasks and processors
#[derive( Debug, Clone)]
pub enum Trace {
    /// For extremely fine-grained information, useful for tracing program execution.
    Messages(TraceRecord),
}

impl Display for Trace {
    /// Prints the value of this metric
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Messages(_event) => write!(f, "Messages"),
        }
    }
}