use std::fmt::Display;

use serde_json::{Map, Value};

use crate::{diagnostics::JSONObjectTrait, traces::tracer::TraceRecord, Tracer};

/// Traces to track the flow of subject messages through tasks and processors
#[derive( Debug, Clone)]
pub enum Trace {
    /// For extremely fine-grained information, useful for tracing program execution.
    Messages(TraceRecord),
}

impl Trace {
    pub fn entered(&self) -> Vec<Tracer> {
        match self {
            Self::Messages(record) => record.entered().unwrap_or_default(),
        }
    }
    pub fn exited(&self) -> Vec<Tracer> {
        match self {
            Self::Messages(record) => record.exited().unwrap_or_default(),
        }
    }
}

impl Display for Trace {
    /// Prints the value of this metric
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Messages(_event) => write!(f, "Messages"),
        }
    }
}

impl JSONObjectTrait for Trace {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        let mut object = Vec::new();
        for tracer in self.entered().into_iter() {
            let mut map = Map::new();
            map.insert("tracer_type".to_string(), self.to_string().into());
            map.insert("tracer_event".to_string(), Value::String("entered".to_string()));
            map.insert("message_name".to_string(), tracer.message_name.into());
            map.insert("subject_name".to_string(), tracer.subject_name.into());
            object.push(map);
        }
        for tracer in self.exited().into_iter() {
            let mut map = Map::new();
            map.insert("tracer_type".to_string(), self.to_string().into());
            map.insert("tracer_event".to_string(), Value::String("exited".to_string()));
            map.insert("message_name".to_string(), tracer.message_name.into());
            map.insert("subject_name".to_string(), tracer.subject_name.into());
            object.push(map);
        }
        object
    }
}