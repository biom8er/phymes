use std::fmt::Display;

use serde_json::{Map, Value};

use crate::{Tracer, diagnostics::JSONObjectTrait, traces::tracer::TraceRecord};

/// Traces to track the flow of subject messages through tasks and processors
#[derive(Debug, Clone)]
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
            map.insert(
                "tracer_event".to_string(),
                Value::String("entered".to_string()),
            );
            map.insert("message_name".to_string(), tracer.message_name.into());
            map.insert("subject_name".to_string(), tracer.subject_name.into());
            object.push(map);
        }
        for tracer in self.exited().into_iter() {
            let mut map = Map::new();
            map.insert("tracer_type".to_string(), self.to_string().into());
            map.insert(
                "tracer_event".to_string(),
                Value::String("exited".to_string()),
            );
            map.insert("message_name".to_string(), tracer.message_name.into());
            map.insert("subject_name".to_string(), tracer.subject_name.into());
            object.push(map);
        }
        object
    }
}

// DM: linting does not realize that it is used in `diagnostic_set` tests
#[allow(unused)]
pub mod available_tracers_tests {
    use crate::TraceableTrait;

    use super::*;

    pub struct Message {
        message_name: String,
        subject_name: String,
    }

    impl Message {
        pub fn new(message_name: &str, subject_name: &str) -> Self {
            Message {
                message_name: message_name.to_string(),
                subject_name: subject_name.to_string(),
            }
        }
    }

    impl TraceableTrait for Message {
        fn to_trace(&self) -> Tracer {
            Tracer::new(&self.message_name, &self.subject_name)
        }
    }
}

#[cfg(test)]
mod tests {
    use available_tracers_tests::Message;

    use super::*;

    #[test]
    fn test_tracers_records() {
        let record = TraceRecord::new();
        let trace = Trace::Messages(record.clone());
        record.enter(&[
            &Message::new("m1", "s1"),
            &Message::new("m2", "s2"),
            &Message::new("m3", "s3"),
        ]);
        record.exit(&[
            &Message::new("m2", "s2"),
            &Message::new("m3", "s3"),
            &Message::new("m4", "s4"),
        ]);

        let object = trace.to_json_object();
        assert_eq!(object.len(), 6);
        assert_eq!(
            object
                .first()
                .unwrap()
                .get("tracer_type")
                .unwrap()
                .as_str()
                .unwrap(),
            trace.to_string().as_str()
        );
        assert_eq!(
            object
                .first()
                .unwrap()
                .get("tracer_event")
                .unwrap()
                .as_str()
                .unwrap(),
            "entered"
        );
        assert_eq!(
            object
                .first()
                .unwrap()
                .get("message_name")
                .unwrap()
                .as_str()
                .unwrap(),
            "m1"
        );
        assert_eq!(
            object
                .first()
                .unwrap()
                .get("subject_name")
                .unwrap()
                .as_str()
                .unwrap(),
            "s1"
        );
        assert_eq!(
            object
                .last()
                .unwrap()
                .get("tracer_type")
                .unwrap()
                .as_str()
                .unwrap(),
            trace.to_string().as_str()
        );
        assert_eq!(
            object
                .last()
                .unwrap()
                .get("tracer_event")
                .unwrap()
                .as_str()
                .unwrap(),
            "exited"
        );
        assert_eq!(
            object
                .last()
                .unwrap()
                .get("message_name")
                .unwrap()
                .as_str()
                .unwrap(),
            "m4"
        );
        assert_eq!(
            object
                .last()
                .unwrap()
                .get("subject_name")
                .unwrap()
                .as_str()
                .unwrap(),
            "s4"
        );
    }
}
