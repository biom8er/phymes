use std::thread::ThreadId;

use serde_json::{Map, Value};

use crate::{diagnostics::JSONObjectTrait, metrics::create_timestamp_micros};

/// The current context of the trace, event, or metric
#[derive(Debug, Clone)]
pub struct CurrentContext {
    /// using std::line!()
    line: u32,
    /// using std::file!()
    file: String,
    /// using std::thread::current().id().as_u64
    thread: ThreadId,
    /// no std
    function: String,
    /// using create_timestamp_micros()
    timestamp: i64,
}

impl CurrentContext {
    /// Create a new current context
    pub fn new(function: &str, line: u32, file: &str) -> Self {
        let file = file.to_string();
        let thread = std::thread::current().id();
        let function = function.to_string();
        let timestamp = create_timestamp_micros();
        Self {
            line,
            file,
            thread,
            function,
            timestamp,
        }
    }

    pub fn line(&self) -> &u32 {
        &self.line
    }

    pub fn file(&self) -> &str {
        &self.file
    }

    pub fn thread(&self) -> &ThreadId {
        &self.thread
    }

    pub fn function(&self) -> &str {
        &self.function
    }

    pub fn timestamp(&self) -> &i64 {
        &self.timestamp
    }
}

impl JSONObjectTrait for CurrentContext {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        let mut map = Map::new();
        map.insert("line".to_string(), self.line.to_owned().into());
        map.insert("file".to_string(), self.file.to_owned().into());
        let thread = format!("{:?}", self.thread());
        map.insert("thread".to_string(), thread.into());
        map.insert("function".to_string(), self.function.to_owned().into());
        map.insert("timestamp".to_string(), self.timestamp.to_owned().into());
        vec![map]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_context() {
        let current_context = CurrentContext::new("my_function", line!(), file!());
        assert_eq!(current_context.line(), &78);
        assert_eq!(
            current_context.file(),
            "phymes-diagnostics/src/traces/current_context.rs"
        );
        assert_eq!(current_context.thread(), &std::thread::current().id());
        assert_eq!(current_context.function(), "my_function");
        #[cfg(not(target_family = "wasm"))]
        assert!(*current_context.timestamp() < create_timestamp_micros());
    }
}
