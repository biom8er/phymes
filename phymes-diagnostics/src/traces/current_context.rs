use std::thread::ThreadId;

use crate::metrics::create_timestamp_micros;

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
    pub fn new(function: &str) -> Self {
        let line = std::line!();
        let file = std::file!().to_string();
        let thread = std::thread::current().id();
        let function = function.to_string();
        let timestamp = create_timestamp_micros();
        Self { line, file, thread, function, timestamp }
    }
}