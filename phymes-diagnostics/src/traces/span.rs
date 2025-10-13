use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Create a (pseudo)random ID
pub fn create_random_id() -> u64 {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).unwrap();
    let id = u64::from_ne_bytes(buf);
    id
}

/// The span
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Span {
    /// The parent span name of execution
    parent_name: Option<String>,
    /// The parent id name of execution
    parent_id: Option<u64>,
    /// The scope of execution name
    span_name: String,
    /// The scope of execution id
    span_id: u64
}

impl Span {
    /// Create a new [Span]
    pub fn new(parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Self {
        Self { parent_name: parent_name.map(String::from), parent_id, span_name: span_name.to_string(), span_id }
    }
    /// Access the parent span
    pub fn parent(&self) -> (&Option<String>, &Option<u64>) {
        (&self.parent_name, &self.parent_id)
    }
    /// Access the current span
    pub fn span(&self) -> (&String, &u64) {
        (&self.span_name, &self.span_id)
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
/// Entrypoint for building a new span
pub struct SpanBuilder {
    parent_name: Option<String>,
    parent_id: Option<u64>,
    span_name: Option<String>,
    span_id: Option<u64>,
}

impl SpanBuilder {
    pub fn with_parent_span(mut self, parent_span: (&String, &u64)) -> Self {
        self.parent_name = Some(parent_span.0.to_owned());
        self.parent_id = Some(parent_span.1.to_owned());
        self
    }
    pub fn with_parent(mut self, parent_name: &str) -> Result<Self> {
        self.parent_name = Some(parent_name.to_string());
        self.parent_id = Some(create_random_id());
        Ok(self)
    }
    pub fn with_span(mut self, span_name: &str) -> Result<Self> {
        self.span_name = Some(span_name.to_string());
        self.span_id = Some(create_random_id());
        Ok(self)
    }
    pub fn build(self) -> Result<Span> {
        let span_name = match self.span_name {
            Some(name) => name,
            None => return Err(anyhow!("Add a span_name before building the span!"))
        };
        let span_id = match self.span_id {
            Some(id) => id,
            None => return Err(anyhow!("Add a span_id before building the span!"))
        };
        Ok(Span { 
            parent_name: self.parent_name, 
            parent_id: self.parent_id, 
            span_name, 
            span_id 
        })
    }
}