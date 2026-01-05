use crate::{
    MappableTrait, ProcessorTrait, RuntimeEnv, SendableRecordBatchStreamMessageMap, TablePublication
};
use anyhow::Result;
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, TraceBuilderTrait};
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Level, event};

/// Processor that returns (i.e., echos) the [RecordBatch]es
#[derive(Debug)]
pub struct ProcessorEcho {
    name: String,
    r#type: String,
}

impl MappableTrait for ProcessorEcho {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for ProcessorEcho {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn process(
        &self,
        message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        _runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Trace the inbox
        let trace = if let Some(diagnostic_builder) = diagnostic_builder {
            let trace_builder = diagnostic_builder.clone().to_child(self.get_name())?;
            let trace = trace_builder
                .clone()
                .messages(line!(), file!(), self.get_name());
            trace.enter(&message.values().collect::<Vec<_>>());
            Some((trace, trace_builder))
        } else {
            None
        };

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&message.values().collect::<Vec<_>>());
        }

        Ok(message)
    }
}
