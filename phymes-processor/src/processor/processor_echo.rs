use anyhow::Result;
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilder,
    SendableRecordBatchStreamMessageBuilderMap, SendableRecordBatchStreamMessageMap,
};
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Level, event};

use crate::ProcessorTrait;

/// Processor that returns (i.e., echos) the [RecordBatch]es
///
/// [RecordBatch]: arrow::record_batch::RecordBatch
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

    fn line_and_file(&self) -> (u32, String) {
        (line!(), file!().to_string())
    }

    fn process(
        &self,
        message: SendableRecordBatchStreamMessageMap,
        _diagnostic_builder: Option<&DiagnosticBuilder>,
        _runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        for (k, v) in message {
            let builder = SendableRecordBatchStreamMessage::get_builder()
                .with_name(v.get_name())
                .with_subject(v.get_subject())
                .with_message(v.get_message_own());
            let _ = builder_map.insert(k, builder);
        }

        Ok(builder_map)
    }
}
