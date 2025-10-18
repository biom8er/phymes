use std::collections::HashMap;

use anyhow::Result;
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    schemas::{chat_completion, mermaid::{create_mermaid_sequence_diagram_messages_template_batch, create_mermaid_sequence_diagram_participants_template_batch}, types},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::table_trait::{Table, TableBuilderTrait, TableTrait},
};
use phymes_diagnostics::{HashMap, HashSet};

use crate::{
    candle_data::data_config::DataConfig,
    candle_operators::data_operator::DataOperatorTrait,
};

/// Compute the normalized start and end times in a [RecordBatch]
#[derive(Debug)]
pub struct FromTracesToMessages {}

impl MappableTrait for FromTracesToMessages {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for FromTracesToMessages {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        from_traces_to_messages(lhs_args, rhs_args.unwrap(), device)
    }
    fn new(_config: &DataConfig) -> Self {
        FromTracesToMessages {}
    }
    fn get_description() -> String {
        ""
            .to_string()
    }
    fn get_json_tool_schema() -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "lhs_values".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::Array),
                description: Some(
                    "A list of value column identifiers for the left hand side table".to_string(),
                ),
                ..Default::default()
            }),
        );
        properties.insert(
            "op_kwargs".to_string(),
            Box::new(types::JSONSchemaDefine {
                schema_type: Some(types::JSONSchemaType::String),
                description: Some(
                    "DataCastOperator and DataType with optional column renaming and template injection in the form of a JSON object".to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = types::Function {
            name: Self::get_static_name().to_string(),
            description: Some(Self::get_description()),
            parameters: types::FunctionParameters {
                schema_type: types::JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec![
                    "lhs_name".to_string(),
                    "lhs_values".to_string(),
                    "op_kwargs".to_string(),
                ]),
            },
        };
        let tool = chat_completion::Tool {
            r#type: chat_completion::ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

/// Custom function to convert `SessionTasks` and `Traces` to Sequence Diagram Messages
/// 
/// # Notes
/// 
/// * LHS is SessionTasks
/// * RHS is Traces
/// * Output schema is MermaidSequenceDiagramMessagesTemplate
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `rhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
pub fn from_traces_to_messages(lhs_args: &[RecordBatch], rhs_args: &[RecordBatch], _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs and rhs into tables
    let lhs_table = Table::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("")
        .build()?;
    let rhs_table = Table::get_builder()
        .with_record_batches(rhs_args.to_vec())?
        .with_name("")
        .build()?;

    // The first trace is always the user -> session and session -> user
    // Initialize the messages vecs
    let mut subject_name_vec: Vec<String> = Vec::new();
    let mut object_name_vec: Vec<String> = Vec::new();
    let mut message_type_vec: Vec<String> = Vec::new();
    let mut activation_type_vec: Vec<String> = Vec::new();
    let mut message_content_vec: Vec<String> = Vec::new();
    let mut note_content_vec: Vec<String> = Vec::new();
    let mut note_location_vec: Vec<String> = Vec::new();

    // Get the unique tasks and processors
    let task_name_set = lhs_table.get_column_as_vec_nonprimitive::<String>("task_name")?.into_iter().collect::<HashSet<_>>();
    let processor_name_set = lhs_table.get_column_as_vec_nonprimitive::<String>("processor_name")?.into_iter().collect::<HashSet<_>>();

    // Get the trace fields sorted by time
    let mut combined = rhs_table.get_column_as_vec_nonprimitive::<String>("tracer_event")?.into_iter()
        .zip(rhs_table.get_column_as_vec_nonprimitive::<String>("span_name")?.into_iter())
        .zip(rhs_table.get_column_as_vec_nonprimitive::<String>("parent_name")?.into_iter())
        .zip(rhs_table.get_column_as_vec_primitive::<u64>("span_id")?.into_iter())
        .zip(rhs_table.get_column_as_vec_primitive::<u64>("parent_id")?.into_iter())
        .zip(rhs_table.get_column_as_vec_nonprimitive::<String>("subject_name")?.into_iter())
        .zip(rhs_table.get_column_as_vec_nonprimitive::<String>("message_name")?.into_iter())
        .zip(rhs_table.get_column_as_vec_primitive::<i64>("timestamp")?.into_iter())
        .map(|(((((((a, b), c), d), e), f), g), h)| (a, b, c, d, e, f, g, h))
        .collect::<Vec<_>>();

    // Group by the messages according to tracer_event and span
    // (message_name, subject_name): unique_id: tracer_event: timestamp
    let mut messages_map = HashMap::<(String, String), HashMap<(String, String, u64, u64), HashMap<String, i64>>>::new();    
    for (tracer_event, span_name, parent_name, span_id, parent_id, subject_name, message_name, timestamp) in combined.iter() {
        let unique_id = (span_name.to_string(), parent_name.to_string(), span_id.to_owned(), parent_id.to_owned());
        let message_id = (message_name.to_string(), subject_name.to_string());
        if let Some(message) = messages_map.get_mut(&message_id) {
            if let Some(span) = message.get_mut(&unique_id) {
                if let Some(tracer) = span.get_mut(tracer_event) {
                    *tracer = timestamp.to_owned();
                } else {
                    span.insert(tracer_event.to_string(), timestamp.to_owned());
                }
            } else {
                let mut span = HashMap::<String, i64>::new();
                span.insert(tracer_event.to_string(), timestamp.to_owned());
                message.insert(unique_id, span);
            }
        } else {
            let mut span = HashMap::<String, i64>::new();
            span.insert(tracer_event.to_string(), timestamp.to_owned());
            let mut message = HashMap::<(String, String, u64, u64), HashMap<String, i64>>::new();
            message.insert(unique_id, span);
            messages_map.insert(message_id, message);
        }
    }

    // Ungroup and sort according to timestamp
    let combined = messages_map.into_iter()
        .map(|((message_name, subject_name), v)| v
            .into_iter().map(move |((span_name, parent_name, span_id, parent_id), v)| v
                // Check for both entered and exited
                .into_iter()
                .map(|(tracer_event, timestamp)| (message_name.to_owned(), subject_name.to_owned(), span_name.to_owned(), parent_name.to_owned(), span_id, parent_id, tracer_event, timestamp))
                .collect::<Vec<_>>()
                )
            )
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    combined.sort_by(|a, b| a.6.cmp(&b.6)); // message_name
    combined.sort_by(|a, b| a.1.cmp(&b.1)); // tracer_event
    combined.sort_by(|a, b| a.3.cmp(&b.3)); // parent_name
    combined.sort_by(|a, b| a.4.cmp(&b.4)); // span_name
    combined.sort_by(|a, b| a.7.cmp(&b.7)); // timestmap

    create_mermaid_sequence_diagram_messages_template_batch(subject_name_vec,
        object_name_vec,
        message_type_vec,
        activation_type_vec,
        message_content_vec,
        note_content_vec,
        note_location_vec)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, StringArray};
    use phymes_core::session::common_traits::device;

    use super::*;

    #[test]
    fn test_from_traces_to_messages() -> Result<()> {
        // Make the test record batches
        let lhs_1_vec = vec!["t1", "t1", "t2", "t2", "t3"];
        let lhs_1_array: ArrayRef = Arc::new(StringArray::from(lhs_1_vec));
        let lhs_2_vec = vec!["p1", "p2", "p3", "p4", "p1"];
        let lhs_2_array: ArrayRef = Arc::new(StringArray::from(lhs_2_vec));
        let lhs_batch = RecordBatch::try_from_iter(vec![
            ("task_name", lhs_1_array),
            ("processor_name", lhs_2_array),
        ])?;
        let rhs_1_vec = vec!["s", "t1", "p1", "p2", "t1", "s", "t2", "p3", "p4", "t2", "s", "t3", "p1", "t3"];
        let rhs_1_array: ArrayRef = Arc::new(StringArray::from(rhs_1_vec));
        let rhs_2_vec = vec!["t1", "p1", "p2", "t1", "s", "t2", "p3", "p4", "t2", "s", "t3", "p1", "t3", "s"];
        let rhs_2_array: ArrayRef = Arc::new(StringArray::from(rhs_2_vec));
        let rhs_batch = RecordBatch::try_from_iter(vec![
            ("subject_name", rhs_1_array),
            ("object_name", rhs_2_array),
        ])?;

        // Make the device
        let device = device(false)?;

        let result = from_traces_to_messages(
            &[lhs_batch],
            &[rhs_batch],
            &device,
        )?;
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let participants = result_table.get_column_as_vec_str("participant_name");
        assert_eq!(participants, ["User", "State", "t1", "p1", "p2", "t2", "p3", "p4", "t3"]);
        let participants = result_table.get_column_as_vec_str("participant_type");
        assert_eq!(participants, ["actor", "database", "collections", "participant", "participant", "collections", "participant", "participant", "collections"]);

        Ok(())
    }
}
