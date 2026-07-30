#![allow(unused)]
// DM: https://github.com/biom8er/phymes/issues/111#issue-3492849457

use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray},
    datatypes::Schema,
};
use candle_core::Device;
use phymes_diagnostics::HashSet;
use phymes_schemas::{
    AvailableSubjects, CsvFormat, DataEncoding, DataFormat, Function, FunctionParameters,
    JSONSchemaDefine, JSONSchemaType, Tool, ToolType, create_parse_owl_batch,
    create_parse_xml_batch,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use serde::{Deserialize, Serialize};

use crate::{DataConfig, DataOperatorTrait, operators::sort::sort};

/// Compute the normalized start and end times in a [RecordBatch]
#[derive(Debug, Default, Serialize, Deserialize)]
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
    fn new(_config: &DataConfig) -> Result<Self> {
        Ok(FromTracesToMessages {})
    }
}

/// Custom function to convert `NetworkTasks` and `Traces` to Sequence Diagram Messages
///
/// # Notes
///
/// * LHS is Traces
/// * RHS is NetworkTasks
/// * Output schema is MermaidSequenceDiagramMessagesTemplate
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `rhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
pub fn from_traces_to_messages(
    lhs_args: &[RecordBatch],
    rhs_args: &[RecordBatch],
    device: &Device,
) -> Result<RecordBatch> {
    // Get the unique tasks and processors
    let rhs_table = Subject::get_builder()
        .with_record_batches(rhs_args.to_vec())?
        .with_name("")
        .build()?;
    let task_name_set = rhs_table
        .get_column_as_vec_nonprimitive::<String>("task_name")?
        .into_iter()
        .collect::<HashSet<_>>();
    let processor_name_set = rhs_table
        .get_column_as_vec_nonprimitive::<String>("processor_name")?
        .into_iter()
        .collect::<HashSet<_>>();

    // Presort the traces by columns
    let lhs_values = [
        "tracer_event",
        "parent_name",
        "span_name",
        "tracer_timestamp",
        "subject_name",
    ];
    let mut lhs_sorted = RecordBatch::new_empty(Arc::new(Schema::empty()));
    for (iter, column_name) in lhs_values.iter().enumerate() {
        if iter > 0 {
            lhs_sorted = sort(column_name, &[lhs_sorted], true, device)?;
        } else {
            lhs_sorted = sort(column_name, lhs_args, true, device)?;
        }
    }
    let lhs_table = Subject::get_builder()
        .with_record_batches(vec![lhs_sorted])?
        .with_name("")
        .build()?;

    // Convert to messages vec
    let combined = lhs_table
        .get_column_as_vec_nonprimitive::<String>("tracer_event")?
        .into_iter()
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("span_name")?)
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("parent_name")?)
        .zip(lhs_table.get_column_as_vec_primitive::<i64>("span_id")?)
        .zip(lhs_table.get_column_as_vec_primitive::<i64>("parent_id")?)
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("subject_name")?)
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("message_name")?)
        .zip(lhs_table.get_column_as_vec_primitive::<i64>("tracer_timestamp")?)
        .map(|(((((((a, b), c), d), e), f), g), h)| (a, b, c, d, e, f, g, h))
        .collect::<Vec<_>>();
    let mut subject_name_vec: Vec<String> = Vec::new();
    let mut object_name_vec: Vec<String> = Vec::new();
    let mut message_type_vec: Vec<String> = Vec::new();
    let mut activation_type_vec: Vec<String> = Vec::new();
    let mut message_content_vec: Vec<String> = Vec::new();
    let mut note_content_vec: Vec<String> = Vec::new();
    let mut note_location_vec: Vec<String> = Vec::new();
    let mut timestamp_messages_vec = Vec::new();

    #[allow(clippy::type_complexity)]
    let mut entered: Option<(String, String, String, i64, i64, String, String, i64)> = None;
    #[allow(clippy::type_complexity)]
    let mut exited: Option<(String, String, String, i64, i64, String, String, i64)> = None;
    for (
        tracer_event,
        span_name,
        parent_name,
        span_id,
        parent_id,
        subject_name,
        message_name,
        timestamp,
    ) in combined.into_iter()
    {
        // Check for either an entered or exited state
        if tracer_event == "entered" {
            // Always record the trace
            if let Some(previous) = entered.take() {
                // User -> network
                let (s_name, o_name) = if parent_name.is_empty() {
                    let subject_name = "User".to_string();
                    let object_name = span_name.clone();
                    (subject_name, object_name)
                // _ -> enter (task or processor)
                } else {
                    let subject_name = parent_name.clone();
                    let object_name = span_name.clone();
                    (subject_name, object_name)
                };

                // Record the trace
                subject_name_vec.push(s_name);
                object_name_vec.push(o_name);
                message_type_vec.push("->>".to_string());
                activation_type_vec.push(String::new());
                message_content_vec.push(subject_name.clone());
                // message_content_vec.push(format!("subject: {subject_name}<br>name: {message_name}"));
                note_content_vec.push(String::new());
                note_location_vec.push(String::new());
                timestamp_messages_vec.push(timestamp.to_owned());

                // Update entered
                entered.replace((
                    tracer_event,
                    span_name,
                    parent_name,
                    span_id,
                    parent_id,
                    subject_name,
                    message_name,
                    timestamp,
                ));
            } else if let Some(previous) = exited.take() {
                // Split into two traces for the exit and for the enter
                // exit -> _ (task or processor) BUT don't double count an exit to the User
                if !previous.2.is_empty() {
                    subject_name_vec.push(previous.1.clone());
                    object_name_vec.push(previous.2.clone());
                    message_type_vec.push("->>".to_string());
                    activation_type_vec.push(String::new());
                    message_content_vec.push(subject_name.clone());
                    // message_content_vec.push(format!("subject: {subject_name}<br>name: {message_name}"));
                    note_content_vec.push(String::new());
                    note_location_vec.push(String::new());
                    timestamp_messages_vec.push(previous.7.to_owned());
                }

                // User -> network
                let (s_name, o_name) = if parent_name.is_empty() {
                    let subject_name = "User".to_string();
                    let object_name = span_name.clone();
                    (subject_name, object_name)
                // _ -> enter (task or processor)
                } else {
                    let subject_name = parent_name.clone();
                    let object_name = span_name.clone();
                    (subject_name, object_name)
                };

                // Record the trace
                subject_name_vec.push(s_name);
                object_name_vec.push(o_name);
                message_type_vec.push("->>".to_string());
                activation_type_vec.push(String::new());
                message_content_vec.push(subject_name.clone());
                // message_content_vec.push(format!("subject: {subject_name}<br>name: {message_name}"));
                note_content_vec.push(String::new());
                note_location_vec.push(String::new());
                timestamp_messages_vec.push(timestamp.to_owned());

                // Update entered
                entered.replace((
                    tracer_event,
                    span_name,
                    parent_name,
                    span_id,
                    parent_id,
                    subject_name,
                    message_name,
                    timestamp,
                ));
            } else {
                // Update entered
                entered.replace((
                    tracer_event,
                    span_name,
                    parent_name,
                    span_id,
                    parent_id,
                    subject_name,
                    message_name,
                    timestamp,
                ));
            }
        } else if tracer_event == "exited" {
            if let Some(previous) = entered.take() {
                // Network -> User
                if parent_name.is_empty() {
                    subject_name_vec.push(span_name.clone());
                    object_name_vec.push("User".to_string());
                    message_type_vec.push("->>".to_string());
                    activation_type_vec.push(String::new());
                    message_content_vec.push(subject_name.clone());
                    // message_content_vec.push(format!("subject: {subject_name}<br>name: {message_name}"));
                    note_content_vec.push(String::new());
                    note_location_vec.push(String::new());
                    timestamp_messages_vec.push(timestamp.to_owned());
                }

                // Update exited
                exited.replace((
                    tracer_event,
                    span_name,
                    parent_name,
                    span_id,
                    parent_id,
                    subject_name,
                    message_name,
                    timestamp,
                ));
            } else if let Some(previous) = exited.take() {
                // Network -> User
                if parent_name.is_empty() {
                    // Record the current exit
                    subject_name_vec.push(span_name.clone());
                    object_name_vec.push("User".to_string());
                    message_type_vec.push("->>".to_string());
                    activation_type_vec.push(String::new());
                    message_content_vec.push(subject_name.clone());
                    // message_content_vec.push(format!("subject: {subject_name}<br>name: {message_name}"));
                    note_content_vec.push(String::new());
                    note_location_vec.push(String::new());
                    timestamp_messages_vec.push(timestamp.to_owned());
                }

                // Record the previous exit
                if !previous.2.is_empty() {
                    subject_name_vec.push(previous.1);
                    object_name_vec.push(previous.2);
                    message_type_vec.push("->>".to_string());
                    activation_type_vec.push(String::new());
                    message_content_vec.push(previous.5);
                    // message_content_vec.push(format!("subject: {subject_name}<br>name: {message_name}"));
                    note_content_vec.push(String::new());
                    note_location_vec.push(String::new());
                    timestamp_messages_vec.push(previous.7);
                }

                // Update exited
                exited.replace((
                    tracer_event,
                    span_name,
                    parent_name,
                    span_id,
                    parent_id,
                    subject_name,
                    message_name,
                    timestamp,
                ));
            } else {
                // Update exited
                exited.replace((
                    tracer_event,
                    span_name,
                    parent_name,
                    span_id,
                    parent_id,
                    subject_name,
                    message_name,
                    timestamp,
                ));
            }
        } else {
            return Err(anyhow!(
                "Unexpected enter/exit trace found {tracer_event}, {span_name}, {parent_name}, {span_id}, {parent_id}, {subject_name}, {message_name}, {timestamp}"
            ));
        }
    }

    // Re-sort by timestamp
    let subject_name: ArrayRef = Arc::new(StringArray::from(subject_name_vec));
    let object_name: ArrayRef = Arc::new(StringArray::from(object_name_vec));
    let message_type: ArrayRef = Arc::new(StringArray::from(message_type_vec));
    let activation_type: ArrayRef = Arc::new(StringArray::from(activation_type_vec));
    let message_content: ArrayRef = Arc::new(StringArray::from(message_content_vec));
    let note_content: ArrayRef = Arc::new(StringArray::from(note_content_vec));
    let note_location: ArrayRef = Arc::new(StringArray::from(note_location_vec));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp_messages_vec));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_name),
        ("object_name", object_name),
        ("message_type", message_type),
        ("activation_type", activation_type),
        ("message_content", message_content),
        ("note_content", note_content),
        ("note_location", note_location),
        ("timestamp", timestamp),
    ])?;
    let mut batch = sort("timestamp", &[batch], true, device)?;
    let _ = batch.remove_column(batch.num_columns() - 1);
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::device;
    use arrow::array::{ArrayRef, StringArray};

    use super::*;

    #[test]
    fn test_from_traces_to_messages() -> Result<()> {
        // Make the test traces which are based on chat network with two interactions
        let tracer_type = [
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages", "Messages", "Messages",
            "Messages", "Messages", "Messages", "Messages", "Messages",
        ];
        let tracer_event = [
            "entered", "entered", "entered", "exited", "entered", "entered", "exited", "entered",
            "entered", "exited", "entered", "entered", "exited", "exited", "entered", "exited",
            "entered", "exited", "entered", "entered", "entered", "exited", "entered", "entered",
            "entered", "exited", "entered", "entered", "entered", "entered", "exited", "entered",
            "entered", "entered", "exited", "entered", "entered", "exited", "entered", "entered",
            "exited", "exited", "entered", "exited", "entered", "exited", "entered", "entered",
            "entered", "exited", "entered", "entered", "entered", "exited",
        ];
        let message_name = [
            "from_contactbiom8ercomChat_on_UserMessages",
            "UserMessages_336755146826193674558598646911320250777",
            "message_aggregator_1_240209703317829971128596458254746874195",
            "from_message_aggregator_task_1_on_chat_task_1",
            "UserMessages_336755146826193674558598646911320250777",
            "message_aggregator_1_240209703317829971128596458254746874195",
            "from_message_aggregator_1_on_chat_task_1",
            "chat_task_1_135480752921071220051689844034448398990",
            "chat_processor_1_272383947361372855682470157595452953961",
            "from_chat_task_1_on_AssistantMessages",
            "chat_task_1_135480752921071220051689844034448398990",
            "chat_processor_1_272383947361372855682470157595452953961",
            "from_chat_processor_1_on_AssistantMessages",
            "from_contactbiom8ercomChat_on_AssistantMessages",
            "AssistantMessages_152245860784190681801447320465701877133",
            "from_contactbiom8ercomChat_on_AssistantMessages",
            "AssistantMessages_152245860784190681801447320465701877133",
            "AssistantMessages_152245860784190681801447320465701877133",
            "UserMessages_186135087994148068496551680515614784075",
            "AssistantMessages_241124620828277571948638243843444336216",
            "message_aggregator_2_305770844663281506055820666779255212610",
            "from_message_aggregator_task_2_on_AggregatedMessages",
            "UserMessages_186135087994148068496551680515614784075",
            "AssistantMessages_241124620828277571948638243843444336216",
            "message_aggregator_2_305770844663281506055820666779255212610",
            "from_message_aggregator_2_on_AggregatedMessages",
            "from_contactbiom8ercomChat_on_UserMessages",
            "AssistantMessages_157336070741302015672910225198795432608",
            "message_aggregator_1_49904971912152367753684983521976556855",
            "UserMessages_239034416761099333322888321976723638008",
            "from_message_aggregator_task_1_on_chat_task_1",
            "AssistantMessages_157336070741302015672910225198795432608",
            "message_aggregator_1_49904971912152367753684983521976556855",
            "UserMessages_239034416761099333322888321976723638008",
            "from_message_aggregator_1_on_chat_task_1",
            "chat_task_1_337200085168065351548915970648389364830",
            "chat_processor_1_260744394084496639907163803786907002860",
            "from_chat_task_1_on_AssistantMessages",
            "chat_task_1_337200085168065351548915970648389364830",
            "chat_processor_1_260744394084496639907163803786907002860",
            "from_chat_processor_1_on_AssistantMessages",
            "from_contactbiom8ercomChat_on_AssistantMessages",
            "AssistantMessages_295546099496625232658165418514664934409",
            "from_contactbiom8ercomChat_on_AssistantMessages",
            "AssistantMessages_295546099496625232658165418514664934409",
            "AssistantMessages_295546099496625232658165418514664934409",
            "AssistantMessages_301756319473837870842740763107819233568",
            "message_aggregator_2_198642670475012020917323289448933916779",
            "UserMessages_17089268937073073383526759130015932611",
            "from_message_aggregator_task_2_on_AggregatedMessages",
            "AssistantMessages_301756319473837870842740763107819233568",
            "message_aggregator_2_198642670475012020917323289448933916779",
            "UserMessages_17089268937073073383526759130015932611",
            "from_message_aggregator_2_on_AggregatedMessages",
        ];
        let subject_name = [
            "UserMessages",
            "UserMessages",
            "message_aggregator_1",
            "chat_task_1",
            "UserMessages",
            "message_aggregator_1",
            "chat_task_1",
            "chat_task_1",
            "chat_processor_1",
            "AssistantMessages",
            "chat_task_1",
            "chat_processor_1",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "UserMessages",
            "AssistantMessages",
            "message_aggregator_2",
            "AggregatedMessages",
            "UserMessages",
            "AssistantMessages",
            "message_aggregator_2",
            "AggregatedMessages",
            "UserMessages",
            "AssistantMessages",
            "message_aggregator_1",
            "UserMessages",
            "chat_task_1",
            "AssistantMessages",
            "message_aggregator_1",
            "UserMessages",
            "chat_task_1",
            "chat_task_1",
            "chat_processor_1",
            "AssistantMessages",
            "chat_task_1",
            "chat_processor_1",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "AssistantMessages",
            "message_aggregator_2",
            "UserMessages",
            "AggregatedMessages",
            "AssistantMessages",
            "message_aggregator_2",
            "UserMessages",
            "AggregatedMessages",
        ];
        let span_name = [
            "contactbiom8ercomChat",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "chat_task_1",
            "chat_task_1",
            "chat_task_1",
            "chat_processor_1",
            "chat_processor_1",
            "chat_processor_1",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "contactbiom8ercomChat",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "chat_task_1",
            "chat_task_1",
            "chat_task_1",
            "chat_processor_1",
            "chat_processor_1",
            "chat_processor_1",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
        ];
        let parent_name = [
            "",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "chat_task_1",
            "chat_task_1",
            "chat_task_1",
            "",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "chat_task_1",
            "chat_task_1",
            "chat_task_1",
            "",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
        ];
        let file = [
            "phymes-subject/src/network/network_stream_step.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-ml/src/candle_chat/chat_processor.rs",
            "phymes-ml/src/candle_chat/chat_processor.rs",
            "phymes-ml/src/candle_chat/chat_processor.rs",
            "phymes-subject/src/network/network_stream_step.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/processor.rs",
            "phymes-subject/src/task/processor.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-subject/src/network/network_stream_step.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-ml/src/candle_chat/chat_processor.rs",
            "phymes-ml/src/candle_chat/chat_processor.rs",
            "phymes-ml/src/candle_chat/chat_processor.rs",
            "phymes-subject/src/network/network_stream_step.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/processor.rs",
            "phymes-subject/src/task/processor.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-subject/src/task/task_trait.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
            "phymes-ml/src/candle_chat/message_aggregator_processor.rs",
        ];
        let thread = [
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
            "ThreadId(23)",
        ];
        let function = [
            "contactbiom8ercomChat",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "chat_task_1",
            "chat_task_1",
            "chat_task_1",
            "chat_processor_1",
            "chat_processor_1",
            "chat_processor_1",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "contactbiom8ercomChat",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "message_aggregator_1",
            "chat_task_1",
            "chat_task_1",
            "chat_task_1",
            "chat_processor_1",
            "chat_processor_1",
            "chat_processor_1",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "contactbiom8ercomChat",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_task_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
            "message_aggregator_2",
        ];
        let line = [
            169, 158, 158, 158, 91, 91, 91, 158, 158, 158, 107, 107, 107, 169, 158, 158, 269, 269,
            158, 158, 158, 158, 91, 91, 91, 91, 169, 158, 158, 158, 158, 91, 91, 91, 91, 158, 158,
            158, 107, 107, 107, 169, 158, 158, 269, 269, 158, 158, 158, 158, 91, 91, 91, 91,
        ];
        let span_id: [i64; 54] = [
            2423125168095290289,
            -6290470452849520000,
            -6290470452849520000,
            -6290470452849520000,
            725936955998133113,
            725936955998133113,
            725936955998133113,
            -2976063072544730000,
            -2976063072544730000,
            -2976063072544730000,
            4672773661716737199,
            4672773661716737199,
            4672773661716737199,
            1627393037850540817,
            8205752543250100092,
            8205752543250100092,
            -1736242488042860000,
            -1736242488042860000,
            4056747569687788544,
            4056747569687788544,
            4056747569687788544,
            4056747569687788544,
            -729888377209831000,
            -729888377209831000,
            -729888377209831000,
            -729888377209831000,
            3217246414627855450,
            -3086786702679540000,
            -3086786702679540000,
            -3086786702679540000,
            -3086786702679540000,
            -1411873010140140000,
            -1411873010140140000,
            -1411873010140140000,
            -1411873010140140000,
            7833809350697111777,
            7833809350697111777,
            7833809350697111777,
            1038913386499437312,
            1038913386499437312,
            1038913386499437312,
            -3041123569125720000,
            8191798950735311332,
            8191798950735311332,
            1055250054529791544,
            1055250054529791544,
            1381392355749085769,
            1381392355749085769,
            1381392355749085769,
            1381392355749085769,
            6758425213845207512,
            6758425213845207512,
            6758425213845207512,
            6758425213845207512,
        ];
        let parent_id: [i64; 54] = [
            0,
            2423125168095290289,
            2423125168095290289,
            2423125168095290289,
            -6290470452849520000,
            -6290470452849520000,
            -6290470452849520000,
            5808914344790812863,
            5808914344790812863,
            5808914344790812863,
            -2976063072544730000,
            -2976063072544730000,
            -2976063072544730000,
            0,
            1627393037850540817,
            1627393037850540817,
            8205752543250100092,
            8205752543250100092,
            1627393037850540817,
            1627393037850540817,
            1627393037850540817,
            1627393037850540817,
            4056747569687788544,
            4056747569687788544,
            4056747569687788544,
            4056747569687788544,
            0,
            3217246414627855450,
            3217246414627855450,
            3217246414627855450,
            3217246414627855450,
            -3086786702679540000,
            -3086786702679540000,
            -3086786702679540000,
            -3086786702679540000,
            1119528872034831779,
            1119528872034831779,
            1119528872034831779,
            7833809350697111777,
            7833809350697111777,
            7833809350697111777,
            0,
            -3041123569125720000,
            -3041123569125720000,
            8191798950735311332,
            8191798950735311332,
            -3041123569125720000,
            -3041123569125720000,
            -3041123569125720000,
            -3041123569125720000,
            1381392355749085769,
            1381392355749085769,
            1381392355749085769,
            1381392355749085769,
        ];
        let tracer_timestamp = [
            1765368584283582,
            1765368584283841,
            1765368584283842,
            1765368584283918,
            1765368584283858,
            1765368584283859,
            1765368584283907,
            1765368584294174,
            1765368584294175,
            1765368584294710,
            1765368584294664,
            1765368584294664,
            1765368584294698,
            1765368588878155,
            1765368588876990,
            1765368588877002,
            1765368588876994,
            1765368588876995,
            1765368588877030,
            1765368588877030,
            1765368588877030,
            1765368588877072,
            1765368588877036,
            1765368588877037,
            1765368588877037,
            1765368588877066,
            1765368822456781,
            1765368822457057,
            1765368822457058,
            1765368822457058,
            1765368822457136,
            1765368822457073,
            1765368822457073,
            1765368822457074,
            1765368822457122,
            1765368822459006,
            1765368822459006,
            1765368822459036,
            1765368822459012,
            1765368822459012,
            1765368822459030,
            1765368823421759,
            1765368823420489,
            1765368823420507,
            1765368823420496,
            1765368823420498,
            1765368823420545,
            1765368823420545,
            1765368823420546,
            1765368823420598,
            1765368823420554,
            1765368823420554,
            1765368823420554,
            1765368823420590,
        ];
        let timestamp: [i64; 54] = [
            1765368584283566,
            1765368584283833,
            1765368584283833,
            1765368584283833,
            1765368584283856,
            1765368584283856,
            1765368584283856,
            1765368584294171,
            1765368584294171,
            1765368584294171,
            1765368584294657,
            1765368584294657,
            1765368584294657,
            1765368588876946,
            1765368588876986,
            1765368588876986,
            1765368588876993,
            1765368588876993,
            1765368588877028,
            1765368588877028,
            1765368588877028,
            1765368588877028,
            1765368588877035,
            1765368588877035,
            1765368588877035,
            1765368588877035,
            1765368822456764,
            1765368822457049,
            1765368822457049,
            1765368822457049,
            1765368822457049,
            1765368822457068,
            1765368822457068,
            1765368822457068,
            1765368822457068,
            1765368822459003,
            1765368822459003,
            1765368822459003,
            1765368822459011,
            1765368822459011,
            1765368822459011,
            1765368823420432,
            1765368823420484,
            1765368823420484,
            1765368823420493,
            1765368823420493,
            1765368823420542,
            1765368823420542,
            1765368823420542,
            1765368823420542,
            1765368823420551,
            1765368823420551,
            1765368823420551,
            1765368823420551,
        ];

        let tracer_type: ArrayRef = Arc::new(StringArray::from(
            tracer_type
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ));
        let tracer_event: ArrayRef = Arc::new(StringArray::from(
            tracer_event
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ));
        let message_name: ArrayRef = Arc::new(StringArray::from(
            message_name
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ));
        let subject_name: ArrayRef = Arc::new(StringArray::from(
            subject_name
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ));
        let span_name: ArrayRef = Arc::new(StringArray::from(
            span_name.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ));
        let parent_name: ArrayRef = Arc::new(StringArray::from(
            parent_name
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ));
        let span_id: ArrayRef = Arc::new(Int64Array::from_iter(
            span_id.into_iter().collect::<Vec<_>>(),
        ));
        let parent_id: ArrayRef = Arc::new(Int64Array::from_iter(
            parent_id.into_iter().collect::<Vec<_>>(),
        ));
        let tracer_timestamp: ArrayRef = Arc::new(Int64Array::from_iter(tracer_timestamp));
        let lhs_batch = RecordBatch::try_from_iter(vec![
            ("tracer_type", tracer_type),
            ("tracer_event", tracer_event),
            ("message_name", message_name),
            ("subject_name", subject_name),
            ("span_name", span_name),
            ("parent_name", parent_name),
            ("span_id", span_id),
            ("parent_id", parent_id),
            ("tracer_timestamp", tracer_timestamp),
        ])?;

        let tasks = [
            "chat_task_1",
            "message_aggregator_task_1",
            "message_aggregator_task_1",
            "contactbiom8ercomChat",
        ];
        let processors = [
            "message_aggregator_1",
            "message_aggregator_2",
            "chat_processor_1",
            "contactbiom8ercomChat",
        ];
        let tasks: ArrayRef = Arc::new(StringArray::from(
            tasks.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ));
        let processors: ArrayRef = Arc::new(StringArray::from(
            processors.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ));
        let rhs_batch =
            RecordBatch::try_from_iter(vec![("task_name", tasks), ("processor_name", processors)])?;

        // Make the device
        // DM: GPU sorting is different than cPU sorting
        let device = device(true)?;

        let result = from_traces_to_messages(&[lhs_batch], &[rhs_batch], &device)?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        // let bytes = result_table.to_csv(b',', true)?;
        // let string = String::from_utf8(bytes)?;
        // dbg!(&string);
        let results = result_table.get_column_as_vec_str("subject_name");
        assert_eq!(
            results,
            [
                "User",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "message_aggregator_task_1",
                "message_aggregator_task_1",
                "message_aggregator_1",
                "message_aggregator_task_1",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "chat_task_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_task_1",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_2",
                "message_aggregator_task_2",
                "contactbiom8ercomChat",
                "User",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "message_aggregator_task_1",
                "message_aggregator_task_1",
                "message_aggregator_task_1",
                "message_aggregator_1",
                "message_aggregator_task_1",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "chat_task_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_task_1",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_2",
                "message_aggregator_task_2",
                "contactbiom8ercomChat"
            ]
        );
        let results = result_table.get_column_as_vec_str("object_name");
        assert_eq!(
            results,
            [
                "contactbiom8ercomChat",
                "message_aggregator_task_1",
                "message_aggregator_task_1",
                "message_aggregator_1",
                "message_aggregator_1",
                "message_aggregator_task_1",
                "contactbiom8ercomChat",
                "chat_task_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_processor_1",
                "chat_task_1",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_2",
                "message_aggregator_2",
                "message_aggregator_2",
                "message_aggregator_task_2",
                "contactbiom8ercomChat",
                "User",
                "contactbiom8ercomChat",
                "message_aggregator_task_1",
                "message_aggregator_task_1",
                "message_aggregator_task_1",
                "message_aggregator_1",
                "message_aggregator_1",
                "message_aggregator_1",
                "message_aggregator_task_1",
                "contactbiom8ercomChat",
                "chat_task_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_processor_1",
                "chat_task_1",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "contactbiom8ercomChat",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_task_2",
                "message_aggregator_2",
                "message_aggregator_2",
                "message_aggregator_2",
                "message_aggregator_task_2",
                "contactbiom8ercomChat",
                "User"
            ]
        );
        let results = result_table.get_column_as_vec_str("message_type");
        assert_eq!(
            results,
            [
                "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>",
                "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>",
                "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>",
                "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>", "->>",
                "->>", "->>", "->>", "->>", "->>", "->>"
            ]
        );
        let results = result_table.get_column_as_vec_str("activation_type");
        assert_eq!(
            results,
            [
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", ""
            ]
        );
        let results = result_table.get_column_as_vec_str("message_content");
        assert_eq!(
            results,
            [
                "UserMessages",
                "UserMessages",
                "message_aggregator_1",
                "UserMessages",
                "message_aggregator_1",
                "chat_task_1",
                "chat_task_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_processor_1",
                "chat_task_1",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "UserMessages",
                "message_aggregator_2",
                "UserMessages",
                "AssistantMessages",
                "message_aggregator_2",
                "AggregatedMessages",
                "AggregatedMessages",
                "AssistantMessages",
                "UserMessages",
                "AssistantMessages",
                "UserMessages",
                "message_aggregator_1",
                "AssistantMessages",
                "message_aggregator_1",
                "UserMessages",
                "chat_task_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_task_1",
                "chat_processor_1",
                "chat_task_1",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "AssistantMessages",
                "message_aggregator_2",
                "UserMessages",
                "AssistantMessages",
                "UserMessages",
                "message_aggregator_2",
                "AssistantMessages",
                "AggregatedMessages",
                "AssistantMessages"
            ]
        );
        let results = result_table.get_column_as_vec_str("note_content");
        assert_eq!(
            results,
            [
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", ""
            ]
        );
        let results = result_table.get_column_as_vec_str("note_location");
        assert_eq!(
            results,
            [
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", ""
            ]
        );

        Ok(())
    }
}
