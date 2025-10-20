use std::{collections::HashMap, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray}, datatypes::Schema};
use candle_core::Device;
use phymes_core::{
    schemas::{chat_completion, types},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::table_trait::{Table, TableBuilderTrait, TableTrait},
};
use phymes_diagnostics::HashSet;

use crate::{
    candle_data::data_config::DataConfig,
    candle_operators::{data_operator::DataOperatorTrait, sort_column_and_indices::sort_column_and_indices},
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
/// * LHS is Traces
/// * RHS is SessionTasks
/// * Output schema is MermaidSequenceDiagramMessagesTemplate
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `rhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
pub fn from_traces_to_messages(lhs_args: &[RecordBatch], rhs_args: &[RecordBatch], device: &Device,
) -> Result<RecordBatch> {

    // Get the unique tasks and processors
    let rhs_table = Table::get_builder()
        .with_record_batches(rhs_args.to_vec())?
        .with_name("")
        .build()?;
    let task_name_set = rhs_table.get_column_as_vec_nonprimitive::<String>("task_name")?.into_iter().collect::<HashSet<_>>();
    let processor_name_set = rhs_table.get_column_as_vec_nonprimitive::<String>("processor_name")?.into_iter().collect::<HashSet<_>>();

    // Presort the traces by columns
    let lhs_values = ["tracer_event", "parent_name", "span_name", "timestamp", "subject_name"];
    let mut lhs_sorted = RecordBatch::new_empty(Arc::new(Schema::empty()));
    for (iter, column_name) in lhs_values.iter().enumerate() {
        if iter > 0 {
            lhs_sorted = sort_column_and_indices(column_name, &[lhs_sorted], true, device)?;
        } else {
            lhs_sorted = sort_column_and_indices(column_name, lhs_args, true, device)?;
        }
    }
    let lhs_table = Table::get_builder()
        .with_record_batches(vec![lhs_sorted])?
        .with_name("")
        .build()?;

    // Convert to messages vec
    let combined = lhs_table.get_column_as_vec_nonprimitive::<String>("tracer_event")?.into_iter()
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("span_name")?.into_iter())
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("parent_name")?.into_iter())
        .zip(lhs_table.get_column_as_vec_primitive::<i64>("span_id")?.into_iter())
        .zip(lhs_table.get_column_as_vec_primitive::<i64>("parent_id")?.into_iter())
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("subject_name")?.into_iter())
        .zip(lhs_table.get_column_as_vec_nonprimitive::<String>("message_name")?.into_iter())
        .zip(lhs_table.get_column_as_vec_primitive::<i64>("timestamp")?.into_iter())
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

    // DM: TODO: add in dimensions of tracer run depth

    // Track the beginning and end of a message triple
    // let mut subject = &combined.first().unwrap().5;
    // let mut entered = None;
    // let mut exited = None;
    for (tracer_event, span_name, parent_name, span_id, parent_id, subject_name, message_name, timestamp) in combined.iter() {
        subject_name_vec.push("State".to_string());
        object_name_vec.push(span_name.to_string());
        message_type_vec.push("->>".to_string());
        activation_type_vec.push(String::new());
        message_content_vec.push(format!("subject: {}", subject_name));
        note_content_vec.push(String::new());
        note_location_vec.push(String::new());
        timestamp_messages_vec.push(timestamp.to_owned());
        
        // if subject_name != subject && tracer_event == "entered" && parent_name.is_empty() {
        //     // From user to state: enter() only with no parent
        //     subject_name_vec.push("User".to_string());
        //     object_name_vec.push(span_name.to_string());
        //     message_type_vec.push("->>".to_string());
        //     activation_type_vec.push(String::new());
        //     message_content_vec.push(format!("subject: {}", subject_name));
        //     note_content_vec.push(String::new());
        //     note_location_vec.push(String::new());
        //     timestamp_messages_vec.push(timestamp.to_owned());
        //     subject = subject_name;
        //     entered.replace((tracer_event, span_name, parent_name, span_id, parent_id, subject_name, message_name, timestamp));
        // } else if subject_name != subject && tracer_event == "entered" && !parent_name.is_empty() {
        //     // From state to task: enter() only
        //     subject_name_vec.push("State".to_string());
        //     object_name_vec.push(span_name.to_string());
        //     message_type_vec.push("->>".to_string());
        //     activation_type_vec.push(String::new());
        //     message_content_vec.push(format!("subject: {}", subject_name));
        //     note_content_vec.push(String::new());
        //     note_location_vec.push(String::new());
        //     timestamp_messages_vec.push(timestamp.to_owned());
        //     subject = subject_name;
        //     entered.replace((tracer_event, span_name, parent_name, span_id, parent_id, subject_name, message_name, timestamp));
        // } else if subject_name != subject && tracer_event == "exited" && !parent_name.is_empty() {
        //     // From task to state: exit() only
        //     subject_name_vec.push(span_name.to_string());
        //     object_name_vec.push("State".to_string());
        //     message_type_vec.push("->>".to_string());
        //     activation_type_vec.push(String::new());
        //     message_content_vec.push(format!("subject: {}", subject_name));
        //     note_content_vec.push(String::new());
        //     note_location_vec.push(String::new());
        //     timestamp_messages_vec.push(timestamp.to_owned());
        //     subject = subject_name;
        //     exited.replace((tracer_event, span_name, parent_name, span_id, parent_id, subject_name, message_name, timestamp));
        // } else if subject_name == subject_name && tracer_event == "entered" && parent_name == entered.unwrap().1 {
        //     // Parent to child: enter() -> enter() where parent name of child matches parent
        //     subject = subject_name;

        // } else if subject_name == subject_name && tracer_event == "exited" && span_name == exited.unwrap().2 {
        //     // Child to parent: exit() -> exit() where parent name of child matches parent
        //     subject = subject_name;

        // } else if subject_name == subject_name && tracer_event == "entered" && parent_name == exited.unwrap().2 {
        //     // Span to span at the same hierarchy: exit() -> enter() where parent_name is the same
        //     subject = subject_name;

        // } else {
        //     return Err(anyhow!("Unexpected enter/exit trace found {}, {}, {}, {}, {}, {}, {}, {}", tracer_event, span_name, parent_name, span_id, parent_id, subject_name, message_name, timestamp));
        // }
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
    let mut batch = sort_column_and_indices("timestamp", &[batch], true, device)?;
    let _ = batch.remove_column(batch.num_columns() - 1);
    Ok(batch)
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
        let tracer_type = ["Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages","Messages"];
        let tracer_event = ["exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","entered","exited","exited","entered","exited","entered","exited","exited","entered","exited","entered","exited","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered","exited","entered"];
        let message_name = ["state_1_145400443860007532692884143026107463224","state_1_145400443860007532692884143026107463224","state_1_186565055984087162001738051867427998330","state_1_186565055984087162001738051867427998330","state_1_261489347384847644355362190768032642075","state_1_261489347384847644355362190768032642075","state_1_228173522346316916885579131611404317204","state_1_228173522346316916885579131611404317204","state_1_243773822264980459180624445440384838248","state_1_243773822264980459180624445440384838248","state_1_37631294003838125574527916569345721805","state_1_37631294003838125574527916569345721805",
            "state_1_26064730536607422073993808491441966412","state_1_26064730536607422073993808491441966412","state_1_80362914748176319296854015531470579852","state_1_80362914748176319296854015531470579852","task_1","from_session_1_on_state_1","from_session_1_on_state_1","state_1_41487522459421544509868281222405207298","state_1_41487522459421544509868281222405207298","state_1_41487522459421544509868281222405207298","from_session_1_on_state_1","from_session_1_on_state_1","state_1_244362090279249268952833896150924961301","state_1_244362090279249268952833896150924961301","state_1_244362090279249268952833896150924961301","from_session_1_on_state_1","from_session_1_on_state_1",
            "state_1_238106427844863423206506895364015149421","state_1_238106427844863423206506895364015149421","state_1_238106427844863423206506895364015149421","from_task_1_on_state_1","state_1_145400443860007532692884143026107463224","from_task_1_on_state_1","state_1_186565055984087162001738051867427998330","from_task_1_on_state_1","state_1_261489347384847644355362190768032642075","from_task_1_on_state_1","state_1_228173522346316916885579131611404317204","from_task_2_on_state_1","state_1_243773822264980459180624445440384838248","from_task_2_on_state_1","state_1_37631294003838125574527916569345721805","from_task_2_on_state_1","state_1_26064730536607422073993808491441966412","from_task_2_on_state_1","state_1_80362914748176319296854015531470579852"];
        let subject_name = ["state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1","state_1"];
        let span_name = ["processor_1","processor_1","processor_1","processor_1","processor_1","processor_1","processor_1","processor_1","processor_2","processor_2","processor_2","processor_2","processor_2","processor_2","processor_2","processor_2","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","task_1","task_1","task_1","task_1","task_1","task_1","task_1","task_1","task_2","task_2","task_2","task_2","task_2","task_2","task_2","task_2"];
        let parent_name = ["task_1","task_1","task_1","task_1","task_1","task_1","task_1","task_1","task_2","task_2","task_2","task_2","task_2","task_2","task_2","task_2","","","session_1","session_1","session_1","session_1","","session_1","session_1","session_1","session_1","","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1","session_1"];
        let span_id: [i128; 48] = [16184082946072572162,16184082946072572162,2186547869047037942,2186547869047037942,3256812710579872580,3256812710579872580,14510244696094898731,14510244696094898731,14456963133862203229,14456963133862203229,1712640329960440735,1712640329960440735,13165260631272995056,13165260631272995056,16797564894134450851,16797564894134450851,13469193432104994765,7391730627494635845,14870679813591216503,14870679813591216503,10817813420396259474,10817813420396259474,188058941559050804,5644835324380142827,5644835324380142827,14010568442619868195,14010568442619868195,15574280659243175955,1298132286589933723,1298132286589933723,17936549320446211319,17936549320446211319,14509531658266103530,14509531658266103530,2257952061453635836,2257952061453635836,7283200596918334309,7283200596918334309,7053620394350563160,7053620394350563160,14042324002796455054,14042324002796455054,1187106305792110079,1187106305792110079,1038112178665446702,1038112178665446702,558667721896770324,558667721896770324];
        let parent_id: [i128; 48] = [14509531658266103530,14509531658266103530,2257952061453635836,2257952061453635836,7283200596918334309,7283200596918334309,7053620394350563160,7053620394350563160,14042324002796455054,14042324002796455054,1187106305792110079,1187106305792110079,1038112178665446702,1038112178665446702,558667721896770324,558667721896770324,0,0,7391730627494635845,7391730627494635845,14870679813591216503,14870679813591216503,0,188058941559050804,188058941559050804,5644835324380142827,5644835324380142827,0,15574280659243175955,15574280659243175955,1298132286589933723,1298132286589933723,13469193432104994765,13469193432104994765,7391730627494635845,7391730627494635845,188058941559050804,188058941559050804,15574280659243175955,15574280659243175955,13469193432104994765,13469193432104994765,7391730627494635845,7391730627494635845,188058941559050804,188058941559050804,15574280659243175955,15574280659243175955];
        let timestamp = [1760807167220159,1760807167220159,1760807167226960,1760807167226960,1760807167236238,1760807167236238,1760807167268072,1760807167268072,1760807167220113,1760807167220113,1760807167226933,1760807167226933,1760807167236201,1760807167236201,1760807167268017,1760807167268017,1760807167217971,1760807167226865,1760807167226896,1760807167226896,1760807167226901,1760807167226901,1760807167236091,1760807167236122,1760807167236122,1760807167236128,1760807167236128,1760807167267856,1760807167267892,1760807167267892,1760807167267898,1760807167267898,1760807167220155,1760807167220155,1760807167226957,1760807167226957,1760807167236234,1760807167236234,1760807167268068,1760807167268068,1760807167220101,1760807167220101,1760807167226929,1760807167226929,1760807167236195,1760807167236195,1760807167268013,1760807167268013];

        let tracer_type: ArrayRef = Arc::new(StringArray::from(tracer_type.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let tracer_event: ArrayRef = Arc::new(StringArray::from(tracer_event.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let message_name: ArrayRef = Arc::new(StringArray::from(message_name.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let subject_name: ArrayRef = Arc::new(StringArray::from(subject_name.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let span_name: ArrayRef = Arc::new(StringArray::from(span_name.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let parent_name: ArrayRef = Arc::new(StringArray::from(parent_name.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let span_id: ArrayRef = Arc::new(Int64Array::from_iter(span_id.into_iter().map(|s| s as i64).collect::<Vec<_>>()));
        let parent_id: ArrayRef = Arc::new(Int64Array::from_iter(parent_id.into_iter().map(|s| s as i64).collect::<Vec<_>>()));
        let timestamp: ArrayRef = Arc::new(Int64Array::from_iter(timestamp));
        let lhs_batch = RecordBatch::try_from_iter(vec![
            ("tracer_type", tracer_type),
            ("tracer_event", tracer_event),
            ("message_name", message_name),
            ("subject_name", subject_name),
            ("span_name", span_name),
            ("parent_name", parent_name),
            ("span_id", span_id),
            ("parent_id", parent_id),
            ("timestamp", timestamp),
        ])?;

        let tasks = ["task_1", "task_2"];
        let processors = ["processor_1", "processor_2"];
        let tasks: ArrayRef = Arc::new(StringArray::from(tasks.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let processors: ArrayRef = Arc::new(StringArray::from(processors.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
        let rhs_batch = RecordBatch::try_from_iter(vec![
            ("task_name", tasks),
            ("processor_name", processors),
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
        
        let bytes = result_table.to_csv(b',', true)?;
        let string = String::from_utf8(bytes)?;
        dbg!(&string);
        let participants = result_table.get_column_as_vec_str("participant_name");
        assert_eq!(participants, ["User", "State", "t1", "p1", "p2", "t2", "p3", "p4", "t3"]);
        let participants = result_table.get_column_as_vec_str("participant_type");
        assert_eq!(participants, ["actor", "database", "collections", "participant", "participant", "collections", "participant", "participant", "collections"]);

        Ok(())
    }
}
