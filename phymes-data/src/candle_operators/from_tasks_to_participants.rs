use anyhow::Result;
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, Table, TableBuilderTrait, TableTrait,
    create_mermaid_sequence_diagram_participants_template_batch,
};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

use crate::{candle_data::DataConfig, candle_operators::DataOperatorTrait};

/// Compute the normalized start and end times in a [RecordBatch]
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FromTasksToParticipants {}

impl MappableTrait for FromTasksToParticipants {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for FromTasksToParticipants {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        from_tasks_to_participants(lhs_args, rhs_args.unwrap(), device)
    }
    fn new(_config: &DataConfig) -> Result<Self> {
        Ok(FromTasksToParticipants {})
    }
}

/// Custom function to convert `SessionTasks` to Sequence Diagram Participants
///
/// # Notes
///
/// * LHS is SessionTasks
/// * RHS is MermaidSequenceDiagramMessagesTemplate
/// * Output schema is MermaidSequenceDiagramParticipantsTemplate
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es
/// * `rhs_args` - Slice of [RecordBatch]es
/// * `device` - The compute device
pub fn from_tasks_to_participants(
    lhs_args: &[RecordBatch],
    rhs_args: &[RecordBatch],
    _device: &Device,
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

    // Initialize the participants vecs
    // let mut participant_name_vec = vec!["User", "State"];
    // let mut participant_type_vec = vec!["actor", "database"];
    let mut participant_name_vec = Vec::new();
    let mut participant_type_vec = Vec::new();

    // Get the unique tasks and processors
    let task_name_set = lhs_table
        .get_column_as_vec_nonprimitive::<String>("task_name")?
        .into_iter()
        .collect::<HashSet<_>>();
    let processor_name_set = lhs_table
        .get_column_as_vec_nonprimitive::<String>("processor_name")?
        .into_iter()
        .collect::<HashSet<_>>();

    // Get the ordering of participants from MermaidSequenceDiagramMessages
    let subject_name_vec = rhs_table.get_column_as_vec_nonprimitive::<String>("subject_name")?;
    let object_name_vec = rhs_table.get_column_as_vec_nonprimitive::<String>("object_name")?;
    let mut found_set = HashSet::new();
    found_set.insert("User");
    for (i, (subject, object)) in subject_name_vec
        .iter()
        .zip(object_name_vec.iter())
        .enumerate()
    {
        // The first entry should be User -> Session
        if i == 0 {
            participant_name_vec.push(subject);
            participant_type_vec.push("actor");
            found_set.insert(subject);
            participant_name_vec.push(object);
            participant_type_vec.push("database");
            found_set.insert(object);
            continue;
        }

        // Prioritize the tasks when the task_name == processor_name
        // DM: cases of task_name == processor_name will appear as self-loops in the sequence diagram
        if task_name_set.contains(subject) && !found_set.contains(subject.as_str()) {
            // New task from subject
            participant_name_vec.push(subject);
            participant_type_vec.push("collections");
            found_set.insert(subject);
        } else if processor_name_set.contains(subject) && !found_set.contains(subject.as_str()) {
            // New processor from subject
            participant_name_vec.push(subject);
            participant_type_vec.push("participant");
            found_set.insert(subject);
        } else if task_name_set.contains(object) && !found_set.contains(object.as_str()) {
            // New task from object
            participant_name_vec.push(object);
            participant_type_vec.push("collections");
            found_set.insert(object);
        } else if processor_name_set.contains(object) && !found_set.contains(object.as_str()) {
            // New processor from object
            participant_name_vec.push(object);
            participant_type_vec.push("participant");
            found_set.insert(object);
        }
    }

    create_mermaid_sequence_diagram_participants_template_batch(
        participant_name_vec
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        participant_type_vec
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, StringArray};
    use phymes_core::device;

    use super::*;

    #[test]
    fn test_from_tasks_to_participants() -> Result<()> {
        // Make the test record batches
        let lhs_1_vec = vec!["State", "t1", "t1", "t2", "t2", "t3"];
        let lhs_1_array: ArrayRef = Arc::new(StringArray::from(lhs_1_vec));
        let lhs_2_vec = vec!["State", "p1", "p2", "p3", "p4", "p1"];
        let lhs_2_array: ArrayRef = Arc::new(StringArray::from(lhs_2_vec));
        let lhs_batch = RecordBatch::try_from_iter(vec![
            ("task_name", lhs_1_array),
            ("processor_name", lhs_2_array),
        ])?;
        let rhs_1_vec = vec![
            "User", "s", "t1", "p1", "p2", "t1", "s", "t2", "p3", "p4", "t2", "s", "t3", "p1", "t3",
        ];
        let rhs_1_array: ArrayRef = Arc::new(StringArray::from(rhs_1_vec));
        let rhs_2_vec = vec![
            "State", "t1", "p1", "p2", "t1", "s", "t2", "p3", "p4", "t2", "s", "t3", "p1", "t3",
            "s",
        ];
        let rhs_2_array: ArrayRef = Arc::new(StringArray::from(rhs_2_vec));
        let rhs_batch = RecordBatch::try_from_iter(vec![
            ("subject_name", rhs_1_array),
            ("object_name", rhs_2_array),
        ])?;

        // Make the device
        let device = device(false)?;

        let result = from_tasks_to_participants(&[lhs_batch], &[rhs_batch], &device)?;
        let result_table = Table::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let participants = result_table.get_column_as_vec_str("participant_name");
        assert_eq!(
            participants,
            ["User", "State", "t1", "p1", "p2", "t2", "p3", "p4", "t3"]
        );
        let participants = result_table.get_column_as_vec_str("participant_type");
        assert_eq!(
            participants,
            [
                "actor",
                "database",
                "collections",
                "participant",
                "participant",
                "collections",
                "participant",
                "participant",
                "collections"
            ]
        );

        Ok(())
    }
}
