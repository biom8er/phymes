use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

/// The plan for the tasks
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct TaskPlan {
    /// The name of the task
    pub task_name: String,
    /// The name of processors that process the messages stream
    pub processor_names: Vec<String>,
}

impl TaskPlan {   

    /// Extend the current [TaskPlan] with the Processor Names from another
    pub fn extend(self, other: TaskPlan) -> Self {
        let names = self.processor_names
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<_>>();
        let other_processors = other.processor_names
            .into_iter()
            .filter(|t| !names.contains(t))
            .collect::<Vec<_>>();
        let processors = self.processor_names
            .into_iter()
            .chain(other_processors)
            .collect::<Vec<_>>();
        TaskPlan { task_name: self.task_name, processor_names: processors }
    }

    /// Break the [TaskPlan] into multiple individual [TaskPlan]s each with only a single Processor
    pub fn individualize(self) -> Vec<Self> {
        self.processor_names.into_iter()
            .enumerate()
            .map(|(i, p)| {
                let name = format!("{}_{i}", self.task_name);
                TaskPlan { task_name: name, processor_names: vec![p] }
            })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_task;

    use super::*;

    #[test]
    fn test_task_plan_individualize() {
        let (test_task, _test_procesor_subjects) = test_task::make_test_task_chained_processor(
            "test_task",
            "test_processor",
            "test_table",
        ).unwrap();
        let test_task_plan = TaskPlan { task_name: test_task.name, processor_names: test_task.processor.into_iter().map(|p| p.get_name().to_string()).collect::<Vec<_>>()};
        let individuals = test_task_plan.individualize();
        assert_eq!(individuals.first().unwrap().task_name, "test_task_0");
        assert_eq!(individuals.first().unwrap().processor_names.last().unwrap(), "test_processor_1");
        assert_eq!(individuals.get(1).unwrap().task_name, "test_task_1");
        assert_eq!(individuals.get(1).unwrap().processor_names.last().unwrap(), "test_processor_2");
        assert_eq!(individuals.get(2).unwrap().task_name, "test_task_2");
        assert_eq!(individuals.get(2).unwrap().processor_names.last().unwrap(), "test_processor_3");
    }

    #[test]
    fn test_task_plan_extend() {
        let (test_task, _test_procesor_subjects) = test_task::make_test_task_single_processor(
            "test_task",
            "test_processor_1",
            "test_table_1",
        ).unwrap();
        let test_task_plan_0 = TaskPlan { task_name: test_task.name, processor_names: test_task.processor.into_iter().map(|p| p.get_name().to_string()).collect::<Vec<_>>()};
        let (test_task, _test_procesor_subjects) = test_task::make_test_task_single_processor(
            "test_task",
            "test_processor_2",
            "test_table_2",
        ).unwrap();
        let test_task_plan_1 = TaskPlan { task_name: test_task.name, processor_names: test_task.processor.into_iter().map(|p| p.get_name().to_string()).collect::<Vec<_>>()};
        let test_task_plan = test_task_plan_0.extend(test_task_plan_1);
        assert_eq!(test_task_plan.task_name, "test_task");
        assert_eq!(test_task_plan.processor_names.first().unwrap(), "test_processor_1");
        assert_eq!(test_task_plan.processor_names.last().unwrap(), "test_processor_2");
    }
}