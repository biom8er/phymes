use std::sync::Arc;

use crate::{
    Network, NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderTrait,
    core::{NextSuperstepNetwork, NextTaskNetwork},
};
use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Field, Schema},
};
use clap::ValueEnum;
use phymes_data::{
    MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE, MERMAID_ER_DIAGRAM_TEMPLATE, SubjectScript,
};
use phymes_diagnostics::{HashMap, HashSet};
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_message::IPCMessageMap;
use phymes_processor::{AvailableProcessors, ProcessorBuilder, ProcessorPlanBuilder};
use phymes_schemas::{
    check_agent_subjects, from_data_type_to_str, from_str_to_data_type, parse_str_to_data_type,
};
use phymes_streams::extract_tool_calls_str;
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, Subject, SubjectBuilderTrait,
    SubjectPlanBuilder, SubjectPlanBuilderTrait, SubjectPlanTrait, SubjectTrait,
};
use phymes_task::TaskPlanBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Map;

/// Trait extension for [NetworkBuilderTrait] to enable exporting to and importing from mermaid.js
pub trait NetworkBuilderMermaidTrait {
    /// Make a mermaid.js flowchart of the session
    ///
    /// # Arguments
    /// * `with_processor_configs` - whether to add the processor config subjects to the diagram
    /// * `with_session_interface` - whether to add the session interface tasks, processors, and runtime environments to the diagram
    /// * `with_next_tasks` - whether to add the tasks subscribe and publish tasks, processors, and runtime environments to the diagram
    ///
    /// # Notes
    /// * Tool processors (i.e., task_name = processor_name) are triggered by updates on their config subject,
    ///   so they are always included even when `with_processor_configs` is false
    fn to_mermaid_flowchart(
        &self,
        with_processor_configs: bool,
        with_session_interface: bool,
    ) -> Result<String>;

    /// Make a mermaid.js erDiagram of the session
    ///
    /// # Arguments
    /// * `with_example_data` - whether to add last row of data as an example to the diagram
    /// * `with_processor_config_data` - whether to add ONLY processor related config data
    fn to_mermaid_erdiagram(
        &self,
        with_example_data: bool,
        with_processor_config_data: bool,
    ) -> Result<String>;

    /// Create a session builder from a mermaid flowchart
    ///
    /// # Arguments
    /// * `flowchart`: the flowchart diagram String
    /// * `agent_subjects`: whether to check for the presence of [AvailableInterfaceSubjects] with [check_agent_subjects]
    ///
    /// [AvailableInterfaceSubjects]: crate::AvailableInterfaceSubjects
    fn from_mermaid_flowchart(flowchart: &str, agent_subjects: bool) -> Result<Self>
    where
        Self: Sized;

    /// Create the state from a mermaid ER Diagram
    ///
    /// # Arguments
    /// * `erdiagram`: the ER diagram String
    /// * `agent_subjects`: whether to check for the presence of [AvailableInterfaceSubjects] with [check_agent_subjects]
    /// * `with_values`: whether to add the example values or leave the [RecordBatch]es empty
    ///
    /// [AvailableInterfaceSubjects]: crate::AvailableInterfaceSubjects
    fn with_subjects_from_mermaid_erdiagram(
        self,
        erdiagram: &str,
        agent_subjects: bool,
        with_values: bool,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl NetworkBuilderMermaidTrait for NetworkBuilder {
    fn to_mermaid_flowchart(
        &self,
        with_configs: bool,
        with_session_interface: bool,
    ) -> Result<String> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!(
                "Add task plans before making the Mermaid Flowchart."
            ));
        }
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processors before making the Mermaid Flowchart."
            ));
        }
        if self.runtime_env.is_none() {
            return Err(anyhow!(
                "Add runtime environments before making the Mermaid Flowchart."
            ));
        }
        if self.subjects.is_none() {
            return Err(anyhow!(
                "Add state subjects before making the Mermaid Flowchart."
            ));
        }
        if self.name.is_none() {
            return Err(anyhow!(
                "Add a session name before making the Mermaid Flowchart."
            ));
        }
        let session_name = self.name.as_ref().unwrap().to_string();

        // Tasks, Processors, and Runtime_envs to exclude
        let mut tasks_exclude = HashSet::new();
        let mut processors_exclude = HashSet::new();
        let mut subjects_exclude = HashSet::new();
        {
            // Exclusions from `NextTaskNetwork`
            let next_task_network = NextTaskNetwork::default();
            let tasks_publish_subscribe = NetworkBuilder::from_mermaid_flowchart(
                next_task_network.as_mermaid_flowchart(),
                false,
            )?
            .with_subjects_from_mermaid_erdiagram(
                next_task_network.as_mermaid_erdiagram(),
                false,
                false,
            )?
            .with_name(next_task_network.network_name)
            .add_processor_subjects()?;
            if let Some(tasks) = tasks_publish_subscribe.tasks {
                for task in tasks {
                    tasks_exclude.insert(task.task_name);
                    let processors = task.processor_names.into_iter().collect::<HashSet<_>>();
                    processors_exclude.extend(processors);
                }
            }
            if let Some(subjects) = tasks_publish_subscribe.subjects {
                for table in subjects {
                    subjects_exclude.insert(table.get_name().to_string());
                }
            }
        }
        {
            // Exclusions from `NextSuperstepNetwork`
            let next_superstep_network = NextSuperstepNetwork::default();
            let tasks_next_superstep = NetworkBuilder::from_mermaid_flowchart(
                next_superstep_network.as_mermaid_flowchart(),
                false,
            )?
            .with_subjects_from_mermaid_erdiagram(
                next_superstep_network.as_mermaid_erdiagram(),
                false,
                false,
            )?
            .with_name(next_superstep_network.network_name)
            .add_processor_subjects()?;
            if let Some(tasks) = tasks_next_superstep.tasks {
                for task in tasks {
                    tasks_exclude.insert(task.task_name);
                    let processors = task.processor_names.into_iter().collect::<HashSet<_>>();
                    processors_exclude.extend(processors);
                }
            }
            if let Some(subjects) = tasks_next_superstep.subjects {
                for table in subjects {
                    subjects_exclude.insert(table.get_name().to_string());
                }
            }
        }
        if !with_configs {
            let mut subjects = self.get_processor_names_from_tasks();
            if let Some(tasks) = self.tasks.as_ref() {
                for task in tasks {
                    let _ = subjects.remove(&task.task_name);
                }
            }
            subjects_exclude.extend(subjects);
        }
        if !with_session_interface {
            tasks_exclude.insert(session_name.to_owned());
            processors_exclude.insert(session_name.to_owned());
        }

        // Entities with expanded shape/label attributes that will be appended to flowchart
        let processors_vec = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|p| {
                if !processors_exclude.contains(p.get_name()) {
                    Some(format!(
                        "\t{}-processor@{{shape: rect, label: {}}}",
                        p.get_name(),
                        p.get_type()
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut subjects_vec = Vec::new();
        let mut sorted_subject_names = self
            .get_subject_names_from_processors()
            .into_iter()
            .filter(|p| !subjects_exclude.contains(p))
            .collect::<Vec<_>>();
        sorted_subject_names.sort();
        for subject_name in sorted_subject_names {
            subjects_vec.push(format!(
                "\t{subject_name}-subject@{{shape: doc, label: {subject_name}}}"
            ));
        }

        let mut runtime_envs_vec = Vec::new();
        runtime_envs_vec.push(format!(
            "\t{}-rt@{{shape: subproc, label: {}}}",
            self.runtime_env.as_ref().unwrap().get_name(),
            self.runtime_env.as_ref().unwrap().get_name()
        ));

        // Subgraphs
        let mut tasks_vec = Vec::new();
        let mut subscriptions_vec = Vec::new();
        let mut publications_vec = Vec::new();
        let mut runtime_envs_to_tasks_vec = Vec::new();
        let tasks = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|p| !tasks_exclude.contains(&p.task_name))
            .collect::<Vec<_>>();
        for task in tasks {
            tasks_vec.push(format!("\tsubgraph {}", task.task_name));
            runtime_envs_to_tasks_vec.push(format!(
                "\t{}-rt-->{}",
                self.runtime_env.as_ref().unwrap().get_name(),
                task.task_name
            ));

            // Iterate through each processor
            for processor_name in task.processor_names.iter() {
                for processor in self.processors.as_ref().unwrap().iter() {
                    if processor_name == processor.get_name() {
                        // Subscriptions
                        subscriptions_vec.push(format!(
                            "\t{processor_name}-subscribe@{{shape: diamond, label: {}}}",
                            processor.get_subscribe_policy().get_name()
                        ));
                        let subscriptions = processor
                            .get_subscriptions()
                            .iter()
                            .filter(|p| !subjects_exclude.contains(p.subject_name()))
                            .collect::<Vec<_>>();
                        for subscription in subscriptions {
                            if subscription.is_update() {
                                tasks_vec.push(format!(
                                    "\t\t{}-subject-.->|{}|{processor_name}-subscribe",
                                    subscription.subject_name(),
                                    subscription.short_name()
                                ));
                            } else {
                                tasks_vec.push(format!(
                                    "\t\t{}-subject-->|{}|{processor_name}-subscribe",
                                    subscription.subject_name(),
                                    subscription.short_name()
                                ));
                            }
                        }
                        tasks_vec.push(format!(
                            "\t\t{processor_name}-subscribe-->{processor_name}-processor"
                        ));

                        // Publications
                        publications_vec
                            .push(format!("\t{processor_name}-publish@{{shape: fork}}"));
                        tasks_vec.push(format!(
                            "\t\t{processor_name}-processor-->{processor_name}-publish"
                        ));
                        let publications = processor
                            .get_publications()
                            .iter()
                            .filter(|p| !subjects_exclude.contains(p.subject_name()))
                            .collect::<Vec<_>>();
                        for publication in publications {
                            tasks_vec.push(format!(
                                "\t\t{processor_name}-publish-->|{}|{}-subject",
                                publication.short_name(),
                                publication.subject_name()
                            ));
                        }

                        break;
                    }
                }
            }
            tasks_vec.push("\tend".to_string());
        }
        subscriptions_vec.sort();
        publications_vec.sort();

        // Create the final mermaid.js flowchart script
        let mut mermaid_js = vec!["flowchart TD".to_string()];
        mermaid_js.extend(tasks_vec);
        mermaid_js.extend(runtime_envs_to_tasks_vec);
        mermaid_js.extend(processors_vec);
        mermaid_js.extend(runtime_envs_vec);
        mermaid_js.extend(subjects_vec);
        mermaid_js.extend(publications_vec);
        mermaid_js.extend(subscriptions_vec);
        Ok(mermaid_js.join("\n"))
    }

    fn to_mermaid_erdiagram(&self, example_data: bool, config_data: bool) -> Result<String> {
        if self.subjects.is_none() {
            return Err(anyhow!(
                "Add state subjects before making the Mermaid ER Diagram."
            ));
        }
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processors before making the Mermaid Flowchart."
            ));
        }
        let mut entities_vec = Vec::new();
        // let mut relations_vec = Vec::new(); // DM: todo when relations are explicitly added between subjects based on FK/PK
        let processor_names = self.get_processor_names_from_tasks();

        // Extract the subjects
        let mut sorted_map = self.subjects.as_ref().unwrap().iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.get_name().cmp(b.get_name()));
        for subject in sorted_map {
            for field in subject.subject().get_schema().fields().iter() {
                let mut row = Map::new();
                row.insert("entity_alias".to_string(), subject.get_name().into());
                row.insert(
                    "entity_name".to_string(),
                    subject
                        .get_name()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join("")
                        .into(),
                );
                let data_type = from_data_type_to_str(field.data_type());
                row.insert("attribute_type".to_string(), data_type.into());
                row.insert("attribute_name".to_string(), field.name().as_str().into());
                let value = if config_data && processor_names.contains(subject.get_name())
                    || example_data
                {
                    if let Ok(mut example_data) =
                        subject.subject().get_column_as_vec_string(field.name())
                    {
                        example_data.pop().unwrap_or_default()
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                let value = value.replace("\"", "'");
                row.insert("attribute_comment".to_string(), value.into());
                row.insert("attribute_key".to_string(), String::new().into());
                entities_vec.push(row);
            }
        }

        // Create the final mermaid.js flowchart script
        let inputs = serde_json::json!({
            "rows": entities_vec
        });
        let entities_string =
            SubjectScript::new_from_template(MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE.to_string())
                .apply_template(&inputs)?;
        let inputs = serde_json::json!({
            "direction": "TB",
            "rows": [{"content": entities_string}]
        });
        let script_string =
            SubjectScript::new_from_template(MERMAID_ER_DIAGRAM_TEMPLATE.to_string())
                .apply_template(&inputs)?;

        Ok(script_string.trim().to_owned())
    }

    fn from_mermaid_flowchart(flowchart: &str, agent_subjects: bool) -> Result<Self> {
        // The members that we will build
        let mut task_plan_builders = HashMap::<String, TaskPlanBuilder>::new();
        let mut processor_builders = HashMap::<String, ProcessorBuilder>::new();
        let mut processor_plan_builders = HashMap::<String, ProcessorPlanBuilder>::new();

        // Track consistency of subjects and processors between subgraphs and labels
        let mut task_names = HashSet::new();
        let mut subject_names = HashSet::new();
        let mut processor_names = HashSet::new();
        let mut runtime_envs_names = HashSet::new();

        // Track the order of processors
        let mut processor_names_vec = Vec::new();
        let mut subject_names_vec = Vec::new();
        let mut runtime_env_names_vec = Vec::new();
        let mut task_names_vec = Vec::new();

        let is_first_line = |line: &str| -> bool {
            match line.split_whitespace().next() {
                Some(line) => line == "flowchart",
                None => false,
            }
        };

        // Parse the mermaid.js flowchart string
        let flowchart_lines = flowchart.split("\n").collect::<Vec<_>>();
        let mut iter = 0;
        if !is_first_line(flowchart_lines.first().unwrap()) {
            return Err(anyhow!(
                "Parsing Error on line {iter}: {}. Unrecognized mermaid.js flowchart type",
                flowchart_lines.get(iter).unwrap()
            ));
        }
        while iter < flowchart_lines.len() {
            // Check the chart type
            if is_first_line(flowchart_lines.get(iter).unwrap()) {

                // Ignore blank lines and comments
            } else if flowchart_lines.get(iter).unwrap().trim().is_empty()
                || flowchart_lines.get(iter).unwrap().trim().starts_with("%%")
            {

                // Task section
            } else if flowchart_lines.get(iter).unwrap().contains("subgraph") {
                // Start building the task plan
                let task_name = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("subgraph")
                    .last()
                    .unwrap()
                    .trim()
                    .to_string();
                if !task_plan_builders.contains_key(&task_name) {
                    let mut builder = TaskPlanBuilder::default();
                    builder.name.replace(task_name.to_owned());
                    task_plan_builders.insert(task_name.to_owned(), builder);
                }
                task_names_vec.push(task_name.to_owned());

                iter += 1;
                while iter < flowchart_lines.len() {
                    if flowchart_lines.get(iter).unwrap().trim() == "end" {
                        break;

                    // Subject, Subscription, Subscribe triple
                    // e.g., state_1-subject-.->|AllRecordBatches|processor_1-subscribe
                    } else if flowchart_lines.get(iter).unwrap().contains("-subject")
                        & flowchart_lines.get(iter).unwrap().contains("->")
                        & flowchart_lines.get(iter).unwrap().contains("|")
                        & flowchart_lines.get(iter).unwrap().contains("-subscribe")
                    {
                        // Extract out the subscription
                        let split_line = flowchart_lines
                            .get(iter)
                            .unwrap()
                            .split("-subject")
                            .collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. There are two subjects in task {task_name}",
                                flowchart_lines.get(iter).unwrap()
                            ));
                        }
                        let subject = split_line.first().unwrap().trim().to_string();
                        let subscription = match Subscription::from_str_mermaid(
                            split_line.last().unwrap(),
                            &subject,
                        ) {
                            Ok(subscription) => subscription,
                            Err(err) => {
                                return Err(anyhow!(
                                    "Parsing Error on line {iter}: {} for task {task_name}. {err}",
                                    flowchart_lines.get(iter).unwrap()
                                ));
                            }
                        };

                        // Check the processor name
                        let split_line = split_line
                            .last()
                            .unwrap()
                            .split("|")
                            .collect::<Vec<_>>()
                            .last()
                            .unwrap()
                            .split("-subscribe")
                            .collect::<Vec<_>>();
                        let processor = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor) {
                            let builder = ProcessorBuilder::default().with_name(&processor);
                            processor_builders.insert(processor.to_owned(), builder);
                        }
                        if !processor_plan_builders.contains_key(&processor) {
                            let builder =
                                ProcessorPlanBuilder::default().with_subscriptions(&[subscription]);
                            processor_plan_builders.insert(processor.to_owned(), builder);
                        } else if processor_plan_builders
                            .get(&processor)
                            .unwrap()
                            .subscriptions
                            .is_none()
                        {
                            processor_plan_builders
                                .get_mut(&processor)
                                .unwrap()
                                .subscriptions
                                .replace(vec![subscription]);
                        } else {
                            processor_plan_builders
                                .get_mut(&processor)
                                .unwrap()
                                .subscriptions
                                .as_mut()
                                .unwrap()
                                .push(subscription);
                        }

                        // Update
                        if task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .is_none()
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .replace(vec![processor.to_owned()]);
                        } else if !task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .as_ref()
                            .unwrap()
                            .contains(&processor)
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .as_mut()
                                .unwrap()
                                .push(processor.to_owned());
                        }
                        processor_names.insert(processor);
                        // ArrowTableSubscribe::None will not have a subject
                        if !subject.is_empty() {
                            subject_names.insert(subject);
                        }

                    // Subscribe, Processor triple
                    // e.g., processor_1-subscribe-->processor_1-processor
                    } else if flowchart_lines.get(iter).unwrap().contains("-subscribe")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
                        & flowchart_lines.get(iter).unwrap().contains("-processor")
                    {
                        // Check the processor name
                        let split_line = flowchart_lines
                            .get(iter)
                            .unwrap()
                            .split("-subscribe")
                            .collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. There are two subscribes in task {task_name}",
                                flowchart_lines.get(iter).unwrap()
                            ));
                        }
                        let processor_1 = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor_1) {
                            let builder = ProcessorBuilder::default().with_name(&processor_1);
                            processor_builders.insert(processor_1.to_owned(), builder);
                        }
                        if !processor_plan_builders.contains_key(&processor_1) {
                            let builder = ProcessorPlanBuilder::default();
                            processor_plan_builders.insert(processor_1.to_owned(), builder);
                        }

                        // Check the processor name
                        let split_line = split_line
                            .last()
                            .unwrap()
                            .split("-->")
                            .collect::<Vec<_>>()
                            .last()
                            .unwrap()
                            .split("-processor")
                            .collect::<Vec<_>>();
                        let processor_2 = split_line.first().unwrap().trim().to_string();
                        if processor_1 != processor_2 {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. Processor name {processor_1} does not match processor name {processor_2} in task {task_name}",
                                flowchart_lines.get(iter).unwrap()
                            ));
                        }

                        // Update
                        if task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .is_none()
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .replace(vec![processor_1.to_owned()]);
                        } else if !task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .as_ref()
                            .unwrap()
                            .contains(&processor_1)
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .as_mut()
                                .unwrap()
                                .push(processor_1.to_owned());
                        }
                        processor_names.insert(processor_1);

                    // Processor, Publish triple
                    // e.g., processor_1-processor-->processor_1-publish
                    } else if flowchart_lines.get(iter).unwrap().contains("-processor")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
                        & flowchart_lines.get(iter).unwrap().contains("-publish")
                    {
                        // Check the processor name
                        let split_line = flowchart_lines
                            .get(iter)
                            .unwrap()
                            .split("-processor")
                            .collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. There are two subscribes in task {task_name}",
                                flowchart_lines.get(iter).unwrap()
                            ));
                        }
                        let processor_1 = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor_1) {
                            let builder = ProcessorBuilder::default().with_name(&processor_1);
                            processor_builders.insert(processor_1.to_owned(), builder);
                        }
                        if !processor_plan_builders.contains_key(&processor_1) {
                            let builder = ProcessorPlanBuilder::default();
                            processor_plan_builders.insert(processor_1.to_owned(), builder);
                        }

                        // Check the processor name
                        let split_line = split_line
                            .last()
                            .unwrap()
                            .split("-->")
                            .collect::<Vec<_>>()
                            .last()
                            .unwrap()
                            .split("-publish")
                            .collect::<Vec<_>>();
                        let processor_2 = split_line.first().unwrap().trim().to_string();
                        if processor_1 != processor_2 {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. Processor name {processor_1} does not match processor name {processor_2} in task {task_name}",
                                flowchart_lines.get(iter).unwrap()
                            ));
                        }

                        // Update
                        if task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .is_none()
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .replace(vec![processor_1.to_owned()]);
                        } else if !task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .as_ref()
                            .unwrap()
                            .contains(&processor_1)
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .as_mut()
                                .unwrap()
                                .push(processor_1.to_owned());
                        }
                        processor_names.insert(processor_1);

                    // Publish, Publication, Subject triple
                    // e.g., processor_1-publish-->|Extend|state_1-subject
                    } else if flowchart_lines.get(iter).unwrap().contains("-publish")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
                        & flowchart_lines.get(iter).unwrap().contains("|")
                        & flowchart_lines.get(iter).unwrap().contains("-subject")
                    {
                        // Check the processor name
                        let split_line = flowchart_lines
                            .get(iter)
                            .unwrap()
                            .split("-publish")
                            .collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. There are two subscribes in task {task_name}",
                                flowchart_lines.get(iter).unwrap()
                            ));
                        }
                        let processor = split_line.first().unwrap().trim().to_string();

                        // Extract the publication
                        let subject = split_line
                            .last()
                            .unwrap()
                            .split("|")
                            .collect::<Vec<_>>()
                            .last()
                            .unwrap()
                            .split("-subject")
                            .collect::<Vec<_>>()
                            .first()
                            .unwrap()
                            .trim()
                            .to_string();
                        let publication = match Publication::from_str_mermaid(
                            split_line.last().unwrap(),
                            &subject,
                        ) {
                            Ok(publication) => publication,
                            Err(err) => {
                                return Err(anyhow!(
                                    "Parsing Error on line {iter}: {} for task {task_name}. {err}",
                                    flowchart_lines.get(iter).unwrap()
                                ));
                            }
                        };
                        if !processor_builders.contains_key(&processor) {
                            let builder = ProcessorBuilder::default().with_name(&processor);
                            processor_builders.insert(processor.to_owned(), builder);
                        }
                        if !processor_plan_builders.contains_key(&processor) {
                            let builder =
                                ProcessorPlanBuilder::default().with_publications(&[publication]);
                            processor_plan_builders.insert(processor.to_owned(), builder);
                        } else if processor_plan_builders
                            .get(&processor)
                            .unwrap()
                            .publications
                            .is_none()
                        {
                            processor_plan_builders
                                .get_mut(&processor)
                                .unwrap()
                                .publications
                                .replace(vec![publication]);
                        } else {
                            processor_plan_builders
                                .get_mut(&processor)
                                .unwrap()
                                .publications
                                .as_mut()
                                .unwrap()
                                .push(publication);
                        }

                        // Update
                        if task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .is_none()
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .replace(vec![processor.to_owned()]);
                        } else if !task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .processor_names
                            .as_ref()
                            .unwrap()
                            .contains(&processor)
                        {
                            task_plan_builders
                                .get_mut(&task_name)
                                .unwrap()
                                .processor_names
                                .as_mut()
                                .unwrap()
                                .push(processor.to_owned());
                        }
                        processor_names.insert(processor);
                        subject_names.insert(subject);

                    // Unrecognized arrows
                    } else if flowchart_lines.get(iter).unwrap().contains("---")
                        | flowchart_lines.get(iter).unwrap().contains("-.-")
                        | flowchart_lines.get(iter).unwrap().contains("==")
                        | flowchart_lines.get(iter).unwrap().contains("~~")
                        | flowchart_lines.get(iter).unwrap().contains("--o")
                        | flowchart_lines.get(iter).unwrap().contains("--x")
                        | flowchart_lines.get(iter).unwrap().contains("<--")
                        | flowchart_lines.get(iter).unwrap().contains("o--")
                        | flowchart_lines.get(iter).unwrap().contains("x--")
                    {
                        return Err(anyhow!(
                            "Parsing Error on line {iter}: {}. Unsupported arrow type in subgraph {task_name}. Only --> and .-> arrows are supported in PHYMES.",
                            flowchart_lines.get(iter).unwrap()
                        ));

                    // Unrecognized qualifier
                    } else if !flowchart_lines.get(iter).unwrap().contains("subject")
                        | !flowchart_lines.get(iter).unwrap().contains("subscribe")
                        | !flowchart_lines.get(iter).unwrap().contains("processor")
                        | !flowchart_lines.get(iter).unwrap().contains("publish")
                    {
                        return Err(anyhow!(
                            "Parsing Error on line {iter}: {}. Unsupported processor or subject qualifier in subgraph {task_name}. Only -subject, -subscribe, -processor, and -publish qualifiers are supported in PHYMES.",
                            flowchart_lines.get(iter).unwrap()
                        ));

                    // Any others
                    } else {
                        return Err(anyhow!(
                            "Parsing Error on line {iter}: {}. Unrecognized line in subgraph {task_name}",
                            flowchart_lines.get(iter).unwrap()
                        ));
                    }
                    iter += 1;
                }

            // Extract out the task runtime environments
            } else if flowchart_lines.get(iter).unwrap().contains("-rt-->") {
                // Extract the runtime and task names
                let split_line = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("-rt-->")
                    .collect::<Vec<_>>();
                if split_line.len() > 2 {
                    return Err(anyhow!(
                        "Parsing Error on line {iter}: {}. There are two runtime environments",
                        flowchart_lines.get(iter).unwrap()
                    ));
                }
                let runtime_env_name = split_line.first().unwrap().trim().to_string();
                let task_name = split_line.last().unwrap().trim().to_string();
                if !task_plan_builders.contains_key(&task_name) {
                    let mut builder = TaskPlanBuilder::default();
                    builder.name.replace(task_name.to_owned());
                    task_plan_builders.insert(task_name.to_owned(), builder);
                }

                // Update
                task_names.insert(task_name);
                runtime_envs_names.insert(runtime_env_name);

            // Extract out the runtime environments
            } else if flowchart_lines
                .get(iter)
                .unwrap()
                .contains("-rt@{shape: subproc,")
            {
                // Extract the runtime and task names
                let split_line = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("-rt@{shape: subproc,")
                    .collect::<Vec<_>>();
                let runtime_env_name = split_line.first().unwrap().trim().to_string();

                // Update
                runtime_env_names_vec.push(runtime_env_name.to_owned());

            // Extract out the processors
            } else if flowchart_lines
                .get(iter)
                .unwrap()
                .contains("-processor@{shape: rect,")
            {
                // Extract the processor name
                let split_line = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("-processor@{shape: rect,")
                    .collect::<Vec<_>>();
                let processor_name = split_line.first().unwrap().trim().to_string();
                let processor_type = match AvailableProcessors::from_str_fuzzy(
                    split_line.last().unwrap(),
                ) {
                    Ok(p) => p.to_string(),
                    Err(err) => {
                        return Err(anyhow!(
                            "Parsing Error on line {iter}: {}. Processor type for processor {processor_name} was not recognized. {err}",
                            flowchart_lines.get(iter).unwrap()
                        ));
                    }
                };

                // Update
                if !processor_builders.contains_key(&processor_name) {
                    let builder = ProcessorBuilder::default()
                        .with_name(&processor_name)
                        .with_type(&processor_type);
                    processor_builders.insert(processor_name.to_owned(), builder);
                } else if processor_builders
                    .get(&processor_name)
                    .unwrap()
                    .r#type
                    .is_none()
                {
                    processor_builders
                        .get_mut(&processor_name)
                        .unwrap()
                        .r#type
                        .replace(processor_type);
                }
                if !processor_plan_builders.contains_key(&processor_name) {
                    let builder = ProcessorPlanBuilder::default();
                    processor_plan_builders.insert(processor_name.to_owned(), builder);
                }
                processor_names_vec.push(processor_name.to_owned());

            // Extract out the subjects
            } else if flowchart_lines
                .get(iter)
                .unwrap()
                .contains("-subject@{shape: doc")
            {
                // Extract the subject name
                let split_line = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("-subject@{shape: doc")
                    .collect::<Vec<_>>();
                let subject_name = split_line.first().unwrap().trim().to_string();

                // Update
                subject_names_vec.push(subject_name.to_owned());

            // Extract out the subscribe
            } else if flowchart_lines
                .get(iter)
                .unwrap()
                .contains("-subscribe@{shape: diamond, label:")
            {
                // Extract the processor name
                let split_line = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("-subscribe@{shape: diamond, label:")
                    .collect::<Vec<_>>();
                let processor_name = split_line.first().unwrap().trim().to_string();
                let subscribe = match AvailableSubscribeEvents::from_str_fuzzy(
                    split_line.last().unwrap(),
                ) {
                    Ok(subscribe) => subscribe.build(),
                    Err(_e) => {
                        return Err(anyhow!(
                            "Parsing Error on line {iter}: {}. Subscribe policy for processor {processor_name} was not recognized.",
                            split_line.last().unwrap()
                        ));
                    }
                };
                if !processor_builders.contains_key(&processor_name) {
                    let builder = ProcessorBuilder::default().with_name(&processor_name);
                    processor_builders.insert(processor_name.to_owned(), builder);
                }

                // Update
                if !processor_plan_builders.contains_key(&processor_name) {
                    let builder = ProcessorPlanBuilder::default();
                    processor_plan_builders.insert(processor_name.to_owned(), builder);
                } else if processor_plan_builders
                    .get(&processor_name)
                    .unwrap()
                    .subscribe_policy
                    .as_ref()
                    .is_some()
                    && processor_plan_builders
                        .get(&processor_name)
                        .unwrap()
                        .subscribe_policy
                        .as_ref()
                        .unwrap()
                        .get_name()
                        != subscribe.get_name()
                {
                    return Err(anyhow!(
                        "Parsing Error on line {iter}: {}. Subscribe {} does not match processor {} subscribe {}.",
                        subscribe.get_name(),
                        processor_name,
                        processor_plan_builders
                            .get(&processor_name)
                            .unwrap()
                            .subscribe_policy
                            .as_ref()
                            .unwrap()
                            .get_name(),
                        flowchart_lines.get(iter).unwrap()
                    ));
                } else if processor_plan_builders
                    .get(&processor_name)
                    .unwrap()
                    .subscribe_policy
                    .as_ref()
                    .is_none()
                {
                    processor_plan_builders
                        .get_mut(&processor_name)
                        .unwrap()
                        .subscribe_policy
                        .replace(subscribe);
                }
                processor_names.insert(processor_name);

            // Extract out the publish
            } else if flowchart_lines
                .get(iter)
                .unwrap()
                .contains("-publish@{shape: fork}")
            {
                // Extract the processor name
                let split_line = flowchart_lines
                    .get(iter)
                    .unwrap()
                    .split("-publish@{shape: fork}")
                    .collect::<Vec<_>>();
                let processor_name = split_line.first().unwrap().trim().to_string();
                if !processor_builders.contains_key(&processor_name) {
                    let builder = ProcessorBuilder::default().with_name(&processor_name);
                    processor_builders.insert(processor_name.to_owned(), builder);
                }
                if !processor_plan_builders.contains_key(&processor_name) {
                    let builder = ProcessorPlanBuilder::default();
                    processor_plan_builders.insert(processor_name.to_owned(), builder);
                }

                // Update
                processor_names.insert(processor_name);
            } else {
                return Err(anyhow!(
                    "Parsing Error on line {iter}: {}. Unrecognized line ",
                    flowchart_lines.get(iter).unwrap()
                ));
            }
            iter += 1;
        }

        // Build the task plans in order
        let mut task_plans = Vec::new();
        let tasks_names_set = task_names_vec.clone().into_iter().collect::<HashSet<_>>();
        if task_names_vec.len() != task_names.len() || tasks_names_set != task_names {
            task_names_vec.sort();
            let mut task_names_sorted = task_names.into_iter().collect::<Vec<_>>();
            task_names_sorted.sort();
            return Err(anyhow!(
                "There is an inconsistency in the task labels {task_names_vec:?} and task mentions {task_names_sorted:?}"
            ));
        }
        for name in task_names_vec {
            let task_plan = task_plan_builders.remove(&name).unwrap().build()?;
            task_plans.push(task_plan);
        }

        // Build the runtime environments in order
        let mut runtime_envs = Vec::new();
        let runtime_env_names_set = runtime_env_names_vec
            .clone()
            .into_iter()
            .collect::<HashSet<_>>();
        if runtime_env_names_vec.len() != runtime_envs_names.len()
            || runtime_env_names_set != runtime_envs_names
        {
            let mut runtime_envs_names_sorted = runtime_envs_names.into_iter().collect::<Vec<_>>();
            runtime_envs_names_sorted.sort();
            return Err(anyhow!(
                "There is an inconsistency in the runtime environment labels {runtime_env_names_vec:?} and runtime environment mentions {runtime_envs_names_sorted:?}"
            ));
        }
        if runtime_env_names_vec.is_empty() {
            return Err(anyhow!("No runtime environment was found."));
        }
        if runtime_env_names_vec.len() > 1 {
            return Err(anyhow!(
                "More than one runtime environment labels was found {runtime_env_names_vec:?}"
            ));
        }
        for name in runtime_env_names_vec {
            // DM, todo!(): Need to update so that the backend and bucket can be included...
            let runtime_env = RuntimeEnv::get_builder().with_name(&name).build()?;
            runtime_envs.push(runtime_env);
        }

        // Build the processors in order
        let mut processors = Vec::new();
        let processor_names_set = processor_names_vec
            .clone()
            .into_iter()
            .collect::<HashSet<_>>();
        if processor_names_vec.len() != processor_names.len()
            || processor_names_set != processor_names
        {
            processor_names_vec.sort();
            let mut processor_names_sorted = processor_names.into_iter().collect::<Vec<_>>();
            processor_names_sorted.sort();
            return Err(anyhow!(
                "There is an inconsistency in the processor labels {processor_names_vec:?} and processor mentions {processor_names_sorted:?}"
            ));
        }
        for name in processor_names_vec {
            let processor_builder = processor_builders.remove(&name).unwrap();
            let processor = AvailableProcessors::from_str(
                processor_builder.r#type.as_ref().unwrap().as_str(),
                false,
            )
            .map_err(|e| anyhow!("{e:?}",))?
            .build_arc(processor_builder.name.as_ref().unwrap().as_str());
            let processor_plan_builder = processor_plan_builders.remove(&name).unwrap();
            let processor_plan = processor_plan_builder.with_processor(processor).build()?;
            processors.push(processor_plan);
        }

        // Check the subjects
        let subject_names_set = subject_names_vec
            .clone()
            .into_iter()
            .collect::<HashSet<_>>();
        if subject_names_vec.len() != subject_names.len() || subject_names_set != subject_names {
            subject_names_vec.sort();
            let mut subject_names_sorted = subject_names.into_iter().collect::<Vec<_>>();
            subject_names_sorted.sort();
            return Err(anyhow!(
                "There is an inconsistency in the subject labels {subject_names_vec:?} and subject mentions {subject_names_sorted:?}"
            ));
        }
        if agent_subjects {
            check_agent_subjects(&subject_names_vec)?;
        }

        let builder = Self::new()
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_env(runtime_envs.pop().unwrap().into());
        Ok(builder)
    }

    fn with_subjects_from_mermaid_erdiagram(
        self,
        erdiagram: &str,
        agent_subjects: bool,
        with_values: bool,
    ) -> Result<Self> {
        // Subjects to be collected
        let mut subject_builders = Vec::<SubjectPlanBuilder>::new();
        let mut subject_names = HashSet::<String>::new();

        // Parse the mermaid.js flowchart string
        let erdiagram_lines = erdiagram.lines().collect::<Vec<_>>();
        let mut iter = 0;
        if let Some(er_diagram) = erdiagram_lines.first() {
            if !er_diagram.contains("erDiagram") {
                return Err(anyhow!(
                    "Parsing Error on line {iter}: {}. Unrecognized mermaid.js erDiagram type.",
                    erdiagram_lines.get(iter).unwrap()
                ));
            }
        } else {
            return Err(anyhow!(
                "Parsing Error on line {iter}. No mermaid.js erDiagram provided."
            ));
        }
        while iter < erdiagram_lines.len() {
            // Check the chart type
            if erdiagram_lines.get(iter).unwrap().contains("erDiagram") {

                // Ignore blank lines and comments
            } else if erdiagram_lines.get(iter).unwrap().trim().is_empty() {

                // Ignore relationship connectors
            } else if erdiagram_lines.get(iter).unwrap().contains("|o")
                || erdiagram_lines.get(iter).unwrap().contains("o|")
                || erdiagram_lines.get(iter).unwrap().contains("||")
                || erdiagram_lines.get(iter).unwrap().contains("}o")
                || erdiagram_lines.get(iter).unwrap().contains("o{")
                || erdiagram_lines.get(iter).unwrap().contains("}|")
                || erdiagram_lines.get(iter).unwrap().contains("|{")
            {

                // Subject section
            } else if erdiagram_lines.get(iter).unwrap().contains("{")
                && erdiagram_lines.get(iter).unwrap().contains("[\"")
                && erdiagram_lines.get(iter).unwrap().contains("\"]")
            {
                // Extract the subject name
                let subject_name = extract_tool_calls_str(
                    erdiagram_lines.get(iter).unwrap(),
                    Some("[\""),
                    Some("\"]"),
                );
                subject_names.insert(subject_name.to_string());

                // Initialize the schema fields
                let mut fields = Vec::new();
                let mut data = Map::new();

                iter += 1;
                while iter < erdiagram_lines.len() {
                    // Check for end of subject section (and exclude jinja2 templating brackets)
                    if erdiagram_lines.get(iter).unwrap().trim_end() == "    }"
                        || erdiagram_lines.get(iter).unwrap().trim_end() == "\t}"
                    {
                        // Build and add the table to the subjects list
                        let schema = Arc::new(Schema::new(fields));
                        let table = if with_values && !data.is_empty() {
                            Subject::get_builder()
                                .with_schema(schema)
                                .with_json_values(&[serde_json::Value::Object(data)])?
                                .with_name(subject_name)
                                .build()?
                        } else {
                            Subject::get_builder()
                                .with_record_batches(vec![RecordBatch::new_empty(schema)])?
                                .with_name(subject_name)
                                .build()?
                        };
                        subject_builders.push(SubjectPlanBuilder::new().with_subject(table));
                        break;

                    // Extract the field and data type
                    } else {
                        let line = erdiagram_lines.get(iter).unwrap().trim();
                        let split_line = line.split_whitespace().collect::<Vec<_>>();

                        // Match the DataType
                        let field_name = if let Some(field_name) = split_line.get(1) {
                            field_name.to_string()
                        } else {
                            return Err(anyhow!(
                                "Parsing Error on line {iter}: {}. Unrecognized field name in subject {subject_name}",
                                erdiagram_lines.get(iter).unwrap(),
                            ));
                        };
                        let data_type = match from_str_to_data_type(split_line.first().unwrap()) {
                            Ok(data_type) => data_type,
                            Err(_e) => {
                                return Err(anyhow!(
                                    "Parsing Error on line {iter}: {}. Unrecognized data type {} in subject {subject_name} for field {field_name}. Supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, FixedSizeList, and List.",
                                    erdiagram_lines.get(iter).unwrap(),
                                    split_line.first().unwrap()
                                ));
                            }
                        };

                        // Extract the value
                        if with_values && split_line.len() > 2 {
                            // DM: we assume that all inner double quotes have been replaced by single quotes
                            if let Some(start) = line.find("\"") {
                                // Same line
                                let line = &line[start + 1..];
                                if let Some(end) = line.find("\"") {
                                    let value = &line[..end].replace("'", "\"");
                                    let _ = data.insert(
                                        field_name.to_owned(),
                                        parse_str_to_data_type(value, &data_type)?,
                                    );

                                // Multi-line
                                } else {
                                    let start_line = iter;
                                    iter += 1;
                                    while iter < erdiagram_lines.len() {
                                        if let Some(_end) =
                                            erdiagram_lines.get(iter).unwrap().find("\"")
                                        {
                                            let value =
                                                erdiagram_lines[start_line..iter].join("\n");
                                            let value = extract_tool_calls_str(
                                                &value,
                                                Some("\""),
                                                Some("\""),
                                            )
                                            .replace("'", "\"");
                                            let _ = data.insert(
                                                field_name.to_owned(),
                                                parse_str_to_data_type(&value, &data_type)?,
                                            );
                                            break;
                                        } else {
                                            iter += 1;
                                        }
                                    }
                                }
                            }
                        }
                        let field = Field::new(field_name, data_type, false);
                        fields.push(field);
                    }

                    iter += 1;
                }
            } else {
                return Err(anyhow!(
                    "Parsing Error on line {iter}: {}. Unrecognized line.",
                    erdiagram_lines.get(iter).unwrap()
                ));
            }

            iter += 1;
        }

        // Check the subjects
        let mut subjects_vec = subject_builders
            .iter()
            .map(|t| t.get_name().unwrap())
            .collect::<Vec<_>>();
        let subjects_set = subjects_vec.clone().into_iter().collect::<HashSet<_>>();
        if subjects_vec.len() != subject_names.len() || subjects_set != subject_names {
            subjects_vec.sort();
            let mut subject_names_sorted = subject_names.into_iter().collect::<Vec<_>>();
            subject_names_sorted.sort();
            return Err(anyhow!(
                "There is an inconsistency in the subject tables {subjects_vec:?} and subject mentions {subject_names_sorted:?}"
            ));
        }
        if agent_subjects {
            check_agent_subjects(&subject_names.into_iter().collect::<Vec<_>>())?;
        }

        let subjects = subject_builders
            .into_iter()
            .map(|s| s.build().unwrap())
            .collect::<Vec<_>>();
        Ok(self.with_subjects(subjects))
    }
}

/// [NetworkBuilder] specialized for Mermaid diagrams
#[derive(Default, Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct NetworkBuilderMermaid {
    pub name: Option<String>,
    pub flowchart: Option<String>,
    pub erdiagram: Option<String>,
}

impl NetworkBuilderMermaid {
    pub fn with_flowchart(mut self, flowchart: &str) -> Self {
        self.flowchart = Some(flowchart.to_string());
        self
    }
    pub fn with_erdiagram(mut self, erdiagram: &str) -> Self {
        self.erdiagram = Some(erdiagram.to_string());
        self
    }
}

impl BuilderTrait for NetworkBuilderMermaid {
    type T = (Network, Option<IPCMessageMap>);
    fn new() -> Self {
        Self {
            name: None,
            flowchart: None,
            erdiagram: None,
        }
    }

    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        // Handle the mermaid.js input
        let flowchart = match self.flowchart {
            Some(flowchart) => flowchart,
            None => {
                return Err(anyhow!(
                    "Please add a mermaid.js flowchart diagram before trying to build the `Network`."
                ));
            }
        };
        let name = match self.name {
            Some(name) => name,
            None => {
                return Err(anyhow!(
                    "Please provide a for the `Network` before trying to build the `Network`."
                ));
            }
        };
        let erdiagram = match self.erdiagram {
            Some(erdiagram) => erdiagram,
            None => {
                return Err(anyhow!(
                    "Please add a mermaid.js ER diagram before trying to build the `Network`."
                ));
            }
        };

        // Use defaults for diagnostics and max iters
        NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_name(&name)
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, true)?
            .add_processor_subjects()?
            .add_network_interface(None)?
            .with_diagnostics(true)
            .build_with_tables()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ChatAgentNetwork, DocumentRAGNetwork, NetworkBuilderAppsTrait, NetworkBuilderCustomTrait,
        ToolAgentNetwork, test_network_builder_apps,
    };

    use super::*;
    #[test]
    fn test_to_mermaid_flowchart() -> Result<()> {
        let builder = test_network_builder_apps::make_test_network_builder_apps("session_1")?
            .add_network_interface(Some(&["state_1", "state_2", "state_3"]))?
            .add_next_tasks()?;

        // Test to flowchart
        let mermaid_js = builder.to_mermaid_flowchart(false, false)?;
        assert_eq!(
            mermaid_js,
            "flowchart TD\n\tsubgraph task_1\n\t\tstate_1-subject-.->|AllRecordBatches|processor_1-subscribe\n\t\tprocessor_1-subscribe-->processor_1-processor\n\t\tprocessor_1-processor-->processor_1-publish\n\t\tprocessor_1-publish-->|Extend|state_1-subject\n\tend\n\tsubgraph task_2\n\t\tstate_2-subject-.->|AllRecordBatches|processor_2-subscribe\n\t\tprocessor_2-subscribe-->processor_2-processor\n\t\tprocessor_2-processor-->processor_2-publish\n\t\tprocessor_2-publish-->|Extend|state_2-subject\n\tend\n\tsubgraph task_3\n\t\tstate_1-subject-.->|AllRecordBatches|processor_3-subscribe\n\t\tstate_2-subject-.->|AllRecordBatches|processor_3-subscribe\n\t\tprocessor_3-subscribe-->processor_3-processor\n\t\tprocessor_3-processor-->processor_3-publish\n\t\tprocessor_3-publish-->|Extend|state_3-subject\n\tend\n\trt_1-rt-->task_1\n\trt_1-rt-->task_2\n\trt_1-rt-->task_3\n\tprocessor_1-processor@{shape: rect, label: ProcessorMock}\n\tprocessor_2-processor@{shape: rect, label: ProcessorMock}\n\tprocessor_3-processor@{shape: rect, label: Join}\n\trt_1-rt@{shape: subproc, label: rt_1}\n\tstate_1-subject@{shape: doc, label: state_1}\n\tstate_2-subject@{shape: doc, label: state_2}\n\tstate_3-subject@{shape: doc, label: state_3}\n\tprocessor_1-publish@{shape: fork}\n\tprocessor_2-publish@{shape: fork}\n\tprocessor_3-publish@{shape: fork}\n\tprocessor_1-subscribe@{shape: diamond, label: All}\n\tprocessor_2-subscribe@{shape: diamond, label: All}\n\tprocessor_3-subscribe@{shape: diamond, label: All}"
        );
        let mermaid_js = builder.to_mermaid_flowchart(false, true)?;
        assert_eq!(
            mermaid_js,
            "flowchart TD\n\tsubgraph task_1\n\t\tstate_1-subject-.->|AllRecordBatches|processor_1-subscribe\n\t\tprocessor_1-subscribe-->processor_1-processor\n\t\tprocessor_1-processor-->processor_1-publish\n\t\tprocessor_1-publish-->|Extend|state_1-subject\n\tend\n\tsubgraph task_2\n\t\tstate_2-subject-.->|AllRecordBatches|processor_2-subscribe\n\t\tprocessor_2-subscribe-->processor_2-processor\n\t\tprocessor_2-processor-->processor_2-publish\n\t\tprocessor_2-publish-->|Extend|state_2-subject\n\tend\n\tsubgraph task_3\n\t\tstate_1-subject-.->|AllRecordBatches|processor_3-subscribe\n\t\tstate_2-subject-.->|AllRecordBatches|processor_3-subscribe\n\t\tprocessor_3-subscribe-->processor_3-processor\n\t\tprocessor_3-processor-->processor_3-publish\n\t\tprocessor_3-publish-->|Extend|state_3-subject\n\tend\n\tsubgraph session_1\n\t\tstate_1-subject-.->|LastRecordBatch|session_1-subscribe\n\t\tstate_2-subject-.->|LastRecordBatch|session_1-subscribe\n\t\tstate_3-subject-.->|LastRecordBatch|session_1-subscribe\n\t\tsession_1-subscribe-->session_1-processor\n\t\tsession_1-processor-->session_1-publish\n\t\tsession_1-publish-->|Extend|state_1-subject\n\t\tsession_1-publish-->|Extend|state_2-subject\n\t\tsession_1-publish-->|Extend|state_3-subject\n\tend\n\trt_1-rt-->task_1\n\trt_1-rt-->task_2\n\trt_1-rt-->task_3\n\trt_1-rt-->session_1\n\tprocessor_1-processor@{shape: rect, label: ProcessorMock}\n\tprocessor_2-processor@{shape: rect, label: ProcessorMock}\n\tprocessor_3-processor@{shape: rect, label: Join}\n\tsession_1-processor@{shape: rect, label: ProcessorEcho}\n\trt_1-rt@{shape: subproc, label: rt_1}\n\tstate_1-subject@{shape: doc, label: state_1}\n\tstate_2-subject@{shape: doc, label: state_2}\n\tstate_3-subject@{shape: doc, label: state_3}\n\tprocessor_1-publish@{shape: fork}\n\tprocessor_2-publish@{shape: fork}\n\tprocessor_3-publish@{shape: fork}\n\tsession_1-publish@{shape: fork}\n\tprocessor_1-subscribe@{shape: diamond, label: All}\n\tprocessor_2-subscribe@{shape: diamond, label: All}\n\tprocessor_3-subscribe@{shape: diamond, label: All}\n\tsession_1-subscribe@{shape: diamond, label: Any}"
        );

        Ok(())
    }

    #[test]
    fn test_to_mermaid_erdiagram() -> Result<()> {
        let builder = test_network_builder_apps::make_test_network_builder_apps("session_1")?;

        // Make the ER Diagram
        let mermaid_js = builder.to_mermaid_erdiagram(false, false)?;
        assert_eq!(
            mermaid_js,
            "erDiagram\n    \n    processor_1[\"processor_1\"] {\n        Boolean cpu\n        Utf8 lhs_name\n        Utf8 lhs_stream\n        Utf8 operator\n    }\n    processor_2[\"processor_2\"] {\n        Boolean cpu\n        Utf8 lhs_name\n        Utf8 lhs_stream\n        Utf8 operator\n    }\n    processor_3[\"processor_3\"] {\n        Boolean cpu\n        Utf8 join_operators\n        Utf8 lhs_fk\n        Utf8 lhs_name\n        Utf8 lhs_pk\n        Utf8 lhs_stream\n        Utf8 operator\n        Utf8 rhs_fk\n        Utf8 rhs_name\n        Utf8 rhs_pk\n    }\n    state_1[\"state_1\"] {\n        UInt32 id\n        Utf8 collection\n        Utf8 title\n        Utf8 text\n        Utf8 metadata\n        Float32 score\n        FixedSizeList-Float32-8 embedding\n    }\n    state_2[\"state_2\"] {\n        UInt32 id\n        Utf8 collection\n        Utf8 title\n        Utf8 text\n        Utf8 metadata\n        Float32 score\n        FixedSizeList-Float32-8 embedding\n    }\n    state_3[\"state_3\"] {\n        UInt32 id\n        Utf8 collection\n        Utf8 title\n        Utf8 text\n        Utf8 metadata\n        Float32 score\n        FixedSizeList-Float32-8 embedding\n    }"
        );
        let mermaid_js = builder.to_mermaid_erdiagram(true, false)?;
        assert_eq!(
            mermaid_js,
            "erDiagram\n    \n    processor_1[\"processor_1\"] {\n        Boolean cpu \"false\"\n        Utf8 lhs_name \"state_1\"\n        Utf8 lhs_stream \"Accumulate\"\n        Utf8 operator \"HumanInTheLoop\"\n    }\n    processor_2[\"processor_2\"] {\n        Boolean cpu \"false\"\n        Utf8 lhs_name \"state_2\"\n        Utf8 lhs_stream \"Accumulate\"\n        Utf8 operator \"HumanInTheLoop\"\n    }\n    processor_3[\"processor_3\"] {\n        Boolean cpu \"false\"\n        Utf8 join_operators \"Inner\"\n        Utf8 lhs_fk \"id\"\n        Utf8 lhs_name \"state_1\"\n        Utf8 lhs_pk \"id\"\n        Utf8 lhs_stream \"Accumulate\"\n        Utf8 operator \"Join\"\n        Utf8 rhs_fk \"id\"\n        Utf8 rhs_name \"state_2\"\n        Utf8 rhs_pk \"id\"\n    }\n    state_1[\"state_1\"] {\n        UInt32 id \"3\"\n        Utf8 collection \"collection3\"\n        Utf8 title \"title3\"\n        Utf8 text \"text3\"\n        Utf8 metadata \"metadata3\"\n        Float32 score \"3.0\"\n        FixedSizeList-Float32-8 embedding \"[3.4e-44,3.5000000000000003e-44,3.6000000000000004e-44,3.8e-44,3.9e-44,4.0000000000000003e-44,4.2000000000000005e-44,4.3e-44]\"\n    }\n    state_2[\"state_2\"] {\n        UInt32 id \"3\"\n        Utf8 collection \"collection3\"\n        Utf8 title \"title3\"\n        Utf8 text \"text3\"\n        Utf8 metadata \"metadata3\"\n        Float32 score \"3.0\"\n        FixedSizeList-Float32-8 embedding \"[3.4e-44,3.5000000000000003e-44,3.6000000000000004e-44,3.8e-44,3.9e-44,4.0000000000000003e-44,4.2000000000000005e-44,4.3e-44]\"\n    }\n    state_3[\"state_3\"] {\n        UInt32 id \"3\"\n        Utf8 collection \"collection3\"\n        Utf8 title \"title3\"\n        Utf8 text \"text3\"\n        Utf8 metadata \"metadata3\"\n        Float32 score \"3.0\"\n        FixedSizeList-Float32-8 embedding \"[3.4e-44,3.5000000000000003e-44,3.6000000000000004e-44,3.8e-44,3.9e-44,4.0000000000000003e-44,4.2000000000000005e-44,4.3e-44]\"\n    }"
        );
        let mermaid_js = builder.to_mermaid_erdiagram(false, true)?;
        assert_eq!(
            mermaid_js,
            "erDiagram\n    \n    processor_1[\"processor_1\"] {\n        Boolean cpu \"false\"\n        Utf8 lhs_name \"state_1\"\n        Utf8 lhs_stream \"Accumulate\"\n        Utf8 operator \"HumanInTheLoop\"\n    }\n    processor_2[\"processor_2\"] {\n        Boolean cpu \"false\"\n        Utf8 lhs_name \"state_2\"\n        Utf8 lhs_stream \"Accumulate\"\n        Utf8 operator \"HumanInTheLoop\"\n    }\n    processor_3[\"processor_3\"] {\n        Boolean cpu \"false\"\n        Utf8 join_operators \"Inner\"\n        Utf8 lhs_fk \"id\"\n        Utf8 lhs_name \"state_1\"\n        Utf8 lhs_pk \"id\"\n        Utf8 lhs_stream \"Accumulate\"\n        Utf8 operator \"Join\"\n        Utf8 rhs_fk \"id\"\n        Utf8 rhs_name \"state_2\"\n        Utf8 rhs_pk \"id\"\n    }\n    state_1[\"state_1\"] {\n        UInt32 id\n        Utf8 collection\n        Utf8 title\n        Utf8 text\n        Utf8 metadata\n        Float32 score\n        FixedSizeList-Float32-8 embedding\n    }\n    state_2[\"state_2\"] {\n        UInt32 id\n        Utf8 collection\n        Utf8 title\n        Utf8 text\n        Utf8 metadata\n        Float32 score\n        FixedSizeList-Float32-8 embedding\n    }\n    state_3[\"state_3\"] {\n        UInt32 id\n        Utf8 collection\n        Utf8 title\n        Utf8 text\n        Utf8 metadata\n        Float32 score\n        FixedSizeList-Float32-8 embedding\n    }"
        );

        Ok(())
    }

    #[test]
    fn test_from_mermaid_parallel_with_config_and_session_no_data() -> Result<()> {
        let builder = test_network_builder_apps::make_test_network_builder_apps("session_1")?
            .add_network_interface(Some(&["state_1", "state_2", "state_3"]))?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(true, true)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, false)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, false, false)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that the schemas match
        {
            let test = builder_test
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            let expected = builder
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            for key in expected.keys() {
                assert!(expected.get(key).eq(&test.get(key)));
            }
        }

        // Test that we can build the session
        let _ = builder_test.with_name("session_1").build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_parallel_with_config_and_session_with_data() -> Result<()> {
        let builder = test_network_builder_apps::make_test_network_builder_apps("session_1")?
            .add_network_interface(Some(&["state_1", "state_2", "state_3"]))?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(true, true)?;
        let erdiagram = builder.to_mermaid_erdiagram(true, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, false)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, false, true)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that the schemas match
        {
            let test = builder_test
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            let expected = builder
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            for key in expected.keys() {
                assert!(expected.get(key).eq(&test.get(key)));
            }
        }

        // Test that the first row was captured
        for table in builder_test.subjects.as_ref().unwrap().iter() {
            assert_eq!(table.subject().count_rows(), 1)
        }

        // Test that we can build the session
        let _ = builder_test.with_name("session_1").build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_parallel_no_config_with_session_and_data() -> Result<()> {
        let builder = test_network_builder_apps::make_test_network_builder_apps("session_1")?
            .add_network_interface(Some(&["state_1", "state_2", "state_3"]))?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(false, true)?;
        let erdiagram = builder.to_mermaid_erdiagram(true, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, false)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, false, true)?;

        // Test that the tasks match
        assert_eq!(builder_test.tasks, builder.tasks);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that the schemas match
        {
            let test = builder_test
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            let expected = builder
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            for key in expected.keys() {
                assert!(expected.get(key).eq(&test.get(key)));
            }
        }

        // Test that the first row was captured
        for table in builder_test.subjects.as_ref().unwrap().iter() {
            assert_eq!(table.subject().count_rows(), 1)
        }

        // Test that the processor configs were added to the subscriptions
        let builder_test = builder_test
            .with_name("session_1")
            .add_processor_subjects()?;
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_chat_agent_network_with_configs_and_session() -> Result<()> {
        // initialize the session
        let builder = ChatAgentNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(true, true)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, false)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.with_name("session_1").build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_doc_rag_network_with_configs_and_session() -> Result<()> {
        // initialize the session
        let builder = DocumentRAGNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(true, true)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, false)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.with_name("session_1").build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_tool_agent_network_with_configs_and_session() -> Result<()> {
        // initialize the session
        let builder = ToolAgentNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(true, true)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, false)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.with_name("session_1").build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_chat_agent_network_without_configs_and_session() -> Result<()> {
        // initialize the session
        let builder = ChatAgentNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(false, false)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, false)?
            .with_name("session_1")
            .add_processor_subjects()?
            .add_network_interface(None)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_doc_rag_network_without_configs_and_session() -> Result<()> {
        // initialize the session
        let builder = DocumentRAGNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(false, false)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, false)?
            .with_name("session_1")
            .add_processor_subjects()?
            .add_network_interface(None)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_tool_agent_network_without_configs_and_session() -> Result<()> {
        // initialize the session
        let builder = ToolAgentNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(false, false)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, false)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, false)?
            .with_name("session_1")
            .add_processor_subjects()?
            .add_network_interface(None)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that we can build the session
        let _ = builder_test.build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_doc_rag_network_with_data() -> Result<()> {
        // initialize the session
        let builder = DocumentRAGNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(false, false)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, true)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, true)?
            .with_name("session_1")
            .add_processor_subjects()?
            .add_network_interface(None)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that the schemas match
        {
            let test = builder_test
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            let expected = builder
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            for key in expected.keys() {
                assert!(expected.get(key).eq(&test.get(key)));
            }
        }

        // Test that the first row was captured
        for table in builder_test.subjects.as_ref().unwrap().iter() {
            if builder_test
                .get_processor_names_from_tasks()
                .contains(table.get_name())
                && !builder_test
                    .tasks
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|t| t.task_name.as_str())
                    .collect::<Vec<_>>()
                    .contains(&table.get_name())
            {
                assert_eq!(table.subject().count_rows(), 1)
            } else {
                assert_eq!(table.subject().count_rows(), 0)
            }
        }

        // Test that we can build the session
        let _ = builder_test.build()?;

        Ok(())
    }

    #[test]
    fn test_from_mermaid_tool_agent_network_with_data() -> Result<()> {
        // initialize the session
        let builder = ToolAgentNetwork::new_with_network_name("session_1")
            .build()
            .with_name("session_1")
            .add_network_interface(None)?;

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart(false, false)?;
        let erdiagram = builder.to_mermaid_erdiagram(false, true)?;

        // Remake the builder
        let builder_test = NetworkBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_subjects_from_mermaid_erdiagram(&erdiagram, true, true)?
            .with_name("session_1")
            .add_processor_subjects()?
            .add_network_interface(None)?;

        // Test that the names match
        assert_eq!(builder_test.tasks, builder.tasks);
        let mut test = builder_test
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_subject_names_from_processors()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);

        // Test the order of the processors
        let test = builder_test
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        let expected = builder
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, expected);

        // Test that the schemas match
        {
            let test = builder_test
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            let expected = builder
                .subjects
                .as_ref()
                .unwrap()
                .iter()
                .map(|p| (p.get_name(), p.subject().get_schema()))
                .collect::<HashMap<_, _>>();
            for key in expected.keys() {
                assert!(expected.get(key).eq(&test.get(key)));
            }
        }

        // Test that the first row was captured
        for table in builder_test.subjects.as_ref().unwrap().iter() {
            if builder_test
                .get_processor_names_from_tasks()
                .contains(table.get_name())
                && !builder_test
                    .tasks
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|t| t.task_name.as_str())
                    .collect::<Vec<_>>()
                    .contains(&table.get_name())
            {
                assert_eq!(table.subject().count_rows(), 1)
            } else {
                assert_eq!(table.subject().count_rows(), 0)
            }
        }

        // Test that we can build the session
        let _ = builder_test.build()?;

        Ok(())
    }
}
