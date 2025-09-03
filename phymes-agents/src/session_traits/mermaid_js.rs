use std::sync::Arc;

use crate::session_plans::{available_interface_subjects::check_agent_subjects, available_processors::AvailableProcessors};
use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{DataType, Field, Schema},
};
use clap::ValueEnum;
use phymes_core::{
    metrics::{HashMap, HashSet},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::{
            SessionContextBuilder, SessionContextBuilderTrait, TaskPlanBuilder,
        },
    },
    table::{
        table::{Table, TableBuilderTrait, TableTrait},
        table_publish::TablePublish,
        table_subscribe::{TableSubscribe, from_str_to_subscribe},
    },
    task::processor::{
        ProcessorBuilder, ProcessorEcho, test_processor::ProcessorMock,
    },
};
use phymes_data::candle_data::{
    data_processor::CandleDataProcessor, summary_processor::DataSummaryProcessor,
};
use phymes_ml::{
    candle_chat::{
        chat_processor::CandleChatProcessor,
        message_aggregator_processor::MessageAggregatorProcessor,
        message_parser_processor::MessageParserProcessor,
    },
    candle_embed::embed_processor::CandleEmbedProcessor,
};
#[cfg(feature = "openai_api")]
use phymes_ml::{
    openai_chat::chat_processor::OpenAIChatProcessor,
    openai_embed::embed_processor::OpenAIEmbedProcessor,
};

/// Helper function to convert an arrow [DataType] to a [String]
pub fn from_data_type_to_str(data_type: &DataType) -> String {
    match data_type {
        DataType::FixedSizeList(f, s) => {
            format!("FixedSizeList-{}-{}", f.data_type(), s)
        }
        DataType::List(f) => {
            format!("List-{}", f.data_type())
        }
        _ => data_type.to_string(),
    }
}

/// Helper function to convert a [String] to an arrow [DataType]
pub fn from_str_to_data_type(data_type: &str) -> Result<DataType> {
    let data_type = match data_type {
        s if s == DataType::UInt8.to_string() => DataType::UInt8,
        s if s == DataType::UInt16.to_string() => DataType::UInt16,
        s if s == DataType::UInt32.to_string() => DataType::UInt32,
        s if s == DataType::Int64.to_string() => DataType::Int64,
        s if s == DataType::Float32.to_string() => DataType::Float32,
        s if s == DataType::Float64.to_string() => DataType::Float64,
        s if s == DataType::Utf8.to_string() => DataType::Utf8,
        s if s == DataType::Null.to_string() => DataType::Null,
        s if s == DataType::Boolean.to_string() => DataType::Boolean,
        s if s.contains("FixedSizeList-UInt8-") => {
            let size = data_type
                .split("FixedSizeList-UInt8-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::UInt8, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-UInt32-") => {
            let size = data_type
                .split("FixedSizeList-UInt32-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::UInt32, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Int64-") => {
            let size = data_type
                .split("FixedSizeList-Int64-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Int64, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Float32-") => {
            let size = data_type
                .split("FixedSizeList-Float32-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Float64-") => {
            let size = data_type
                .split("FixedSizeList-Float64-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float64, false)),
                size,
            )
        }
        s if s.contains("FixedSizeList-Utf8-") => {
            let size = data_type
                .split("FixedSizeList-Utf8-")
                .last()
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            DataType::FixedSizeList(Arc::new(Field::new_list_field(DataType::Utf8, false)), size)
        }
        s if s.contains("List-UInt8") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::UInt8, false)))
        }
        s if s.contains("List-UInt32") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)))
        }
        s if s.contains("List-Int64") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false)))
        }
        s if s.contains("List-Float32") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)))
        }
        s if s.contains("List-Float64") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, false)))
        }
        s if s.contains("List-Utf8") => {
            DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)))
        }
        _ => return Err(anyhow!("Unrecognized data type {data_type}")),
    };
    Ok(data_type)
}

/// Trait extension for [SessionContextBuilderTrait] to enable exporting to and importing from mermaid.js
pub trait SessionContextBuilderMermaidTrait {
    /// Make a mermaid.js flowchart of the session
    fn to_mermaid_flowchart(&self) -> Result<String>;

    /// Make a mermaid.js erDiagram of the session
    fn to_mermaid_erdiagram(&self) -> Result<String>;

    /// Create a session builder from a mermaid flowchart
    fn from_mermaid_flowchart(flowchart: &str, agent_subjects: bool) -> Result<Self>
    where
        Self: Sized;

    /// Create the state from a mermaid ER Diagram
    fn with_state_from_mermaid_erdiagram(self, erdiagram: &str, agent_subjects: bool) -> Result<Self>
    where
        Self: Sized;
}

impl SessionContextBuilderMermaidTrait for SessionContextBuilder {
    fn to_mermaid_flowchart(&self) -> Result<String> {
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
        if self.runtime_envs.is_none() {
            return Err(anyhow!(
                "Add runtime environments before making the Mermaid Flowchart."
            ));
        }
        if self.state.is_none() {
            return Err(anyhow!(
                "Add state subjects before making the Mermaid Flowchart."
            ));
        }

        // Entities with expanded shape/label attributes that will be appended to flowchart
        let mut processors_vec = Vec::new();
        for processor in self.processors.as_ref().unwrap().iter() {
            processors_vec.push(format!(
                "\t{}-processor@{{shape: rect, label: {}}}",
                processor.get_name(),
                processor.get_type()
            ));
        }

        let mut subjects_vec = Vec::new();
        let mut sorted_subject_names = self.get_subject_names().into_iter().collect::<Vec<_>>();
        sorted_subject_names.sort();
        for subject_name in sorted_subject_names {
            subjects_vec.push(format!(
                "\t{subject_name}-subject@{{shape: doc, label: {subject_name}}}"
            ));
        }

        let mut runtime_envs_vec = Vec::new();
        let mut sorted_runtime_env_names =
            self.get_runtime_env_names().into_iter().collect::<Vec<_>>();
        sorted_runtime_env_names.sort();
        for runtime_env_name in sorted_runtime_env_names {
            runtime_envs_vec.push(format!(
                "\t{runtime_env_name}-rt@{{shape: subproc, label: {runtime_env_name}}}"
            ));
        }

        // Subgraphs
        let mut tasks_vec = Vec::new();
        let mut subscriptions_vec = Vec::new();
        let mut publications_vec = Vec::new();
        let mut runtime_envs_to_tasks_vec = Vec::new();
        for task in self.tasks.as_ref().unwrap().iter() {
            tasks_vec.push(format!("\tsubgraph {}", task.task_name));
            runtime_envs_to_tasks_vec.push(format!(
                "\t{}-rt-->{}",
                task.runtime_env_name, task.task_name
            ));

            // Iterate through each processor
            for processor_name in task.processor_names.iter() {
                for processor in self.processors.as_ref().unwrap().iter() {
                    if processor_name == processor.get_name() {
                        // Subscriptions
                        subscriptions_vec.push(format!(
                            "\t{processor_name}-subscribe@{{shape: diamond, label: {}}}",
                            processor.get_subscribe().get_name()
                        ));
                        for subscription in processor.get_subscriptions().iter() {
                            if subscription.is_update() {
                                tasks_vec.push(format!(
                                    "\t\t{}-subject-.{}.->{processor_name}-subscribe",
                                    subscription.get_table_name(),
                                    subscription.get_short_name()
                                ));
                            } else {
                                tasks_vec.push(format!(
                                    "\t\t{}-subject--{}-->{processor_name}-subscribe",
                                    subscription.get_table_name(),
                                    subscription.get_short_name()
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
                        for publication in processor.get_publications().iter() {
                            tasks_vec.push(format!(
                                "\t\t{processor_name}-publish--{}-->{}-subject",
                                publication.get_short_name(),
                                publication.get_table_name()
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

    fn to_mermaid_erdiagram(&self) -> Result<String> {
        if self.state.is_none() {
            return Err(anyhow!(
                "Add state subjects before making the Mermaid ER Diagram."
            ));
        }

        // Extract the subjects
        let mut subjects = Vec::new();
        let mut sorted_map = self.state.as_ref().unwrap().iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.get_name().cmp(b.get_name()));
        for subject in sorted_map {
            subjects.push(format!("\t{}{{", subject.get_name()));
            for field in subject.get_schema().fields().iter() {
                let data_type = from_data_type_to_str(field.data_type());
                subjects.push(format!("\t\t{data_type}\t{}", field.name()));
            }
            subjects.push("\t}".to_string());
        }

        // Create the final mermaid.js flowchart script
        let mut mermaid_js = vec!["erDiagram".to_string()];
        mermaid_js.extend(subjects);
        Ok(mermaid_js.join("\n"))
    }

    fn from_mermaid_flowchart(flowchart: &str, agent_subjects: bool) -> Result<Self> {
        // The members that we will build
        let mut task_plan_builders = HashMap::<String, TaskPlanBuilder>::new();
        let mut processor_builders = HashMap::<String, ProcessorBuilder>::new();

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

        // Closures to create the subscriptions and publications
        let subscription_from_str = |line: &str,
                                     iter: usize,
                                     subject: &str,
                                     task: &str|
         -> Result<TableSubscribe> {
            if line.contains("-.") & line.contains(".->") & line.contains("FullTable") {
                Ok(TableSubscribe::OnUpdateFullTable {
                    table_name: subject.to_string(),
                })
            } else if line.contains("--") & line.contains("-->") & line.contains("FullTable") {
                Ok(TableSubscribe::AlwaysFullTable {
                    table_name: subject.to_string(),
                })
            } else if line.contains("-.") & line.contains(".->") & line.contains("LastRecordBatch")
            {
                Ok(TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: subject.to_string(),
                })
            } else if line.contains("--") & line.contains("-->") & line.contains("LastRecordBatch")
            {
                Ok(TableSubscribe::AlwaysLastRecordBatch {
                    table_name: subject.to_string(),
                })
            } else if line.contains("None") {
                Ok(TableSubscribe::None {})
            } else {
                Err(anyhow!(
                    "Parsing Error on line {iter}: {line}. Variant for ArrowTableSubscribe with subject {subject} for task {task} was not recognized."
                ))
            }
        };
        let publication_from_str = |line: &str,
                                    iter: usize,
                                    subject: &str,
                                    task: &str|
         -> Result<TablePublish> {
            if line.contains("--") & line.contains("-->") & line.contains("ExtendChunks") {
                Ok(TablePublish::ExtendChunks {
                    table_name: subject.to_string(),
                    col_name: "content".to_string(),
                })
            } else if line.contains("--") & line.contains("-->") & line.contains("Extend") {
                Ok(TablePublish::Extend {
                    table_name: subject.to_string(),
                })
            } else if line.contains("--") & line.contains("-->") & line.contains("ReplaceLast") {
                Ok(TablePublish::ReplaceLast {
                    table_name: subject.to_string(),
                })
            } else if line.contains("--") & line.contains("-->") & line.contains("Replace") {
                Ok(TablePublish::Replace {
                    table_name: subject.to_string(),
                })
            } else if line.contains("None") {
                Ok(TablePublish::None {})
            } else {
                Err(anyhow!(
                    "Parsing Error on line {iter}: {line}. Variant for ArrowTablePublish with subject {subject} for task {task} was not recognized."
                ))
            }
        };
        let processor_from_str = |line: &str, iter: usize, processor: &str| -> Result<String> {
            if line.contains(ProcessorMock::get_static_name()) {
                Ok(ProcessorMock::get_static_name().to_string())
            } else if line.contains(ProcessorEcho::get_static_name()) {
                Ok(ProcessorEcho::get_static_name().to_string())
            } else if line.contains(CandleDataProcessor::get_static_name()) {
                Ok(CandleDataProcessor::get_static_name().to_string())
            } else if line.contains(DataSummaryProcessor::get_static_name()) {
                Ok(DataSummaryProcessor::get_static_name().to_string())
            } else if line.contains(CandleChatProcessor::get_static_name()) {
                Ok(CandleChatProcessor::get_static_name().to_string())
            } else if line.contains(MessageAggregatorProcessor::get_static_name()) {
                Ok(MessageAggregatorProcessor::get_static_name().to_string())
            } else if line.contains(MessageParserProcessor::get_static_name()) {
                Ok(MessageParserProcessor::get_static_name().to_string())
            } else if line.contains(CandleEmbedProcessor::get_static_name()) {
                Ok(CandleEmbedProcessor::get_static_name().to_string())
            } else {
                #[cfg(feature = "openai_api")]
                if line.contains(OpenAIChatProcessor::get_static_name()) {
                    Ok(OpenAIChatProcessor::get_static_name().to_string())
                } else if line.contains(OpenAIEmbedProcessor::get_static_name()) {
                    Ok(OpenAIEmbedProcessor::get_static_name().to_string())
                } else {
                    Err(anyhow!(
                        "Parsing Error on line {iter}: {line}. Processor type for processor {processor} was not recognized."
                    ))
                }
                #[cfg(not(feature = "openai_api"))]
                Err(anyhow!(
                    "Parsing Error on line {iter}: {line}. Processor type for processor {processor} was not recognized."
                ))
            }
        };

        // Parse the mermaid.js flowchart string
        let flowchart_lines = flowchart.split("\n").collect::<Vec<_>>();
        let mut iter = 0;
        if !flowchart_lines.first().unwrap().contains("flowchart") {
            return Err(anyhow!(
                "Parsing Error on line {iter}: {}. Unrecognized mermaid.js flowchart type",
                flowchart_lines.get(iter).unwrap()
            ));
        }
        while iter < flowchart_lines.len() {
            // Check the chart type
            if flowchart_lines.get(iter).unwrap().contains("flowchart") {

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
                    builder.task_name.replace(task_name.to_owned());
                    task_plan_builders.insert(task_name.to_owned(), builder);
                }
                task_names_vec.push(task_name.to_owned());

                iter += 1;
                while iter < flowchart_lines.len() {
                    if flowchart_lines.get(iter).unwrap().trim() == "end" {
                        break;

                    // Subject, Subscription, Subscribe triple
                    // e.g., state_1-subject-.FullTable.->processor_1-subscribe
                    } else if flowchart_lines.get(iter).unwrap().contains("-subject")
                        & flowchart_lines.get(iter).unwrap().contains("->")
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
                        let subscription = subscription_from_str(
                            split_line.last().unwrap(),
                            iter,
                            &subject,
                            &task_name,
                        )?;

                        // Check the processor name
                        let split_line = split_line
                            .last()
                            .unwrap()
                            .split("->")
                            .collect::<Vec<_>>()
                            .last()
                            .unwrap()
                            .split("-subscribe")
                            .collect::<Vec<_>>();
                        let processor = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor) {
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor.to_owned());
                            builder.subscriptions.replace(vec![subscription]);
                            processor_builders.insert(processor.to_owned(), builder);
                        } else if processor_builders
                            .get(&processor)
                            .unwrap()
                            .subscriptions
                            .is_none()
                        {
                            processor_builders
                                .get_mut(&processor)
                                .unwrap()
                                .subscriptions
                                .replace(vec![subscription]);
                        } else {
                            processor_builders
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
                        } else {
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
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor_1.to_owned());
                            processor_builders.insert(processor_1.to_owned(), builder);
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
                        task_plan_builders
                            .get_mut(&task_name)
                            .unwrap()
                            .processor_names
                            .as_mut()
                            .unwrap()
                            .push(processor_1.to_owned());
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
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor_1.to_owned());
                            processor_builders.insert(processor_1.to_owned(), builder);
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
                        task_plan_builders
                            .get_mut(&task_name)
                            .unwrap()
                            .processor_names
                            .as_mut()
                            .unwrap()
                            .push(processor_1.to_owned());
                        processor_names.insert(processor_1);

                    // Publish, Publication, Subject triple
                    // e.g., processor_1-publish--Extend-->state_1-subject
                    } else if flowchart_lines.get(iter).unwrap().contains("-publish")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
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
                            .split("-->")
                            .collect::<Vec<_>>()
                            .last()
                            .unwrap()
                            .split("-subject")
                            .collect::<Vec<_>>()
                            .first()
                            .unwrap()
                            .trim()
                            .to_string();
                        let publication = publication_from_str(
                            split_line.last().unwrap(),
                            iter,
                            &subject,
                            &task_name,
                        )?;
                        if !processor_builders.contains_key(&processor) {
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor.to_owned());
                            builder.publications.replace(vec![publication]);
                            processor_builders.insert(processor.to_owned(), builder);
                        } else if processor_builders
                            .get(&processor)
                            .unwrap()
                            .publications
                            .is_none()
                        {
                            processor_builders
                                .get_mut(&processor)
                                .unwrap()
                                .publications
                                .replace(vec![publication]);
                        } else {
                            processor_builders
                                .get_mut(&processor)
                                .unwrap()
                                .publications
                                .as_mut()
                                .unwrap()
                                .push(publication);
                        }

                        // Update
                        task_plan_builders
                            .get_mut(&task_name)
                            .unwrap()
                            .processor_names
                            .as_mut()
                            .unwrap()
                            .push(processor.to_owned());
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
                    builder.task_name.replace(task_name.to_owned());
                    task_plan_builders.insert(task_name.to_owned(), builder);
                }

                // Update
                if task_plan_builders
                    .get(&task_name)
                    .unwrap()
                    .runtime_env_name
                    .as_ref()
                    .is_some()
                    && task_plan_builders
                        .get(&task_name)
                        .unwrap()
                        .runtime_env_name
                        .as_ref()
                        .unwrap()
                        != &runtime_env_name
                {
                    return Err(anyhow!(
                        "Parsing Error on line {iter}: {}. Runtime environment {} does not match task {} runtime environment {}.",
                        runtime_env_name,
                        task_name,
                        task_plan_builders
                            .get(&task_name)
                            .unwrap()
                            .runtime_env_name
                            .as_ref()
                            .unwrap(),
                        flowchart_lines.get(iter).unwrap()
                    ));
                } else if task_plan_builders
                    .get(&task_name)
                    .unwrap()
                    .runtime_env_name
                    .as_ref()
                    .is_none()
                {
                    task_plan_builders
                        .get_mut(&task_name)
                        .unwrap()
                        .runtime_env_name
                        .replace(runtime_env_name.to_owned());
                }
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
                let processor_type =
                    processor_from_str(split_line.last().unwrap(), iter, &processor_name)?;

                // Update
                if !processor_builders.contains_key(&processor_name) {
                    let mut builder = ProcessorBuilder::default();
                    builder.processor_name.replace(processor_name.to_owned());
                    builder.processor_type.replace(processor_type);
                    processor_builders.insert(processor_name.to_owned(), builder);
                } else if processor_builders
                    .get(&processor_name)
                    .unwrap()
                    .processor_type
                    .is_none()
                {
                    processor_builders
                        .get_mut(&processor_name)
                        .unwrap()
                        .processor_type
                        .replace(processor_type);
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
                let subscribe = match from_str_to_subscribe(split_line.last().unwrap()) {
                    Ok(subscribe) => subscribe,
                    Err(_e) => {
                        return Err(anyhow!(
                            "Parsing Error on line {iter}: {}. Subscribe policy for processor {processor_name} was not recognized.",
                            split_line.last().unwrap()
                        ));
                    }
                };
                if !processor_builders.contains_key(&processor_name) {
                    let mut builder = ProcessorBuilder::default();
                    builder.processor_name.replace(processor_name.to_owned());
                    processor_builders.insert(processor_name.to_owned(), builder);
                }

                // Update
                if processor_builders
                    .get(&processor_name)
                    .unwrap()
                    .subscribe
                    .as_ref()
                    .is_some()
                    && processor_builders
                        .get(&processor_name)
                        .unwrap()
                        .subscribe
                        .as_ref()
                        .unwrap()
                        .get_name()
                        != subscribe.get_name()
                {
                    return Err(anyhow!(
                        "Parsing Error on line {iter}: {}. Subscribe {} does not match processor {} subscribe {}.",
                        subscribe.get_name(),
                        processor_name,
                        processor_builders
                            .get(&processor_name)
                            .unwrap()
                            .subscribe
                            .as_ref()
                            .unwrap()
                            .get_name(),
                        flowchart_lines.get(iter).unwrap()
                    ));
                } else if processor_builders
                    .get(&processor_name)
                    .unwrap()
                    .subscribe
                    .as_ref()
                    .is_none()
                {
                    processor_builders
                        .get_mut(&processor_name)
                        .unwrap()
                        .subscribe
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
                    let mut builder = ProcessorBuilder::default();
                    builder.processor_name.replace(processor_name.to_owned());
                    processor_builders.insert(processor_name.to_owned(), builder);
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
        if task_names_vec.len() != task_names.len()
            || task_names_vec.clone().into_iter().collect::<HashSet<_>>() != task_names
        {
            return Err(anyhow!(
                "There is an inconsistency in the task labels {:?} and task mentions {:?}",
                task_names_vec,
                task_names
            ));
        }
        for name in task_names_vec {
            let task_plan = task_plan_builders.remove(&name).unwrap().build()?;
            task_plans.push(task_plan);
        }

        // Build the runtime environments in order
        let mut runtime_envs = Vec::new();
        if runtime_env_names_vec.len() != runtime_envs_names.len()
            || runtime_env_names_vec
                .clone()
                .into_iter()
                .collect::<HashSet<_>>()
                != runtime_envs_names
        {
            return Err(anyhow!(
                "There is an inconsistency in the runtime environment labels {:?} and runtime environment mentions {:?}",
                runtime_env_names_vec,
                runtime_envs_names
            ));
        }
        for name in runtime_env_names_vec {
            let runtime_env = RuntimeEnv::new().with_name(&name);
            runtime_envs.push(runtime_env);
        }

        // Build the processors in order
        let mut processors = Vec::new();
        if processor_names_vec.len() != processor_names.len()
            || processor_names_vec
                .clone()
                .into_iter()
                .collect::<HashSet<_>>()
                != processor_names
        {
            return Err(anyhow!(
                "There is an inconsistency in the processor labels {:?} and processor mentions {:?}",
                processor_names_vec,
                processor_names
            ));
        }
        for name in processor_names_vec {
            let builder = processor_builders.remove(&name).unwrap();
            let available_processor = AvailableProcessors::from_str(
                builder.processor_type.as_ref().unwrap().as_str(), 
                false
            )
            .unwrap();
            let processor = available_processor.build_with_builder(builder)?;
            processors.push(processor);
        }

        // Check the subjects
        if subject_names_vec.len() != subject_names.len()
            || subject_names_vec
                .clone()
                .into_iter()
                .collect::<HashSet<_>>()
                != subject_names
        {
            return Err(anyhow!(
                "There is an inconsistency in the subject labels {:?} and subject mentions {:?}",
                subject_names_vec,
                subject_names
            ));
        }
        if agent_subjects {
            check_agent_subjects(&subject_names_vec)?;
        }        

        let builder = Self::new()
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_envs(runtime_envs);
        Ok(builder)
    }

    fn with_state_from_mermaid_erdiagram(self, erdiagram: &str, agent_subjects: bool) -> Result<Self> {
        // Subjects to be collected
        let mut subjects = Vec::new();
        let mut subject_names = HashSet::new();

        // Parse the mermaid.js flowchart string
        let erdiagram_lines = erdiagram.split("\n").collect::<Vec<_>>();
        let mut iter = 0;
        if !erdiagram_lines.first().unwrap().contains("erDiagram") {
            return Err(anyhow!(
                "Parsing Error on line {iter}: {}. Unrecognized mermaid.js erDiagram type",
                erdiagram_lines.get(iter).unwrap()
            ));
        }
        while iter < erdiagram_lines.len() {
            // Check the chart type
            if erdiagram_lines.get(iter).unwrap().contains("erDiagram") {

                // Subject section
            } else if erdiagram_lines.get(iter).unwrap().contains("{") {
                // Extract the subject name
                let subject_name = erdiagram_lines
                    .get(iter)
                    .unwrap()
                    .split("{")
                    .collect::<Vec<_>>()
                    .first()
                    .unwrap()
                    .trim();
                subject_names.insert(subject_name.to_string());

                // Initialize the schema fields
                let mut fields = Vec::new();

                iter += 1;
                while iter < erdiagram_lines.len() {
                    // Check for end of subject section
                    if erdiagram_lines.get(iter).unwrap().contains("}") {
                        // Build and add the table to the subjects list
                        let schema = Arc::new(Schema::new(fields));
                        let batch = RecordBatch::new_empty(schema);
                        let table = Table::get_builder()
                            .with_record_batches(vec![batch])?
                            .with_name(subject_name)
                            .build()?;
                        subjects.push(table);
                        break;

                    // Extract the field and data type
                    } else {
                        let line = erdiagram_lines.get(iter).unwrap().trim();
                        let split_line = line.split_whitespace().collect::<Vec<_>>();

                        // Match the DataType
                        let field_name = split_line.last().unwrap().to_string();
                        let data_type = match from_str_to_data_type(split_line.first().unwrap()) {
                            Ok(data_type) => data_type,
                            Err(_e) => {
                                return Err(anyhow!(
                                    "Parsing Error on line {iter}: {}. Unrecognized data type {} in subject {subject_name} for field {field_name}. Supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, FixedSizeList, and List, ",
                                    erdiagram_lines.get(iter).unwrap(),
                                    split_line.first().unwrap()
                                ));
                            }
                        };
                        let field = Field::new(field_name, data_type, false);
                        fields.push(field);
                    }

                    iter += 1;
                }
            } else {
                return Err(anyhow!(
                    "Parsing Error on line {iter}: {}. Unrecognized line ",
                    erdiagram_lines.get(iter).unwrap()
                ));
            }

            iter += 1;
        }

        // Check the subjects
        let subjects_vec =  subjects
            .iter()
            .map(|t| t.get_name().to_string())
            .collect::<Vec<_>>();
        if subjects_vec.len() != subject_names.len()
            || subjects_vec
                .clone()
                .into_iter()
                .collect::<HashSet<_>>()
                != subject_names
        {
            return Err(anyhow!(
                "There is an inconsistency in the subject tables {:?} and subject mentions {:?}",
                subjects_vec,
                subject_names
            ));
        }
        if agent_subjects {
            check_agent_subjects(&subject_names.into_iter().collect::<Vec<_>>())?;
        }        

        Ok(self.with_state(subjects))
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{
        session::session_context_builder::test_session_context_builder::make_test_session_builder_parallel_task,
        task::task::test_task::{make_runtime_env, make_state_tables},
    };

    use crate::{
        session_plans::{
            chat_agent_session::ChatAgentSession, document_rag_session::DocumentRAGSession,
            tool_agent_session::ToolAgentSession,
        },
        session_traits::agents::CustomAgentsBuilderTrait,
    };

    use super::*;
    #[test]
    fn test_to_mermaid_flowchart() -> Result<()> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);

        // Make the builder
        let builder = make_test_session_builder_parallel_task()
            .with_runtime_envs(runtime_envs)
            .with_state(state);

        // Test to flowchart
        let mermaid_js = builder.to_mermaid_flowchart()?;
        assert_eq!(mermaid_js, "flowchart TD\n\tsubgraph task_1\n\t\tstate_1-subject-.FullTable.->processor_1-subscribe\n\t\tconfig_1-subject--FullTable-->processor_1-subscribe\n\t\tprocessor_1-subscribe-->processor_1-processor\n\t\tprocessor_1-processor-->processor_1-publish\n\t\tprocessor_1-publish--Extend-->state_1-subject\n\tend\n\tsubgraph task_2\n\t\tstate_2-subject-.FullTable.->processor_2-subscribe\n\t\tconfig_2-subject--FullTable-->processor_2-subscribe\n\t\tprocessor_2-subscribe-->processor_2-processor\n\t\tprocessor_2-processor-->processor_2-publish\n\t\tprocessor_2-publish--Extend-->state_2-subject\n\tend\n\tsubgraph task_3\n\t\tstate_3-subject-.FullTable.->processor_3-subscribe\n\t\tconfig_3-subject--FullTable-->processor_3-subscribe\n\t\tprocessor_3-subscribe-->processor_3-processor\n\t\tprocessor_3-processor-->processor_3-publish\n\t\tprocessor_3-publish--Extend-->state_3-subject\n\tend\n\tsubgraph session_1\n\t\tstate_1-subject-.LastRecordBatch.->session_1-subscribe\n\t\tstate_2-subject-.LastRecordBatch.->session_1-subscribe\n\t\tstate_3-subject-.LastRecordBatch.->session_1-subscribe\n\t\tsession_1-subscribe-->session_1-processor\n\t\tsession_1-processor-->session_1-publish\n\t\tsession_1-publish--Extend-->state_1-subject\n\t\tsession_1-publish--Extend-->state_2-subject\n\t\tsession_1-publish--Extend-->state_3-subject\n\tend\n\trt_1-rt-->task_1\n\trt_1-rt-->task_2\n\trt_1-rt-->task_3\n\trt_1-rt-->session_1\n\tprocessor_1-processor@{shape: rect, label: ArrowProcessorMock}\n\tprocessor_2-processor@{shape: rect, label: ArrowProcessorMock}\n\tprocessor_3-processor@{shape: rect, label: ArrowProcessorMock}\n\tsession_1-processor@{shape: rect, label: ArrowProcessorMock}\n\trt_1-rt@{shape: subproc, label: rt_1}\n\tconfig_1-subject@{shape: doc, label: config_1}\n\tconfig_2-subject@{shape: doc, label: config_2}\n\tconfig_3-subject@{shape: doc, label: config_3}\n\tstate_1-subject@{shape: doc, label: state_1}\n\tstate_2-subject@{shape: doc, label: state_2}\n\tstate_3-subject@{shape: doc, label: state_3}\n\tprocessor_1-publish@{shape: fork}\n\tprocessor_2-publish@{shape: fork}\n\tprocessor_3-publish@{shape: fork}\n\tsession_1-publish@{shape: fork}\n\tprocessor_1-subscribe@{shape: diamond, label: All}\n\tprocessor_2-subscribe@{shape: diamond, label: All}\n\tprocessor_3-subscribe@{shape: diamond, label: All}\n\tsession_1-subscribe@{shape: diamond, label: All}".to_string());
        Ok(())
    }

    #[test]
    fn test_to_mermaid_erdiagram() -> Result<()> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);

        // Make the builder
        let builder = make_test_session_builder_parallel_task()
            .with_runtime_envs(runtime_envs)
            .with_state(state);

        // Make the ER Diagram
        let mermaid_js = builder.to_mermaid_erdiagram()?;
        assert_eq!(mermaid_js, "erDiagram\n\tconfig_1{\n\t\tUtf8\ta\n\t\tUInt32\tb\n\t\tUInt16\tc\n\t}\n\tconfig_2{\n\t\tUtf8\ta\n\t\tUInt32\tb\n\t\tUInt16\tc\n\t}\n\tconfig_3{\n\t\tUtf8\ta\n\t\tUInt32\tb\n\t\tUInt16\tc\n\t}\n\tstate_1{\n\t\tUInt32\tid\n\t\tUtf8\tcollection\n\t\tUtf8\ttitle\n\t\tUtf8\ttext\n\t\tUtf8\tmetadata\n\t\tFloat32\tscore\n\t\tFixedSizeList-Float32-8\tembedding\n\t}\n\tstate_2{\n\t\tUInt32\tid\n\t\tUtf8\tcollection\n\t\tUtf8\ttitle\n\t\tUtf8\ttext\n\t\tUtf8\tmetadata\n\t\tFloat32\tscore\n\t\tFixedSizeList-Float32-8\tembedding\n\t}\n\tstate_3{\n\t\tUInt32\tid\n\t\tUtf8\tcollection\n\t\tUtf8\ttitle\n\t\tUtf8\ttext\n\t\tUtf8\tmetadata\n\t\tFloat32\tscore\n\t\tFixedSizeList-Float32-8\tembedding\n\t}".to_string());
        Ok(())
    }

    #[test]
    fn test_from_mermaid_parallel_task() -> Result<()> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);

        // Make the builder
        let builder = make_test_session_builder_parallel_task()
            .with_runtime_envs(runtime_envs)
            .with_state(state);

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart()?;
        let erdiagram = builder.to_mermaid_erdiagram()?;

        // Remake the builder
        let builder_test = SessionContextBuilder::from_mermaid_flowchart(&flowchart, false)?
            .with_state_from_mermaid_erdiagram(&erdiagram, false)?;

        // Test that the names match
        let mut test = builder_test
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_subject_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder.get_subject_names().into_iter().collect::<Vec<_>>();
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
    fn test_from_mermaid_chat_agent_session() -> Result<()> {
        // initialize the session
        let builder = ChatAgentSession::new_with_session_name("session_1").build();

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart()?;
        let erdiagram = builder.to_mermaid_erdiagram()?;

        // Remake the builder
        let builder_test = SessionContextBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_state_from_mermaid_erdiagram(&erdiagram, true)?;

        // Test that the names match
        let mut test = builder_test
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_subject_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder.get_subject_names().into_iter().collect::<Vec<_>>();
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
    fn test_from_mermaid_doc_rag_session() -> Result<()> {
        // initialize the session
        let builder = DocumentRAGSession::new_with_session_name("session_1").build();

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart()?;
        let erdiagram = builder.to_mermaid_erdiagram()?;

        // Remake the builder
        let builder_test = SessionContextBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_state_from_mermaid_erdiagram(&erdiagram, true)?;

        // Test that the names match
        let mut test = builder_test
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_subject_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder.get_subject_names().into_iter().collect::<Vec<_>>();
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
    fn test_from_mermaid_tool_agent_session() -> Result<()> {
        // initialize the session
        let builder = ToolAgentSession::new_with_session_name("session_1").build();

        // Make the flowchart and erdiagram
        let flowchart = builder.to_mermaid_flowchart()?;
        let erdiagram = builder.to_mermaid_erdiagram()?;

        // Remake the builder
        let builder_test = SessionContextBuilder::from_mermaid_flowchart(&flowchart, true)?
            .with_state_from_mermaid_erdiagram(&erdiagram, true)?;

        // Test that the names match
        let mut test = builder_test
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_processor_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder
            .get_runtime_env_names()
            .into_iter()
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(test, expected);
        let mut test = builder_test
            .get_subject_names()
            .into_iter()
            .collect::<Vec<_>>();
        test.sort();
        let mut expected = builder.get_subject_names().into_iter().collect::<Vec<_>>();
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
}
