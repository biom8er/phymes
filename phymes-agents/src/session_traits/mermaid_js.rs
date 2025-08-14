use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow::{array::RecordBatch, datatypes::{DataType, Field, Schema}};
use phymes_core::{metrics::{HashMap, HashSet}, session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, runtime_env::{RuntimeEnv, RuntimeEnvTrait}, session_context_builder::{SessionContextBuilder, SessionContextBuilderTrait, TaskPlan}}, table::{arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}, arrow_table_publish::ArrowTablePublish, arrow_table_subscribe::{AllTableNamesSubscribe, AllTableSchemasSubscribe, AlwaysSubscribe, AnyTableNameSubscribe, AnyTableSchemaSubscribe, ArrowTableSubscribe, SubscribeTrait}}, task::arrow_processor::{ArrowProcessorEcho, ArrowProcessorTrait}};

pub trait SessionContextBuilderMermaidTrait {
    /// Make a mermaid.js flowchart of the session
    fn to_mermaid_flowchart(&self) -> Result<String>;

    /// Make a mermaid.js erDiagram of the session
    fn to_mermaid_erdiagram(&self) -> Result<String>;

    /// Create a session builder from a mermaid flowchart
    fn from_mermaid_flowchart(flowchart: &str) -> Result<Self> where Self: Sized;
    
    /// Create the state from a mermaid ER Diagram
    fn with_state_from_mermaid_erdiagram(self, erdiagram: &str) -> Result<Self> where Self: Sized;
}

impl SessionContextBuilderMermaidTrait for SessionContextBuilder { 
    fn to_mermaid_flowchart(&self) -> Result<String> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the Mermaid Flowchart."));
        }
        if self.processors.is_none() {
            return Err(anyhow!("Add processors before making the Mermaid Flowchart."));
        }
        if self.runtime_envs.is_none() {
            return Err(anyhow!("Add runtime environments before making the Mermaid Flowchart."));
        }
        if self.state.is_none() {
            return Err(anyhow!("Add state subjects before making the Mermaid Flowchart."));
        }        

        // Entities with expanded shape/label attributes that will be appended to flowchart
        let mut processors_vec = Vec::new();
        let mut sorted_processor_names = self.get_processor_names().into_iter().collect::<Vec<_>>();
        sorted_processor_names.sort();
        for processor_name in sorted_processor_names {
            processors_vec.push(format!("\t{processor_name}-processor@{{shape:rect,label:{processor_name}}}"));
        }

        let mut subjects_vec = Vec::new();
        let mut sorted_subject_names = self.get_subject_names().into_iter().collect::<Vec<_>>();
        sorted_subject_names.sort();
        for subject_name in sorted_subject_names {
            subjects_vec.push(format!("\t{subject_name}-subject@{{shape:doc,label:{subject_name}}}"));
        }

        let mut runtime_envs_vec = Vec::new();
        let mut sorted_runtime_env_names = self.get_runtime_env_names().into_iter().collect::<Vec<_>>();
        sorted_runtime_env_names.sort();
        for runtime_env_name in sorted_runtime_env_names {
            runtime_envs_vec.push(format!("\t{runtime_env_name}-rt@{{shape:subproc,label:{runtime_env_name}}}"));
        }

        // Subgraphs
        let mut tasks_vec = Vec::new();
        let mut subscriptions_vec = Vec::new();
        let mut publications_vec = Vec::new();
        let mut runtime_envs_to_tasks_vec = Vec::new();
        for task in self.tasks.as_ref().unwrap().iter() {
            tasks_vec.push(format!("\tsubgraph {}", task.task_name));
            runtime_envs_to_tasks_vec.push(format!("\t{}-rt-->{}", task.runtime_env_name, task.task_name));

            // Iterate through each processor
            for processor_name in task.processor_names.iter() {
                for processor in self.processors.as_ref().unwrap().iter() {
                    if processor_name == processor.get_name() {

                        // Subscriptions
                        let subscribe = processor.get_subscribe().get_name();
                        subscriptions_vec.push(format!("\t{processor_name}-subscribe@{{shape:diamond,label:{subscribe}}}"));
                        for subscription in processor.get_subscriptions().iter() {
                            if subscription.is_update() {
                                tasks_vec.push(format!("\t\t{}-subject-.{}.->{processor_name}-subscribe", subscription.get_table_name(), subscription.get_short_name()));
                            } else {
                                tasks_vec.push(format!("\t\t{}-subject--{}-->{processor_name}-subscribe", subscription.get_table_name(), subscription.get_short_name()));
                            }                            
                        }
                        tasks_vec.push(format!("\t\t{processor_name}-subscribe-->{processor_name}-processor"));

                        // Publications
                        publications_vec.push(format!("\t{processor_name}-publish@{{shape:fork}}"));
                        tasks_vec.push(format!("\t\t{processor_name}-processor-->{processor_name}-publish"));
                        for publication in processor.get_publications().iter() {
                            tasks_vec.push(format!("\t\t{processor_name}-publish--{}-->{}-subject", publication.get_short_name(), publication.get_table_name()));
                        }

                        break;
                    }
                }                
            }
            tasks_vec.push(format!("\tend"));
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
            return Err(anyhow!("Add state subjects before making the Mermaid ER Diagram."));
        }

        // Extract the subjects
        let mut subjects = Vec::new();
        let mut sorted_map = self.state.as_ref().unwrap().iter().collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.get_name().cmp(b.get_name()));
        for subject in sorted_map {
            subjects.push(format!("\t{}{{", subject.get_name()));
            for field in subject.get_schema().fields().iter() {
                match field.data_type() {
                    DataType::FixedSizeList(f, s) => {
                        subjects.push(format!("\t\tFixedSizeList-{}-{}\t{}", f.data_type(), s, field.name()));
                    }
                    DataType::List(f) => {
                        subjects.push(format!("\t\tList-{}\t{}", f.data_type(), field.name()));
                    }
                    _ => {
                        subjects.push(format!("\t\t{}\t{}", field.data_type(), field.name()));
                    }
                }
            }
            subjects.push("\t}".to_string());
        }        

        // Create the final mermaid.js flowchart script
        let mut mermaid_js = vec!["erDiagram".to_string()];
        mermaid_js.extend(subjects);
        Ok(mermaid_js.join("\n"))
    }

    fn from_mermaid_flowchart(flowchart: &str) -> Result<Self> {
        // Builders for tasks and processors
        #[derive(Default)]
        struct TaskPlanBuilder {
            pub task_name: Option<String>,
            pub runtime_env_name: Option<String>,
            pub processor_names: Option<Vec<String>>,
        }
        impl TaskPlanBuilder {
            pub fn build(mut self) -> Result<TaskPlan> {
                if self.task_name.is_none() {
                    return Err(anyhow!("Missing task name"));
                } else if self.runtime_env_name.as_ref().is_none() {
                    return Err(anyhow!("Missing runtime_env_name for task {}", self.task_name.as_ref().unwrap()));
                } else if self.processor_names.as_ref().is_none() {
                    return Err(anyhow!("Missing processor_names for task {}", self.task_name.as_ref().unwrap()));
                }

                let task_plan = TaskPlan {
                    task_name: self.task_name.take().unwrap(),
                    runtime_env_name: self.runtime_env_name.take().unwrap(),
                    processor_names: self.processor_names.take().unwrap(),
                };
                Ok(task_plan)
            }
        }
        #[derive(Default)]
        struct ProcessorBuilder {
            pub publications: Option<Vec<ArrowTablePublish>>,
            pub subscriptions: Option<Vec<ArrowTableSubscribe>>,
            pub subscribe: Option<Box<dyn SubscribeTrait>>,
            pub processor_name: Option<String>,
        }
        impl ProcessorBuilder {
            pub fn build(mut self) -> Result<Arc<dyn ArrowProcessorTrait>> {                
                if self.processor_name.as_ref().is_none() {
                    return Err(anyhow!("Missing processor name"));
                } else if self.publications.as_ref().is_none() {
                    return Err(anyhow!("Missing publications for processor {}", self.processor_name.as_ref().unwrap()));
                } else if self.subscriptions.as_ref().is_none() {
                    return Err(anyhow!("Missing subscriptions for processor {}", self.processor_name.as_ref().unwrap()));
                } else if self.subscribe.as_ref().is_none() {
                    return Err(anyhow!("Missing subscribe for processor {}", self.processor_name.as_ref().unwrap()));
                }
                
                let processor = ArrowProcessorEcho::new_arc_with_pub_sub(
                    &self.processor_name.take().unwrap(), 
                    &self.publications.take().unwrap(), 
                    &self.subscriptions.take().unwrap(), 
                    self.subscribe.take().unwrap());
                Ok(processor)
            }
        }

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
        let subscription_from_str = |line: &str, iter: usize, subject: &str, task: &str| -> Result<ArrowTableSubscribe> {
            if line.contains("-.") & line.contains(".->") & line.contains("FullTable") {
                Ok(ArrowTableSubscribe::OnUpdateFullTable { table_name: subject.to_string() })
            } else if line.contains("--") & line.contains("-->") & line.contains("FullTable") {
                Ok(ArrowTableSubscribe::AlwaysFullTable { table_name: subject.to_string() })
            } else if line.contains("-.") & line.contains(".->") & line.contains("LastRecordBatch") {
                Ok(ArrowTableSubscribe::OnUpdateLastRecordBatch { table_name: subject.to_string() })
            } else if line.contains("--") & line.contains("-->") & line.contains("LastRecordBatch") {
                Ok(ArrowTableSubscribe::AlwaysLastRecordBatch { table_name: subject.to_string() })
            } else if line.contains("None") {
                Ok(ArrowTableSubscribe::None {})
            } else {
                Err(anyhow!("Parsing Error on line {iter}: {line}. Variant for ArrowTableSubscribe with subject {subject} for task {task} was not recognized."))
            }
        };
        let publication_from_str = |line: &str, iter: usize, subject: &str, task: &str| -> Result<ArrowTablePublish> {
            if line.contains("--") & line.contains("-->") & line.contains("Extend") {
                Ok(ArrowTablePublish::Extend { table_name: subject.to_string() })
            } else if line.contains("--") & line.contains("-->") & line.contains("Replace") {
                Ok(ArrowTablePublish::Replace { table_name: subject.to_string() })
            } else if line.contains("--") & line.contains("-->") & line.contains("ReplaceLast") {
                Ok(ArrowTablePublish::ReplaceLast { table_name: subject.to_string() })
            } else if line.contains("None") {
                Ok(ArrowTablePublish::None {})
            } else {
                Err(anyhow!("Parsing Error on line {iter}: {line}. Variant for ArrowTablePublish with subject {subject} for task {task} was not recognized."))
            }
        };
        let subscribe_from_str = |line: &str, iter: usize, processor: &str| -> Result<Box<dyn SubscribeTrait>> {
            if line.contains("All") {
                Ok(AllTableNamesSubscribe::new_box())
            } else if line.contains("Any") {
                Ok(AnyTableNameSubscribe::new_box())
            } else if line.contains("AllSchemas") {
                Ok(AllTableSchemasSubscribe::new_box())
            } else if line.contains("AnySchema") {
                Ok(AnyTableSchemaSubscribe::new_box())
            } else if line.contains("Always") {
                Ok(AlwaysSubscribe::new_box())
            } else {
                Err(anyhow!("Parsing Error on line {iter}: {line}. Subscribe policy for processor {processor} was not recognized."))
            }
        };

        // Parse the mermaid.js flowchart string
        let flowchart_lines = flowchart.split("\n").collect::<Vec<_>>();
        let mut iter = 0;                      
        if !flowchart_lines.first().unwrap().contains("flowchart") {
            return Err(anyhow!("Parsing Error on line {iter}: {}. Unrecognized mermaid.js flowchart type", flowchart_lines.get(iter).unwrap()));
        }
        while iter < flowchart_lines.len() {

            // Check the chart type
            if flowchart_lines.get(iter).unwrap().contains("flowchart") {

            // Task section
            } else if flowchart_lines.get(iter).unwrap().contains("subgraph") {

                // Start building the task plan
                let task_name = flowchart_lines.get(iter).unwrap().split("subgraph").last().unwrap().trim().to_string();
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
                        let split_line = flowchart_lines.get(iter).unwrap().split("-subject").collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. There are two subjects in task {task_name}", flowchart_lines.get(iter).unwrap()));
                        }
                        let subject = split_line.first().unwrap().trim().to_string();
                        let subscription = subscription_from_str(split_line.last().unwrap(), iter, &subject, &task_name)?;

                        // Check the processor name
                        let split_line = split_line.last().unwrap().split("->").collect::<Vec<_>>().last().unwrap().split("-subscribe").collect::<Vec<_>>();
                        let processor = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor) {
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor.to_owned());
                            builder.subscriptions.replace(vec![subscription]);
                            processor_builders.insert(processor.to_owned(), builder);
                        } else if processor_builders.get(&processor).unwrap().subscriptions.is_none() {
                            processor_builders.get_mut(&processor).unwrap().subscriptions.replace(vec![subscription]);
                        } else {
                            processor_builders.get_mut(&processor).unwrap().subscriptions.as_mut().unwrap().push(subscription);
                        }                        
                        
                        // Update
                        if task_plan_builders.get(&task_name).unwrap().processor_names.is_none() {
                            task_plan_builders.get_mut(&task_name).unwrap().processor_names.replace(vec![processor.to_owned()]);
                        } else {
                            task_plan_builders.get_mut(&task_name).unwrap().processor_names.as_mut().unwrap().push(processor.to_owned());
                        }                        
                        processor_names.insert(processor);
                        subject_names.insert(subject);

                    // Subscribe, Processor triple
                    // e.g., processor_1-subscribe-->processor_1-processor
                    } else if flowchart_lines.get(iter).unwrap().contains("-subscribe")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
                        & flowchart_lines.get(iter).unwrap().contains("-processor")
                    {
                        // Check the processor name
                        let split_line = flowchart_lines.get(iter).unwrap().split("-subscribe").collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. There are two subscribes in task {task_name}", flowchart_lines.get(iter).unwrap()));
                        }
                        let processor_1 = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor_1) {
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor_1.to_owned());
                            processor_builders.insert(processor_1.to_owned(), builder);
                        }

                        // Check the processor name
                        let split_line = split_line.last().unwrap().split("-->").collect::<Vec<_>>().last().unwrap().split("-processor").collect::<Vec<_>>();
                        let processor_2 = split_line.first().unwrap().trim().to_string();
                        if processor_1 != processor_2 {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. Processor name {processor_1} does not match processor name {processor_2} in task {task_name}", flowchart_lines.get(iter).unwrap()));
                        }

                        // Update
                        task_plan_builders.get_mut(&task_name).unwrap().processor_names.as_mut().unwrap().push(processor_1.to_owned());
                        processor_names.insert(processor_1);

                    // Processor, Publish triple
                    // e.g., processor_1-processor-->processor_1-publish
                    } else if flowchart_lines.get(iter).unwrap().contains("-processor")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
                        & flowchart_lines.get(iter).unwrap().contains("-publish")
                    {
                        // Check the processor name
                        let split_line = flowchart_lines.get(iter).unwrap().split("-processor").collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. There are two subscribes in task {task_name}", flowchart_lines.get(iter).unwrap()));
                        }
                        let processor_1 = split_line.first().unwrap().trim().to_string();
                        if !processor_builders.contains_key(&processor_1) {
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor_1.to_owned());
                            processor_builders.insert(processor_1.to_owned(), builder);
                        }
                        
                        // Check the processor name
                        let split_line = split_line.last().unwrap().split("-->").collect::<Vec<_>>().last().unwrap().split("-publish").collect::<Vec<_>>();
                        let processor_2 = split_line.first().unwrap().trim().to_string();
                        if processor_1 != processor_2 {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. Processor name {processor_1} does not match processor name {processor_2} in task {task_name}", flowchart_lines.get(iter).unwrap()));
                        }

                        // Update
                        task_plan_builders.get_mut(&task_name).unwrap().processor_names.as_mut().unwrap().push(processor_1.to_owned());
                        processor_names.insert(processor_1);

                    // Publish, Publication, Subject triple
                    // e.g., processor_1-publish--Extend-->state_1-subject
                    } else if flowchart_lines.get(iter).unwrap().contains("-publish")
                        & flowchart_lines.get(iter).unwrap().contains("-->")
                        & flowchart_lines.get(iter).unwrap().contains("-subject")
                    {
                        // Check the processor name
                        let split_line = flowchart_lines.get(iter).unwrap().split("-publish").collect::<Vec<_>>();
                        if split_line.len() > 2 {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. There are two subscribes in task {task_name}", flowchart_lines.get(iter).unwrap()));
                        }
                        let processor = split_line.first().unwrap().trim().to_string();

                        // Extract the publication
                        let subject = split_line.last().unwrap().split("-->").collect::<Vec<_>>()
                            .last().unwrap().split("-subject").collect::<Vec<_>>()
                            .first().unwrap().trim().to_string();
                        let publication = publication_from_str(split_line.last().unwrap(), iter, &subject, &task_name)?;                        
                        if !processor_builders.contains_key(&processor) {
                            let mut builder = ProcessorBuilder::default();
                            builder.processor_name.replace(processor.to_owned());
                            builder.publications.replace(vec![publication]);
                            processor_builders.insert(processor.to_owned(), builder);
                        } else if processor_builders.get(&processor).unwrap().publications.is_none() {
                            processor_builders.get_mut(&processor).unwrap().publications.replace(vec![publication]);
                        } else {
                            processor_builders.get_mut(&processor).unwrap().publications.as_mut().unwrap().push(publication);
                        } 

                        // Update
                        task_plan_builders.get_mut(&task_name).unwrap().processor_names.as_mut().unwrap().push(processor.to_owned());
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
                        return Err(anyhow!("Parsing Error on line {iter}: {}. Unsupported arrow type in subgraph {task_name}. Only --> and .-> arrows are supported in PHYMES.", flowchart_lines.get(iter).unwrap()));

                    // Unrecognized qualifier
                    } else if !flowchart_lines.get(iter).unwrap().contains("subject")
                        | !flowchart_lines.get(iter).unwrap().contains("subscribe")
                        | !flowchart_lines.get(iter).unwrap().contains("processor")
                        | !flowchart_lines.get(iter).unwrap().contains("publish")
                    {
                        return Err(anyhow!("Parsing Error on line {iter}: {}. Unsupported processor or subject qualifier in subgraph {task_name}. Only -subject, -subscribe, -processor, and -publish qualifiers are supported in PHYMES.", flowchart_lines.get(iter).unwrap()));

                    // Any others
                    } else {
                        return Err(anyhow!("Parsing Error on line {iter}: {}. Unrecognized line in subgraph {task_name}", flowchart_lines.get(iter).unwrap()));                   
                    }
                    iter += 1;                    
                }
                
            // Extract out the task runtime environments
            } else if flowchart_lines.get(iter).unwrap().contains("-rt-->") {

                // Extract the runtime and task names
                let split_line = flowchart_lines.get(iter).unwrap().split("-rt-->").collect::<Vec<_>>();
                if split_line.len() > 2 {
                    return Err(anyhow!("Parsing Error on line {iter}: {}. There are two runtime environments", flowchart_lines.get(iter).unwrap()));
                }
                let runtime_env_name = split_line.first().unwrap().trim().to_string();
                let task_name = split_line.last().unwrap().trim().to_string();
                if !task_plan_builders.contains_key(&task_name) {
                    let mut builder = TaskPlanBuilder::default();
                    builder.task_name.replace(task_name.to_owned());
                    task_plan_builders.insert(task_name.to_owned(), builder);
                }
                
                // Update
                if task_plan_builders.get(&task_name).unwrap().runtime_env_name.as_ref().is_some()
                    && task_plan_builders.get(&task_name).unwrap().runtime_env_name.as_ref().unwrap() != &runtime_env_name
                {
                    return Err(anyhow!("Parsing Error on line {iter}: {}. Runtime environment {} does not match task {} runtime environment {}.", 
                        runtime_env_name,
                        task_name,
                        task_plan_builders.get(&task_name).unwrap().runtime_env_name.as_ref().unwrap(),
                        flowchart_lines.get(iter).unwrap()));
                } else if task_plan_builders.get(&task_name).unwrap().runtime_env_name.as_ref().is_none() {
                    task_plan_builders.get_mut(&task_name).unwrap().runtime_env_name.replace(runtime_env_name.to_owned());
                }
                task_names.insert(task_name);
                runtime_envs_names.insert(runtime_env_name);

            // Extract out the runtime environments
            } else if flowchart_lines.get(iter).unwrap().contains("-rt@{shape:subproc,") {

                // Extract the runtime and task names
                let split_line = flowchart_lines.get(iter).unwrap().split("-rt@{shape:subproc,").collect::<Vec<_>>();
                let runtime_env_name = split_line.first().unwrap().trim().to_string();
                
                // Update
                runtime_env_names_vec.push(runtime_env_name.to_owned());
                
            // Extract out the processors
            } else if flowchart_lines.get(iter).unwrap().contains("-processor@{shape:rect,") {

                // Extract the processor name
                let split_line = flowchart_lines.get(iter).unwrap().split("-processor@{shape:rect,").collect::<Vec<_>>();
                let processor_name = split_line.first().unwrap().trim().to_string();
                
                // Update
                processor_names_vec.push(processor_name.to_owned());

            // Extract out the subjects
            } else if flowchart_lines.get(iter).unwrap().contains("-subject@{shape:doc") {

                // Extract the subject name
                let split_line = flowchart_lines.get(iter).unwrap().split("-subject@{shape:doc").collect::<Vec<_>>();
                let subject_name = split_line.first().unwrap().trim().to_string();
                
                // Update
                subject_names_vec.push(subject_name.to_owned());

            // Extract out the subscribe
            } else if flowchart_lines.get(iter).unwrap().contains("-subscribe@{shape:diamond,label:") {

                // Extract the processor name
                let split_line = flowchart_lines.get(iter).unwrap().split("-subscribe@{shape:diamond,label:").collect::<Vec<_>>();
                let processor_name = split_line.first().unwrap().trim().to_string();
                let subscribe = subscribe_from_str(split_line.last().unwrap(), iter, &processor_name)?;
                if !processor_builders.contains_key(&processor_name) {
                    let mut builder = ProcessorBuilder::default();
                    builder.processor_name.replace(processor_name.to_owned());
                    processor_builders.insert(processor_name.to_owned(), builder);
                }
                
                // Update
                if processor_builders.get(&processor_name).unwrap().subscribe.as_ref().is_some()
                    && processor_builders.get(&processor_name).unwrap().subscribe.as_ref().unwrap().get_name() != subscribe.get_name()
                {
                    return Err(anyhow!("Parsing Error on line {iter}: {}. Subscribe {} does not match processor {} subscribe {}.", 
                        subscribe.get_name(),
                        processor_name,
                        processor_builders.get(&processor_name).unwrap().subscribe.as_ref().unwrap().get_name(),
                        flowchart_lines.get(iter).unwrap()));
                } else if processor_builders.get(&processor_name).unwrap().subscribe.as_ref().is_none() {
                    processor_builders.get_mut(&processor_name).unwrap().subscribe.replace(subscribe);
                }
                processor_names.insert(processor_name);

            // Extract out the publish
            } else if flowchart_lines.get(iter).unwrap().contains("-publish@{shape:fork}") {

                // Extract the processor name
                let split_line = flowchart_lines.get(iter).unwrap().split("-publish@{shape:fork}").collect::<Vec<_>>();
                let processor_name = split_line.first().unwrap().trim().to_string();
                if !processor_builders.contains_key(&processor_name) {
                    let mut builder = ProcessorBuilder::default();
                    builder.processor_name.replace(processor_name.to_owned());
                    processor_builders.insert(processor_name.to_owned(), builder);
                }

                // Update
                processor_names.insert(processor_name);

            } else {
                return Err(anyhow!("Parsing Error on line {iter}: {}. Unrecognized line ", flowchart_lines.get(iter).unwrap()));
            }
            iter += 1;   
        }

        // Build the task plans in order
        let mut task_plans = Vec::new();
        if task_names_vec.len() != task_names.len() || task_names_vec.clone().into_iter().collect::<HashSet<_>>() != task_names {
            return Err(anyhow!("There is an inconsistency in the task labels {:?} and task mentions {:?}", task_names_vec, task_names));
        }
        for name in task_names_vec {
            let task_plan = task_plan_builders.remove(&name).unwrap().build()?;
            task_plans.push(task_plan);
        }

        // Build the runtime environments in order
        let mut runtime_envs = Vec::new();
        if runtime_env_names_vec.len() != runtime_envs_names.len() || runtime_env_names_vec.clone().into_iter().collect::<HashSet<_>>() != runtime_envs_names {
            return Err(anyhow!("There is an inconsistency in the runtime environment labels {:?} and runtime environment mentions {:?}", runtime_env_names_vec, runtime_envs_names));
        }
        for name in runtime_env_names_vec {            
            let runtime_env = RuntimeEnv::new().with_name(&name);
            runtime_envs.push(runtime_env);
        }

        // Build the processors in order
        let mut processors = Vec::new();
        if processor_names_vec.len() != processor_names.len() || processor_names_vec.clone().into_iter().collect::<HashSet<_>>() != processor_names {
            return Err(anyhow!("There is an inconsistency in the processor labels {:?} and processor mentions {:?}", processor_names_vec, processor_names));
        }
        for name in processor_names_vec {
            let processor = processor_builders.remove(&name).unwrap().build()?;
            processors.push(processor);
        }

        // Check the subjects
        if subject_names_vec.len() != subject_names.len() || subject_names_vec.clone().into_iter().collect::<HashSet<_>>() != subject_names {
            return Err(anyhow!("There is an inconsistency in the subject labels {:?} and subject mentions {:?}", subject_names_vec, subject_names));
        }

        let builder = Self::new()
            .with_tasks(task_plans)
            .with_processors(processors)
            .with_runtime_envs(runtime_envs);
        Ok(builder)
    }

    fn with_state_from_mermaid_erdiagram(self, erdiagram: &str) -> Result<Self> {
        // Subjects to be collected
        let mut subjects = Vec::new();
        let mut subject_names = HashSet::new();

        // Supported List types

        // Parse the mermaid.js flowchart string
        let erdiagram_lines = erdiagram.split("\n").collect::<Vec<_>>();
        let mut iter = 0;                      
        if !erdiagram_lines.first().unwrap().contains("erDiagram") {
            return Err(anyhow!("Parsing Error on line {iter}: {}. Unrecognized mermaid.js erDiagram type", erdiagram_lines.get(iter).unwrap()));
        }
        while iter < erdiagram_lines.len() {

            // Check the chart type
            if erdiagram_lines.get(iter).unwrap().contains("erDiagram") {

            // Subject section
            } else if erdiagram_lines.get(iter).unwrap().contains("{") {

                // Extract the subject name
                let subject_name = erdiagram_lines.get(iter).unwrap().split("{").collect::<Vec<_>>().first().unwrap().trim();
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
                        let table = ArrowTable::get_builder()
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
                        let data_type = split_line.first().unwrap();
                        if *data_type == &DataType::UInt8.to_string() {
                            let field = Field::new(field_name, DataType::UInt8, false);
                            fields.push(field);
                        } else if *data_type == &DataType::UInt16.to_string() {
                            let field = Field::new(field_name, DataType::UInt16, false);
                            fields.push(field);
                        } else if *data_type == &DataType::UInt32.to_string() {
                            let field = Field::new(field_name, DataType::UInt32, false);
                            fields.push(field);
                        } else if *data_type == &DataType::Int64.to_string() {
                            let field = Field::new(field_name, DataType::Int64, false);
                            fields.push(field);
                        } else if *data_type == &DataType::Float32.to_string() {
                            let field = Field::new(field_name, DataType::Float32, false);
                            fields.push(field);
                        } else if *data_type == &DataType::Float64.to_string() {
                            let field = Field::new(field_name, DataType::Float64, false);
                            fields.push(field);
                        } else if *data_type == &DataType::Utf8.to_string() {
                            let field = Field::new(field_name, DataType::Utf8, false);
                            fields.push(field);
                        } else if data_type.contains("FixedSizeList-UInt8-") {
                            let size = data_type.split("FixedSizeList-UInt8-").last().unwrap().trim().parse::<i32>().unwrap();                       
                            let list_data_type = DataType::FixedSizeList(
                                Arc::new(Field::new_list_field(DataType::UInt8, false)),
                                size,
                            );
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("FixedSizeList-UInt32-") {
                            let size = data_type.split("FixedSizeList-UInt32-").last().unwrap().trim().parse::<i32>().unwrap();                       
                            let list_data_type = DataType::FixedSizeList(
                                Arc::new(Field::new_list_field(DataType::UInt32, false)),
                                size,
                            );
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("FixedSizeList-Int64-") {
                            let size = data_type.split("FixedSizeList-Int64-").last().unwrap().trim().parse::<i32>().unwrap();                       
                            let list_data_type = DataType::FixedSizeList(
                                Arc::new(Field::new_list_field(DataType::Int64, false)),
                                size,
                            );
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("FixedSizeList-Float32-") {
                            let size = data_type.split("FixedSizeList-Float32-").last().unwrap().trim().parse::<i32>().unwrap();                       
                            let list_data_type = DataType::FixedSizeList(
                                Arc::new(Field::new_list_field(DataType::Float32, false)),
                                size,
                            );
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("FixedSizeList-Float64-") {
                            let size = data_type.split("FixedSizeList-Float64-").last().unwrap().trim().parse::<i32>().unwrap();                       
                            let list_data_type = DataType::FixedSizeList(
                                Arc::new(Field::new_list_field(DataType::Float64, false)),
                                size,
                            );
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("FixedSizeList-Utf8-") {
                            let size = data_type.split("FixedSizeList-Utf8-").last().unwrap().trim().parse::<i32>().unwrap();                       
                            let list_data_type = DataType::FixedSizeList(
                                Arc::new(Field::new_list_field(DataType::Utf8, false)),
                                size,
                            );
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("List-UInt8") {                      
                            let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::UInt8, false)));
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("List-UInt32") {                     
                            let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false)));
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("List-Int64") {                      
                            let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false)));
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("List-Float32") {                      
                            let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)));
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("List-Float64") {                      
                            let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Float64, false)));
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else if data_type.contains("List-Utf8") {                       
                            let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, false)));
                            let field = Field::new(field_name, list_data_type, false);
                            fields.push(field);
                        } else {
                            return Err(anyhow!("Parsing Error on line {iter}: {}. Unrecognized data type {data_type} in subject {subject_name} for field {field_name}. Supported data types are UInt8, UInt32, Int64, Float32, Float64, Utf8, FixedSizeList, and List, ", erdiagram_lines.get(iter).unwrap()));
                        }
                    }
                    
                    iter += 1; 
                }

            } else {
                return Err(anyhow!("Parsing Error on line {iter}: {}. Unrecognized line ", erdiagram_lines.get(iter).unwrap()));
            }

            iter += 1;
        }

        // Check the subjects
        if subjects.len() != subject_names.len() || subjects.iter().map(|t| t.get_name().to_string()).collect::<HashSet<_>>() != subject_names {
            return Err(anyhow!("There is an inconsistency in the subject tables {:?} and subject mentions {:?}", subjects.iter().map(|t| t.get_name().to_string()).collect::<HashSet<_>>(), subject_names));
        }

        Ok(self.with_state(subjects))
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{session::session_context_builder::test_session_context_builder::make_test_session_builder_parallel_task, task::arrow_task::test_task::{make_runtime_env, make_state_tables}};

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
        assert_eq!(mermaid_js, "flowchart TD\n\tsubgraph task_1\n\t\tstate_1-subject-.FullTable.->processor_1-subscribe\n\t\tconfig_1-subject--FullTable-->processor_1-subscribe\n\t\tprocessor_1-subscribe-->processor_1-processor\n\t\tprocessor_1-processor-->processor_1-publish\n\t\tprocessor_1-publish--Extend-->state_1-subject\n\tend\n\tsubgraph task_2\n\t\tstate_2-subject-.FullTable.->processor_2-subscribe\n\t\tconfig_2-subject--FullTable-->processor_2-subscribe\n\t\tprocessor_2-subscribe-->processor_2-processor\n\t\tprocessor_2-processor-->processor_2-publish\n\t\tprocessor_2-publish--Extend-->state_2-subject\n\tend\n\tsubgraph task_3\n\t\tstate_3-subject-.FullTable.->processor_3-subscribe\n\t\tconfig_3-subject--FullTable-->processor_3-subscribe\n\t\tprocessor_3-subscribe-->processor_3-processor\n\t\tprocessor_3-processor-->processor_3-publish\n\t\tprocessor_3-publish--Extend-->state_3-subject\n\tend\n\tsubgraph session_1\n\t\tstate_1-subject-.LastRecordBatch.->session_1-subscribe\n\t\tstate_2-subject-.LastRecordBatch.->session_1-subscribe\n\t\tstate_3-subject-.LastRecordBatch.->session_1-subscribe\n\t\tsession_1-subscribe-->session_1-processor\n\t\tsession_1-processor-->session_1-publish\n\t\tsession_1-publish--Extend-->state_1-subject\n\t\tsession_1-publish--Extend-->state_2-subject\n\t\tsession_1-publish--Extend-->state_3-subject\n\tend\n\trt_1-rt-->task_1\n\trt_1-rt-->task_2\n\trt_1-rt-->task_3\n\trt_1-rt-->session_1\n\tprocessor_1-processor@{shape:rect,label:processor_1}\n\tprocessor_2-processor@{shape:rect,label:processor_2}\n\tprocessor_3-processor@{shape:rect,label:processor_3}\n\tsession_1-processor@{shape:rect,label:session_1}\n\trt_1-rt@{shape:subproc,label:rt_1}\n\tconfig_1-subject@{shape:doc,label:config_1}\n\tconfig_2-subject@{shape:doc,label:config_2}\n\tconfig_3-subject@{shape:doc,label:config_3}\n\tstate_1-subject@{shape:doc,label:state_1}\n\tstate_2-subject@{shape:doc,label:state_2}\n\tstate_3-subject@{shape:doc,label:state_3}\n\tprocessor_1-publish@{shape:fork}\n\tprocessor_2-publish@{shape:fork}\n\tprocessor_3-publish@{shape:fork}\n\tsession_1-publish@{shape:fork}\n\tprocessor_1-subscribe@{shape:diamond,label:All}\n\tprocessor_2-subscribe@{shape:diamond,label:All}\n\tprocessor_3-subscribe@{shape:diamond,label:All}\n\tsession_1-subscribe@{shape:diamond,label:All}".to_string());
        
        // Test from flowchart
        let _builder_test = SessionContextBuilder::from_mermaid_flowchart(&mermaid_js)?;
        
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

        // Make the builder from the ER Diagram
        let _builder_test = SessionContextBuilder::new().with_state_from_mermaid_erdiagram(&mermaid_js)?;
        Ok(())
    }
}