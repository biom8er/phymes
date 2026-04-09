use std::sync::Arc;

use crate::ProcessorTrait;
use phymes_core::MappableTrait;
use phymes_event::{Publication, SubscribeEventTrait, Subscription, UpdateEventTrait};

/// The plan for the processors
#[derive(Debug)]
pub struct ProcessorPlan {
    /// The processor
    processor: Arc<dyn ProcessorTrait>,
    /// The subjects the processor publishes on
    publications: Vec<Publication>,
    /// The subjects the processor subscribes to
    subscriptions: Vec<Subscription>,
    /// The policy for subscribing to subjects
    subscribe_policy: Box<dyn SubscribeEventTrait>,
    /// The policy for determining when tables have been updated
    update_policy: Box<dyn UpdateEventTrait>,
}

impl PartialEq for ProcessorPlan {
    fn eq(&self, other: &Self) -> bool {
        self.processor.get_name() == other.processor.get_name()
            && self.processor.get_type() == other.processor.get_type()
            && self.publications == other.publications
            && self.subscriptions == other.subscriptions
            && self.subscribe_policy.get_name() == other.subscribe_policy.get_name()
            && self.update_policy.get_name() == other.update_policy.get_name()
    }
}

impl ProcessorPlan {
    pub fn new(
        processor: Arc<dyn ProcessorTrait>,
        publications: &[Publication],
        subscriptions: &[Subscription],
        subscribe_policy: Box<dyn SubscribeEventTrait>,
        update_policy: Box<dyn UpdateEventTrait>,
    ) -> Self {
        ProcessorPlan {
            processor,
            publications: publications.to_vec(),
            subscriptions: subscriptions.to_vec(),
            subscribe_policy,
            update_policy,
        }
    }
    pub fn get_processor(&self) -> &Arc<dyn ProcessorTrait> {
        &self.processor
    }
    pub fn get_processor_owned(self) -> Arc<dyn ProcessorTrait> {
        self.processor
    }
    pub fn get_type(&self) -> &str {
        self.processor.get_type()
    }
    pub fn get_subscriptions(&self) -> &Vec<Subscription> {
        &self.subscriptions
    }
    pub fn get_subscriptions_owned(self) -> Vec<Subscription> {
        self.subscriptions
    }
    pub fn get_publications(&self) -> &Vec<Publication> {
        &self.publications
    }
    pub fn get_publications_owned(self) -> Vec<Publication> {
        self.publications
    }
    #[allow(clippy::borrowed_box)]
    pub fn get_subscribe_policy(&self) -> &Box<dyn SubscribeEventTrait> {
        &self.subscribe_policy
    }
    pub fn get_subscribe_policy_owned(self) -> Box<dyn SubscribeEventTrait> {
        self.subscribe_policy
    }
    #[allow(clippy::borrowed_box)]
    pub fn get_update_policy(&self) -> &Box<dyn UpdateEventTrait> {
        &self.update_policy
    }
    pub fn get_update_policy_owned(self) -> Box<dyn UpdateEventTrait> {
        self.update_policy
    }
}

impl MappableTrait for ProcessorPlan {
    fn get_name(&self) -> &str {
        self.processor.get_name()
    }
}

/// The publications and subscriptions to run the processor with
#[derive(Debug, PartialEq)]
pub struct ProcessorSubjects {
    /// Name of the processor
    pub name: String,
    /// The subjects the processor publishes on
    pub publications: Vec<Publication>,
    /// The subjects the processor subscribes to
    pub subscriptions: Vec<Subscription>,
}

impl MappableTrait for ProcessorSubjects {
    fn get_name(&self) -> &str {
        &self.name
    }
}
