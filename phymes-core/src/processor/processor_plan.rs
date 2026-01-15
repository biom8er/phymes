use std::sync::Arc;

use crate::{
    MappableTrait, ProcessorTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription,
    TableUpdatePolicyTrait,
};

/// The plan for the processors
#[derive(Debug)]
pub struct ProcessorPlan {
    /// The processor
    processor: Arc<dyn ProcessorTrait>,
    /// The subjects the processor publishes on
    publications: Vec<TablePublication>,
    /// The subjects the processor subscribes to
    subscriptions: Vec<TableSubscription>,
    /// The policy for subscribing to subjects
    subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    /// The policy for determining when tables have been updated
    update_policy: Box<dyn TableUpdatePolicyTrait>,
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
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
        update_policy: Box<dyn TableUpdatePolicyTrait>,
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
    pub fn get_subscriptions(&self) -> &Vec<TableSubscription> {
        &self.subscriptions
    }
    pub fn get_subscriptions_owned(self) -> Vec<TableSubscription> {
        self.subscriptions
    }
    pub fn get_publications(&self) -> &Vec<TablePublication> {
        &self.publications
    }
    pub fn get_publications_owned(self) -> Vec<TablePublication> {
        self.publications
    }
    #[allow(clippy::borrowed_box)]
    pub fn get_subscribe_policy(&self) -> &Box<dyn TableSubscribePolicyTrait> {
        &self.subscribe_policy
    }
    pub fn get_subscribe_policy_owned(self) -> Box<dyn TableSubscribePolicyTrait> {
        self.subscribe_policy
    }
    #[allow(clippy::borrowed_box)]
    pub fn get_update_policy(&self) -> &Box<dyn TableUpdatePolicyTrait> {
        &self.update_policy
    }
    pub fn get_update_policy_owned(self) -> Box<dyn TableUpdatePolicyTrait> {
        self.update_policy
    }
}

impl MappableTrait for ProcessorPlan {
    fn get_name(&self) -> &str {
        self.processor.get_name()
    }
}

/// The publications and subscriptions to run the processor with
#[derive(Debug)]
pub struct ProcessorSubjects {
    /// Name of the processor
    pub name: String,
    /// The subjects the processor publishes on
    pub publications: Vec<TablePublication>,
    /// The subjects the processor subscribes to
    pub subscriptions: Vec<TableSubscription>,
}

impl MappableTrait for ProcessorSubjects {
    fn get_name(&self) -> &str {
        &self.name
    }
}
