use std::sync::Arc;

use crate::{ProcessorTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription};

/// The plan for the processors
#[derive(Debug)]
pub struct ProcessorPlan {
    /// The processor
    pub processor: Arc<dyn ProcessorTrait>,
    /// The subjects the processor publishes on
    pub publications: Vec<TablePublication>,
    /// The subjects the processor subscribes to
    pub subscriptions: Vec<TableSubscription>,
    /// The policy for subscribing to subjects
    pub subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
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
