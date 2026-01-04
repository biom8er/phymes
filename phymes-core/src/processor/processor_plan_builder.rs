use std::sync::Arc;

use anyhow::{Result, anyhow};
use crate::{ProcessorTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription, ProcessorPlan};

/// The builder for the [ProcessorPlan]
#[derive(Debug, Default)]
pub struct ProcessorPlanBuilder {
    pub publications: Option<Vec<TablePublication>>,
    pub subscriptions: Option<Vec<TableSubscription>>,
    pub subscribe_policy: Option<Box<dyn TableSubscribePolicyTrait>>,
    pub processor: Option<Arc<dyn ProcessorTrait>>,
}

impl ProcessorPlanBuilder {
    pub fn with_publications(mut self, publications: &[TablePublication]) -> Self {
        self.publications = Some(publications.to_vec());
        self
    }
    pub fn with_subscriptions(mut self, subscriptions: &[TableSubscription]) -> Self {
        self.subscriptions = Some(subscriptions.to_vec());
        self
    }
    pub fn with_subscribe_policy(
        mut self,
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Self {
        self.subscribe_policy = Some(subscribe_policy);
        self
    }
    pub fn with_processor(mut self, processor: Arc<dyn ProcessorTrait>) -> Self {
        self.processor= Some(processor);
        self
    }
    pub fn build(mut self) -> Result<ProcessorPlan> {
        if self.processor.as_ref().is_none() {
            return Err(anyhow!("Missing processor"));
        } else if self.publications.as_ref().is_none() {
            return Err(anyhow!(
                "Missing publications for processor {}",
                self.processor.as_ref().unwrap().get_name()
            ));
        } else if self.subscriptions.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscriptions for processor {}",
                self.processor.as_ref().unwrap().get_name()
            ));
        } else if self.subscribe_policy.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscribe for processor {}",
                self.processor.as_ref().unwrap().get_name()
            ));
        }
        Ok(ProcessorPlan {
            processor: self.processor.take().unwrap(),
            publications: self.publications.take().unwrap(),
            subscriptions: self.subscriptions.take().unwrap(),
            subscribe_policy: self.subscribe_policy.take().unwrap(),
        })
    }
}
