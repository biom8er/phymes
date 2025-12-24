use crate::{ProcessorTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription};
use anyhow::{Result, anyhow};
use std::sync::Arc;

/// Builder for structures implementing the [ProcessorTrait]
#[derive(Default)]
pub struct ProcessorBuilder {
    pub publications: Option<Vec<TablePublication>>,
    pub subscriptions: Option<Vec<TableSubscription>>,
    pub subscribe_policy: Option<Box<dyn TableSubscribePolicyTrait>>,
    pub name: Option<String>,
    pub r#type: Option<String>,
}

impl ProcessorBuilder {
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
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    pub fn with_type(mut self, r#type: &str) -> Self {
        self.r#type = Some(r#type.to_string());
        self
    }
    pub fn build<T>(mut self) -> Result<T>
    where
        T: ProcessorTrait,
    {
        if self.name.as_ref().is_none() {
            return Err(anyhow!("Missing processor name"));
        } else if self.r#type.as_ref().is_none() {
            return Err(anyhow!(
                "Missing type for processor {}",
                self.name.as_ref().unwrap()
            ));
        } else if self.publications.as_ref().is_none() {
            return Err(anyhow!(
                "Missing publications for processor {}",
                self.name.as_ref().unwrap()
            ));
        } else if self.subscriptions.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscriptions for processor {}",
                self.name.as_ref().unwrap()
            ));
        } else if self.subscribe_policy.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscribe for processor {}",
                self.name.as_ref().unwrap()
            ));
        }
        Ok(T::new(
            &self.name.take().unwrap(),
            &self.r#type.take().unwrap(),
            &self.publications.take().unwrap(),
            &self.subscriptions.take().unwrap(),
            self.subscribe_policy.take().unwrap(),
        ))
    }
    /// convenience method to return an Arc reference instead of the object itself
    pub fn build_arc<T>(self) -> Result<Arc<dyn ProcessorTrait>>
    where
        Self: Sized,
        T: ProcessorTrait + 'static,
    {
        self.build()
            .map(|p: T| Arc::new(p) as Arc<dyn ProcessorTrait>)
    }
}
