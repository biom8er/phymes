use crate::ProcessorTrait;
use anyhow::{Result, anyhow};
use std::sync::Arc;

/// Builder for structures implementing the [ProcessorTrait]
#[derive(Default)]
pub struct ProcessorBuilder {
    pub name: Option<String>,
    pub r#type: Option<String>,
}

impl ProcessorBuilder {
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
        }
        Ok(T::new(
            &self.name.take().unwrap(),
            &self.r#type.take().unwrap(),
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
