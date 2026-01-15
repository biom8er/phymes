use std::fmt::Debug;

use crate::MappableTrait;

/// `BuidableTrait` + `BuilderTraint` - `get_builder` - `build`
///
/// # Notes
/// * A work in progress...
/// * Missing methods for specifying the device or number of devices
/// * Missing methods for disk usage and access
pub trait RuntimeEnvTrait: MappableTrait + Send + Sync {
    fn new() -> Self;
    fn with_name(self, name: &str) -> Self;
}

#[derive(Default, Debug, PartialEq)]
pub struct RuntimeEnv {
    /// name for the runtime environment config
    pub name: String,
    /// the max allowable memory
    pub memory_limit: Option<usize>,
    /// the max allowable compute time
    pub time_limit: Option<usize>,
}

impl MappableTrait for RuntimeEnv {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl RuntimeEnvTrait for RuntimeEnv {
    fn new() -> Self {
        Self {
            name: "".to_string(),
            memory_limit: None,
            time_limit: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }
}
