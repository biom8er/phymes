use parking_lot::RwLock;
use std::fmt::Debug;
use std::sync::Arc;

use crate::session::common_traits::{MappableTrait, TensorProcessorTrait, TokenProcessorTrait};

/// `BuidableTrait` + `BuilderTraint` - `get_builder` - `build`
pub trait RuntimeEnvTrait: MappableTrait + Send + Sync {
    fn new() -> Self;
    fn with_name(self, name: &str) -> Self;
}

#[derive(Default, Clone, Debug)]
pub struct RuntimeEnv {
    /// the service for generating and consuming tokens
    pub token_service: Option<Arc<RwLock<dyn TokenProcessorTrait>>>,
    /// the service for operating over tensors
    pub tensor_service: Option<Arc<RwLock<dyn TensorProcessorTrait>>>,
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
            token_service: None,
            tensor_service: None,
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
