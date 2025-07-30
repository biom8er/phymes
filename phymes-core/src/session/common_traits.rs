use crate::metrics::HashMap;
use crate::session::runtime_env::RuntimeEnv;
use crate::table::{
    arrow_table::ArrowTable, arrow_table_publish::ArrowTablePublish,
    arrow_table_subscribe::ArrowTableSubscribe,
};
use crate::task::{
    arrow_message::{ArrowIncomingIPCMessage, ArrowIncomingMessage, ArrowOutgoingMessage},
    arrow_processor::ArrowProcessorTrait,
    arrow_task::ArrowTask,
};

/// General imports
use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

/// Imports from Candle
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

/// Runtime environment HashMap with Arc/Mutex for thread-safe mutability
pub type RuntimeEnvMap = HashMap<String, Arc<Mutex<RuntimeEnv>>>;

/// Processor HashMap with Arc-based abstraction
pub type ProcessorMap = HashMap<String, Arc<dyn ArrowProcessorTrait>>;

/// Task HashMap
pub type TaskMap = HashMap<String, Arc<ArrowTask>>;

/// Table HashMap with Arc/RwLock for thread-safe multiple reads
pub type StateMap = HashMap<String, Arc<RwLock<ArrowTable>>>;

/// Incoming Message HashMap
pub type IncomingMessageMap = HashMap<String, ArrowIncomingMessage>;

/// Outgoing Message HashMap
pub type OutgoingMessageMap = HashMap<String, ArrowOutgoingMessage>;

/// Incoming IPCMessage HashMap
pub type IPCMessageMap = HashMap<String, ArrowIncomingIPCMessage>;

/// For all objects that can be inserted into a HashMap
/// based on their `name` attribute
pub trait MappableTrait {
    /// name of the object
    fn get_name(&self) -> &str;
    /// send the object to a HashMap
    /// only works with concrete types and not traits!
    fn to_map(self, map: &mut HashMap<String, Arc<Self>>) -> Option<Arc<Self>>
    where
        Self: Sized,
    {
        map.insert(self.get_name().to_string(), Arc::new(self))
    }
}

/// For objects built using a T builder object
pub trait BuildableTrait {
    type T;
    /// get the builder for the method
    /// should just be a call to `T::default()`
    fn get_builder() -> Self::T
    where
        Self: Sized;
}

/// For builder objects that build a T object
pub trait BuilderTrait {
    type T;
    /// expected for builder objects even if
    /// there is a default implementation
    fn new() -> Self
    where
        Self: Sized;
    /// add name to the builder
    fn with_name(self, name: &str) -> Self
    where
        Self: Sized;
    /// build the target object
    fn build(self) -> Result<Self::T>
    where
        Self: Sized;
    /// convenience method to return an Arc reference instead
    /// of the object itself
    fn build_arc(self) -> Result<Arc<Self::T>>
    where
        Self: Sized,
    {
        self.build().map(Arc::new)
    }
}

/// For task or processor objects that publish and
/// subscribe to messages
pub trait PubSubTrait {
    /// Get an immutable list of subscription subject names
    fn get_subscriptions(&self) -> &Vec<ArrowTableSubscribe>;

    /// Get an immutable list of publication subject names
    fn get_publications(&self) -> &Vec<ArrowTablePublish>;
}

/// For task objects that run computation and send/recieve
/// streaming `RecordBatch`es as messages
pub trait RunnableTrait {
    /// Run the computation
    fn run(&self, messages: OutgoingMessageMap) -> Result<OutgoingMessageMap>;
}

/// For services that process Tensors
pub trait TensorProcessorTrait: Send + Sync + Debug {
    /// Device
    fn get_device(&self) -> &Device;
}

/// Tokens representations in different dimensions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenWrapper {
    ///ComputableInput Text generation input
    D1(Vec<u32>),
    /// Embedding generation input
    D2(Vec<Vec<u32>>),
}

/// Tokenizer configurations and templates
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: Option<usize>,
    pub chat_template: Option<String>, // Jinja2 template is provided in tokenizer_config.json
    pub eos_token: Option<String>, // can be inferred from vocab.json and config.json and provided in tokenizer_config.json
    pub eos_token_id: Option<u32>, // provided in config.json
    pub bos_token: Option<String>,
    pub bos_token_id: Option<u32>, // provided in config.json
                                   // pub completion_template: Option<String>, // provided in tokenizer_config.json
                                   // pub tokenizer_class: Option<String>, // provided in tokenizer_config.json
}

/// for services that process tokens
pub trait TokenProcessorTrait: TensorProcessorTrait + Send + Sync + Debug {
    /// Backward: propogation of the error signal for updating the tensor weights
    //fn backward(&mut self) -> Result<Self::T>;
    /// Forward: inference using the tensor weights on the specified device
    fn forward(
        &mut self,
        input: &TokenWrapper,
        index_position: usize,
        attention_mask: Option<&TokenWrapper>,
        include_lm_head: bool,
    ) -> Result<Tensor>;

    fn get_tokenizer(&self) -> &Tokenizer;

    fn get_tokenizer_config(&self) -> &TokenizerConfig;
}
