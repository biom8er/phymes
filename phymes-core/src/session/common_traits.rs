use crate::{
    IPCMessage, ProcessorTrait, RuntimeEnv, SendableRecordBatchStreamMessage, Table, Task,
};

/// General imports
use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

/// Runtime environment HashMap with Arc/Mutex for thread-safe mutability
pub type RuntimeEnvMap = HashMap<String, Arc<Mutex<RuntimeEnv>>>;

/// Processor HashMap with Arc-based abstraction
pub type ProcessorMap = HashMap<String, Arc<dyn ProcessorTrait>>;

/// Task HashMap
pub type TaskMap = HashMap<String, Arc<Task>>;

/// Table HashMap with Arc/RwLock for thread-safe multiple reads
pub type StateMap = HashMap<String, Arc<RwLock<Table>>>;

/// Incoming Message HashMap
pub type IPCMessageMap = HashMap<String, IPCMessage>;

/// Outgoing Message HashMap
pub type SendableRecordBatchStreamMessageMap = HashMap<String, SendableRecordBatchStreamMessage>;

/// For all objects that can be inserted into a HashMap
/// based on their `name` attribute
pub trait MappableTrait {
    /// Short name for the Task, Processor, or any other struct, such as 'AddRows'.
    /// that can be called without an instance.
    fn get_static_name() -> &'static str
    where
        Self: Sized,
    {
        let full_name = std::any::type_name::<Self>();
        let maybe_start_idx = full_name.rfind(':');
        match maybe_start_idx {
            Some(start_idx) => &full_name[start_idx + 1..],
            None => "UNKNOWN",
        }
    }

    /// Name of the object
    /// defaults to the static name
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

/// For task objects that run computation and send/recieve
/// streaming `RecordBatch`es as messages
pub trait RunnableTrait {
    /// Run the computation
    fn run(
        &self,
        messages: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
    ) -> Result<SendableRecordBatchStreamMessageMap>;
}