use std::sync::Arc;
use anyhow::Result;
use phymes_diagnostics::HashMap;

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