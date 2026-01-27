mod common_traits;
mod runtime_env_trait;

pub use common_traits::{BuildableTrait, BuilderTrait, MappableTrait};
pub use runtime_env_trait::{RuntimeEnv, RuntimeEnvTrait};

use phymes_diagnostics::HashMap;
use std::sync::Arc;

pub type RuntimeEnvMap = HashMap<String, Arc<RuntimeEnv>>;
