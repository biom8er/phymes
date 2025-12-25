mod common_traits;
mod runtime_env;

pub use common_traits::{BuildableTrait, BuilderTrait, MappableTrait};
pub use runtime_env::{RuntimeEnv, RuntimeEnvTrait};

use std::sync::Arc;
use phymes_diagnostics::HashMap;

pub type RuntimeEnvMap = HashMap<String, Arc<RuntimeEnv>>;