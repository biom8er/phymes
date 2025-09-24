use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct UserState {
    user: ArrowTable,
    session_plans: ArrowTable,
    session_builders: ArrowTable,
}