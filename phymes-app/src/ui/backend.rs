use serde::{Deserialize, Serialize};

// Backend URL
// DM: need to change this to an environmental variable
//  to better stay in sync with the server url.
pub const ADDR_BACKEND: &str = "http://127.0.0.1:4000";

/// Server session info get request
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct GetSessionState {
    pub session_name: String,
    pub subject_name: String,
}

/// Create the session name by combining the user ID
/// with the session plan
pub fn create_session_name(email: &str, session_plan: &str) -> String {
    let session_name = format!("{email}{session_plan}");
    session_name
}
