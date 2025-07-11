use serde::{Deserialize, Serialize};

// Backend URL
// DM: need to change this to an environmental variable
//  to better stay in sync with the server url.
pub const ADDR_BACKEND: &str = "http://127.0.0.1:4000";

/// Server session info get request
/// same as in phymes-server/src/handlers/session_info.rs
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct GetSessionState {
    /// The name of the session
    pub session_name: String,
    /// The subject name if known else blank
    pub subject_name: String,
    /// The format of the response
    /// Options are "csv_str" and "json_obj"
    pub format: String,
}

/// Create the session name by combining the user ID
/// with the session plan
pub fn create_session_name(email: &str, session_plan: &str) -> String {
    let session_name = format!("{email}{session_plan}");
    session_name
}
