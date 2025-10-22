// Backend URL
// DM: need to change this to an environmental variable
//  to better stay in sync with the server url.
#[cfg(not(target_os = "android"))]
pub const ADDR_BACKEND: &str = "http://127.0.0.1:4000";
#[cfg(target_os = "android")]
pub const ADDR_BACKEND: &str = "http://10.0.2.2:4000";
