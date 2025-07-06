// Server related imports
use axum::{extract::Json, http::StatusCode, response::IntoResponse};

// General imports
use serde::Serialize;

// from https://github.com/EricLBuehler/candle-vllm/blob/master/src/openai/responses.rs#L117
pub trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize)]
pub struct JsonError {
    message: String,
}

impl JsonError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl ErrorToResponse for JsonError {}

// attempt to extract the inner `serde_path_to_error::Error<serde_json::Error>`,
// if that succeeds we can provide a more specific error.
//
// `Json` uses `serde_path_to_error` so the error will be wrapped in `serde_path_to_error::Error`.
pub fn serde_json_error_response<E>(err: E) -> (StatusCode, String)
where
    E: std::error::Error + 'static,
{
    if let Some(err) = find_error_source::<serde_path_to_error::Error<serde_json::Error>>(&err) {
        let serde_json_err = err.inner();
        (
            StatusCode::BAD_REQUEST,
            format!(
                "Invalid JSON at line {} column {}",
                serde_json_err.line(),
                serde_json_err.column()
            ),
        )
    } else {
        (StatusCode::BAD_REQUEST, "Unknown error".to_string())
    }
}

// attempt to downcast `err` into a `T` and if that fails recursively try and
// downcast `err`'s source
fn find_error_source<'a, T>(err: &'a (dyn std::error::Error + 'static)) -> Option<&'a T>
where
    T: std::error::Error + 'static,
{
    if let Some(err) = err.downcast_ref::<T>() {
        Some(err)
    } else if let Some(source) = err.source() {
        find_error_source(source)
    } else {
        None
    }
}
