// Server related imports
use axum::{
    Extension,
    body::Body,
    extract::{Json, State, rejection::JsonRejection},
    http::StatusCode,
    response::IntoResponse,
};

// General imports
use anyhow::Result;
use bytes::Bytes;
use phymes_core::table::{arrow_table::ArrowTableTrait, arrow_table_publish::ArrowTablePublish};
use serde::{Deserialize, Serialize};

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::handlers::sign_in::CurrentUser;
use crate::server::server_state::ServerState;

use super::session_info::GetSessionState;

/// Dioxus application put request
/// same as phymes-server/src/handlers/session_state.rs
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PutSessionState {
    /// Session name to publish on
    pub session_name: String,
    /// Subject name to publish on
    pub subject_name: String,
    /// (Optional) document title
    pub document_name: String,
    pub text: String,
    /// Publish method
    /// Options are "Extend" or "Replace"
    /// see phymes-core/src/table/arrow_table_publish.rs
    pub publish: String
}

/// Chat inference endpoint
#[axum::debug_handler]
pub async fn session_put_state(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<PutSessionState>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Put session state for session_name {}",
                payload.session_name.as_str()
            );
            if !state.check_email_in_state(&current_user.email)
                && let Err(e) = state.read_state_by_email(
                    &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                    &current_user.email,
                )
            {
                tracing::error!(
                    "Failed to read the session stream state {e:?}. Creating new session stream state."
                );
                if state
                    .create_session_names_by_email(&current_user.email)
                    .is_none()
                {
                    return JsonError::new("Failed to get the session stream state".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
            match state
                .session_contexts
                .try_write()
                .unwrap()
                .get(payload.session_name.as_str())
            {
                Some(session_stream_state) => {
                    // Update the state and superstep_updates
                    let update = if payload.publish.contains("Replace") {
                        ArrowTablePublish::Replace { table_name: payload.subject_name.to_owned() }
                    } else if payload.publish.contains("Extend") {
                        ArrowTablePublish::Extend { table_name: payload.subject_name.to_owned() }
                    } else {
                        ArrowTablePublish::Extend { table_name: payload.subject_name.to_owned() }
                    };
                    let schema = session_stream_state
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(payload.subject_name.as_str())
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_schema();
                    let update = session_stream_state
                        .try_write()
                        .unwrap()
                        .update_state_from_csv_str(
                            &schema,
                            payload.text.as_str(),
                            &update,
                            b',',
                            true,
                            1024,
                        )
                        .unwrap();
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .extend_superstep_updates(update);
                }
                None => {
                    return JsonError::new("Failed to get the session stream state".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            // Write the updates to disk
            if let Err(e) = state.write_state_by_email(
                &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                &current_user.email,
            ) {
                return JsonError::new(format!(
                    "Failed to write the session stream state {e:?}"
                ))
                .to_response(StatusCode::INTERNAL_SERVER_ERROR);
            }

            // Send the response
            Body::from(serde_json::to_string("State updated").unwrap()).into_response()
        }
        Err(JsonRejection::MissingJsonContentType(_err)) => {
            // Request didn't have `Content-Type: application/json`
            // header
            JsonError::new("Missing `Content-Type: application/json` header".to_string())
                .to_response(StatusCode::BAD_REQUEST)
        }
        Err(JsonRejection::JsonDataError(err)) => {
            // Couldn't deserialize the body into the target type
            let (e_code, e_str) = serde_json_error_response(err);
            JsonError::new(e_str).to_response(e_code)
        }
        Err(JsonRejection::JsonSyntaxError(err)) => {
            // Syntax error in the body
            let (e_code, e_str) = serde_json_error_response(err);
            JsonError::new(e_str).to_response(e_code)
        }
        Err(JsonRejection::BytesRejection(_err)) => {
            // Failed to extract the request body
            JsonError::new("Failed to buffer request body".to_string())
                .to_response(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(_err) => {
            // `JsonRejection` is marked `#[non_exhaustive]` so match must
            // include a catch-all case.
            JsonError::new("Unknown error".to_string())
                .to_response(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Chat inference endpoint
#[axum::debug_handler]
pub async fn session_get_state(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<GetSessionState>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Get session state for session_name {}",
                payload.session_name.as_str()
            );
            if !state.check_email_in_state(&current_user.email)
                && let Err(e) = state.read_state_by_email(
                    &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                    &current_user.email,
                )
            {
                tracing::error!(
                    "Failed to read the session stream state {e:?}. Creating new session stream state."
                );
                if state
                    .create_session_names_by_email(&current_user.email)
                    .is_none()
                {
                    return JsonError::new("Failed to get the session stream state".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
            match state
                .session_contexts
                .try_write()
                .unwrap()
                .get(payload.session_name.as_str())
            {
                Some(session_stream_state) => {
                    if payload.format == "json_obj" {
                        // Get the subject table as a json object
                        let object = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_json_object()
                            .unwrap();
                        let content = serde_json::to_string(&object).unwrap();
                        let buf = Bytes::from(content);
                        Body::from(buf).into_response()
                    } else {                        
                        // Get the subject table as a csv string
                        let csv = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_csv(b',', true)
                            .unwrap();
                        let buf = Bytes::from(csv);
                        Body::from(buf).into_response()
                    }
                }
                None => JsonError::new("Failed to get the session stream state".to_string())
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
        Err(JsonRejection::MissingJsonContentType(_err)) => {
            // Request didn't have `Content-Type: application/json`
            // header
            JsonError::new("Missing `Content-Type: application/json` header".to_string())
                .to_response(StatusCode::BAD_REQUEST)
        }
        Err(JsonRejection::JsonDataError(err)) => {
            // Couldn't deserialize the body into the target type
            let (e_code, e_str) = serde_json_error_response(err);
            JsonError::new(e_str).to_response(e_code)
        }
        Err(JsonRejection::JsonSyntaxError(err)) => {
            // Syntax error in the body
            let (e_code, e_str) = serde_json_error_response(err);
            JsonError::new(e_str).to_response(e_code)
        }
        Err(JsonRejection::BytesRejection(_err)) => {
            // Failed to extract the request body
            JsonError::new("Failed to buffer request body".to_string())
                .to_response(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(_err) => {
            // `JsonRejection` is marked `#[non_exhaustive]` so match must
            // include a catch-all case.
            JsonError::new("Unknown error".to_string())
                .to_response(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
