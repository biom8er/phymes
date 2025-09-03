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
use phymes_agents::session_plans::available_interface_subjects::{AttachmentInterface, MessageInterface};
use phymes_core::session::common_traits::MappableTrait;
use phymes_core::{
    session::session_context::SessionContextTableNames,
    table::table::TableTrait,
};
use serde::{Deserialize, Serialize};

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::handlers::sign_in::CurrentUser;
use crate::server::server_state::ServerState;

/// Session information endpoint for subjects
#[axum::debug_handler]
pub async fn session_mermaid_js(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Getting mermaid.js strings for session_name {}",
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
                    let object = session_stream_state
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(SessionContextTableNames::MermaidJS.get_name())
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .to_json_object()
                        .unwrap();
                    let content = serde_json::to_string(&object).unwrap();
                    let buf = Bytes::from(content);
                    Body::from(buf).into_response()
                }
                None => {
                    tracing::debug!(
                        "session_name {} not found in sessions {:?}",
                        payload.session_name.as_str(),
                        state.session_contexts.try_read().unwrap().keys()
                    );
                    JsonError::new("Failed to get the session".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR)
                }
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

/// Session information endpoint for subjects
#[axum::debug_handler]
pub async fn session_subjects_num_rows(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Getting subjects number of rows for session_name {}",
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
                    let info = session_stream_state
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_subject_num_rows_as_table("")
                        .unwrap();

                    // DM: Reqwest `byte_stream` will automatically chunk the stream sent to it
                    //  therefore we need to use the serde_json object to ensure the chunks are broken
                    //  and formatted properly for decoding on the app side
                    let object = info.to_json_object().unwrap();
                    let content = serde_json::to_string(&object).unwrap();
                    let buf = Bytes::from(content);
                    Body::from(buf).into_response()
                }
                None => {
                    tracing::debug!(
                        "session_name {} not found in sessions {:?}",
                        payload.session_name.as_str(),
                        state.session_contexts.try_read().unwrap().keys()
                    );
                    JsonError::new("Failed to get the session".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR)
                }
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

/// Session information endpoint for subjects
#[axum::debug_handler]
pub async fn session_subjects_schema(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Getting subjects schema for session_name {}",
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
                    let object = session_stream_state
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(SessionContextTableNames::Subjects.get_name())
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .to_json_object()
                        .unwrap();
                    let content = serde_json::to_string(&object).unwrap();
                    let buf = Bytes::from(content);
                    Body::from(buf).into_response()
                }
                None => {
                    tracing::debug!(
                        "session_name {} not found in sessions {:?}",
                        payload.session_name.as_str(),
                        state.session_contexts.try_read().unwrap().keys()
                    );
                    JsonError::new("Failed to get the session".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR)
                }
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

/// Session information endpoint for metrics
#[axum::debug_handler]
pub async fn session_metrics_info(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Getting metrics info for session_name {}",
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
                    // Add the metrics to the state just in case...
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .get_session_context_mut()
                        .update_metrics_table()
                        .unwrap();

                    // Get the mermaid string
                    let info = session_stream_state
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_metrics_as_mermaid_gantt()
                        .unwrap();

                    // DM: Reqwest `byte_stream` will automatically chunk the stream sent to it
                    //  therefore we need to use the serde_json object to ensure the chunks are broken
                    //  and formatted properly for decoding on the app side
                    let object = info.to_json_object().unwrap();
                    let content = serde_json::to_string(&object).unwrap();
                    let buf = Bytes::from(content);
                    Body::from(buf).into_response()
                }
                None => {
                    tracing::debug!(
                        "session_name {} not found in sessions {:?}",
                        payload.session_name.as_str(),
                        state.session_contexts.try_read().unwrap().keys()
                    );
                    JsonError::new("Failed to get the session".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR)
                }
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
