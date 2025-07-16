// Server related imports
use axum::{
    Extension,
    body::Body,
    extract::{Json, State, rejection::JsonRejection},
    http::StatusCode,
    response::IntoResponse,
};

// Streaming imports
use bytes::Bytes;
use futures::prelude::*;
use phymes_core::{
    metrics::HashMap,
    session::{
        common_traits::{BuilderTrait, MappableTrait},
        session_context::SessionStream,
    },
    table::{
        arrow_table::{ArrowTableBuilder, ArrowTableTrait},
        arrow_table_publish::ArrowTablePublish,
    },
    task::arrow_message::{
        ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
        ArrowIncomingMessageTrait, ArrowMessageBuilderTrait,
    },
};

// General imports
use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Library imports
use crate::{
    handlers::{
        json_error::{ErrorToResponse, JsonError, serde_json_error_response},
        session_info::{SessionResponse, SessionResponseFormat},
        sign_in::CurrentUser,
    },
    server::server_state::ServerState,
};

// Crate imports
use phymes_agents::candle_chat::message_history::MessageHistoryBuilderTraitExt;

/// Chat inference endpoint
#[axum::debug_handler]
pub async fn session_stream(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Running chat session for session_name {}",
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
            let session_stream_state = match state
                .session_contexts
                .try_write()
                .unwrap()
                .get(payload.session_name.as_str())
            {
                // Continue an existing session
                Some(session) => {
                    // Reset the iter
                    session.try_write().unwrap().set_iter(0);

                    // Copy
                    Arc::clone(session)
                }
                // Create new session
                None => {
                    return JsonError::new("Failed to get the session stream state".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            // Make the system prompt and add the user query
            let message_builder = ArrowTableBuilder::new()
                .with_name(payload.subject_name.as_str())
                .append_new_user_query_str(payload.content.as_str(), "user")
                .unwrap();

            // Build the incoming message
            let incoming_message = ArrowIncomingMessageBuilder::new()
                .with_name("Chat")
                .with_subject(payload.subject_name.as_str())
                .with_publisher(payload.session_name.as_str())
                .with_message(message_builder.build().unwrap())
                .with_update(&ArrowTablePublish::Extend {
                    table_name: payload.subject_name.to_owned(),
                })
                .build()
                .unwrap();
            let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
            incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

            // Make the session stream
            let session_stream =
                SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

            // Run and update the session
            // DM: we cannot just forward the stream because we want to
            //  update the session which requires executing the stream first
            //  i.e., we cannot do something like the following

            //  and then send the response after optionally converting to a
            //  a byte stream

            // Convert the output to the user specified format
            // Note: that we cannot write state updates to disk for
            //   streaming responses since we need to execute the stream first
            match (&payload.format, payload.stream) {
                (SessionResponseFormat::Bytes, true) => {
                    // Convert the output to bytes
                    let response = session_stream.into_stream().map_ok(move |f| {
                        f.into_iter()
                            .filter(|(_k, v)| v.get_name().contains(payload.session_name.as_str()))
                            .flat_map(|(_k, v)| v.get_message_own().to_bytes().unwrap())
                            .collect::<Vec<_>>()
                    });

                    // Send the stream
                    Body::from_stream(response).into_response()
                }
                (SessionResponseFormat::Bytes, false) => {
                    // Convert the output to bytes
                    let response: Vec<HashMap<String, ArrowIncomingMessage>> =
                        session_stream.try_collect().await.unwrap();
                    let response = response
                        .into_iter()
                        .flatten()
                        .filter(|(_k, v)| v.get_name().contains(payload.session_name.as_str()))
                        .map(|(_k, v)| v.get_message_own().to_json_object().unwrap())
                        .flat_map(|v| Bytes::from(serde_json::to_string(&v).unwrap()))
                        .collect::<Vec<_>>();

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

                    // Send the stream
                    Body::from(response).into_response()
                }
                (SessionResponseFormat::IPC, true) => {
                    // Convert the output to IPC
                    let response = session_stream.into_stream().map_ok(move |f| {
                        f.into_iter()
                            .filter(|(_k, v)| v.get_name().contains(payload.session_name.as_str()))
                            .flat_map(|(_k, v)| v.get_message_own().to_ipc_stream().unwrap())
                            .collect::<Vec<_>>()
                    });

                    // Send the stream
                    Body::from_stream(response).into_response()
                }
                (SessionResponseFormat::IPC, false) => {
                    // Convert the output to bytes
                    let response: Vec<HashMap<String, ArrowIncomingMessage>> =
                        session_stream.try_collect().await.unwrap();
                    let response = response
                        .into_iter()
                        .flatten()
                        .filter(|(_k, v)| v.get_name().contains(payload.session_name.as_str()))
                        .flat_map(|(_k, v)| v.get_message_own().to_ipc_stream().unwrap())
                        .collect::<Vec<_>>();

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

                    // Send the stream
                    Body::from(response).into_response()
                }
                _ => unimplemented!(),
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

pub mod test_chat_handler {
    use super::*;

    #[derive(Serialize, Deserialize)]
    pub struct StreamBytesInput {
        pub num_bytes: u16,
        pub greeting: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct StreamBytesOutput {
        pub message: String,
    }

    /// Chat inference endpoint
    pub async fn stream_bytes(
        Extension(_current_user): Extension<CurrentUser>,
        payload: Result<Json<StreamBytesInput>, JsonRejection>,
    ) -> impl IntoResponse {
        // Extract and process the payload
        match payload {
            Ok(payload) => {
                // We got a valid JSON payload
                let stream = stream::iter((0..payload.num_bytes).map(move |_idx| {
                    let response = StreamBytesOutput {
                        message: payload.greeting.clone(),
                    };
                    let buf = Bytes::from(serde_json::to_string(&response)?);
                    Ok::<Bytes, Error>(buf)
                }));
                Body::from_stream(stream).into_response()
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
}
