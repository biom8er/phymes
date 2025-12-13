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
use clap::ValueEnum;
use futures::prelude::*;
use phymes_agents::{AvailableInterfaceSubjects, create_message_map};
use phymes_core::{
    AvailableSubjectsTrait, BuildableTrait, BuilderTrait, DataFormat, IPCMessage,
    JoinUserInboxSessionContextsMermaidDiagrams, MappableTrait, MessageBuilderTrait, MessageTrait,
    SessionInterfaceMessage, SessionInterfaceMessageTrait, SessionStream, Table, TableBuilder,
    TableBuilderTrait, TableTrait,
};

// General imports
use anyhow::Result;
use phymes_diagnostics::HashMap;
use std::sync::Arc;

// Library imports
use crate::{
    handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response},
    state::{ServerState, UserState},
};

/// Chat inference endpoint
#[axum::debug_handler]
pub async fn session_stream(
    Extension((current_user, user_session_contexts)): Extension<(
        String,
        Vec<JoinUserInboxSessionContextsMermaidDiagrams>,
    )>,
    State((_, mut state)): State<(UserState, ServerState)>,
    payload: Result<Json<SessionInterfaceMessage>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Running chat session for session_name {}",
                payload.get_session_name()
            );

            // Add user state if it does not exist already
            if !state
                .user_session_names
                .try_read()
                .unwrap()
                .contains_key(&current_user)
            {
                // Initialize the user session contexts
                let _session_names = match state.make_session_contexts(&user_session_contexts, true)
                {
                    Ok(session_names) => session_names,
                    Err(err) => {
                        return JsonError::new(err.to_string())
                            .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                };

                // Read in any updates to the session context
                match state.read_session_contexts(
                    &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                    &current_user,
                ) {
                    Ok(()) => tracing::info!("Read state for {}", current_user),
                    Err(e) => tracing::info!(
                        "Failed to read the session stream state {e:?} for {}",
                        current_user
                    ),
                }
            }

            let session_stream_state = match state
                .session_contexts
                .try_write()
                .unwrap()
                .get(payload.get_session_name())
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

            // Convert the message to IPC if not already
            // DM: this can be optimized so that the message payload is consumed...
            let bytes = match &payload.get_format() {
                DataFormat::Ipc => payload.get_message().to_owned(),
                DataFormat::Bytes => {
                    let schema =
                        match AvailableInterfaceSubjects::from_str(payload.get_subject(), false) {
                            Ok(subject) => subject.to_schema(),
                            Err(err) => {
                                return JsonError::new(err.to_string())
                                    .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                            }
                        };
                    Table::get_builder()
                        .with_name(payload.get_subject())
                        .with_schema(schema)
                        .with_bytes(payload.get_message())
                        .unwrap()
                        .build()
                        .unwrap()
                        .to_ipc_stream()
                        .unwrap()
                }
                _ => unimplemented!(),
            };
            let message = IPCMessage::get_builder()
                .with_message(bytes)
                .with_subject(payload.get_subject())
                .with_update(payload.get_update())
                .with_publisher(payload.get_publisher())
                .make_name()
                .unwrap()
                .build()
                .unwrap();

            // Make the session stream
            // DM: we assume only a single message per request
            let message_map = create_message_map(vec![message]);
            let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));

            // Run and update the session and convert the output to the user specified format
            // Note: that we cannot write state updates to disk for
            //   streaming responses since we need to execute the stream first
            match (&payload.get_format(), payload.get_stream()) {
                (DataFormat::Bytes, true) => {
                    // Convert the output to bytes
                    let response = session_stream.into_stream().map_ok(move |f| {
                        f.into_iter()
                            .filter(|(_k, v)| v.get_name().contains(payload.get_session_name()))
                            .flat_map(|(_k, v)| {
                                let name = v.get_name().to_string();
                                TableBuilder::new_from_ipc_stream(&v.get_message_own())
                                    .unwrap()
                                    .with_name(name.as_str())
                                    .build()
                                    .unwrap()
                                    .to_bytes()
                                    .unwrap()
                            })
                            .collect::<Vec<_>>()
                    });

                    // Send the stream
                    Body::from_stream(response).into_response()
                }
                (DataFormat::Bytes, false) => {
                    // Convert the output to bytes
                    let response: Vec<HashMap<String, IPCMessage>> =
                        session_stream.try_collect().await.unwrap();
                    let response = response
                        .into_iter()
                        .flatten()
                        .filter(|(_k, v)| v.get_name().contains(payload.get_session_name()))
                        .flat_map(|(_k, v)| {
                            let name = v.get_name().to_string();
                            TableBuilder::new_from_ipc_stream(&v.get_message_own())
                                .unwrap()
                                .with_name(name.as_str())
                                .build()
                                .unwrap()
                                .to_json_object()
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let response = Bytes::from(serde_json::to_string(&response).unwrap());

                    // Update the row counts
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .get_session_context_mut()
                        .update_subject_num_rows_table();

                    // Write the updates to disk
                    if let Err(e) = state.write_session_contexts(
                        &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                        &current_user,
                    ) {
                        return JsonError::new(format!(
                            "Failed to write the session stream state {e:?}"
                        ))
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }

                    // Send the stream
                    Body::from(response).into_response()
                }
                (DataFormat::Ipc, true) => {
                    // Convert the output to IPC
                    let response = session_stream.into_stream().map_ok(move |f| {
                        f.into_iter()
                            .filter(|(_k, v)| v.get_name().contains(payload.get_session_name()))
                            .flat_map(|(_k, v)| v.get_message_own())
                            .collect::<Vec<_>>()
                    });

                    // Send the stream
                    Body::from_stream(response).into_response()
                }
                (DataFormat::Ipc, false) => {                    
                    // Convert the output to IPC messages
                    // DM: the bytes cannot be flattened and then read as a single table
                    //  because the reader will break at the end of the first batch encountered!
                    let response: Vec<HashMap<String, IPCMessage>> =
                        session_stream.try_collect().await.unwrap();
                    let batches = response
                        .into_iter()
                        .flat_map(|map| map.into_iter()
                            .filter_map(|(_k, v)| if v.get_name().contains(payload.get_session_name()) {
                                let batches = TableBuilder::new_from_ipc_stream(&v.get_message_own()).unwrap()
                                    .with_name("")
                                    .build().unwrap()
                                    .get_record_batches_own();
                                Some(batches)
                            } else {
                                None
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                        )
                        .collect::<Vec<_>>();
                    let response = TableBuilder::new().with_record_batches(batches).unwrap()
                        .with_name("")
                        .build().unwrap()
                        .concat_record_batches().unwrap()
                        .to_ipc_stream().unwrap();

                    // Update the row counts
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .get_session_context_mut()
                        .update_subject_num_rows_table();

                    // Write the updates to disk
                    if let Err(e) = state.write_session_contexts(
                        &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                        &current_user,
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
