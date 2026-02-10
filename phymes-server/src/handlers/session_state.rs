use std::sync::Arc;

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
use phymes_agents::{SessionStreamStep, SessionStreamStepTrait, create_message_map};
use phymes_core::{
    BuilderTrait, CsvFormat, DataFormat, IPCMessageBuilder,
    JoinUserInboxSessionContextsMermaidDiagrams, MessageBuilderTrait, MessageTrait,
    TableBuilder, TableBuilderTrait, TableTrait,
};
use phymes_agents::{SessionInterfaceMessage, SessionInterfaceMessageTrait};

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::state::{ServerState, UserState};

/// Put state input
#[axum::debug_handler]
pub async fn session_put_state(
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
                "Put session state for session_name {}",
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

            let session_ctx_arc = match state
                .session_contexts
                .try_write()
                .unwrap()
                .get(payload.get_session_name())
            {
                Some(session) => Arc::clone(session),
                None => {
                    return JsonError::new("Failed to get the session stream state".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            // Extract the payload as bytes
            let schema = if let Some(subject) = session_ctx_arc
                .try_read()
                .unwrap()
                .get_states()
                .get(payload.get_subject())
            {
                subject.try_read().unwrap().get_schema()
            } else {
                return JsonError::new("Failed to get the session stream state".to_string())
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR);
            };
            let bytes = match payload.get_format() {
                DataFormat::Csv(csv_format) => TableBuilder::new()
                    .with_schema(schema)
                    .with_name(payload.get_subject())
                    .with_csv(
                        payload.get_message(),
                        csv_format.delimiter,
                        csv_format.header,
                        csv_format.batch_size,
                    )
                    .unwrap()
                    .build()
                    .unwrap()
                    .to_ipc_stream()
                    .unwrap(),
                DataFormat::CsvDefault => {
                    let csv_format = CsvFormat::default();
                    TableBuilder::new()
                        .with_schema(schema)
                        .with_name(payload.get_subject())
                        .with_csv(
                            payload.get_message(),
                            csv_format.delimiter,
                            csv_format.header,
                            csv_format.batch_size,
                        )
                        .unwrap()
                        .build()
                        .unwrap()
                        .to_ipc_stream()
                        .unwrap()
                }
                DataFormat::JsonDefault => {
                    let json_value: Vec<serde_json::Value> =
                        serde_json::from_slice(payload.get_message()).unwrap();
                    TableBuilder::new()
                        .with_schema(schema)
                        .with_name(payload.get_subject())
                        .with_json_values(&json_value)
                        .unwrap()
                        .build()
                        .unwrap()
                        .to_ipc_stream()
                        .unwrap()
                }
                DataFormat::Bytes => TableBuilder::new()
                    .with_schema(schema)
                    .with_name(payload.get_subject())
                    .with_bytes(payload.get_message())
                    .unwrap()
                    .build()
                    .unwrap()
                    .to_ipc_stream()
                    .unwrap(),
                DataFormat::Ipc => payload.get_message().to_owned(),
                _ => unimplemented!(),
            };

            // Create the update message
            let message = IPCMessageBuilder::new()
                .with_subject(payload.get_subject())
                .with_publisher(payload.get_publisher())
                .with_message(bytes)
                .with_update(payload.get_update())
                .make_name()
                .unwrap()
                .build()
                .unwrap();
            let messages = create_message_map(vec![message]);

            // Update the session state with the new message
            let _step = SessionStreamStep::current_superstep(&session_ctx_arc).await;
            if let Err(e) = SessionStreamStep::update_subjects_and_changelog_from_messages(
                &session_ctx_arc,
                messages,
            ) {
                return JsonError::new(format!("Failed to update the session stream state {e:?}"))
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR);
            }

            // Write the updates to disk
            if let Err(e) = state.write_session_contexts(
                &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                &current_user,
            ) {
                return JsonError::new(format!("Failed to write the session stream state {e:?}"))
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR);
            }

            // Send the response
            Body::from(serde_json::to_string("State updated with subject content.").unwrap())
                .into_response()
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

/// Get state endpoint
#[axum::debug_handler]
pub async fn session_get_state(
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
                "Get session state for session_name {}",
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

            match state
                .session_contexts
                .write()
                .get(payload.get_session_name())
            {
                Some(session_ctx_arc) => {
                    // Update the row counts just in case...
                    session_ctx_arc
                        .try_write()
                        .unwrap()
                        .update_subject_num_rows_table();

                    match payload.get_format() {
                        DataFormat::Bytes => {
                            // Get the subject table as a json object
                            let buf = session_ctx_arc
                                .try_read()
                                .unwrap()
                                .get_states()
                                .get(payload.get_subject())
                                .unwrap()
                                .try_read()
                                .unwrap()
                                .to_bytes()
                                .unwrap();
                            Body::from(buf).into_response()
                        }
                        DataFormat::Csv(csv_format) => {
                            // Get the subject table as a csv string
                            let out = session_ctx_arc
                                .try_read()
                                .unwrap()
                                .get_states()
                                .get(payload.get_subject())
                                .unwrap()
                                .try_read()
                                .unwrap()
                                .to_csv(csv_format.delimiter, csv_format.header)
                                .unwrap();
                            let buf = Bytes::from(out);
                            Body::from(buf).into_response()
                        }
                        DataFormat::CsvDefault => {
                            // Get the subject table as a csv string
                            let csv_format = CsvFormat::default();
                            let out = session_ctx_arc
                                .try_read()
                                .unwrap()
                                .get_states()
                                .get(payload.get_subject())
                                .unwrap()
                                .try_read()
                                .unwrap()
                                .to_csv(csv_format.delimiter, csv_format.header)
                                .unwrap();
                            let buf = Bytes::from(out);
                            Body::from(buf).into_response()
                        }
                        DataFormat::Json(_) | DataFormat::JsonDefault => {
                            // Get the subject table as a json string
                            let out = session_ctx_arc
                                .try_read()
                                .unwrap()
                                .get_states()
                                .get(payload.get_subject())
                                .unwrap()
                                .try_read()
                                .unwrap()
                                .to_json()
                                .unwrap();
                            let buf = Bytes::from(out);
                            Body::from(buf).into_response()
                        }
                        DataFormat::Ipc => {
                            // Get the subject table as a csv string
                            let out = session_ctx_arc
                                .try_read()
                                .unwrap()
                                .get_states()
                                .get(payload.get_subject())
                                .unwrap()
                                .try_read()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let buf = Bytes::from(out);
                            Body::from(buf).into_response()
                        }
                        _ => unimplemented!(),
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
