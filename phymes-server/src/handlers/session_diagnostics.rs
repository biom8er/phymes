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
use parking_lot::RwLock;
use phymes_agents::{
    AvailableInterfaceSubjects, CustomAgentsBuilderTrait, DiagnosticSession, SessionContextBuilderAgentsTrait, create_message_map
};
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, DataFormat, IPCMessage,
    JoinUserInboxSessionContextsMermaidDiagrams, MappableTrait, MessageBuilderTrait, MessageTrait,
    SessionInterfaceMessage, SessionInterfaceMessageTrait, SessionStream, SessionStreamState,
    TableBuilder, TableBuilderTrait, TablePublication, TableTrait,
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
pub async fn session_diagnostics(
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
                "Running diagnostic session for session_name {}",
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

            // Initialize the diagnostics session
            let diagnostic_session = DiagnosticSession::default();

            // Get the diagnostic information from the session stream state
            let message_map = {
                let session_stream_state = match state
                    .session_contexts
                    .try_write()
                    .unwrap()
                    .get(payload.get_session_name())
                {
                    // Continue an existing session
                    Some(session) => {
                        // Copy
                        Arc::clone(session)
                    }
                    // Create new session
                    None => {
                        return JsonError::new(
                            "Failed to get the session stream state".to_string(),
                        )
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                };
                let sss = session_stream_state.read();
                let table = sss
                    .get_session_context()
                    .get_states()
                    .get(AvailableSubjects::SessionMetrics.to_string().as_str())
                    .unwrap()
                    .read();
                let metrics_message = IPCMessage::get_builder()
                    .with_message(table.to_ipc_stream().unwrap())
                    .with_subject(AvailableSubjects::AnalyticsMetrics.to_string().as_str())
                    .with_update(&TablePublication::Replace {
                        table_name: AvailableSubjects::AnalyticsMetrics.to_string(),
                    })
                    .with_publisher(diagnostic_session.session_context_name)
                    .make_name()
                    .unwrap()
                    .build()
                    .unwrap();
                let table = sss
                    .get_session_context()
                    .get_states()
                    .get(AvailableSubjects::SessionTraces.to_string().as_str())
                    .unwrap()
                    .read();
                let traces_message = IPCMessage::get_builder()
                    .with_message(table.to_ipc_stream().unwrap())
                    .with_subject(AvailableSubjects::AnalyticsTraces.to_string().as_str())
                    .with_update(&TablePublication::Replace {
                        table_name: AvailableSubjects::AnalyticsTraces.to_string(),
                    })
                    .with_publisher(diagnostic_session.session_context_name)
                    .make_name()
                    .unwrap()
                    .build()
                    .unwrap();
                let table = sss
                    .get_session_context()
                    .get_states()
                    .get(AvailableSubjects::SessionEvents.to_string().as_str())
                    .unwrap()
                    .read();
                let events_message = IPCMessage::get_builder()
                    .with_message(table.to_ipc_stream().unwrap())
                    .with_subject(AvailableSubjects::AnalyticsEvents.to_string().as_str())
                    .with_update(&TablePublication::Replace {
                        table_name: AvailableSubjects::AnalyticsEvents.to_string(),
                    })
                    .with_publisher(diagnostic_session.session_context_name)
                    .make_name()
                    .unwrap()
                    .build()
                    .unwrap();
                let table = sss
                    .get_session_context()
                    .get_states()
                    .get(AvailableSubjects::SessionTasks.to_string().as_str())
                    .unwrap()
                    .read();
                let tasks_message = IPCMessage::get_builder()
                    .with_message(table.to_ipc_stream().unwrap())
                    .with_subject(AvailableSubjects::AnalyticsTasks.to_string().as_str())
                    .with_update(&TablePublication::Replace {
                        table_name: AvailableSubjects::AnalyticsTasks.to_string(),
                    })
                    .with_publisher(diagnostic_session.session_context_name)
                    .make_name()
                    .unwrap()
                    .build()
                    .unwrap();
                let table = sss
                    .get_session_context()
                    .get_states()
                    .get(AvailableSubjects::SessionErrors.to_string().as_str())
                    .unwrap()
                    .read();
                if table.count_rows() > 0 {
                    let errors_message = IPCMessage::get_builder()
                        .with_message(table.to_ipc_stream().unwrap())
                        .with_subject(AvailableSubjects::AnalyticsErrors.to_string().as_str())
                        .with_update(&TablePublication::Replace {
                            table_name: AvailableSubjects::AnalyticsErrors.to_string(),
                        })
                        .with_publisher(diagnostic_session.session_context_name)
                        .make_name()
                        .unwrap()
                        .build()
                        .unwrap();

                    create_message_map(vec![
                        metrics_message,
                        traces_message,
                        events_message,
                        errors_message,
                        tasks_message,
                    ])
                } else {
                    create_message_map(vec![
                        metrics_message,
                        traces_message,
                        events_message,
                        tasks_message,
                    ])
                }
            };
            dbg!(&message_map.keys());
            // Make the diagnostics session stream
            let session_ctx = diagnostic_session
                .build()
                .with_name(diagnostic_session.session_context_name)
                .add_session_interface(Some(&[AvailableInterfaceSubjects::AggregatedAttachments
                    .to_string()
                    .as_str()])).unwrap()
                .build_with_tables()
                .unwrap();
            let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));
            let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));

            // Run and update the session and convert the output to the user specified format
            match (&payload.get_format(), payload.get_stream()) {
                (DataFormat::Bytes, true) => {
                    // Convert the output to bytes
                    let response = session_stream.into_stream().map_ok(move |f| {
                        f.into_iter()
                            .filter(|(_k, v)| {
                                v.get_name()
                                    .contains(diagnostic_session.session_context_name)
                            })
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
                        .filter(|(_k, v)| {
                            v.get_name()
                                .contains(diagnostic_session.session_context_name)
                        })
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

                    // Send the stream
                    Body::from(response).into_response()
                }
                (DataFormat::Ipc, true) => {
                    // Convert the output to IPC
                    let response = session_stream.into_stream().map_ok(move |f| {
                        f.into_iter()
                            .filter(|(_k, v)| {
                                v.get_name()
                                    .contains(diagnostic_session.session_context_name)
                            })
                            .flat_map(|(_k, v)| v.get_message_own())
                            .collect::<Vec<_>>()
                    });

                    // Send the stream
                    Body::from_stream(response).into_response()
                }
                (DataFormat::Ipc, false) => {
                    // Convert the output to bytes
                    let response: Vec<HashMap<String, IPCMessage>> =
                        session_stream.try_collect().await.unwrap();
                    let response = response
                        .into_iter()
                        .flatten()
                        .filter(|(_k, v)| {
                            v.get_name()
                                .contains(diagnostic_session.session_context_name)
                        })
                        .flat_map(|(_k, v)| v.get_message_own())
                        .collect::<Vec<_>>();

                    let sss = session_stream_state.read();
                    let table = sss
                        .get_session_context()
                        .get_states()
                        .get(AvailableSubjects::SessionErrors.to_string().as_str())
                        .unwrap()
                        .read();
                    println!("__ERRORS__");
                    println!("{}", String::from_utf8(table.to_csv(b',', true).unwrap()).unwrap());

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
