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
use parking_lot::RwLock;
use phymes_agents::session_plans::available_interface_subjects::create_message_map;
use phymes_core::{
    session::{common_traits::BuilderTrait, message::{SessionInterfaceMessage, SessionInterfaceMessageTrait}}, 
    table::{data_format::{CsvFormat, DataFormat}, table_trait::{TableBuilder, TableBuilderTrait, TableTrait}}, 
    task::message::{IPCMessageBuilder, MessageBuilderTrait, MessageTrait}};

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::state::server_state::ServerState;

/// Put state input
#[axum::debug_handler]
pub async fn session_put_state(
    Extension(current_user): Extension<String>,
    State(state): State<Arc<RwLock<ServerState>>>,
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

            match state
                .try_write()
                .unwrap()
                .session_contexts
                .get(payload.get_session_name())
            {
                Some(session_stream_state) => {
                    let schema = session_stream_state
                        .try_read()
                        .unwrap()
                        .get_session_context()
                        .get_states()
                        .get(payload.get_subject())
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_schema();
                    let bytes = match payload.get_format() {
                        DataFormat::Csv(csv_format) => TableBuilder::new()
                            .with_schema(schema)
                            .with_name(payload.get_subject())
                            .with_csv(payload.get_message(), csv_format.delimiter, csv_format.header, csv_format.batch_size)
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
                                .with_csv(payload.get_message(), csv_format.delimiter, csv_format.header, csv_format.batch_size)
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap()
                        },
                        DataFormat::JsonDefault => {                            
                            let json_value: Vec<serde_json::Value> = serde_json::from_slice(payload.get_message()).unwrap();                            
                            TableBuilder::new()
                                .with_schema(schema)
                                .with_name(payload.get_subject())
                                .with_json_values(&json_value)
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap()
                        },
                        DataFormat::Bytes => {                                                       
                            TableBuilder::new()
                                .with_schema(schema)
                                .with_name(payload.get_subject())
                                .with_bytes(payload.get_message())
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap()
                        },
                        DataFormat::Ipc => payload.get_message().to_owned(),
                        _ => unimplemented!(),
                    };

                    // Create the update message
                    let message = IPCMessageBuilder::new()
                        .with_name(payload.get_subject())
                        .with_subject(payload.get_subject())
                        .with_publisher(payload.get_publisher())
                        .with_message(bytes)
                        .with_update(payload.get_update())
                        .build()
                        .unwrap();
                    let message_map = create_message_map(vec![message]);

                    // Update the session state with the new message
                    let update = session_stream_state
                        .try_write()
                        .unwrap()
                        .update_state_from_messages(message_map);

                    // Update the superstep
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
            if let Err(e) = state.try_read().unwrap().write_session_contexts(
                &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                &current_user,
            ) {
                return JsonError::new(format!("Failed to write the session stream state {e:?}"))
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

/// Get state endpoint
#[axum::debug_handler]
pub async fn session_get_state(
    Extension(_current_user): Extension<String>,
    State(state): State<Arc<RwLock<ServerState>>>,
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

            match state
                .try_write()
                .unwrap()
                .session_contexts
                .get(payload.get_session_name())
            {
                Some(session_stream_state) => {
                    
                    // Update the metrics and row counts just in case...
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .get_session_context_mut()
                        .update_metrics_table()
                        .unwrap();
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .get_session_context_mut()
                        .update_subject_num_rows_table();

                    match payload.get_format() {
                        DataFormat::Bytes => {
                            // Get the subject table as a json object
                            let buf = session_stream_state
                                .try_read()
                                .unwrap()
                                .get_session_context()
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
                            let out = session_stream_state
                                .try_read()
                                .unwrap()
                                .get_session_context()
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
                            let out = session_stream_state
                                .try_read()
                                .unwrap()
                                .get_session_context()
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
                            let out = session_stream_state
                                .try_read()
                                .unwrap()
                                .get_session_context()
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
                            let out = session_stream_state
                                .try_read()
                                .unwrap()
                                .get_session_context()
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
                },
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
