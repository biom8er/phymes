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
use phymes_core::table::table::TableTrait;
use phymes_data::candle_data::summary_config::{CsvFormat, DataSummaryFormat};

// Library imports
use crate::handlers::sign_in::CurrentUser;
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::server::server_state::ServerState;

use super::session_info::SessionResponse;

/// Chat inference endpoint
#[axum::debug_handler]
pub async fn session_put_state(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
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
                    let bytes = match payload.format {
                        DataSummaryFormat::Csv(csv_format) => TableBuilder::new()
                            .with_schema(schema)
                            .with_name(payload.publish.get_table_name())
                            .with_csv(&payload.content, csv_format.delimiter, csv_format.header, csv_format.batch_size)?
                            .build()
                            .unwrap()
                            .to_ipc_stream()?,
                        DataSummaryFormat::CsvDefault => {
                            let csv_format = CsvFormat::default();
                            TableBuilder::new()
                                .with_schema(schema)
                                .with_name(payload.publish.get_table_name())
                                .with_csv(&payload.content, csv_format.delimiter, csv_format.header, csv_format.batch_size)?
                                .build()
                                .unwrap()
                                .to_ipc_stream()?
                        },
                        DataSummaryFormat::Json(_json_format) | DataSummaryFormat::Json => {                            
                            let json_value: Vec<serde_json::Value> = serde_json::from_slice(&payload.content)?;                            
                            TableBuilder::new()
                                .with_schema(schema)
                                .with_name(payload.publish.get_table_name())
                                .with_json_values(&payload.content)?
                                .build()
                                .unwrap()
                                .to_ipc_stream()?
                        },
                        DataSummaryFormat::Bytes => {                                                       
                            TableBuilder::new()
                                .with_schema(schema)
                                .with_name(payload.publish.get_table_name())
                                .with_bytes(&payload.content)?
                                .build()
                                .unwrap()
                                .to_ipc_stream()?
                        },
                        SessionResponseFormat::Pdf => {
                            // Load the PDF document and extract text
                            let pdf =
                                filter_pdf(load_pdf_document(payload.content.as_slice()).unwrap());
                            let batch =
                                extract_pdf_text([(payload.metadata.clone(), pdf)].as_slice())
                                    .unwrap();
                            ArrowTable::get_builder()
                                .with_name(payload.subject_name.as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()?
                        }
                        _ => unimplemented!(),
                    };

                    // Create the update message
                    let incoming_message = ArrowIncomingMessageBuilder::new()
                        .with_name(payload.subject_name.as_str())
                        .with_subject(payload.subject_name.as_str())
                        .with_publisher(payload.session_name.as_str())
                        .with_message(bytes)
                        .with_update(&payload.publish)
                        .build()
                        .unwrap();
                    let mut incoming_message_map =
                        HashMap::<String, ArrowIncomingMessage>::new();
                    incoming_message_map
                        .insert(incoming_message.get_name().to_string(), incoming_message);

                    // Update the session state with the new message
                    session_stream_state
                        .try_write()
                        .unwrap()
                        .update_state_from_messages(incoming_message_map);

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
            if let Err(e) = state.write_state_by_email(
                &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
                &current_user.email,
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

/// Chat inference endpoint
#[axum::debug_handler]
pub async fn session_get_state(
    Extension(current_user): Extension<CurrentUser>,
    State(mut state): State<ServerState>,
    payload: Result<Json<SessionResponse>, JsonRejection>,
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
                Some(session_stream_state) => match payload.format {
                    DataSummaryFormat::Bytes => {
                        // Get the subject table as a json object
                        let buf = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_bytes()
                            .unwrap();
                        Body::from(buf).into_response()
                    }
                    DataSummaryFormat::Csv(csv_format) => {
                        // Get the subject table as a csv string
                        let out = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_csv(csv_format.delimiter, csv_format.header)
                            .unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    DataSummaryFormat::CsvDefault => {
                        // Get the subject table as a csv string
                        let csv_format = CsvFormat::default();
                        let out = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_csv(csv_format.delimiter, csv_format.header)
                            .unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    DataSummaryFormat::Json(_json_format) | DataSummaryFormat::JsonDefault => {
                        // Get the subject table as a json string
                        let out = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_json()
                            .unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    DataSummaryFormat::Ipc => {
                        // Get the subject table as a csv string
                        let out = session_stream_state
                            .try_read()
                            .unwrap()
                            .get_session_context()
                            .get_states()
                            .get(payload.subject_name.as_str())
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .to_ipc_stream()
                            .unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    _ => unimplemented!(),
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
