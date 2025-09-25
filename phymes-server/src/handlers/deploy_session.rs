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
use phymes_agents::session_plans::available_interface_subjects::create_message_map;
use phymes_core::{
    schemas::{available_subjects::{AvailableSubjects, AvailableSubjectsTrait}, user::UserSubject},
    session::{common_traits::BuilderTrait, message::{SessionInterfaceMessage, SessionInterfaceMessageTrait}}, 
    table::{data_format::{CsvFormat, DataFormat}, table::{TableBuilder, TableBuilderTrait, TableTrait}}, 
    task::message::{IPCMessageBuilder, MessageBuilderTrait, MessageTrait}};

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::state::server_state::ServerState;


/// Put state input
#[axum::debug_handler]
pub async fn deploy_session(
    Extension(current_user): Extension<UserSubject>,
    State(mut state): State<ServerState>,
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
                    .create_session_plans_by_email(&current_user.email)
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
                .get(payload.get_session_name())
            {
                Some(_session_stream_state) => return JsonError::new("Session stream state already exists".to_string())
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR),
                None => {
                    let batches = match payload.get_format() {
                        DataFormat::Csv(csv_format) => TableBuilder::new()
                            .with_schema(AvailableSubjects::Mermaid.to_schema())
                            .with_name(payload.get_subject())
                            .with_csv(payload.get_message(), csv_format.delimiter, csv_format.header, csv_format.batch_size)
                            .unwrap()
                            .build()
                            .unwrap()
                            .get_record_batches_own(),
                        DataFormat::CsvDefault => {
                            let csv_format = CsvFormat::default();
                            TableBuilder::new()
                                .with_schema(AvailableSubjects::Mermaid.to_schema())
                                .with_name(payload.get_subject())
                                .with_csv(payload.get_message(), csv_format.delimiter, csv_format.header, csv_format.batch_size)
                                .unwrap()
                                .build()
                                .unwrap()
                                .get_record_batches_own(),
                        },
                        DataFormat::JsonDefault => {                            
                            let json_value: Vec<serde_json::Value> = serde_json::from_slice(payload.get_message()).unwrap();                            
                            TableBuilder::new()
                                .with_schema(AvailableSubjects::Mermaid.to_schema())
                                .with_name(payload.get_subject())
                                .with_json_values(&json_value)
                                .unwrap()
                                .build()
                                .unwrap()
                                .get_record_batches_own(),
                        },
                        DataFormat::Bytes => TableBuilder::new()
                            .with_schema(AvailableSubjects::Mermaid.to_schema())
                            .with_name(payload.get_subject())
                            .with_bytes(payload.get_message())
                            .unwrap()
                            .build()
                            .unwrap()
                            .get_record_batches_own(),
                        DataFormat::Ipc => TableBuilder::new_from_ipc_stream(payload.get_message())
                            .with_name(payload.get_subject())
                            .unwrap()
                            .build()
                            .unwrap()
                            .get_record_batches_own(),
                        _ => unimplemented!(),
                    };
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
            Body::from(serde_json::to_string("Session deployed").unwrap()).into_response()
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