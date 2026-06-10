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
use futures::TryStreamExt;
use phymes_event::Subscription;
use phymes_message::{
    IPCMessageBuilder, MessageBuilderTrait, MessageTrait, SessionInterfaceMessage,
    SessionInterfaceMessageTrait, create_message_map,
};
use phymes_network::{NetworkStreamStep, NetworkStreamStepTrait};
use phymes_schemas::{CsvFormat, DataFormat, JoinUserInboxNetworksMermaidDiagrams};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait,
    SubjectTrait,
};
use phymes_task::SubscriptionTrait;

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::state::{ServerState, UserState};

/// Put state input
#[axum::debug_handler]
pub async fn network_put_subjects(
    Extension((current_user, user_networks)): Extension<(
        String,
        Vec<JoinUserInboxNetworksMermaidDiagrams>,
    )>,
    State((users, mut state)): State<(UserState, ServerState)>,
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
                .user_network_names
                .try_read()
                .unwrap()
                .contains_key(&current_user)
            {
                // Initialize the user session contexts
                let _session_names = match state
                    .make_networks(&user_networks, true, users.users.runtime_env())
                    .await
                {
                    Ok(session_names) => session_names,
                    Err(err) => {
                        return JsonError::new(err.to_string())
                            .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                };
            }

            let network_arc = match state
                .networks
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
            let schema = if let Some(schema) = network_arc.subjects().get(payload.get_subject()) {
                schema.clone()
            } else {
                return JsonError::new("Failed to get the schema for the session".to_string())
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR);
            };
            let bytes = match payload.get_format() {
                DataFormat::Csv(csv_format) => SubjectBuilder::new()
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
                    SubjectBuilder::new()
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
                    SubjectBuilder::new()
                        .with_schema(schema)
                        .with_name(payload.get_subject())
                        .with_json_values(&json_value)
                        .unwrap()
                        .build()
                        .unwrap()
                        .to_ipc_stream()
                        .unwrap()
                }
                DataFormat::Bytes => SubjectBuilder::new()
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
            let _step = NetworkStreamStep::current_superstep(&network_arc).await;
            if let Err(e) = NetworkStreamStep::update_subjects_and_changelog_from_messages(
                &network_arc,
                messages,
                0,
            )
            .await
            {
                return JsonError::new(format!("Failed to update the session stream state {e:?}"))
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
pub async fn network_get_subjects(
    Extension((current_user, user_networks)): Extension<(
        String,
        Vec<JoinUserInboxNetworksMermaidDiagrams>,
    )>,
    State((users, mut state)): State<(UserState, ServerState)>,
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
                .user_network_names
                .try_read()
                .unwrap()
                .contains_key(&current_user)
            {
                // Initialize the user session contexts
                let _session_names = match state
                    .make_networks(&user_networks, true, users.users.runtime_env())
                    .await
                {
                    Ok(session_names) => session_names,
                    Err(err) => {
                        return JsonError::new(err.to_string())
                            .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                };
            }

            let network_arc = match state.networks.write().get(payload.get_session_name()) {
                Some(network_arc) => network_arc.clone(),
                None => {
                    return JsonError::new("Failed to get the session stream state".to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            // Update the row counts just in case...
            let _ = network_arc.update_subject_num_rows().await;

            // Read the subject
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: payload.get_subject().to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())
            .unwrap()
            .unwrap()
            .try_collect()
            .await
            .unwrap();
            if batches.is_empty() {
                Body::from(Bytes::new()).into_response()
            } else {
                let subject = Subject::get_builder()
                    .with_name(payload.get_subject())
                    .with_record_batches(batches)
                    .unwrap()
                    .build()
                    .unwrap();
                match payload.get_format() {
                    DataFormat::Bytes => {
                        // Get the subject as bytes
                        let buf = subject.to_bytes().unwrap();
                        Body::from(buf).into_response()
                    }
                    DataFormat::Csv(csv_format) => {
                        // Get the subject table as a csv string
                        let out = subject
                            .to_csv(csv_format.delimiter, csv_format.header)
                            .unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    DataFormat::CsvDefault => {
                        // Get the subject table as a csv string
                        let csv_format = CsvFormat::default();
                        let out = subject
                            .to_csv(csv_format.delimiter, csv_format.header)
                            .unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    DataFormat::Json(_) | DataFormat::JsonDefault => {
                        // Get the subject table as a json string
                        let out = subject.to_json().unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    DataFormat::Ipc => {
                        // Get the subject table as a csv string
                        let out = subject.to_ipc_stream().unwrap();
                        let buf = Bytes::from(out);
                        Body::from(buf).into_response()
                    }
                    _ => unimplemented!(),
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
