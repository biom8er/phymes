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
use phymes_agents::{SessionInterfaceMessage, SessionInterfaceMessageTrait};
use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuilderTrait, CsvFormat, DataFormat,
    JoinUserInboxSessionContextsMermaidDiagrams, MessageTrait, SubjectBuilder, SubjectBuilderTrait,
    SubjectTrait,
};

// Library imports
use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::state::{ServerState, UserState};

/// Put state input
#[axum::debug_handler]
pub async fn session_build(
    Extension((current_user, user_session_contexts)): Extension<(
        String,
        Vec<JoinUserInboxSessionContextsMermaidDiagrams>,
    )>,
    State((users, mut state)): State<(UserState, ServerState)>,
    payload: Result<Json<SessionInterfaceMessage>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Build new session with session_name {}",
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
                let _session_names = match state
                    .make_session_contexts(&user_session_contexts, true, users.users.runtime_env())
                    .await
                {
                    Ok(session_names) => session_names,
                    Err(err) => {
                        return JsonError::new(err.to_string())
                            .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                };
            }

            // Extract out the Mermaid table
            let table = match payload.get_format() {
                DataFormat::Csv(csv_format) => SubjectBuilder::new()
                    .with_schema(AvailableSubjects::SessionMermaid.to_schema())
                    .with_name(payload.get_subject())
                    .with_csv(
                        payload.get_message(),
                        csv_format.delimiter,
                        csv_format.header,
                        csv_format.batch_size,
                    )
                    .unwrap()
                    .build()
                    .unwrap(),
                DataFormat::CsvDefault => {
                    let csv_format = CsvFormat::default();
                    SubjectBuilder::new()
                        .with_schema(AvailableSubjects::SessionMermaid.to_schema())
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
                }
                DataFormat::JsonDefault => {
                    let json_value: Vec<serde_json::Value> =
                        serde_json::from_slice(payload.get_message()).unwrap();
                    SubjectBuilder::new()
                        .with_schema(AvailableSubjects::SessionMermaid.to_schema())
                        .with_name(payload.get_subject())
                        .with_json_values(&json_value)
                        .unwrap()
                        .build()
                        .unwrap()
                }
                DataFormat::Bytes => SubjectBuilder::new()
                    .with_schema(AvailableSubjects::SessionMermaid.to_schema())
                    .with_name(payload.get_subject())
                    .with_bytes(payload.get_message())
                    .unwrap()
                    .build()
                    .unwrap(),
                DataFormat::Ipc => SubjectBuilder::new_from_ipc_stream(payload.get_message())
                    .unwrap()
                    .with_name(payload.get_subject())
                    .build()
                    .unwrap(),
                _ => unimplemented!(),
            };

            // Extract out the columns
            let session_context_name = table
                .get_column_as_vec_nonprimitive::<String>("session_context_name")
                .unwrap();
            let flowchart_diagram = table
                .get_column_as_vec_nonprimitive::<String>("flowchart_diagram")
                .unwrap();
            let er_diagram = table
                .get_column_as_vec_nonprimitive::<String>("er_diagram")
                .unwrap();
            let timestamp = table
                .get_column_as_vec_primitive::<i64>("timestamp")
                .unwrap();
            let combined = session_context_name
                .into_iter()
                .zip(flowchart_diagram.into_iter())
                .zip(er_diagram.into_iter())
                .zip(timestamp.into_iter())
                .map(
                    |(((a, b), c), d)| JoinUserInboxSessionContextsMermaidDiagrams {
                        email: current_user.to_owned(),
                        session_context_name: a,
                        flowchart_diagram: b,
                        er_diagram: c,
                        timestamp: d,
                    },
                )
                .collect::<Vec<JoinUserInboxSessionContextsMermaidDiagrams>>();

            // Add the new mermaid diagrams to the user session contexts
            let _session_names = match state
                .make_session_contexts(&combined, true, users.users.runtime_env())
                .await
            {
                Ok(session_names) => session_names,
                Err(err) => {
                    return JsonError::new(err.to_string())
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            // Update the users state with the new sessions
            users
                .update_user_session_contexts(
                    current_user.as_str(),
                    &table
                        .get_column_as_vec_nonprimitive::<String>("session_context_name")
                        .unwrap(),
                    &table
                        .get_column_as_vec_nonprimitive::<String>("flowchart_diagram")
                        .unwrap(),
                    &table
                        .get_column_as_vec_nonprimitive::<String>("er_diagram")
                        .unwrap(),
                    &table
                        .get_column_as_vec_primitive::<i64>("timestamp")
                        .unwrap(),
                )
                .await
                .unwrap();

            // Send the response
            Body::from(serde_json::to_string("State updated with new sessions.").unwrap())
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
