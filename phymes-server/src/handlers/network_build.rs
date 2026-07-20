use axum::{
    Extension,
    body::Body,
    extract::{Json, State, rejection::JsonRejection},
    http::StatusCode,
    response::IntoResponse,
};
use anyhow::Result;
use clap::ValueEnum;
use phymes_message::{MessageTrait, NetworkInterfaceMessage, NetworkInterfaceMessageTrait};
use phymes_network::{NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait};
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, CsvFormat, DataFormat,
    JoinUserInboxNetworksMermaidDiagrams,
};
use phymes_subject::{BuilderTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
use serde::{Deserialize, Serialize};

use crate::handlers::json_error::{ErrorToResponse, JsonError, serde_json_error_response};
use crate::state::{ServerState, UserState};

#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum NetworkBuildSubjects {
    #[value(name = "AddNetwork")]
    AddNetwork,
    /// Build the full network from flowchart and er diagrams
    #[default]
    #[value(name = "CheckFlowchartAndERDiagrams")]
    CheckFlowchartAndERDiagrams,
    /// Try to build from flowchart diagram
    #[value(name = "CheckFlowchartDiagram")]
    CheckFlowchartDiagram,
    /// Try to build from the er diagram
    #[value(name = "CheckERDiagram")]
    CheckERDiagram,
    /// Auto fill er diagram subject parameters
    #[value(name = "AutoFillERDiagram")]
    AutoFillERDiagram,
}

impl std::fmt::Display for NetworkBuildSubjects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddNetwork => write!(f, "AddNetwork"),
            Self::CheckFlowchartAndERDiagrams => write!(f, "CheckFlowchartAndERDiagrams"),
            Self::CheckFlowchartDiagram => write!(f, "CheckFlowchartDiagram"),
            Self::CheckERDiagram => write!(f, "CheckERDiagram"),
            Self::AutoFillERDiagram => write!(f, "AutoFillERDiagram"),
        }
    }
}

/// Put state input
#[axum::debug_handler]
pub async fn network_build(
    Extension((current_user, user_networks)): Extension<(
        String,
        Vec<JoinUserInboxNetworksMermaidDiagrams>,
    )>,
    State((users, mut state)): State<(UserState, ServerState)>,
    payload: Result<Json<NetworkInterfaceMessage>, JsonRejection>,
) -> impl IntoResponse {
    // Extract and process the payload
    match payload {
        Ok(payload) => {
            // We got a valid JSON payload
            tracing::debug!(
                "Build new network with network_name {}",
                payload.get_network_name()
            );

            // Add user state if it does not exist already
            if !state
                .user_network_names
                .try_read()
                .unwrap()
                .contains_key(&current_user)
            {
                // Initialize the user network contexts
                let _network_names = match state
                    .make_networks(&user_networks, true, users.users.runtime_env())
                    .await
                {
                    Ok(network_names) => network_names,
                    Err(err) => {
                        return JsonError::new(err.to_string())
                            .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                };
            }

            // Extract out the Mermaid table
            let subject = match payload.get_format() {
                DataFormat::Csv(csv_format) => SubjectBuilder::new()
                    .with_schema(AvailableSubjects::NetworkMermaid.to_schema())
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
                        .with_schema(AvailableSubjects::NetworkMermaid.to_schema())
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
                        .with_schema(AvailableSubjects::NetworkMermaid.to_schema())
                        .with_name(payload.get_subject())
                        .with_json_values(&json_value)
                        .unwrap()
                        .build()
                        .unwrap()
                }
                DataFormat::Bytes => SubjectBuilder::new()
                    .with_schema(AvailableSubjects::NetworkMermaid.to_schema())
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
            let network_name = subject
                .get_column_as_vec_nonprimitive::<String>("network_name")
                .unwrap();
            let flowchart_diagram = subject
                .get_column_as_vec_nonprimitive::<String>("flowchart_diagram")
                .unwrap();
            let er_diagram = subject
                .get_column_as_vec_nonprimitive::<String>("er_diagram")
                .unwrap();
            let timestamp = subject
                .get_column_as_vec_primitive::<i64>("timestamp")
                .unwrap();

            // Based on the subject
            // 1. Add new to users
            // 2. AutoFill ER diagram
            // 3. Test build

            let network_build_subject = match NetworkBuildSubjects::from_str(payload.get_subject(), false) {
                Ok(nbs) => nbs,
                Err(err) => {
                    return JsonError::new(err.to_string()).to_response(StatusCode::BAD_REQUEST);
                }
            };

            let response = match network_build_subject {
                NetworkBuildSubjects::AddNetwork => {
                    let combined = network_name
                        .into_iter()
                        .zip(flowchart_diagram.into_iter())
                        .zip(er_diagram.into_iter())
                        .zip(timestamp.into_iter())
                        .map(|(((a, b), c), d)| JoinUserInboxNetworksMermaidDiagrams {
                            email: current_user.to_owned(),
                            network_name: a,
                            flowchart_diagram: b,
                            er_diagram: c,
                            timestamp: d,
                        })
                        .collect::<Vec<JoinUserInboxNetworksMermaidDiagrams>>();

                    // Add the new mermaid diagrams to the user network contexts
                    let _network_names = match state
                        .make_networks(&combined, true, users.users.runtime_env())
                        .await
                    {
                        Ok(network_names) => network_names,
                        Err(err) => {
                            return JsonError::new(err.to_string())
                                .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                        }
                    };

                    // Update the users state with the new networks
                    users
                        .update_user_networks(
                            current_user.as_str(),
                            &subject
                                .get_column_as_vec_nonprimitive::<String>("network_name")
                                .unwrap(),
                            &subject
                                .get_column_as_vec_nonprimitive::<String>("flowchart_diagram")
                                .unwrap(),
                            &subject
                                .get_column_as_vec_nonprimitive::<String>("er_diagram")
                                .unwrap(),
                            &subject
                                .get_column_as_vec_primitive::<i64>("timestamp")
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    serde_json::to_string("State updated with new networks.").unwrap()
                }
                NetworkBuildSubjects::CheckFlowchartAndERDiagrams => {
                    let combined = network_name
                        .into_iter()
                        .zip(flowchart_diagram.into_iter())
                        .zip(er_diagram.into_iter())
                        .zip(timestamp.into_iter())
                        .map(|(((a, b), c), d)| {
                            let mut builder = match NetworkBuilder::from_mermaid_flowchart(&b, false) {
                                Ok(builder) => builder,
                                Err(err) => format!("{err:?}"),
                            };
                            builder = match builder.with_subjects_from_mermaid_erdiagram(&c, false, true) {
                                Ok(builder) => builder,
                                Err(err) => {
                                    build_errors.write().push_str(format!("{err:?}").as_str());
                                    return;
                                },
                            };
                            let _network = match builder.with_name(&a)
                                .add_processor_subjects().unwrap()
                                .add_network_interface(None).unwrap()
                                .build_with_tables()
                            {
                                Ok(network) => network,
                                Err(err) => {
                                    build_errors.write().push_str(format!("{err:?}").as_str());
                                    return;
                                },
                            };
                        })
                        .collect::<Vec<_>>();
                    serde_json::to_string(&combined).unwrap()
                }
                NetworkBuildSubjects::CheckFlowchartDiagram => {
                    let combined = network_name
                        .into_iter()
                        .zip(flowchart_diagram.into_iter())
                        .zip(er_diagram.into_iter())
                        .zip(timestamp.into_iter())
                        .map(|(((a, b), c), d)| {
                            match NetworkBuilder::from_mermaid_flowchart(&b, false) {
                                Ok(_res) => String::new(),
                                Err(err) => err.to_string(),
                            }
                        })
                        .collect::<Vec<_>>();
                    serde_json::to_string(&combined).unwrap()
                }
                NetworkBuildSubjects::CheckERDiagram => {
                    let combined = network_name
                        .into_iter()
                        .zip(flowchart_diagram.into_iter())
                        .zip(er_diagram.into_iter())
                        .zip(timestamp.into_iter())
                        .map(|(((a, b), c), d)| {
                            match NetworkBuilder::default().with_subjects_from_mermaid_erdiagram(&c, false, true) {
                                Ok(_res) => String::new(),
                                Err(err) => err.to_string(),
                            }
                        })
                        .collect::<Vec<_>>();
                    serde_json::to_string(&combined).unwrap()
                }
                NetworkBuildSubjects::AutoFillERDiagram => {
                    let combined = network_name
                        .into_iter()
                        .zip(flowchart_diagram.into_iter())
                        .zip(er_diagram.into_iter())
                        .zip(timestamp.into_iter())
                        .map(|(((a, b), c), d)| {
                            // Generate defaults if possible
                            match NetworkBuilder::from_mermaid_flowchart(&b, false) {
                                // Read in what information is available and update the rest
                                Ok(builder) => {
                                    let builder = if c.is_empty() {
                                        builder
                                    } else if let Ok(builder) = builder.with_subjects_from_mermaid_erdiagram(&c, false, true) {
                                        builder
                                    } else {
                                        // Revert
                                        NetworkBuilder::from_mermaid_flowchart(&b, false).unwrap()
                                    };
                                    match builder.with_name(&a).add_processor_subjects() {
                                        // Include the last row of data during the prototyping stage
                                        Ok(builder) => match builder.to_mermaid_erdiagram(true, true) {
                                            Ok(diagram) => diagram,
                                            Err(err) => format!("{err:?}"),
                                        },
                                        Err(err) => format!("{err:?}"),
                                    }
                                },
                                Err(err) => format!("{err:?}"),
                            }
                        })
                        .collect::<Vec<_>>();
                    serde_json::to_string(&combined).unwrap()
                }
            };

            // Send the response
            Body::from(response).into_response()
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
