use dioxus::prelude::*;

use phymes_diagnostics::create_timestamp_micros;
use phymes_event::Publication;
use phymes_message::{
    MessageBuilderTrait, NetworkInterfaceMessage, NetworkInterfaceMessageBuilderTrait,
};
use phymes_schemas::{create_network_mermaid_batch, AvailableSubjects, DataFormat};
use phymes_server::{NetworkBuildSubjects, NetworkBuildResponse, NetworkBuildResult, create_network_name};
use phymes_subject::{BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait, SubjectTrait};
use phymes_templates::AvailableNetworks;

use crate::state::{
    filter_in_mermaid_diagrams_by_network_name, filter_out_mermaid_diagrams_by_network_name,
    get_non_duplicated_sorted_subjects,
    svg_icons::{
        ms_checkmark_circle_icon_svg, b8_save_icon_svg, fa_trash_icon_svg, ms_code_icon_svg, ms_column_arrow_right_icon_svg,
        ms_deploy_icon_svg, ms_edit_icon_svg, ms_search_icon_svg, ms_chevron_circle_icon_svg,
    },
    sync_network_names_state, SyncNetworkNamesState, EMAIL, JWT, SESSION_NAMES,
};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(feature = "serverless")]
use crate::state::RUNTIME_ENV;
#[cfg(feature = "serverless")]
use bytes::Bytes;
#[cfg(feature = "serverless")]
use futures::TryStreamExt;
#[cfg(feature = "serverless")]
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

/// View for the builds drop down menu
#[component]
pub fn builds_dropdown_view(
    mut is_flowchart_shown: Signal<bool>,
    mut active_network_name: Signal<String>,
    mut active_flowchart_diagram: Signal<String>,
    mut active_er_diagram: Signal<String>,
    mut mermaid_network_names: Signal<Vec<String>>,
    mut mermaid_flowchart_diagrams: Signal<Vec<String>>,
    mut mermaid_er_diagrams: Signal<Vec<String>>,
    mut mermaid_timestamps: Signal<Vec<i64>>,
    mut is_saved: Signal<bool>,
    mut build_errors: Signal<String>,
) -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_network_names_state);
    let sync_network_names = use_coroutine_handle::<SyncNetworkNamesState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    let mut subject_dropdown = use_signal(String::new);

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &mermaid_network_names
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(Vec::new);

    rsx! {
        div {
            class: "p-2 rounded bg-neutral-800 grid grid-rows-[auto_1fr] grid-cols-[1fr_auto]",
            form {
                class: "w-full h-full flex row-span-1 col-span-1 row-start-1 col-start-1",
                input {
                    class: "w-full rounded bg-neutral-700",
                    r#type: "text",
                    placeholder: "search network",
                    value: "{subject_dropdown}",
                    onclick: move |_| show_subject_dropdown.set(true),
                    onfocusout: move |_| show_subject_dropdown.set(false),
                    oninput: move |evt| subject_dropdown.set(evt.value()),
                    onkeyup: move |_| {
                        subjects_filtered.set(subjects_vec().iter()
                            .filter(|s| !s.contains(subject_dropdown.read().as_str()))
                            .cloned()
                            .collect::<Vec<_>>());
                    }
                },
            },

            // Dynamic dropdown
            if show_subject_dropdown() {
                div {
                    class: "p-2 rounded bg-neutral-800 list-none flex row-span-1 col-span-1 row-start-2 col-start-1",
                    ul {
                        {subjects_vec().iter().filter(|s| active_network_name.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
                            let sub = sub.clone();
                            rsx! {
                                li {
                                    class: "hover:bg-neutral-700 cursor-pointer",
                                    key: "{i}",
                                    div {
                                        onmouseover: move |_evt| subject_dropdown.set(sub.clone()),
                                        p { "{sub}" },
                                    }
                                }
                            }
                        })}
                    }
                }
            }

            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-2",
                button {
                    class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                    onclick: move |_evt| async move {
                        // Reset any build errors
                        build_errors.set(String::new());
                        
                        // Reset the dropdown
                        active_network_name.set(subject_dropdown.try_read().unwrap().to_string());
                        subject_dropdown.set(String::new());
                    },
                    svg {
                        class: "max-w-[24px] max-h-[24px]",
                        dangerous_inner_html: ms_search_icon_svg()
                    },
                },

                if !active_network_name().is_empty() {
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_evt| async move {
                            // Make a defualt name for the copy of the active network
                            let active_network = format!("{}-copy", active_network_name.read());

                            // Copy the diagrams
                            mermaid_network_names.push(active_network.clone());
                            mermaid_flowchart_diagrams.push(active_flowchart_diagram.read().to_string());
                            mermaid_er_diagrams.push(active_er_diagram.read().to_string());
                            mermaid_timestamps.push(create_timestamp_micros());

                            // Set the active network
                            active_network_name.set(active_network.clone());
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_column_arrow_right_icon_svg()
                        },
                    },
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_evt| async move {
                            // Change the name of all active network diagrams
                            let (network_names, flowchart_diagrams, er_diagrams, timestamps) = filter_in_mermaid_diagrams_by_network_name(
                                &active_network_name(),
                                &mermaid_network_names
                                    .read()
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>(),
                                &mermaid_flowchart_diagrams
                                    .read()
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>(),
                                &mermaid_er_diagrams
                                    .read()
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>(),
                                &mermaid_timestamps());
                            let network_names = network_names.into_iter().map(|s| format!("__deleted__{s}")).collect::<Vec<_>>();
                            let batch_deleted = create_network_mermaid_batch(network_names, flowchart_diagrams, er_diagrams, timestamps).unwrap();

                            // Filter out the active network
                            let (network_names, flowchart_diagrams, er_diagrams, timestamps) = filter_out_mermaid_diagrams_by_network_name(
                                &active_network_name(),
                                &mermaid_network_names
                                    .read()
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>(),
                                &mermaid_flowchart_diagrams
                                    .read()
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>(),
                                &mermaid_er_diagrams
                                    .read()
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>(),
                                &mermaid_timestamps());
                            let active_network = network_names.first().unwrap().to_string();
                            let batch = create_network_mermaid_batch(network_names, flowchart_diagrams, er_diagrams, timestamps).unwrap();

                            // Update the mermaid state with the active diagram
                            let route = "/app/v1/put_state";
                            let message = Subject::get_builder()
                                .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
                                .with_record_batches(vec![batch_deleted, batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let data_serialized = serde_json::to_string(&NetworkInterfaceMessage::get_builder()
                                .with_network_name(&create_network_name(EMAIL().as_str(), AvailableNetworks::Builder.to_string().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_network_name(EMAIL().as_str(), AvailableNetworks::Builder.to_string().as_str()))
                                .with_update(&Publication::Replace { subject_name: AvailableSubjects::BuilderMermaid.to_string() })
                                .with_stream(false)
                                .with_subject(AvailableSubjects::BuilderMermaid.to_string().as_str())
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap()).unwrap();

                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.read().to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(response) => match response.text().await {
                                    Ok(text) => tracing::debug!("{text}"),
                                    Err(err) => tracing::error!("{err:?}"),
                                },
                                Err(err) => tracing::error!("{err:?}"),
                            }

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT.read().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let _text = String::from_utf8_lossy(bytes.first().unwrap()).into_owned();
                                }
                                Err(err) => tracing::error!("{err:?}"),
                            }

                            // Reset the active network to the first network
                            active_network_name.set(active_network);
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: fa_trash_icon_svg()
                        },
                    },
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            let current = is_flowchart_shown.read().to_owned();
                            is_flowchart_shown.set(!current);
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_chevron_circle_icon_svg()
                        },
                    },
                    
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            // Clear any text
                            build_errors.set(String::new());

                            // Determine the subject within build to publish on
                            let subject = if is_flowchart_shown() {
                                NetworkBuildSubjects::CheckFlowchartDiagram.to_string()
                            } else {
                                NetworkBuildSubjects::CheckERDiagram.to_string()
                            };

                            // Check for build errors
                            let batch = create_network_mermaid_batch(vec![active_network_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Subject::get_builder()
                                .with_name(subject.as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let data_serialized = serde_json::to_string(&NetworkInterfaceMessage::get_builder()
                                .with_network_name(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_update(&Publication::None)
                                .with_stream(false)
                                .with_subject(subject.as_str())
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap()).unwrap();

                            let route = "/app/v1/build";
                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            let build_result = match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.read().to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(response) => match response.json::<NetworkBuildResponse>().await {
                                    Ok(mut response) => {
                                        if let Some(mut results) = response.response.take() {
                                            if let Some(result) = results.pop() {
                                                result                                               
                                            } else {
                                                tracing::debug!("No NetworkBuildResponse result found for check diagram.");
                                                NetworkBuildResult::new(None, None)
                                            }
                                        } else {
                                            tracing::debug!("No NetworkBuildResponse found for check diagram.");
                                            NetworkBuildResult::new(None, None)
                                        }
                                    }
                                    Err(err) => {
                                        tracing::error!("{err:?}");
                                        NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                    }
                                },
                                Err(err) => {
                                    tracing::error!("{err:?}");
                                    NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                }
                            };

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT.read().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            let build_result = match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let bytes = bytes.into_iter().flatten().collect::<Vec<_>>();
                                    match serde_json::from_slice::<NetworkBuildResponse>(bytes.as_slice()) { 
                                        Ok(mut response) => {
                                            if let Some(mut results) = response.response.take() {
                                                if let Some(result) = results.pop() {
                                                    result                                                
                                                } else {
                                                    NetworkBuildResult::new(None, None)
                                                }
                                            } else {
                                                NetworkBuildResult::new(None, None)
                                            }
                                        }
                                        Err(err) => {
                                            tracing::error!("{err:?}");
                                            NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                        }
                                    }
                                }
                                Err(err) => {
                                    tracing::error!("{err:?}");
                                    NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                }
                            };
                            if let Some(err) = build_result.error {
                                build_errors.write().push_str(&err);
                                return;
                            }
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_checkmark_circle_icon_svg()
                        },
                    },
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            // Clear any text
                            build_errors.set(String::new());

                            // Check the name of the network
                            if SESSION_NAMES.read().iter().any(|s| s==&active_network_name()) {
                                build_errors.write().push_str(format!("Network name '{}' already exists. Please choose a different name.", active_network_name()).as_str());
                                return;
                            }

                            // Check if the current network can be built
                            let route = "/app/v1/build";
                            let batch = create_network_mermaid_batch(vec![active_network_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Subject::get_builder()
                                .with_name(NetworkBuildSubjects::CheckFlowchartAndERDiagrams.to_string().as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let data_serialized = serde_json::to_string(&NetworkInterfaceMessage::get_builder()
                                .with_network_name(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_update(&Publication::None)
                                .with_stream(false)
                                .with_subject(NetworkBuildSubjects::CheckFlowchartAndERDiagrams.to_string().as_str())
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap()).unwrap();

                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            let build_result = match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.read().to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(response) => match response.json::<NetworkBuildResponse>().await {
                                    Ok(mut response) => {
                                        if let Some(mut results) = response.response.take() {
                                            if let Some(result) = results.pop() {
                                                result
                                            } else {
                                                NetworkBuildResult::new(None, Some("Empty NetworkBuildResponse"))
                                            }
                                        } else {
                                            NetworkBuildResult::new(None, None)
                                        }
                                    }
                                    Err(err) => {
                                        tracing::error!("{err:?}");
                                        NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                    }
                                },
                                Err(err) => {
                                    tracing::error!("{err:?}");
                                    NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                }
                            };

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT.read().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            let build_result = match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let bytes = bytes.into_iter().flatten().collect::<Vec<_>>();
                                    match serde_json::from_slice::<NetworkBuildResponse>(bytes.as_slice()) {                    
                                        Ok(mut response) => {
                                            if let Some(mut results) = response.response.take() {
                                                if let Some(result) = results.pop() {
                                                    result
                                                } else {
                                                    NetworkBuildResult::new(None, Some("Empty NetworkBuildResponse"))
                                                }
                                            } else {
                                                tracing::error!("Missing NetworkBuildResponse");
                                                NetworkBuildResult::new(None, None)
                                            }
                                        }
                                        Err(err) => {
                                            tracing::error!("{err:?}");
                                            NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                        }
                                    }
                                }
                                Err(err) => {
                                    tracing::error!("{err:?}");
                                    NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                }
                            };
                            if let Some(err) = build_result.error {
                                build_errors.write().push_str(&err);
                                return;
                            }

                            // Update the server with the new network
                            let batch = create_network_mermaid_batch(vec![active_network_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Subject::get_builder()
                                .with_name(NetworkBuildSubjects::AddNetwork.to_string().as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let data_serialized = serde_json::to_string(&NetworkInterfaceMessage::get_builder()
                                .with_network_name(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_update(&Publication::None)
                                .with_stream(false)
                                .with_subject(NetworkBuildSubjects::AddNetwork.to_string().as_str())
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap()).unwrap();
                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.read().to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(response) => match response.text().await {
                                    Ok(text) => tracing::debug!("{text}"),
                                    Err(err) => tracing::error!("{err:?}"),
                                },
                                Err(err) => tracing::error!("{err:?}"),
                            }

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT.read().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let _text = String::from_utf8_lossy(bytes.first().unwrap()).into_owned();
                                }
                                Err(err) => tracing::error!("{err:?}"),
                            }

                            // Update the frontend state with the new network so as not to require the user to re-signin
                            let mut network_plans = vec![active_network_name().to_string()];
                            network_plans.extend(SESSION_NAMES.read().iter().filter(|s| *s!=&active_network_name()).cloned());
                            sync_network_names.send(SyncNetworkNamesState { network_plans });
                            build_errors.write().push_str(format!("Network name '{}' has been built successfully.", active_network_name()).as_str());

                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_deploy_icon_svg()
                        },
                    },
                }

                // Show the save button only when modified
                if !is_saved() {
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            // Update the mermaid state with the active diagram
                            let route = "/app/v1/put_state";
                            let batch = create_network_mermaid_batch(vec![active_network_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Subject::get_builder()
                                .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let data_serialized = serde_json::to_string(&NetworkInterfaceMessage::get_builder()
                                .with_network_name(&create_network_name(EMAIL().as_str(), AvailableNetworks::Builder.to_string().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_network_name(EMAIL().as_str(), AvailableNetworks::Builder.to_string().as_str()))
                                .with_update(&Publication::Extend { subject_name: AvailableSubjects::BuilderMermaid.to_string() })
                                .with_stream(false)
                                .with_subject(AvailableSubjects::BuilderMermaid.to_string().as_str())
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap()).unwrap();

                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.read().to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(response) => match response.text().await {
                                    Ok(text) => tracing::debug!("{text}"),
                                    Err(err) => tracing::error!("{err:?}"),
                                },
                                Err(err) => tracing::error!("{err:?}"),
                            }

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT.read().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let _text = String::from_utf8_lossy(bytes.first().unwrap()).into_owned();
                                }
                                Err(err) => tracing::error!("{err:?}"),
                            }

                            // DM: limit the history to 25

                            // Change to saved
                            is_saved.set(true);
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: b8_save_icon_svg()
                        }
                    }
                }

                // Fill in missing ER diagram entries with defaults
                if !is_flowchart_shown() {
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            // Generate defaults if possible
                            let route = "/app/v1/build";
                            let batch = create_network_mermaid_batch(vec![active_network_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Subject::get_builder()
                                .with_name(NetworkBuildSubjects::AutoFillERDiagram.to_string().as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();
                            let data_serialized = serde_json::to_string(&NetworkInterfaceMessage::get_builder()
                                .with_network_name(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_network_name(EMAIL().as_str(), active_network_name().as_str()))
                                .with_update(&Publication::None)
                                .with_stream(false)
                                .with_subject(NetworkBuildSubjects::AutoFillERDiagram.to_string().as_str())
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap()).unwrap();

                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            let build_result = match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.read().to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(response) => match response.json::<NetworkBuildResponse>().await {
                                    Ok(mut response) => {
                                        if let Some(mut results) = response.response.take() {
                                            if let Some(result) = results.pop() {
                                                result
                                            } else {
                                                NetworkBuildResult::new(None, Some("Empty NetworkBuildResponse"))
                                            }
                                        } else {
                                            NetworkBuildResult::new(None, None)
                                        }
                                    }
                                    Err(err) => {
                                        tracing::error!("{err:?}");
                                        NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                    }
                                },
                                Err(err) => {
                                    tracing::error!("{err:?}");
                                    NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                }
                            };

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT.read().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            let build_result = match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let bytes = bytes.into_iter().flatten().collect::<Vec<_>>();
                                    match serde_json::from_slice::<NetworkBuildResponse>(bytes.as_slice()) {                    
                                        Ok(mut response) => {
                                            if let Some(mut results) = response.response.take() {
                                                if let Some(result) = results.pop() {
                                                    result
                                                } else {
                                                    NetworkBuildResult::new(None, Some("Empty NetworkBuildResponse"))
                                                }
                                            } else {
                                                tracing::error!("Missing NetworkBuildResponse");
                                                NetworkBuildResult::new(None, None)
                                            }
                                        }
                                        Err(err) => {
                                            tracing::error!("{err:?}");
                                            NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                        }
                                    }
                                }
                                Err(err) => {
                                    tracing::error!("{err:?}");
                                    NetworkBuildResult::new(None, Some(&format!("{err:?}")))
                                }
                            };

                            if let Some(diagram) = build_result.diagram {
                                active_er_diagram.set(diagram);

                                // Change to saved
                                is_saved.set(false);
                            } else if let Some(err) = build_result.error {
                                tracing::error!("{err}");
                            } else {
                                tracing::error!("Missing NetworkBuildResult for AutoFillERDiagram.");
                            }
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_code_icon_svg()
                        }
                    }
                }
            }
        }
    }
}

/// Code editor element with textarea callbacks
#[component]
pub fn diagram_code_editor(
    is_flowchart_shown: Signal<bool>,
    active_network_name: Signal<String>,
    mut active_flowchart_diagram: Signal<String>,
    mut active_er_diagram: Signal<String>,
    mut is_saved: Signal<bool>,
) -> Element {
    // Determine the code to show
    let code: Memo<String> = use_memo(move || {
        if is_flowchart_shown() {
            active_flowchart_diagram.read().to_string()
        } else {
            active_er_diagram.read().to_string()
        }
    });

    // Call back to update the code
    let on_input = move |event: Event<FormData>| async move {
        // Update the active diagrams
        if is_flowchart_shown() {
            active_flowchart_diagram.set(event.value());
        } else {
            active_er_diagram.set(event.value());
        };

        // Change to unsaved
        is_saved.set(false);
    };

    // Compute the line numbers for the gutter
    // DM: cannot use `lines` or there is a delay for new lines
    let line_count = code.read().split('\n').count().max(1);

    // Listener to synchronize scrolling between the gutter and code
    use_effect(move || {
        let _ = code.read();
        document::eval(
            r#"const gutter = document.getElementById('gutter');
const code = document.getElementById('code');
code.addEventListener('scroll', () => {
    gutter.scrollTop = code.scrollTop;
});"#,
        );
    });

    rsx! {
        div {
            class: "w-full h-full overflow-hidden rounded-md shadow-sm py-2 p-2 snap-y grid grid-cols-[3rem_1fr] font-mono text-sm leading-6 snap-start",
            div {
                id: "gutter",
                class: "h-full text-right flex flex-col whitespace-pre overflow-hidden",
                {(1..=line_count).map(|n| rsx! {
                    div {
                        class: "px-2 text-gray-500 select-none",
                        "{n}"
                    }
                })}
            }

            textarea {
                id: "code",
                value: "{code.to_string()}",
                oninput: on_input,
                class: "w-full h-full grow bg-neutral-800 px-3 resize-none focus:outline-none whitespace-pre overflow-auto"
            }
        }
    }
}

/// View for modifying the network name
#[component]
pub fn network_name_editor(mut active_network_name: Signal<String>) -> Element {
    let mut is_editing = use_signal(|| false);
    let mut network_name = use_signal(String::new);

    if !active_network_name().is_empty() {
        if is_editing() {
            rsx! {
                div {
                    class: "w-full rounded p-2 items-center flex flex-row bg-neutral-800",
                    form {
                        class: "w-full p-2 gap-2 rounded bg-neutral-800",
                        input {
                            r#type: "text",
                            placeholder: "{active_network_name}",
                            oninput: move |event| network_name.set(event.value()),
                            class: "w-full p-2 rounded bg-neutral-700",
                        }
                    }
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            active_network_name.set(network_name());
                            network_name.write().clear();
                            is_editing.set(false)
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: b8_save_icon_svg()
                        },
                    }
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        onclick: move |_| async move {
                            is_editing.set(false)
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: fa_trash_icon_svg()
                        },
                    }
                }
            }
        } else {
            rsx! {
                div {
                    class: "w-full rounded p-2 items-center flex flex-row bg-neutral-800",
                    p {
                        class: "w-full text-center bg-neutral-800",
                        "{active_network_name}"
                    }
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded cursor-pointer",
                        onclick: move |_| async move {
                            is_editing.set(true)
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_edit_icon_svg()
                        },
                    }
                }
            }
        }
    } else {
        rsx! {}
    }
}
