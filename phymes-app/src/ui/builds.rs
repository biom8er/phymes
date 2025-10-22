use dioxus::prelude::*;
use phymes_agents::{session_plans::available_session_plans::AvailableSessionPlans, session_traits::mermaid::SessionContextBuilderMermaidTrait};
use phymes_core::{schemas::{available_subjects::{AvailableSubjects}, mermaid::create_session_mermaid_batch}, session::{common_traits::{BuildableTrait, BuilderTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilderTrait}, session_context_builder::SessionContextBuilder}, table::{DataFormat, table_trait::{Table, TableBuilderTrait, TableTrait}, TablePublish}, task::message::MessageBuilderTrait};
use phymes_diagnostics::create_timestamp_micros;
use phymes_server::handlers::sign_in::create_session_name;

use crate::state::{
    apps::{filter_in_mermaid_diagrams_by_session_name, get_non_duplicated_sorted_subjects, filter_out_mermaid_diagrams_by_session_name},
    sign_in::{sync_session_names_state, SyncSessionNamesState, EMAIL, JWT, SESSION_NAMES},
    svg_icons::{ms_column_arrow_right_icon_svg, ms_deploy_icon_svg, ms_edit_icon_svg, b8_save_icon_svg, ms_sync_icon_svg, fa_trash_icon_svg}
};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(feature = "serverless")]
use bytes::Bytes;
#[cfg(feature = "serverless")]
use futures::TryStreamExt;
#[cfg(feature = "serverless")]
use phymes_server::server::{
    serverless_app::{serverless_app, Serverless},
    serverless_config::ServerlessConfig,
};

/// View for the builds drop down menu
#[component]
pub fn builds_dropdown_view(mut is_flowchart_shown: Signal<bool>, mut active_session_name: Signal<String>, mut active_flowchart_diagram: Signal<String>, mut active_er_diagram: Signal<String>, 
    mut mermaid_session_context_names: Signal<Vec<String>>, mut mermaid_flowchart_diagrams: Signal<Vec<String>>, mut mermaid_er_diagrams: Signal<Vec<String>>, mut mermaid_timestamps: Signal<Vec<i64>>) -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_session_names_state);
    let sync_session_names = use_coroutine_handle::<SyncSessionNamesState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    let mut subject_dropdown = use_signal(String::new);

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &mermaid_session_context_names
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(Vec::new);

    // Error message signal    
    let mut build_errors = use_signal(String::new);

    rsx! {
        div {
            class: "dropdown_form",
            form {
                class: "dropdown_form_input",
                input {
                    r#type: "text",
                    placeholder: "search session",
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
            button {
                class: "dropdown_form_button",
                onclick: move |_evt| async move {
                    // Reset the dropdown
                    active_session_name.set(subject_dropdown.try_read().unwrap().to_string());
                    subject_dropdown.set(String::new());
                },
                svg { dangerous_inner_html: ms_edit_icon_svg() },
            },

            if !active_session_name().is_empty() {
                button { 
                    class: "dropdown_form_button",
                    onclick: move |_evt| async move {
                        // Make a defualt name for the copy of the active session
                        let active_session = format!("{}-copy", active_session_name.read());

                        // Copy the diagrams
                        mermaid_session_context_names.push(active_session.clone());
                        mermaid_flowchart_diagrams.push(active_flowchart_diagram.read().to_string());
                        mermaid_er_diagrams.push(active_er_diagram.read().to_string());
                        mermaid_timestamps.push(create_timestamp_micros());

                        // Set the active session
                        active_session_name.set(active_session.clone());
                    },
                    svg { dangerous_inner_html: ms_column_arrow_right_icon_svg() },
                },
                button {
                    class: "dropdown_form_button",
                    onclick: move |_evt| async move {
                        // Change the name of all active session diagrams
                        let (session_context_names, flowchart_diagrams, er_diagrams, timestamps) = filter_in_mermaid_diagrams_by_session_name(
                            &active_session_name(),
                            &mermaid_session_context_names
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
                        let session_context_names = session_context_names.into_iter().map(|s| format!("__deleted__{s}")).collect::<Vec<_>>();
                        let batch_deleted = create_session_mermaid_batch(session_context_names, flowchart_diagrams, er_diagrams, timestamps).unwrap();

                        // Filter out the active session
                        let (session_context_names, flowchart_diagrams, er_diagrams, timestamps) = filter_out_mermaid_diagrams_by_session_name(                        
                            &active_session_name(),
                            &mermaid_session_context_names
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
                        let active_session = session_context_names.first().unwrap().to_string();
                        let batch = create_session_mermaid_batch(session_context_names, flowchart_diagrams, er_diagrams, timestamps).unwrap();

                        // Update the mermaid state with the active diagram
                        let route = "/app/v1/put_state";
                        let message = Table::get_builder()
                            .with_name(AvailableSubjects::SessionMermaid.to_string().as_str())
                            .with_record_batches(vec![batch_deleted, batch])
                            .unwrap()
                            .build()
                            .unwrap()
                            .to_ipc_stream()
                            .unwrap();    
                        let data_serialized = serde_json::to_string(&SessionInterfaceMessage::get_builder()
                            .with_session_name(&create_session_name(EMAIL().as_str(), AvailableSessionPlans::Builder.to_string().as_str()))
                            .with_format(&DataFormat::Ipc)
                            .with_publisher(&create_session_name(EMAIL().as_str(), AvailableSessionPlans::Builder.to_string().as_str()))
                            .with_update(&TablePublish::Replace { table_name: AvailableSubjects::SessionMermaid.to_string() })
                            .with_stream(false)
                            .with_subject(AvailableSubjects::SessionMermaid.to_string().as_str())
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
                        };
                        #[cfg(feature = "serverless")]
                        let mut serverless = Serverless::new(None);
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

                        // Reset the active session to the first session
                        active_session_name.set(active_session);
                    },
                    svg { dangerous_inner_html: fa_trash_icon_svg() },
                },
                button { 
                    onclick: move |_| async move {
                        let current = is_flowchart_shown.read().to_owned();
                        is_flowchart_shown.set(!current);
                    },
                    svg { dangerous_inner_html: ms_sync_icon_svg() },
                },
                button { 
                    onclick: move |_| async move {
                        // Clear any text
                        build_errors.set(String::new());

                        // Check if the current session can be built
                        let mut builder = match SessionContextBuilder::from_mermaid_flowchart(&active_flowchart_diagram(), true) {
                            Ok(builder) => builder,
                            Err(err) => {
                                build_errors.write().push_str(format!("{err:?}").as_str());
                                return;
                            },
                        };
                        builder = match builder.with_state_from_mermaid_erdiagram(&active_er_diagram(), true) {
                            Ok(builder) => builder,
                            Err(err) => {
                                build_errors.write().push_str(format!("{err:?}").as_str());
                                return;
                            },
                        };
                        if SESSION_NAMES.read().iter().any(|s| s==&active_session_name()) {
                            build_errors.write().push_str(format!("Session name '{}' already exists. Please choose a different name.", active_session_name()).as_str());
                            return;
                        }
                        let _session = match builder.with_name(&active_session_name()).build() {
                            Ok(session) => session,
                            Err(err) => {
                                build_errors.write().push_str(format!("{err:?}").as_str());
                                return;
                            },
                        };

                        // Update the server with the new session
                        let route = "/app/v1/build";
                        let batch = create_session_mermaid_batch(vec![active_session_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                        let message = Table::get_builder()
                            .with_name(AvailableSubjects::SessionMermaid.to_string().as_str())
                            .with_record_batches(vec![batch])
                            .unwrap()
                            .build()
                            .unwrap()
                            .to_ipc_stream()
                            .unwrap();    
                        let data_serialized = serde_json::to_string(&SessionInterfaceMessage::get_builder()
                            .with_session_name(&create_session_name(EMAIL().as_str(), active_session_name().as_str()))
                            .with_format(&DataFormat::Ipc)
                            .with_publisher(&create_session_name(EMAIL().as_str(), active_session_name().as_str()))
                            .with_update(&TablePublish::None)
                            .with_stream(false)
                            .with_subject(AvailableSubjects::SessionMermaid.to_string().as_str())
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
                        };
                        #[cfg(feature = "serverless")]
                        let mut serverless = Serverless::new(None);
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

                        // Update the frontend state with the new session so as not to require the user to re-signin
                        let mut session_plans = vec![active_session_name().to_string()];
                        session_plans.extend(SESSION_NAMES.read().iter().filter(|s| *s!=&active_session_name()).cloned());
                        sync_session_names.send(SyncSessionNamesState { session_plans });

                    },
                    svg { dangerous_inner_html: ms_deploy_icon_svg() },
                },
            }
        }

        // Dynamic dropdown
        if show_subject_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "builds_dropdown_list",
                    {subjects_vec().iter().filter(|s| active_session_name.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
                        let sub = sub.clone();
                        rsx! {
                            li {
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

        if !active_session_name().is_empty() {
            div {
                p { "{active_session_name().to_string()}" },
                if !build_errors.try_read().unwrap().is_empty() {
                    p { "{build_errors}" },
                }
            }
        }
    }
}

/// Diagram code editor
#[component]
pub fn builds_interface_footer(is_flowchart_shown: Signal<bool>, active_session_name: Signal<String>, mut active_flowchart_diagram: Signal<String>, mut active_er_diagram: Signal<String>,) -> Element {
    
    let mut is_saved = use_signal(|| true);

    let diagram_code: Memo<String> = use_memo(move || {
        if is_flowchart_shown() {
            active_flowchart_diagram.read().to_string()
        } else {
            active_er_diagram.read().to_string()
        }        
    });

    rsx! {
        if !JWT.read().is_empty() && !active_session_name.read().is_empty() {
            footer {
                class: "resizable_text_input",
                div {
                    class: "text_input",
                    form {
                        id: "diagram_code_form",
                        textarea {
                            value: "{diagram_code.to_string()}",
                            oninput: move |event| async move {
                                // Update the active diagrams
                                if is_flowchart_shown() {
                                    active_flowchart_diagram.set(event.value());
                                } else {
                                    active_er_diagram.set(event.value());
                                };

                                // Change to unsaved
                                is_saved.set(false);
                            },
                        }
                    }
                }
                
                div {
                    class: "submit_button",
                    // This must be outside the form or it will be refreshed on each submit
                    button {
                        onclick: move |_| async move {
                            // Update the mermaid state with the active diagram
                            let route = "/app/v1/put_state";
                            let batch = create_session_mermaid_batch(vec![active_session_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Table::get_builder()
                                .with_name(AvailableSubjects::SessionMermaid.to_string().as_str())
                                .with_record_batches(vec![batch])
                                .unwrap()
                                .build()
                                .unwrap()
                                .to_ipc_stream()
                                .unwrap();    
                            let data_serialized = serde_json::to_string(&SessionInterfaceMessage::get_builder()
                                .with_session_name(&create_session_name(EMAIL().as_str(), AvailableSessionPlans::Builder.to_string().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_session_name(EMAIL().as_str(), AvailableSessionPlans::Builder.to_string().as_str()))
                                .with_update(&TablePublish::Extend { table_name: AvailableSubjects::SessionMermaid.to_string() })
                                .with_stream(false)
                                .with_subject(AvailableSubjects::SessionMermaid.to_string().as_str())
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
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None);
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
                        // Show the save button only when modified
                        if !is_saved() && !diagram_code().is_empty() {
                            svg { dangerous_inner_html: b8_save_icon_svg() }
                        }
                    }
                }
            }
        }
    }
}
