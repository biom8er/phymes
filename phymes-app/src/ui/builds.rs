use dioxus::prelude::*;
use phymes_agents::{session_plans::available_session_plans::AvailableSessionPlans, session_traits::mermaid::SessionContextBuilderMermaidTrait};
use phymes_core::{schemas::{available_subjects::{create_timestamp_micros, AvailableSubjects}, mermaid::create_mermaid_batch}, session::{common_traits::{BuildableTrait, BuilderTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilderTrait}, session_context_builder::SessionContextBuilder}, table::{data_format::DataFormat, table_trait::{Table, TableBuilderTrait, TableTrait}, table_publish::TablePublish}, task::message::MessageBuilderTrait};
use phymes_server::handlers::sign_in::create_session_name;

use crate::{
    state::{
        apps::{
            sync_current_active_session_state, sync_current_session_mermaid_state, sync_is_flowchart_shown_state, SyncCurrentActiveSessionState, SyncCurrentSessionMermaidJSState, SyncIsFlowchartShownState, ACTIVE_SESSION_NAME, IS_FLOWCHART_SHOWN, SESSION_ER_DIAGRAM, SESSION_FLOWCHART_DIAGRAM, get_non_duplicated_sorted_subjects, filter_in_mermaid_diagrams_by_session_name},
        builds::{sync_current_mermaid_state, SyncCurrentMermaidState, MERMAID_ER_DIAGRAM, MERMAID_FLOWCHART_DIAGRAM, MERMAID_SESSION_CONTEXT_NAME, MERMAID_TIMESTAMP, filter_out_mermaid_diagrams_by_session_name}, 
        messaging::{clear_current_message_state, ClearCurrentMessageState}, sign_in::{EMAIL, JWT}
    },
    ui::svg_icons::{column_arrow_right_icon_svg, deploy_icon_svg, edit_icon_svg, save_icon_svg, sync_icon_svg, trash_icon_svg},
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
pub fn builds_dropdown_view() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_session_state);
    let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();
    use_coroutine(clear_current_message_state);
    let clear_current_message_state = use_coroutine_handle::<ClearCurrentMessageState>();
    use_coroutine(sync_is_flowchart_shown_state);
    let sync_is_flowchart_shown_state = use_coroutine_handle::<SyncIsFlowchartShownState>();
    use_coroutine(sync_current_mermaid_state);
    let sync_current_mermaid_state = use_coroutine_handle::<SyncCurrentMermaidState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &MERMAID_SESSION_CONTEXT_NAME
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    // Error message signal    
    #[allow(clippy::redundant_closure)]
    let mut build_errors = use_signal(|| String::new());

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
                    let active_session = subject_dropdown.try_read().unwrap().to_string();
                    subject_dropdown.set(String::new());

                    // Set the active session
                    sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: active_session.clone() });

                    // Reset the current session messaging
                    clear_current_message_state.send(ClearCurrentMessageState {});
                },
                svg { dangerous_inner_html: edit_icon_svg() },
            },
            button { 
                class: "dropdown_form_button",
                onclick: move |_evt| async move {
                    // Make a defualt name for the copy of the active session
                    let active_session = format!("{}-copy", ACTIVE_SESSION_NAME.read());

                    // Copy the diagrams
                    sync_current_mermaid_state.send(SyncCurrentMermaidState {
                        session_context_name: active_session.clone(),
                        flowchart_diagram: SESSION_FLOWCHART_DIAGRAM.read().to_string(),
                        er_diagram: SESSION_ER_DIAGRAM.read().to_string(),
                        timestamp: create_timestamp_micros()
                    });

                    // Set the active session
                    sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: active_session.clone() });

                    // Reset the current session messaging
                    clear_current_message_state.send(ClearCurrentMessageState {});
                },
                svg { dangerous_inner_html: column_arrow_right_icon_svg() },
            },
            button {
                class: "dropdown_form_button",
                onclick: move |_evt| async move {
                    // Change the name of all active session diagrams
                    let (session_context_names, flowchart_diagrams, er_diagrams, timestamps) = filter_in_mermaid_diagrams_by_session_name(
                        &ACTIVE_SESSION_NAME(),
                        &MERMAID_SESSION_CONTEXT_NAME
                            .read()
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &MERMAID_FLOWCHART_DIAGRAM
                            .read()
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &MERMAID_ER_DIAGRAM
                            .read()
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &MERMAID_TIMESTAMP());
                    let session_context_names = session_context_names.into_iter().map(|s| format!("__deleted__{s}")).collect::<Vec<_>>();
                    let batch_deleted = create_mermaid_batch(session_context_names, flowchart_diagrams, er_diagrams, timestamps).unwrap();

                    // Filter out the active session
                    let (session_context_names, flowchart_diagrams, er_diagrams, timestamps) = filter_out_mermaid_diagrams_by_session_name(                        
                        &ACTIVE_SESSION_NAME(),
                        &MERMAID_SESSION_CONTEXT_NAME
                            .read()
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &MERMAID_FLOWCHART_DIAGRAM
                            .read()
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &MERMAID_ER_DIAGRAM
                            .read()
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &MERMAID_TIMESTAMP());
                    let active_session = session_context_names.first().unwrap().to_string();
                    let batch = create_mermaid_batch(session_context_names, flowchart_diagrams, er_diagrams, timestamps).unwrap();

                    // Update the mermaid state with the active diagram
                    let route = "/app/v1/put_state";
                    let message = Table::get_builder()
                        .with_name(AvailableSubjects::Mermaid.to_string().as_str())
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
                        .with_update(&TablePublish::Replace { table_name: AvailableSubjects::Mermaid.to_string() })
                        .with_stream(false)
                        .with_subject(AvailableSubjects::Mermaid.to_string().as_str())
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
                    let mut serverless = Serverless::new();
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
                    sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: active_session });

                    // Reset the current session messaging
                    clear_current_message_state.send(ClearCurrentMessageState {});
                },
                svg { dangerous_inner_html: trash_icon_svg() },
            },
            button { 
                onclick: move |_| async move {
                    let current = IS_FLOWCHART_SHOWN.read().to_owned();
                    sync_is_flowchart_shown_state.send( SyncIsFlowchartShownState { is_shown: !current} );
                },
                svg { dangerous_inner_html: sync_icon_svg() },
            },
            button { 
                onclick: move |_| async move {
                    // Clear any text
                    build_errors.set(String::new());

                    // Check if the current session can be built
                    let mut builder = match SessionContextBuilder::from_mermaid_flowchart(&SESSION_FLOWCHART_DIAGRAM(), true) {
                        Ok(builder) => builder,
                        Err(err) => {
                            build_errors.write().push_str(format!("{err:?}").as_str());
                            return;
                        },
                    };
                    builder = match builder.with_state_from_mermaid_erdiagram(&SESSION_ER_DIAGRAM(), true) {
                        Ok(builder) => builder,
                        Err(err) => {
                            build_errors.write().push_str(format!("{err:?}").as_str());
                            return;
                        },
                    };
                    let _session = match builder.with_name(&ACTIVE_SESSION_NAME()).build() {
                        Ok(session) => session,
                        Err(err) => {
                            build_errors.write().push_str(format!("{err:?}").as_str());
                            return;
                        },
                    };

                    // Update the server
                    let route = "/app/v1/deploy_session";
                    let batch = create_mermaid_batch(vec![ACTIVE_SESSION_NAME()], vec![SESSION_FLOWCHART_DIAGRAM()], vec![SESSION_ER_DIAGRAM()], vec![create_timestamp_micros()]).unwrap();
                    let message = Table::get_builder()
                        .with_name(AvailableSubjects::Mermaid.to_string().as_str())
                        .with_record_batches(vec![batch])
                        .unwrap()
                        .build()
                        .unwrap()
                        .to_ipc_stream()
                        .unwrap();    
                    let data_serialized = serde_json::to_string(&SessionInterfaceMessage::get_builder()
                        .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                        .with_format(&DataFormat::Ipc)
                        .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                        .with_update(&TablePublish::None)
                        .with_stream(false)
                        .with_subject(AvailableSubjects::Mermaid.to_string().as_str())
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
                    let mut serverless = Serverless::new();
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
                },
                svg { dangerous_inner_html: deploy_icon_svg() },
            },
        }

        // Dynamic dropdown
        if show_subject_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "builds_dropdown_list",
                    {subjects_vec().iter().filter(|s| ACTIVE_SESSION_NAME.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
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

        if !ACTIVE_SESSION_NAME().is_empty() {
            div {
                p { "{ACTIVE_SESSION_NAME().to_string()}" },
                if !build_errors.try_read().unwrap().is_empty() {
                    p { "{build_errors}" },
                }
            }
        }
    }
}

/// Diagram code editor
#[component]
pub fn builds_interface_footer() -> Element {
    use_coroutine(sync_current_session_mermaid_state);
    let sync_current_session_mermaid_state = use_coroutine_handle::<SyncCurrentSessionMermaidJSState>();
    
    let mut is_saved = use_signal(|| true);

    let diagram_code: Memo<String> = use_memo(move || {
        if IS_FLOWCHART_SHOWN() {
            SESSION_FLOWCHART_DIAGRAM.read().to_string()
        } else {
            SESSION_ER_DIAGRAM.read().to_string()
        }        
    });

    rsx! {
        if !JWT.read().is_empty() && !ACTIVE_SESSION_NAME.read().is_empty() {
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
                                let current_session_mermaid_js = if IS_FLOWCHART_SHOWN() {
                                    SyncCurrentSessionMermaidJSState {
                                        flowchart_diagram: Some(event.value()),
                                        er_diagram: None,
                                    }
                                } else {
                                    SyncCurrentSessionMermaidJSState {
                                        flowchart_diagram: None,
                                        er_diagram: Some(event.value()),
                                    }
                                };
                                sync_current_session_mermaid_state.send(current_session_mermaid_js);

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
                            let batch = create_mermaid_batch(vec![ACTIVE_SESSION_NAME()], vec![SESSION_FLOWCHART_DIAGRAM()], vec![SESSION_ER_DIAGRAM()], vec![create_timestamp_micros()]).unwrap();
                            let message = Table::get_builder()
                                .with_name(AvailableSubjects::Mermaid.to_string().as_str())
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
                                .with_update(&TablePublish::Extend { table_name: AvailableSubjects::Mermaid.to_string() })
                                .with_stream(false)
                                .with_subject(AvailableSubjects::Mermaid.to_string().as_str())
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
                            let mut serverless = Serverless::new();
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
                            svg { dangerous_inner_html: save_icon_svg() }
                        }
                    }
                }
            }
        }
    }
}
