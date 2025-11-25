use std::future::Future;

use dioxus::prelude::*;
use phymes_agents::{
    AvailableSessionPlans, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
};
use phymes_core::{
    create_session_mermaid_batch, AvailableSubjects, BuildableTrait, BuilderTrait, DataFormat,
    MessageBuilderTrait, SessionContextBuilder, SessionInterfaceMessage,
    SessionInterfaceMessageBuilderTrait, Table, TableBuilderTrait, TablePublication, TableTrait,
};
use phymes_diagnostics::create_timestamp_micros;
use phymes_server::create_session_name;

use crate::state::{
    filter_in_mermaid_diagrams_by_session_name, filter_out_mermaid_diagrams_by_session_name,
    get_non_duplicated_sorted_subjects,
    svg_icons::{
        b8_save_icon_svg, fa_trash_icon_svg, ms_code_icon_svg, ms_column_arrow_right_icon_svg,
        ms_deploy_icon_svg, ms_edit_icon_svg, ms_sync_icon_svg,
    },
    sync_session_names_state, SyncSessionNamesState, EMAIL, JWT, SESSION_NAMES,
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
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

/// View for the builds drop down menu
#[component]
pub fn builds_dropdown_view(
    mut is_flowchart_shown: Signal<bool>,
    mut active_session_name: Signal<String>,
    mut active_flowchart_diagram: Signal<String>,
    mut active_er_diagram: Signal<String>,
    mut mermaid_session_context_names: Signal<Vec<String>>,
    mut mermaid_flowchart_diagrams: Signal<Vec<String>>,
    mut mermaid_er_diagrams: Signal<Vec<String>>,
    mut mermaid_timestamps: Signal<Vec<i64>>,
    mut is_saved: Signal<bool>,
) -> Element {
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
            // input + 5 buttons of 64 px by 64 px
            class: "p-2 gap-2 rounded bg-gray-800 grid grid-rows-[64px_1fr] grid-cols-[1fr_418px]",
            form {
                class: "w-full h-full flex row-span-1 col-span-1 row-start-1 col-start-1",
                input {
                    class: "w-full h-full bg-gray-700",
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

            // Dynamic dropdown
            if show_subject_dropdown() {
                div {
                    class: "p-2 rounded bg-gray-800 list-none flex row-span-1 col-span-1 row-start-2 col-start-1",
                    ul {
                        {subjects_vec().iter().filter(|s| active_session_name.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
                            let sub = sub.clone();
                            rsx! {
                                li {
                                    class: "hover:bg-gray-700 cursor-pointer",
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
                    class: "p-1 rounded hover:bg-gray-700 cursor-pointer flex-none",
                    onclick: move |_evt| async move {
                        // Reset the dropdown
                        active_session_name.set(subject_dropdown.try_read().unwrap().to_string());
                        subject_dropdown.set(String::new());
                    },
                    svg {
                        class: "max-w-[48px] max-h-[48px]",
                        dangerous_inner_html: ms_edit_icon_svg()
                    },
                },

                if !active_session_name().is_empty() {
                    button {
                        class: "p-1 rounded hover:bg-gray-700 cursor-pointer flex-none",
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
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: ms_column_arrow_right_icon_svg()
                        },
                    },
                    button {
                        class: "p-1 rounded hover:bg-gray-700 cursor-pointer flex-none",
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
                                .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
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
                                .with_update(&TablePublication::Replace { table_name: AvailableSubjects::BuilderMermaid.to_string() })
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
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: fa_trash_icon_svg()
                        },
                    },
                    button {
                        class: "p-1 rounded hover:bg-gray-700 cursor-pointer flex-none",
                        onclick: move |_| async move {
                            let current = is_flowchart_shown.read().to_owned();
                            is_flowchart_shown.set(!current);
                        },
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: ms_sync_icon_svg()
                        },
                    },
                    button {
                        class: "p-1 rounded hover:bg-gray-700 cursor-pointer flex-none",
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
                            builder = match builder.with_state_from_mermaid_erdiagram(&active_er_diagram(), true, true) {
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
                            let _session = match builder.with_name(&active_session_name())
                                .add_processor_subjects().unwrap()
                                .add_session_interface(None).unwrap()
                                .build_with_tables()
                            {
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
                                .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
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
                                .with_update(&TablePublication::None)
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
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: ms_deploy_icon_svg()
                        },
                    },
                }

                // Show the save button only when modified
                if !is_saved() {
                    button {
                        class: "p-1 hover:bg-gray-700 rounded bg-gray-800 cursor-pointer",
                        onclick: move |_| async move {
                            // Update the mermaid state with the active diagram
                            let route = "/app/v1/put_state";
                            let batch = create_session_mermaid_batch(vec![active_session_name()], vec![active_flowchart_diagram()], vec![active_er_diagram()], vec![create_timestamp_micros()]).unwrap();
                            let message = Table::get_builder()
                                .with_name(AvailableSubjects::BuilderMermaid.to_string().as_str())
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
                                .with_update(&TablePublication::Extend { table_name: AvailableSubjects::BuilderMermaid.to_string() })
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
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: b8_save_icon_svg()
                        }
                    }
                }

                // Show the save button only when modified
                if !is_flowchart_shown() && active_er_diagram().is_empty() {
                    button {
                        class: "p-1 hover:bg-gray-700 rounded bg-gray-800 cursor-pointer",
                        onclick: move |_| async move {
                            // Generate defaults if possible
                            match SessionContextBuilder::from_mermaid_flowchart(&active_flowchart_diagram(), true) {
                                Ok(builder) => match builder.with_name(&active_session_name()).add_processor_subjects() {
                                    Ok(builder) => match builder.to_mermaid_erdiagram(false, true) {
                                        Ok(diagram) => {
                                            active_er_diagram.set(diagram);

                                            // Change to saved
                                            is_saved.set(false);
                                        },
                                        Err(err) => tracing::error!("{err:?}"),
                                    },
                                    Err(err) => tracing::error!("{err:?}"),
                                },
                                Err(err) => tracing::error!("{err:?}"),
                            }
                        },
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: ms_code_icon_svg()
                        }
                    }
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

/// Code editor element with textarea callbacks
#[component]
pub fn diagram_code_editor(
    is_flowchart_shown: Signal<bool>,
    active_session_name: Signal<String>,
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
            format!(
                r#"const gutter = document.getElementById('gutter');
const code = document.getElementById('code');
code.addEventListener('scroll', () => {{
    gutter.scrollTop = code.scrollTop;
}});"#
            )
            .as_str(),
        );
    });

    rsx! {
        div {
            class: "w-full h-full rounded-md shadow-sm py-2 p-2 snap-y overflow-auto grid grid-cols-[3rem_1fr] font-mono text-sm leading-6 snap-start",
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
                class: "w-full h-full grow bg-gray-800 px-3 resize-none focus:outline-none whitespace-pre overflow-hidden"
            }
        }
    }
}
