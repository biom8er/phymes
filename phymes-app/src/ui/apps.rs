use dioxus::prelude::*;
use phymes_agents::AvailableSessionPlans;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, DataFormat, MessageBuilderTrait,
    SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait,
    TablePublication,
};
use phymes_server::create_session_name;
use serde_json::{Map, Value};

use crate::{
    state::{
        filter_in_mermaid_diagrams_by_session_name, get_non_duplicated_sorted_subjects,
        svg_icons::{ms_search_icon_svg, ms_sync_icon_svg},
        sync_current_active_session_state, SyncCurrentActiveSessionState, ACTIVE_SESSION_NAME,
        BUILDER, EMAIL, JWT, SESSION_NAMES,
    },
    ui::{
        builds_dropdown_view, diagram_code_editor,
        main_window::{split_panel, SnapPct},
    },
};

#[cfg(not(feature = "serverless"))]
use futures::StreamExt;

#[cfg(feature = "mermaid_js")]
use crate::state::MermaidJsObject;
#[cfg(feature = "mermaid_js")]
use phymes_agents::SessionContextBuilderMermaidTrait;
#[cfg(feature = "mermaid_js")]
use phymes_core::SessionContextBuilder;

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

/// View for the per runtime settings
#[component]
pub fn apps_interface_view() -> Element {
    // Intialize signals
    let is_flowchart_shown = use_signal(|| true);
    let active_session_name = use_signal(String::new);
    let mut active_flowchart_diagram = use_signal(String::new);
    let mut active_er_diagram = use_signal(String::new);

    let mut mermaid_session_context_names = use_signal(Vec::<String>::new);
    let mut mermaid_flowchart_diagrams = use_signal(Vec::<String>::new);
    let mut mermaid_er_diagrams = use_signal(Vec::<String>::new);
    let mut mermaid_timestamps = use_signal(Vec::<i64>::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || {
        let session_name = if BUILDER() {
            // DM: this can be better optimized to prevent redundant API calls each time the active session is changed in Builder mode
            create_session_name(
                EMAIL().as_str(),
                AvailableSessionPlans::Builder.to_string().as_str(),
            )
        } else {
            create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str())
        };
        SessionInterfaceMessage::get_builder()
            .with_session_name(&session_name)
            .with_format(&DataFormat::Bytes)
            .with_publisher(&session_name)
            .with_update(&TablePublication::None)
            .with_stream(false)
    });

    // Get the mermaid.js diagrams for the session
    let _ = use_resource(move || async move {
        // clear the current mermaid state
        mermaid_session_context_names.set(Vec::new());
        mermaid_flowchart_diagrams.set(Vec::new());
        mermaid_er_diagrams.set(Vec::new());
        mermaid_timestamps.set(Vec::new());

        // get the mermaid state
        let route = "/app/v1/get_state";
        let subject = if BUILDER() {
            AvailableSubjects::BuilderMermaid.to_string()
        } else {
            AvailableSubjects::SessionMermaid.to_string()
        };
        let data_serialized = serde_json::to_string(
            &get_session_state()
                .with_subject(subject.as_str())
                .make_name()
                .unwrap()
                .build()
                .unwrap(),
        )
        .unwrap();

        #[cfg(not(feature = "serverless"))]
        let addr = format!("{ADDR_BACKEND}{route}");
        #[cfg(not(feature = "serverless"))]
        match reqwest::Client::new()
            .post(addr)
            .bearer_auth(JWT().to_string())
            .header(CONTENT_TYPE, "application/json")
            .body(data_serialized)
            .send()
            .await
        {
            Ok(stream) => {
                let mut stream = stream.bytes_stream();
                while let Some(Ok(bytes)) = stream.next().await {
                    let json_str = String::from_utf8_lossy(bytes.as_ref()).into_owned();
                    let json_rows: Vec<Map<String, Value>> =
                        serde_json::from_str(json_str.as_str()).unwrap_or_else(|err| {
                            tracing::error!("There was a error parsing mermaid state {err}.");
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        // Check for deleted sessions
                        let session_context_name = row
                            .get("session_context_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string();
                        if !session_context_name.contains("__deleted__") {
                            // Update the mermaid state
                            let timestamp = if let Some(Value::Number(val)) = row.get("timestamp") {
                                val.as_u64().unwrap().try_into().unwrap()
                            } else {
                                0
                            };
                            mermaid_session_context_names.push(session_context_name);
                            mermaid_flowchart_diagrams.push(
                                row.get("flowchart_diagram")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                            );
                            mermaid_er_diagrams
                                .push(row.get("er_diagram").unwrap().as_str().unwrap().to_string());
                            mermaid_timestamps.push(timestamp);
                        }
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }

        #[cfg(feature = "serverless")]
        let config = ServerlessConfig {
            route: route.to_string(),
            basic_auth: None,
            bearer_auth: Some(JWT().to_string()),
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
                for byte in bytes.iter() {
                    let json_rows: Vec<Map<String, Value>> =
                        serde_json::from_slice(byte).unwrap_or_else(|_err| Vec::new());
                    for row in json_rows.iter() {
                        // Check for deleted sessions
                        let session_context_name = row
                            .get("session_context_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string();
                        if !session_context_name.contains("__deleted__") {
                            // Update the mermaid state
                            let timestamp = if let Some(Value::Number(val)) = row.get("timestamp") {
                                val.as_u64().unwrap().try_into().unwrap()
                            } else {
                                0
                            };
                            mermaid_session_context_names.push(session_context_name);
                            mermaid_flowchart_diagrams.push(
                                row.get("flowchart_diagram")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                            );
                            mermaid_er_diagrams
                                .push(row.get("er_diagram").unwrap().as_str().unwrap().to_string());
                            mermaid_timestamps.push(timestamp);
                        }
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

    // Filter the mermaid.js diagrams for the session
    let filtered_diagrams = use_memo(move || {
        // The builder session names do not contain the email
        let session_name = if BUILDER() {
            active_session_name()
        } else {
            create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str())
        };

        // Filter in the active diagrams
        let (_session_context_names, flowchart_diagrams, er_diagrams, timestamps) =
            filter_in_mermaid_diagrams_by_session_name(
                &session_name,
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
                &mermaid_timestamps(),
            );

        // Sort by timestamp
        let mut combined = flowchart_diagrams
            .into_iter()
            .zip(er_diagrams.into_iter())
            .zip(timestamps.into_iter())
            .map(|((a, b), c)| (a, b, c))
            .collect::<Vec<_>>();
        combined.sort_by(|a, b| a.2.cmp(&b.2));

        // last is most recent
        match combined.last() {
            Some(diagrams) => (Some(diagrams.0.to_owned()), Some(diagrams.1.to_owned())),
            None => (None, None),
        }
    });

    // Update the active mermaid.js diagrams for the session
    let _ = use_resource(move || async move {
        if let Some(diagram) = filtered_diagrams().0 {
            active_flowchart_diagram.set(diagram.to_string());
        }
        if let Some(diagram) = filtered_diagrams().1 {
            active_er_diagram.set(diagram.to_string());
        }
    });

    let diagram_code: Memo<(String, Option<String>)> = use_memo(move || {
        // Get the active diagram code
        let diagram_code = if is_flowchart_shown() {
            active_flowchart_diagram.read().to_string()
        } else {
            active_er_diagram.read().to_string()
        };

        // Check for build warnings
        let builder_error = if is_flowchart_shown() {
            match SessionContextBuilder::from_mermaid_flowchart(&diagram_code, true) {
                Ok(_res) => None,
                Err(err) => Some(err.to_string()),
            }
        } else {
            match SessionContextBuilder::default().with_state_from_mermaid_erdiagram(
                &diagram_code,
                true,
                true,
            ) {
                Ok(_res) => None,
                Err(err) => Some(err.to_string()),
            }
        };
        (diagram_code, builder_error)
    });

    // Track when the diagram code changes
    let mut is_saved = use_signal(|| true);

    rsx! {
        if JWT.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please sign-in before activating a session." },
            }
        } else if SESSION_NAMES.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Waiting to retrieve available session plans..." },
            }
        } else {
            if BUILDER() {
                split_panel {
                    top: rsx! {
                        div {
                            class: "h-full w-full p-2 flex flex-col items-center",
                            builds_dropdown_view { is_flowchart_shown, active_session_name, active_flowchart_diagram, active_er_diagram, mermaid_session_context_names, mermaid_flowchart_diagrams, mermaid_er_diagrams, mermaid_timestamps, is_saved }
                            diagram_code_editor { is_flowchart_shown, active_session_name, active_flowchart_diagram, active_er_diagram, is_saved }
                        }
                    },
                    bottom: rsx! {
                        div {
                            class: "h-full w-full p-2 flex flex-col items-center",
                            mermaid_view { diagram_code }
                        }
                    },
                    initial_top_pct: SnapPct::Pct50,
                    horizontal: false,
                }
            } else {
                div {
                    class: "h-full w-full p-2 flex flex-col items-center",
                    apps_dropdown_view { is_flowchart_shown }
                    mermaid_view { diagram_code }
                }
            }
        }
    }
}

/// View for the per runtime settings
#[component]
pub fn apps_dropdown_view(mut is_flowchart_shown: Signal<bool>) -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_session_state);
    let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    let mut subject_dropdown = use_signal(String::new);
    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &SESSION_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    rsx! {
        div {
            // input + 2 buttons of 64 px by 64 px
            class: "p-2 gap-2 rounded bg-gray-800 grid grid-rows-[48px_1fr] grid-cols-[1fr_128px] w-full sm:max-w-3/4",
            form {
                class: "w-full h-full flex row-span-1 col-span-1 row-start-1 col-start-1",
                input {
                    class: "w-full h-full bg-gray-700",
                    r#type: "text",
                    placeholder: "search apps",
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
                        {subjects_vec().iter().filter(|s| ACTIVE_SESSION_NAME.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
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
                        let active_session = subject_dropdown.try_read().unwrap().to_string();
                        subject_dropdown.set(String::new());

                        // Set the active session
                        sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: active_session.clone() });
                    },
                    svg {
                        class: "max-w-[48px] max-h-[48px]",
                        dangerous_inner_html: ms_search_icon_svg()
                    },
                },

                if !ACTIVE_SESSION_NAME().is_empty() {
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
                }
            }
        }

        if !ACTIVE_SESSION_NAME().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "{ACTIVE_SESSION_NAME().to_string()}" },
            }
        }
    }
}

#[component]
pub fn mermaid_view(diagram_code: Memo<(String, Option<String>)>) -> Element {
    let mut diagram_svg = use_signal(String::new);
    let mut error_mjs = use_signal(String::new);
    let id = use_signal(|| "graphDiv".to_string());
    // Temporary DOM elemented created by Mermaid.js breaks Dioxus
    // when the actual SVG target ID is used...
    let id_decoy = use_signal(|| "GraphDiv".to_string());

    // Render the mermaid.js diagram
    let _ = use_resource(move || async move {
        let eval = document::eval(
            format!(
                r#"
        try {{
            let code = await dioxus.recv();
            const {{ svg }} = await mermaid.render("{id_decoy}", code);
            return {{ svg: svg, error: null }};
        }} catch (error) {{
            return {{ svg: null, error: error.message }};
        }}"#
            )
            .as_str(),
        );
        eval.send(diagram_code().0).unwrap();
        let mermaid_js_object = match eval.await {
            Ok(res) => {
                let res: MermaidJsObject = serde_json::from_value(res).unwrap();
                res
            }
            Err(err) => {
                tracing::error!("Mermaid.js err {err:?}");
                MermaidJsObject {
                    svg: None,
                    error: Some(err.to_string()),
                }
            }
        };

        // Update the signals
        if mermaid_js_object.error.is_none() {
            diagram_svg.set(mermaid_js_object.svg.unwrap_or_default());
        }
        error_mjs.set(mermaid_js_object.error.unwrap_or_default());
    });

    // Add pan and zoom to the svg
    use_effect(move || {
        let _ = diagram_svg(); // needed to trigger the effect
        document::eval(
            format!(
                r#"
            const container = document.getElementById("{id}");
            const svgElement = container.querySelector("svg");

            // Initialize Panzoom
            const panzoomInstance = Panzoom(svgElement, {{
                maxScale: 100,
                minScale: 0.1,
                step: 0.1,
            }});

            // Add mouse wheel zoom
            container.addEventListener("wheel", (event) => {{
                panzoomInstance.zoomWithWheel(event);
            }});
            "#
            )
            .as_str(),
        );
    });

    rsx! {
        if !error_mjs().is_empty() {
            div {
                class: "rounded p-2 items-center text-gray-200 bg-gray-700",
                p {
                    class: "text-gray-200",
                    "{error_mjs}"
                },
            }
        }
        if let Some(error_ctxb) = diagram_code().1 {
            div {
                class: "rounded p-2 items-center bg-gray-700",
                p {
                    class: "text-gray-200",
                    "{error_ctxb}"
                },
            }
        }
        if !diagram_svg().is_empty() {
            mermaid_div { diagram_svg, id }
        }
    }
}

#[component]
pub fn mermaid_div(diagram_svg: Signal<String>, id: Signal<String>) -> Element {
    rsx! {
        div {
            id: id(),
            class: "w-full h-full",
            svg { dangerous_inner_html: diagram_svg() }
        }
    }
}
