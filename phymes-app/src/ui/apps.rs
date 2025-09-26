use dioxus::prelude::*;
use futures::StreamExt;
use phymes_agents::session_plans::available_session_plans::AvailableSessionPlans;
use phymes_core::{
    schemas::available_subjects::AvailableSubjects, session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait}, session_context::SessionContextTableNames}, table::{data_format::DataFormat, table_publish::TablePublish}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;
use serde_json::{Map, Value};

use crate::{
    state::{
        apps::{
            filter_in_mermaid_diagrams_by_session_name, get_non_duplicated_sorted_subjects, sync_current_active_session_state, sync_current_session_mermaid_state, sync_is_flowchart_shown_state, SyncCurrentActiveSessionState, SyncCurrentSessionMermaidJSState, SyncIsFlowchartShownState, ACTIVE_SESSION_NAME, IS_FLOWCHART_SHOWN, SESSION_ER_DIAGRAM, SESSION_FLOWCHART_DIAGRAM
        }, builds::{clear_current_mermaid_state, sync_current_mermaid_state, ClearCurrentMermaidState, SyncCurrentMermaidState, MERMAID_ER_DIAGRAM, MERMAID_FLOWCHART_DIAGRAM, MERMAID_SESSION_CONTEXT_NAME, MERMAID_TIMESTAMP}, messaging::{clear_current_message_state, ClearCurrentMessageState}, sign_in::{BUILDER, EMAIL, JWT, SESSION_NAMES}
    },
    ui::{builds::builds_dropdown_view, svg_icons::{search_icon_svg, sync_icon_svg}},
};


#[cfg(feature = "mermaid_js")]
use crate::state::apps::MermaidJsObject;
#[cfg(feature = "mermaid_js")]
use phymes_agents::session_traits::mermaid::SessionContextBuilderMermaidTrait;
#[cfg(feature = "mermaid_js")]
use phymes_core::session::session_context_builder::SessionContextBuilder;

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

/// View for the per runtime settings
#[component]
pub fn apps_interface_view() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_session_mermaid_state);
    use_coroutine(sync_current_mermaid_state);
    use_coroutine(clear_current_mermaid_state);
    let sync_current_mermaid_state = use_coroutine_handle::<SyncCurrentMermaidState>();
    let clear_current_mermaid_state = use_coroutine_handle::<ClearCurrentMermaidState>();
    let sync_current_session_mermaid_state = use_coroutine_handle::<SyncCurrentSessionMermaidJSState>();

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || {
        let session_name = if BUILDER() {
            // DM: this can be better optimized to prevent redundant API calls each time the active session is changed in Builder mode
            create_session_name(EMAIL().as_str(), AvailableSessionPlans::Builder.to_string().as_str())
        } else {
            create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str())
        };
        SessionInterfaceMessage::get_builder()
            .with_session_name(&session_name)
            .with_format(&DataFormat::Bytes)
            .with_publisher(&session_name)
            .with_update(&TablePublish::None)
            .with_stream(false)
    });

    // Get the mermaid.js diagrams for the session
    let _ = use_resource(move || async move {
        
        // clear the current mermaid state
        clear_current_mermaid_state.send(ClearCurrentMermaidState {});

        // get the mermaid state
        let route = "/app/v1/get_state";
        let subject = if BUILDER() {
            AvailableSubjects::Mermaid.to_string()
        } else {
            SessionContextTableNames::MermaidJS.get_name().to_string()
        };
        let data_serialized = serde_json::to_string(&get_session_state()
            .with_subject(subject.as_str())
            .make_name()
            .unwrap()
            .build()
            .unwrap()).unwrap();

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
                            tracing::error!(
                                "There was a error parsing SyncCurrentMermaidState {err}."
                            );
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
                            sync_current_mermaid_state.send(SyncCurrentMermaidState {
                                session_context_name,
                                flowchart_diagram: row
                                    .get("flowchart_diagram")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                                er_diagram: row
                                    .get("er_diagram")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                                timestamp
                            });
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
                            sync_current_mermaid_state.send(SyncCurrentMermaidState {
                                session_context_name,
                                flowchart_diagram: row
                                    .get("flowchart_diagram")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                                er_diagram: row
                                    .get("er_diagram")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                                timestamp: timestamp
                            });
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
            ACTIVE_SESSION_NAME()
        } else {
            create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str())
        };

        // Filter in the active diagrams
        let (_session_context_names, flowchart_diagrams, er_diagrams, timestamps) = filter_in_mermaid_diagrams_by_session_name(
            &session_name,
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
            None => (None, None)
        }
    });

    // Update the active mermaid.js diagrams for the session
    let _ = use_resource(move || async move {
        sync_current_session_mermaid_state.send(SyncCurrentSessionMermaidJSState {
            flowchart_diagram: filtered_diagrams().0,
            er_diagram: filtered_diagrams().1,
        });
    });

    // DM: we have to re-render the entire virtual DOM everytime the mermaid svg changes...
    let diagram_code: Memo<String> = use_memo(move || {
        if IS_FLOWCHART_SHOWN() {
            SESSION_FLOWCHART_DIAGRAM.read().to_string()
        } else {
            SESSION_ER_DIAGRAM.read().to_string()
        }        
    });
    #[allow(clippy::redundant_closure)]
    let is_flowchart_shown: Memo<bool> = use_memo(move || IS_FLOWCHART_SHOWN());    
    let rendered_html = render_mermaid_svg(diagram_code, "graphDiv", true, is_flowchart_shown);

    let out = if let Some(result) = &*rendered_html.read() {
        match result {
            // Mermaid.js or SessionContextBuilder error
            (_, Some(error), None) | (_, None, Some(error)) => {
                rsx! {
                    if JWT.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please sign-in before activating a session." },
                        }
                    } else if SESSION_NAMES.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve available session plans..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            if BUILDER() {
                                builds_dropdown_view {}
                            } else {
                                apps_dropdown_view {}
                            }                            
                            p { "{error}" },
                        }
                    }
                }
            }
            // Mermaid.js and SessionContextBuilder error
            (_, Some(error_mjs), Some(error_ctxb)) => {
                rsx! {
                    if JWT.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please sign-in before activating a session." },
                        }
                    } else if SESSION_NAMES.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve available session plans..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            if BUILDER() {
                                builds_dropdown_view {}
                            } else {
                                apps_dropdown_view {}
                            } 
                            p { "{error_mjs}" },
                            p { "{error_ctxb}" },
                        }
                    }
                }
            }
            // Valid SVG with no errors
            (Some(svg), _, _) => {
                rsx! {
                    if JWT.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please sign-in before activating a session." },
                        }
                    } else if SESSION_NAMES.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve available session plans..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            if BUILDER() {
                                builds_dropdown_view {}
                            } else {
                                apps_dropdown_view {}
                            } 
                            div {
                                id: "graphDiv",
                                class: "mermaid",
                                svg { dangerous_inner_html: svg.to_string() }
                            }
                        }
                    }
                }
            }
            // All other cases
            (_, _, _) => {
                rsx! {
                    if JWT.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please sign-in before activating a session." },
                        }
                    } else if SESSION_NAMES.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve available session plans..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            if BUILDER() {
                                builds_dropdown_view {}
                            } else {
                                apps_dropdown_view {}
                            } 
                        }
                    }
                }
            }
        }
    } else {
        rsx! {
            if JWT.read().is_empty() {
                div {
                    class: "messaging_list",
                    p { "Please sign-in before activating a session." },
                }
            } else if SESSION_NAMES.read().is_empty() {
                div {
                    class: "messaging_list",
                    p { "Waiting to retrieve available session plans..." },
                }
            } else {
                div {
                    class: "messaging_list",
                    if BUILDER() {
                        builds_dropdown_view {}
                    } else {
                        apps_dropdown_view {}
                    } 
                }
            }
        }
    };
    out
}

/// View for the per runtime settings
#[component]
pub fn apps_dropdown_view() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_session_state);
    use_coroutine(clear_current_message_state);
    use_coroutine(sync_is_flowchart_shown_state);
    let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();
    let clear_current_message_state = use_coroutine_handle::<ClearCurrentMessageState>();
    let sync_is_flowchart_shown_state = use_coroutine_handle::<SyncIsFlowchartShownState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());
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
            class: "dropdown_form",
            form {
                class: "dropdown_form_input",
                input {
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
                svg { dangerous_inner_html: search_icon_svg() },
            },
            button { 
                onclick: move |_| async move {
                    let current = IS_FLOWCHART_SHOWN.read().to_owned();
                    sync_is_flowchart_shown_state.send( SyncIsFlowchartShownState { is_shown: !current} );
                },
                svg { dangerous_inner_html: sync_icon_svg() },
            },
        }

        // Dynamic dropdown
        if show_subject_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "apps_dropdown_list",
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
            }
        }
    }
}

#[cfg(feature = "mermaid_js")]
pub fn render_mermaid_svg(
    diagram_code: Memo<String>,
    id: &str,
    check_build: bool,
    flowchart: Memo<bool>,
) -> Resource<(Option<String>, Option<String>, Option<String>)> {
    let div_id = id.to_string();
    let rendered_html: Resource<(Option<String>, Option<String>, Option<String>)> =
        use_resource(move || {
            let div_id = div_id.clone();
            async move {
                // Render the mermaid.js diagram
                let eval = document::eval(
                    format!(
                        r#"
                try {{
                    let code = await dioxus.recv();
                    const {{ svg }} = await mermaid.render("{div_id}", code);
                    return {{ svg: svg, error: null }};
                }} catch (error) {{
                    return {{ svg: null, error: error.message }};
                }}"#
                    )
                    .as_str(),
                );
                eval.send(diagram_code()).unwrap();
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

                // Build the preliminary session context
                if check_build {
                    let builder_error = if flowchart() {
                        match SessionContextBuilder::from_mermaid_flowchart(&diagram_code(), true) {
                            Ok(_res) => None,
                            Err(err) => Some(err.to_string()),
                        }
                    } else {
                        match SessionContextBuilder::default().with_state_from_mermaid_erdiagram(&diagram_code(), true) {
                            Ok(_res) => None,
                            Err(err) => Some(err.to_string()),
                        }
                    };
                    (
                        mermaid_js_object.svg,
                        mermaid_js_object.error,
                        builder_error,
                    )
                } else {
                    (mermaid_js_object.svg, mermaid_js_object.error, None)
                }
            }
        });

    // add pan and zoom
    let div_id = id.to_string();
    use_effect(move || {
        let div_id = div_id.clone();
        let _ = rendered_html.read();
        document::eval(
            format!(
                r#"
            const container = document.getElementById("{div_id}");
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

    rendered_html
}
