use dioxus::prelude::*;
use futures::StreamExt;
use phymes_core::table::table_publish::TablePublish;
use phymes_data::candle_data::summary_config::DataFormat;
use phymes_server::handlers::{
    session_info::SessionInterfaceMessage,
    sign_in::create_session_name,
};
use serde_json::{Map, Value};

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

use crate::{
    state::{
        metrics::{
            sync_current_active_metric_state, sync_current_metrics_mermaid_state,
            SyncCurrentActiveMetricState, SyncCurrentMetricsMermaidJSState, ACTIVE_METRIC,
            MERMAID_ELAPSED_COMPUTE, MERMAID_OUTPUT_ROWS, MERMAID_PROCESSOR_TRACES,
        },
        settings::ACTIVE_SESSION_NAME,
        sign_in::{EMAIL, JWT},
    },
    ui::{settings::render_mermaid_svg, svg_icons::search_icon_svg},
};

const SESSION_METRICS_HEADERS: [&str; 3] = ["processor_traces", "elapsed_compute", "output_rows"];

#[component]
pub fn metrics_modal() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_metrics_mermaid_state);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || SessionInterfaceMessageBuilder
        .with_session_name(create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_format(DataFormat::Bytes)
        .with_publisher(create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_publish(TablePublish::None)
        .with_stream(false)
    );

    // Get the active session info for the metrics view
    let sync_current_metrics_mermaid_state =
        use_coroutine_handle::<SyncCurrentMetricsMermaidJSState>();
    let _ = use_resource(move || async move {
        let route = "/app/v1/get_state";
        let data_serialized = serde_json::to_string(&get_session_state()
            .with_subject(SessionContextTableNames::MetricsGantt.get_name())
            .make_name()
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
                                "There was a error parsing SyncCurrentMetricsMermaidJSState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        sync_current_metrics_mermaid_state.send(SyncCurrentMetricsMermaidJSState {
                            processor_traces: row
                                .get("processor_traces")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            elapsed_compute: row
                                .get("elapsed_compute")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            output_rows: row
                                .get("output_rows")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                        });
                    }
                }
            }
            Err(err) => tracing::error!("There was a error getting metrics info {err}."),
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
                        serde_json::from_str(json_str.as_str()).unwrap_or_else(|err| {
                            tracing::error!(
                                "There was a error parsing SyncCurrentMetricMermaidJSState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        sync_current_metrics_mermaid_state.send(SyncCurrentMetricMermaidJSState {
                            processor_traces: row
                                .get("processor_traces")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            elapsed_compute: row
                                .get("elapsed_compute")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            output_rows: row
                                .get("output_rows")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                        });
                    }
                }
            }
            Err(err) => tracing::error!("{err}"),
        }
    });

    // DM: we have to re-render the entire virtual DOM everytime the mermaid svg changes...
    let diagram_code: Memo<String> = use_memo(move || {
        if &ACTIVE_METRIC.read().to_string() == SESSION_METRICS_HEADERS.first().unwrap() {
            MERMAID_PROCESSOR_TRACES.read().to_string()
        } else if &ACTIVE_METRIC.read().to_string() == SESSION_METRICS_HEADERS.get(1).unwrap() {
            MERMAID_ELAPSED_COMPUTE.read().to_string()
        } else if &ACTIVE_METRIC.read().to_string() == SESSION_METRICS_HEADERS.get(2).unwrap() {
            MERMAID_OUTPUT_ROWS.read().to_string()
        } else {
            "gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tWaiting to retrieve session plan metrics...".to_string()
        }
    });
    let is_flowchart_shown: Memo<bool> = use_memo(move || false);
    let rendered_html = render_mermaid_svg(diagram_code, "graphDiv", false, is_flowchart_shown);
    let out = if let Some(result) = &*rendered_html.read() {
        match result {
            // Mermaid.js or SessionContextBuilder error
            (_, Some(error), _) => {
                rsx! {
                    if JWT.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please sign-in before searching metrics." },
                        }
                    } else if ACTIVE_SESSION_NAME.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please activate a session before searching metrics." },
                        }
                    } else if MERMAID_ELAPSED_COMPUTE.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve session plan metrics..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            metrics_dropdown {}
                            p { "{error}" },
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
                            p { "Please sign-in before searching metrics." },
                        }
                    } else if ACTIVE_SESSION_NAME.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please activate a session before searching metrics." },
                        }
                    } else if MERMAID_ELAPSED_COMPUTE.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve session plan metrics..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            metrics_dropdown {}
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
                            p { "Please sign-in before searching metrics." },
                        }
                    } else if ACTIVE_SESSION_NAME.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Please activate a session before searching metrics." },
                        }
                    } else if MERMAID_ELAPSED_COMPUTE.read().is_empty() {
                        div {
                            class: "messaging_list",
                            p { "Waiting to retrieve session plan metrics..." },
                        }
                    } else {
                        div {
                            class: "messaging_list",
                            metrics_dropdown {},
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
                    p { "Please sign-in before searching metrics." },
                }
            } else if ACTIVE_SESSION_NAME.read().is_empty() {
                div {
                    class: "messaging_list",
                    p { "Please activate a session before searching metrics." },
                }
            } else if MERMAID_ELAPSED_COMPUTE.read().is_empty() {
                div {
                    class: "messaging_list",
                    p { "Waiting to retrieve session plan metrics..." },
                }
            } else {
                div {
                    class: "messaging_list",
                    metrics_dropdown {},
                }
            }
        }
    };
    out
}

/// Metrics dropdown
#[component]
pub fn metrics_dropdown() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_metric_state);

    // Dropdown signals
    let mut show_metric_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut metric_dropdown = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut metrics_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    rsx! {
        div {
            class: "dropdown_form",
            form {
                class: "dropdown_form_input",
                input {
                    r#type: "text",
                    placeholder: "search session",
                    value: "{metric_dropdown}",
                    onclick: move |_| show_metric_dropdown.set(true),
                    onfocusout: move |_| show_metric_dropdown.set(false),
                    oninput: move |evt| metric_dropdown.set(evt.value()),
                    onkeyup: move |_| {
                        metrics_filtered.set(SESSION_METRICS_HEADERS.iter()
                            .filter(|s| !s.contains(metric_dropdown.read().as_str()))
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>());
                    }
                },
            },
            button {
                class: "dropdown_form_button",
                onclick: move |_evt| async move {
                    // Reset the dropdown
                    let active_metric = metric_dropdown.try_read().unwrap().to_string();
                    metric_dropdown.set(String::new());

                    // Set the active session
                    let sync_current_active_metric_state = use_coroutine_handle::<SyncCurrentActiveMetricState>();
                    sync_current_active_metric_state.send(SyncCurrentActiveMetricState { name: active_metric.clone() });
                },
                svg { dangerous_inner_html: search_icon_svg() },
            },
        }

        // Dynamic dropdown
        if show_metric_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "sessions_dropdown_list",
                    {SESSION_METRICS_HEADERS.iter().filter(|s| ACTIVE_METRIC.read().to_string()!=**s && !metrics_filtered.read().contains(&s.to_string())).enumerate().map(|(i, sub)|  {
                        rsx! {
                            li {
                                key: "{i}",
                                div {
                                    onmouseover: move |_evt| metric_dropdown.set(sub.to_string()),
                                    p { "{sub}" },
                                }
                            }
                        }
                    })}
                }
            }
        }
    }
}
