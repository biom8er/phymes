use dioxus::prelude::*;
use futures::StreamExt;
use phymes_core::{
    session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait}, session_context::SessionContextTableNames}, table::{data_format::DataFormat, table_publish::TablePublish}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;
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
        apps::ACTIVE_SESSION_NAME,
        sign_in::{EMAIL, JWT},
        svg_icons::ms_search_icon_svg
    },
    ui::apps::mermaid_view
};

const SESSION_METRICS_HEADERS: [&str; 3] = ["processor_traces", "elapsed_compute", "output_rows"];

#[component]
pub fn metrics_interface_view() -> Element {
    // Initalize signals
    let active_metric = use_signal(String::new);
    let mut mermaid_processor_traces = use_signal(String::new);
    let mut mermaid_elapsed_compute = use_signal(String::new);
    let mut mermaid_output_rows = use_signal(String::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || SessionInterfaceMessage::get_builder()
        .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_format(&DataFormat::Bytes)
        .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_update(&TablePublish::None)
        .with_stream(false)
    );

    // Get the active session info for the metrics view
    let got_metrics = use_memo(move || !mermaid_processor_traces().is_empty());
    let _ = use_resource(move || async move {
        // Prevent re-fetching metrics if we already have them
        if got_metrics() {
            return;
        }

        let route = "/app/v1/get_state";
        let data_serialized = serde_json::to_string(&get_session_state()
            .with_subject(SessionContextTableNames::MetricsGantt.get_name())
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
                                "There was a error parsing SyncCurrentMetricsMermaidJSState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        mermaid_processor_traces.set(row
                            .get("processor_traces")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        mermaid_elapsed_compute.set(row
                            .get("elapsed_compute")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        mermaid_output_rows.set(row
                            .get("output_rows")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
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
                        mermaid_processor_traces.set(row
                            .get("processor_traces")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        mermaid_elapsed_compute.set(row
                            .get("elapsed_compute")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        mermaid_output_rows.set(row
                            .get("output_rows")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                    }
                }
            }
            Err(err) => tracing::error!("{err}"),
        }
    });

    let diagram_code: Memo<String> = use_memo(move || {
        if &active_metric.read().to_string() == SESSION_METRICS_HEADERS.first().unwrap() {
            mermaid_processor_traces.read().to_string()
        } else if &active_metric.read().to_string() == SESSION_METRICS_HEADERS.get(1).unwrap() {
            mermaid_elapsed_compute.read().to_string()
        } else if &active_metric.read().to_string() == SESSION_METRICS_HEADERS.get(2).unwrap() {
            mermaid_output_rows.read().to_string()
        } else {
            "gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tWaiting to retrieve session plan metrics...".to_string()
        }
    });

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
        } else if mermaid_elapsed_compute.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Waiting to retrieve session plan metrics..." },
            }
        } else if active_metric.read().is_empty() {
            div {
                class: "messaging_list",
                metrics_dropdown {active_metric}
            }
        } else {
            div {
                class: "messaging_list",
                metrics_dropdown {active_metric}
                mermaid_view {diagram_code, check_build: use_signal(|| false), is_flowchart_shown: use_signal(|| false)}
            }
        }
    }
}

/// Metrics dropdown
#[component]
pub fn metrics_dropdown(mut active_metric: Signal<String>) -> Element {

    // Dropdown signals
    let mut show_metric_dropdown = use_signal(|| false);
    let mut metric_dropdown = use_signal(String::new);
    let mut metrics_filtered = use_signal(|| Vec::<String>::new());

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
                    active_metric.set(metric_dropdown.try_read().unwrap().to_string());
                    metric_dropdown.set(String::new());
                },
                svg { dangerous_inner_html: ms_search_icon_svg() },
            },
        }

        // Dynamic dropdown
        if show_metric_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "sessions_dropdown_list",
                    {SESSION_METRICS_HEADERS.iter().filter(|s| active_metric().to_string()!=**s && !metrics_filtered.read().contains(&s.to_string())).enumerate().map(|(i, sub)|  {
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
