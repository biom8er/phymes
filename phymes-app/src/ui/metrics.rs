use dioxus::prelude::*;
use phymes_agents::session_plans::available_interface_subjects::AvailableInterfaceSubjects;
use phymes_core::{
    session::{common_traits::{BuildableTrait, BuilderTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait}}, table::{DataFormat, TablePublish}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;
use serde_json::{Map, Value};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(not(feature = "serverless"))]
use futures::StreamExt;

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
        apps::{get_non_duplicated_sorted_subjects, ACTIVE_SESSION_NAME},
        sign_in::{EMAIL, JWT},
        svg_icons::ms_search_icon_svg
    },
    ui::apps::mermaid_view
};

pub fn get_metric_visualizations_by_metric_name(
    active_subject: &str,
    metric_names: &[&str],
    metric_visualizations: &[&str],
) -> Vec<String> {
    let indices = metric_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    metric_visualizations
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>()
}

#[component]
pub fn metrics_interface_view() -> Element {
    // Initalize signals
    let active_metric = use_signal(String::new);
    let mut metric_names = use_signal(Vec::<String>::new);
    let mut metric_visualizations = use_signal(Vec::<String>::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || SessionInterfaceMessage::get_builder()
        .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_format(&DataFormat::Bytes)
        .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_update(&TablePublish::None)
        .with_stream(false)
    );

    // Get the active session info for the metrics view;
    let _ = use_resource(move || async move {
        // Prevent re-fetching metrics if we already have them
        if !metric_names.is_empty() {
            return;
        }

        let route = "/app/v1/get_state";
        // DM: https://github.com/biom8er/phymes/issues/111#issue-3492849457
        // let route = "/app/v1/diagnostics";
        let data_serialized = serde_json::to_string(&get_session_state()
            .with_subject(AvailableInterfaceSubjects::AggregatedAttachments.to_string().as_str())
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
                    let json_rows: Vec<Map<String, Value>> =
                        serde_json::from_slice(bytes.as_ref()).unwrap_or_else(|err| {
                            tracing::error!("There was a error getting the session diagnostics {err}.");
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        metric_names.push("processor_traces".to_string());
                        metric_visualizations.push(row
                            .get("processor_traces")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        metric_names.push("elapsed_compute".to_string());
                        metric_visualizations.push(row
                            .get("elapsed_compute")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        metric_names.push("output_rows".to_string());
                        metric_visualizations.push(row
                            .get("output_rows")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        // DM: https://github.com/biom8er/phymes/issues/111#issue-3492849457
                        // metric_names.push(row
                        //     .get("filename")
                        //     .unwrap()
                        //     .as_str()
                        //     .unwrap()
                        //     .to_string());
                        // let bytes = row.get("bytes").unwrap()
                        //     .as_array().unwrap()
                        //     .iter()
                        //     .map(|v| v.as_u64().unwrap() as u8)
                        //     .collect::<Vec<u8>>();
                        // metric_visualizations.push(String::from_utf8_lossy(bytes.as_ref()).into_owned());
                    }
                }
            }
            Err(err) => tracing::error!("There was a error getting session diagnostics info {err}."),
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
                        serde_json::from_slice(byte.as_ref()).unwrap_or_else(|err| {
                            tracing::error!("There was a error getting the session diagnostics {err}.");
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        metric_names.push("processor_traces".to_string());
                        metric_visualizations.push(row
                            .get("processor_traces")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        metric_names.push("elapsed_compute".to_string());
                        metric_visualizations.push(row
                            .get("elapsed_compute")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        metric_names.push("output_rows".to_string());
                        metric_visualizations.push(row
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
        let mut visualizations = get_metric_visualizations_by_metric_name(
            active_metric.read().as_str(),
            &metric_names
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            &metric_visualizations
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>());
        visualizations.pop().unwrap_or("gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tWaiting to retrieve session plan metrics...".to_string())
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
        } else if metric_names.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Waiting to retrieve session plan metrics..." },
            }
        } else if active_metric.read().is_empty() {
            div {
                class: "messaging_list",
                metrics_dropdown {active_metric, metric_names}
            }
        } else {
            div {
                class: "messaging_list",
                metrics_dropdown {active_metric, metric_names}
                mermaid_view {diagram_code, check_build: use_signal(|| false), is_flowchart_shown: use_signal(|| false)}
            }
        }
    }
}

/// Metrics dropdown
#[component]
pub fn metrics_dropdown(mut active_metric: Signal<String>, metric_names: Signal<Vec<String>>) -> Element {

    // Dropdown signals
    let mut show_metric_dropdown = use_signal(|| false);
    let mut metric_dropdown = use_signal(String::new);
    let metrics_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &metric_names
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
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
                        metrics_filtered.set(metrics_vec().iter()
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
                    {metrics_vec().iter().filter(|s| active_metric().to_string()!=**s && !metrics_filtered.read().contains(&s.to_string())).enumerate().map(|(i, sub)|  {
                        let sub = sub.clone();
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
