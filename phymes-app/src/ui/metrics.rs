use dioxus::prelude::*;
use phymes_agents::{
    AvailableInterfaceSubjects, SessionInterfaceMessage, SessionInterfaceMessageBuilder,
    SessionInterfaceMessageBuilderTrait,
};
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, MessageBuilderTrait, Publication, SubjectBuilder,
    SubjectBuilderTrait, SubjectTrait,
};
use phymes_server::create_session_name;

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
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

use crate::{
    state::{
        get_metric_visualizations_by_metric_name, get_non_duplicated_sorted_subjects,
        svg_icons::ms_search_icon_svg, ACTIVE_SESSION_NAME, EMAIL, JWT,
    },
    ui::mermaid_view,
};

#[component]
pub fn metrics_interface_view() -> Element {
    // Initalize signals
    let active_metric = use_signal(String::new);
    let mut metric_names = use_signal(Vec::<String>::new);
    let mut metric_visualizations = use_signal(Vec::<String>::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || {
        SessionInterfaceMessage::get_builder()
            .with_session_name(&create_session_name(
                EMAIL().as_str(),
                ACTIVE_SESSION_NAME().as_str(),
            ))
            .with_format(&DataFormat::Ipc)
            .with_publisher(&create_session_name(
                EMAIL().as_str(),
                ACTIVE_SESSION_NAME().as_str(),
            ))
            .with_update(&Publication::None)
            .with_stream(false)
    });

    // Get the active session info for the metrics view;
    use_resource(move || async move {
        // Prevent re-fetching metrics if we already have them
        if !metric_names.is_empty() {
            return;
        }

        // DM: https://github.com/biom8er/phymes/issues/111#issue-3492849457
        let route = "/app/v1/diagnostics";
        let data_serialized = serde_json::to_string(
            &get_session_state()
                .with_subject(
                    AvailableInterfaceSubjects::AggregatedAttachments
                        .to_string()
                        .as_str(),
                )
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
                let mut bytes = Vec::new();
                while let Some(Ok(b)) = stream.next().await {
                    bytes.extend(b);
                }
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        // DM: https://github.com/biom8er/phymes/issues/111#issue-3492849457
                        metric_names.set(
                            table
                                .get_column_as_vec_nonprimitive::<String>("filename")
                                .unwrap(),
                        );
                        let viz_str_vec = table
                            .get_column_as_vec_nested_primitive::<u8>("bytes")
                            .unwrap()
                            .into_iter()
                            .map(|bytes| String::from_utf8_lossy(bytes.as_ref()).into_owned())
                            .collect::<Vec<_>>();
                        metric_visualizations.set(viz_str_vec);
                    }
                    Err(err) => {
                        tracing::error!("{err:?}");
                    }
                }
            }
            Err(err) => {
                tracing::error!("There was a error getting session diagnostics info {err}.")
            }
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
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        // DM: https://github.com/biom8er/phymes/issues/111#issue-3492849457
                        metric_names.set(
                            table
                                .get_column_as_vec_nonprimitive::<String>("filename")
                                .unwrap(),
                        );
                        let viz_str_vec = table
                            .get_column_as_vec_nested_primitive::<u8>("bytes")
                            .unwrap()
                            .into_iter()
                            .map(|bytes| String::from_utf8_lossy(bytes.as_ref()).into_owned())
                            .collect::<Vec<_>>();
                        metric_visualizations.set(viz_str_vec);
                    }
                    Err(err) => {
                        tracing::error!("{err:?}");
                    }
                }
            }
            Err(err) => tracing::error!("{err}"),
        }
    });

    let diagram_code: Memo<(String, Option<String>)> = use_memo(move || {
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
                .collect::<Vec<_>>(),
        );
        let visualization = visualizations.pop().unwrap_or("gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\tWaiting to retrieve session plan metrics...".to_string());
        (visualization, None)
    });

    // Build errors that may have occured
    let build_errors = use_signal(String::new);

    rsx! {
        if JWT.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please sign-in before searching metrics." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please activate a session before searching metrics." },
            }
        } else if metric_names.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Waiting to retrieve session plan metrics..." },
            }
        } else if active_metric.read().is_empty() {
            div {
                class: "h-full w-full p-2 flex flex-col items-center",
                metrics_dropdown {active_metric, metric_names}
            }
        } else {
            div {
                class: "h-full w-full p-2 flex flex-col items-center",
                metrics_dropdown {active_metric, metric_names}
                mermaid_view {diagram_code, build_errors}
            }
        }
    }
}

/// Metrics dropdown
#[component]
pub fn metrics_dropdown(
    mut active_metric: Signal<String>,
    metric_names: Signal<Vec<String>>,
) -> Element {
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
    let mut metrics_filtered = use_signal(Vec::<String>::new);

    rsx! {
        div {
            class: "p-2 rounded bg-neutral-800 grid grid-rows-[auto_1fr] grid-cols-[1fr_auto] md:max-w-3/4",
            form {
                class: "w-full h-full flex row-span-1 col-span-1 row-start-1 col-start-1",
                input {
                    class: "w-full h-full bg-neutral-700",
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

            // Dynamic dropdown
            if show_metric_dropdown() {
                div {
                    class: "p-2 rounded bg-neutral-800 list-none flex row-span-1 col-span-1 row-start-2 col-start-1",
                    ul {
                        {metrics_vec().iter().filter(|s| active_metric()!=**s && !metrics_filtered.read().contains(&s.to_string())).enumerate().map(|(i, sub)|  {
                            let sub = sub.clone();
                            rsx! {
                                li {
                                    class: "hover:bg-neutral-700 cursor-pointer",
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

            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-2",
                button {
                    class: "p-1 rounded hover:bg-neutral-700 cursor-pointer flex-none",
                    onclick: move |_evt| async move {
                        // Reset the dropdown
                        active_metric.set(metric_dropdown.try_read().unwrap().to_string());
                        metric_dropdown.set(String::new());
                    },
                    svg {
                        class: "max-w-[48px] max-h-[48px]",
                        dangerous_inner_html: ms_search_icon_svg()
                    },
                },
            }
        }
    }
}
