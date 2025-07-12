use dioxus::prelude::*;
use futures::StreamExt;
use reqwest::{self, header::CONTENT_TYPE};
use serde_json::{Map, Value};

use crate::ui::{
    backend::{create_session_name, GetSessionState, ADDR_BACKEND},
    metrics_state::{
        clear_metrics_info_state, sync_current_metrics_info_state, ClearMetricsInfoState,
        SyncCurrentMetricsInfoState, METRIC_NAMES, METRIC_TASK_NAMES, METRIC_VALUES,
    },
    settings_interface::get_non_duplicated_sorted_subjects,
    settings_state::ACTIVE_SESSION_NAME,
    sign_in_state::{EMAIL, JWT},
    svg_icons::search_icon_svg,
};

const SESSION_METRICS_HEADERS: [&str; 3] = ["Task", "Metric", "Value"];

/// View to display the subject tables for the session
/// and to allow for easier upload by the user
#[component]
pub fn metrics_modal() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_metrics_info_state);
    use_coroutine(clear_metrics_info_state);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<GetSessionState> = use_memo(move || GetSessionState {
        session_name: create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()),
        subject_name: "".to_string(),
        format: "".to_string(),
    });

    // Get the active session info for the metrics view
    let clear_metrics_info_state = use_coroutine_handle::<ClearMetricsInfoState>();
    let sync_current_metrics_info_state = use_coroutine_handle::<SyncCurrentMetricsInfoState>();
    let _ = use_resource(move || async move {
        let data_serialized = serde_json::to_string(&get_session_state()).unwrap();
        clear_metrics_info_state.send(ClearMetricsInfoState {});
        let addr = format!("{ADDR_BACKEND}/app/v1/metrics_info");
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
                        serde_json::from_str(json_str.as_str()).unwrap_or_else(|_err| {
                            // content.write().push_str(format!("There was a error parsing SyncCurrentMetricsInfoState {err}.").as_str());
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        let metric_value = if let Some(Value::Number(val)) = row.get("metric_value")
                        {
                            val.as_u64().unwrap()
                        } else {
                            0
                        };
                        let metric_task_name =
                            if let Some(Value::String(val)) = row.get("task_name") {
                                val.to_owned()
                            } else {
                                "".to_string()
                            };
                        let metric_name = if let Some(Value::String(val)) = row.get("metric_name") {
                            val.to_owned()
                        } else {
                            "".to_string()
                        };
                        sync_current_metrics_info_state.send(SyncCurrentMetricsInfoState {
                            metric_task_name,
                            metric_name,
                            metric_value,
                        });
                    }
                }
            }
            Err(_err) => (), //content.write().push_str(format!("There was a error getting metrics info {err}.").as_str()),
        }
    });

    // Dropdown signals
    let mut show_metric_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut metric_dropdown = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut metrics_shown: Signal<Vec<String>> = use_signal(|| Vec::new());
    let metrics_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &METRIC_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut metrics_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    let mut show_tasks_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut tasks_dropdown = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut tasks_shown: Signal<Vec<String>> = use_signal(|| Vec::new());
    let tasks_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &METRIC_TASK_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut tasks_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    let indices_filtered = use_memo(move || {
        (0..METRIC_TASK_NAMES.len())
            .filter(|i| {
                metrics_shown().contains(METRIC_NAMES().get(*i).unwrap())
                    && tasks_shown().contains(METRIC_TASK_NAMES().get(*i).unwrap())
            })
            .collect::<Vec<_>>()
    });

    rsx! {
        // Check for sign-in
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
        } else if tasks_vec().is_empty() {
            div {
                class: "messaging_list",
                p { "Waiting to retrieve session plan metrics..." },
            }
        } else {
            div {
                class: "messaging_list",
                // Search metrics
                div {
                    class: "dropdown_form",
                    form {
                        id: "i_search_metrics_form",
                        input {
                            r#type: "text",
                            placeholder: "search metrics",
                            value: "{metric_dropdown}",
                            onclick: move |_| show_metric_dropdown.set(true),
                            onfocusout: move |_| show_metric_dropdown.set(false),
                            oninput: move |evt| metric_dropdown.set(evt.value()),
                            onkeyup: move |_| {
                                metrics_filtered.set(metrics_vec().iter()
                                    .filter(|s| !s.contains(metric_dropdown.read().as_str()))
                                    .cloned()
                                    .collect::<Vec<_>>());
                            }
                        },
                    },
                    button {
                        onclick: move |_evt| {
                            if !metric_dropdown.read().is_empty() {
                                metrics_shown.write().push(metric_dropdown.to_string());
                                metric_dropdown.set(String::new());
                            }
                        },
                        svg { dangerous_inner_html: search_icon_svg() },
                    },
                    button {
                        id: "i_metrics_all",
                        onclick: move |_| {
                            metrics_shown.write().clear();
                            for sub in metrics_vec().iter() {
                                metrics_shown.write().push(sub.to_string());
                            }
                        },
                        "All"
                    },
                    button {
                        id: "i_metrics_one",
                        onclick: move |_| {
                            metrics_shown.write().clear();
                        },
                        "None"
                    }
                }

                // Dynamic dropdown for metrics
                if show_metric_dropdown() {
                    div {
                        class: "dropdown_list",
                        ul {
                            id: "i_search_metrics_dropdown",
                            {metrics_vec().iter().filter(|s| !metrics_shown.read().contains(*s) && !metrics_filtered.read().contains(*s)).enumerate().map(|(i, sub)| {
                                let sub = sub.clone();
                                rsx! {
                                    li {
                                        key: "{i}",
                                        div {
                                            class: "i_search_metrics_dropdown_metric",
                                            onmouseover: move |_evt| metric_dropdown.set(sub.clone()),
                                            h3 { "{sub}" },
                                        }
                                    }
                                }
                            })}
                        }
                    }
                }

                // Search tasks
                div {
                    class: "dropdown_form",
                    form {
                        id: "i_search_tasks_form",
                        input {
                            r#type: "text",
                            placeholder: "search tasks",
                            value: "{tasks_dropdown}",
                            onclick: move |_| show_tasks_dropdown.set(true),
                            onfocusout: move |_| show_tasks_dropdown.set(false),
                            oninput: move |evt| tasks_dropdown.set(evt.value()),
                            onkeyup: move |_| {
                                tasks_filtered.set(tasks_vec().iter()
                                    .filter(|s| !s.contains(tasks_dropdown.read().as_str()))
                                    .cloned()
                                    .collect::<Vec<_>>());
                            }
                        },
                    },
                    button {
                        onclick: move |_evt| {
                            if !tasks_dropdown.read().is_empty(){
                                tasks_shown.write().push(tasks_dropdown.to_string());
                                tasks_dropdown.set(String::new());
                            }
                        },
                        svg { dangerous_inner_html: search_icon_svg() },
                    },
                    button {
                        id: "i_tasks_all",
                        onclick: move |_| {
                            tasks_shown.write().clear();
                            for task in tasks_vec().iter() {
                                tasks_shown.write().push(task.to_string());
                            }
                        },
                        "All"
                    },
                    button {
                        id: "i_tasks_one",
                        onclick: move |_| {
                            tasks_shown.write().clear();
                        },
                        "None"
                    }
                }

                // Dynamic dropdown for tasks
                if show_tasks_dropdown() {
                    div {
                        class: "dropdown_list",
                        ul {
                            id: "i_search_tasks_dropdown",
                            {tasks_vec().iter().filter(|s| !tasks_shown.read().contains(*s) && !tasks_filtered.read().contains(*s)).enumerate().map(|(i, task)| {
                                let task = task.clone();
                                rsx! {
                                    li {
                                        key: "{i}",
                                        div {
                                            class: "i_search_tasks_dropdown_metric",
                                            onmouseover: move |_evt| tasks_dropdown.set(task.clone()),
                                            h3 { "{task}" },
                                        }
                                    }
                                }
                            })}
                        }
                    }
                }

                if !METRIC_TASK_NAMES.read().is_empty() {
                    // Table of the subject schema
                    div {
                        class: "output_table",
                        table {
                            caption { "Metrics for session plan {ACTIVE_SESSION_NAME.read().to_string()}."},
                            tr {
                                {SESSION_METRICS_HEADERS.iter().map(|header| {
                                    rsx! {
                                        th { "{header}" }
                                    }
                                })}
                            },
                            {indices_filtered().iter().map(|i| {
                                let c1 = METRIC_TASK_NAMES.get(*i).unwrap().to_string();
                                let c2 = METRIC_NAMES.get(*i).unwrap().to_string();
                                let c3 = METRIC_VALUES.get(*i).unwrap().to_string();
                                rsx! {
                                    tr {
                                        td { "{c1}" },
                                        td { "{c2}" },
                                        td { "{c3}" },
                                    }
                                }
                            })}
                        }
                    }
                }
            }
        }
    }
}
