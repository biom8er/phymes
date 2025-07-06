use dioxus::prelude::*;
use futures::StreamExt;
use reqwest::{self, header::CONTENT_TYPE};
use serde_json::{Map, Value};
use std::collections::HashSet;

use crate::ui::{
    backend::{GetSessionState, ADDR_BACKEND},
    messaging_interface::create_session_name,
    settings_interface::get_non_duplicated_sorted_subjects,
    settings_state::ACTIVE_SESSION_NAME,
    sign_in_state::{EMAIL, JWT},
    svg_icons::search_icon_svg,
    tasks_state::{
        clear_task_info_state, sync_current_task_info_state, ClearTaskInfoState,
        SyncCurrentTaskInfoState, TASK_PROCESSOR_NAMES, TASK_PUB_OR_SUB, TASK_SUBJECT_NAMES,
        TASK_TASK_NAMES,
    },
};

/// Get the distinct processors from the distinct tasks
pub fn get_processors_shown(
    tasks_shown: &[&str],
    tasks: &[&str],
    processors: &[&str],
) -> Vec<String> {
    let tasks_shown_indices = tasks
        .iter()
        .enumerate()
        .filter(|(_i, t)| tasks_shown.contains(t))
        .map(|(i, _t)| i)
        .collect::<Vec<_>>();
    let processors_shown_set = processors
        .iter()
        .enumerate()
        .filter(|(i, _s)| tasks_shown_indices.contains(i))
        .map(|(_i, s)| s)
        .collect::<HashSet<_>>();
    let mut processors_shown = processors_shown_set
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    processors_shown.sort();
    processors_shown
}

/// View to display the subject tables for the session
/// and to allow for easier upload by the user
#[component]
pub fn tasks_modal() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_task_info_state);
    use_coroutine(clear_task_info_state);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state = use_memo(move || GetSessionState {
        session_name: create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()),
        subject_name: "".to_string(),
    });

    // Get the active session info for the task view
    let clear_tasks_info_state = use_coroutine_handle::<ClearTaskInfoState>();
    let sync_current_tasks_info_state = use_coroutine_handle::<SyncCurrentTaskInfoState>();
    let _ = use_resource(move || async move {
        let data_serialized = serde_json::to_string(&get_session_state()).unwrap();
        clear_tasks_info_state.send(ClearTaskInfoState {});
        let addr = format!("{ADDR_BACKEND}/app/v1/tasks_info");
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
                            // content.write().push_str(format!("There was a error parsing SyncCurrentTaskInfoState {err}.").as_str());
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        sync_current_tasks_info_state.send(SyncCurrentTaskInfoState {
                            task_task_name: row
                                .get("task_names")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            task_processor_name: row
                                .get("processor_names")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            task_subject_name: row
                                .get("subject_names")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            task_pub_or_sub: row
                                .get("pub_or_sub")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                        });
                    }
                }
            }
            Err(_err) => (), //content.write().push_str(format!("There was a error getting tasks info {err}.").as_str()),
        }
    });

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut subjects_shown: Signal<Vec<String>> = use_signal(|| Vec::new());
    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &TASK_SUBJECT_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    let mut show_tasks_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut tasks_dropdown = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut tasks_shown: Signal<Vec<String>> = use_signal(|| Vec::new());
    let tasks_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &TASK_TASK_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut tasks_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    let mut show_processors = use_signal(|| false);

    rsx! {
        // Check for sign-in
        if JWT.read().is_empty() {
            div {
                class: "sign-in-modal",
                p { "Please sign-in before searching tasks." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "sign-in-modal",
                p { "Please activate a session before searching tasks." },
            }
        } else if tasks_vec().is_empty() {
            div {
                class: "sign-in-modal",
                p { "Waiting to retrieve session plan tasks..." },
            }
        } else {
            // Search subjects
            div {
                class: "dropdown_form",
                form {
                    id: "i_search_subjects_form",
                    input {
                        r#type: "text",
                        placeholder: "search subjects",
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
                    onclick: move |_evt| {
                        if !subject_dropdown.read().is_empty() {
                            subjects_shown.write().push(subject_dropdown.to_string());
                            subject_dropdown.set(String::new());
                        }
                    },
                    svg { dangerous_inner_html: search_icon_svg() },
                },
                button {
                    id: "i_subjects_all",
                    onclick: move |_| {
                        subjects_shown.write().clear();
                        for sub in subjects_vec().iter() {
                            subjects_shown.write().push(sub.to_string());
                        }
                    },
                    "All"
                },
                button {
                    id: "i_subjects_one",
                    onclick: move |_| {
                        subjects_shown.write().clear();
                    },
                    "None"
                }
            }

            // Dynamic dropdown for subjects
            if show_subject_dropdown() {
                div {
                    class: "dropdown_list",
                    ul {
                        id: "i_search_subjects_dropdown",
                        {subjects_vec().iter().filter(|s| !subjects_shown.read().contains(*s) && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)| {
                            let sub = sub.clone();
                            rsx! {
                                li {
                                    key: "{i}",
                                    div {
                                        class: "i_search_subjects_dropdown_subject",
                                        onmouseover: move |_evt| subject_dropdown.set(sub.clone()),
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
                                        class: "i_search_tasks_dropdown_subject",
                                        onmouseover: move |_evt| tasks_dropdown.set(task.clone()),
                                        h3 { "{task}" },
                                    }
                                }
                            }
                        })}
                    }
                }
            }

            // Toggle tasks or processors
            div {
                class: "dropdown_form",
                button {
                    id: "tasks or processors",
                    onclick: move |_| {
                        if *show_processors.read() {
                            show_processors.set(false);
                        } else {
                            show_processors.set(true);
                        }
                    },
                    if *show_processors.read() {
                        "Switch to tasks"
                    } else {
                        "Switch to processors"
                    }
                }
            }

            // Incidence table of tasks and subjects
            // DM: convert to SequenceDiagram using e.g., Mermaid.js https://codepen.io/atnyman/pen/gabGXV
            div {
                style: "overflow-x:auto;overflow-y:auto;",
                class: "output_table",
                table {
                    id: "output_table_tasks",
                    caption {
                        if *show_processors.read() {
                            "Incidence matrix for processors and subjects"
                        } else {
                            "Incidence matrix for tasks and subjects"
                        }
                    }
                    tr {
                        td { "" },
                        if *show_processors.read() {
                            {get_processors_shown(
                                &tasks_shown.read().iter().map(|task| task.as_str()).collect::<Vec<_>>(),
                                &TASK_TASK_NAMES.read().iter().map(|task| task.as_str()).collect::<Vec<_>>(),
                                &TASK_PROCESSOR_NAMES.read().iter().map(|task| task.as_str()).collect::<Vec<_>>()
                            ).iter().map(|processor| {
                                rsx! {
                                    th { "{processor}" }
                                }
                            })}
                        } else {
                            {tasks_shown.read().iter().map(|task| {
                                rsx! {
                                    th { "{task}" }
                                }
                            })}
                        }
                    },
                    {subjects_shown.read().iter().map(|subject| {
                        rsx! {
                            tr {
                                th { "{subject}" },
                                if *show_processors.read() {
                                    {get_processors_shown(
                                        &tasks_shown.read().iter().map(|task| task.as_str()).collect::<Vec<_>>(),
                                        &TASK_TASK_NAMES.read().iter().map(|task| task.as_str()).collect::<Vec<_>>(),
                                        &TASK_PROCESSOR_NAMES.read().iter().map(|task| task.as_str()).collect::<Vec<_>>()
                                    ).iter().map(|processor| {
                                        // get the indices of the tasks
                                        let indices = TASK_PROCESSOR_NAMES.read().iter()
                                            .enumerate()
                                            .filter(|(_i, p)| **p == processor.as_str())
                                            .map(|(i, _p)| i)
                                            .collect::<Vec<_>>();

                                        // filter subscriptions and publications
                                        let sub_indices = TASK_SUBJECT_NAMES.iter()
                                            .enumerate().filter(|(i, s)|
                                                indices.contains(i) && **s == subject.as_str()
                                            )
                                            .map(|(i, _s)| i)
                                            .collect::<Vec<_>>();
                                        let publications = TASK_PUB_OR_SUB.iter()
                                            .enumerate().filter(|(i, _s)|
                                                sub_indices.contains(i)
                                            )
                                            .map(|(_i, s)| s.to_string())
                                            .collect::<Vec<String>>();

                                        // create the cell entry
                                        let mut symbol = String::new();
                                        if publications.contains(&"+".to_string()) && publications.contains(&"-".to_string()) {
                                            symbol = "+/-".to_string();
                                        } else if publications.contains(&"-".to_string()) {
                                            symbol = "-".to_string();
                                        } else if publications.contains(&"+".to_string()) {
                                            symbol = "+".to_string();
                                        }

                                        rsx! {
                                            td { "{symbol}" }
                                        }
                                    })}
                                } else {
                                    {tasks_shown.read().iter().map(|task| {
                                        // get the indices of the tasks
                                        let indices = TASK_TASK_NAMES.iter()
                                            .enumerate()
                                            .filter(|(_i, t)| **t == task.as_str())
                                            .map(|(i, _t)| i)
                                            .collect::<Vec<_>>();

                                        // filter subscriptions and publications
                                        let sub_indices = TASK_SUBJECT_NAMES.iter()
                                            .enumerate().filter(|(i, s)|
                                                indices.contains(i) && **s == subject.as_str()
                                            )
                                            .map(|(i, _s)| i)
                                            .collect::<Vec<_>>();
                                        let publications = TASK_PUB_OR_SUB.iter()
                                            .enumerate().filter(|(i, _s)|
                                                sub_indices.contains(i)
                                            )
                                            .map(|(_i, s)| s.to_string())
                                            .collect::<Vec<String>>();

                                        // create the cell entry
                                        let mut symbol = String::new();
                                        if publications.contains(&"+".to_string()) && publications.contains(&"-".to_string()) {
                                            symbol = "+/-".to_string();
                                        } else if publications.contains(&"-".to_string()) {
                                            symbol = "-".to_string();
                                        } else if publications.contains(&"+".to_string()) {
                                            symbol = "+".to_string();
                                        }

                                        rsx! {
                                            td { "{symbol}" }
                                        }
                                    })}
                                }
                            }
                        }
                    })}
                }
            }
        }
    }
}
