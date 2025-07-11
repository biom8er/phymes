use dioxus::prelude::*;
use futures::StreamExt;
use reqwest::{self, header::CONTENT_TYPE};

// File upload imports
use dioxus::prelude::dioxus_elements::FileEngine;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;

use super::svg_icons::{arrow_add_icon_svg, arrow_down_icon_svg, search_icon_svg, table_icon_svg};

use crate::ui::{
    backend::{create_session_name, ADDR_BACKEND, GetSessionState},
    settings_interface::get_non_duplicated_sorted_subjects,
    settings_state::ACTIVE_SESSION_NAME,
    sign_in_state::{EMAIL, JWT},
    subjects_state::{
        clear_subject_info_state, sync_current_subject_info_state, ClearSubjectInfoState,
        SyncCurrentSubjectInfoState, SUBJECT_SCHEMA_COLUMNS, SUBJECT_SCHEMA_NAMES,
        SUBJECT_SCHEMA_ROWS, SUBJECT_SCHEMA_TYPES,
    },
};

const SUBJECT_SCHEMA_HEADERS: [&str; 3] = ["Column", "Type", "Rows"];

/// File upload
#[derive(Debug, Default, Serialize, Deserialize)]
struct PutSessionState {
    pub session_name: String,
    pub subject_name: String,
    pub document_name: String,
    pub text: String,
}

/// File download
#[derive(Debug, Default, Serialize, Deserialize)]
struct DownloadSubject {
    pub download: String,
    pub href: String,
}

/// Chunk a document
///
/// # Arguments
///
/// * `contents` - A string
/// * `chunk_size` - The number of chars (each char is 4 bytes)
///
/// # Returns
///
/// * vector of chunks
#[allow(dead_code)]
fn chunk_document(mut doc: String, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    while doc.len() > chunk_size {
        let (s1, s2) = doc.split_at(chunk_size);
        chunks.push(s1.to_string());
        doc = s2.to_string();
    }
    chunks.push(doc);
    chunks
}

fn get_subject_schema_col_type_rows_by_subject_name(
    active_subject: &str,
    subject_schema_names: &[&str],
    subject_schema_columns: &[&str],
    subject_schema_types: &[&str],
    subject_schema_rows: &[&usize],
) -> (Vec<String>, Vec<String>, Vec<usize>) {
    let indices = subject_schema_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    let columns = subject_schema_columns
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let types = subject_schema_types
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let rows = subject_schema_rows
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_owned().to_owned())
        .collect::<Vec<_>>();
    (columns, types, rows)
}

/// View to display the subject tables for the session
/// and to allow for easier upload by the user
#[component]
pub fn subjects_modal() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_subject_info_state);
    use_coroutine(clear_subject_info_state);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<GetSessionState> = use_memo(move || GetSessionState {
        session_name: create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()),
        subject_name: "".to_string(),
    });

    // Get the active session info for the subject view
    let clear_subjects_info_state = use_coroutine_handle::<ClearSubjectInfoState>();
    let sync_current_subjects_info_state = use_coroutine_handle::<SyncCurrentSubjectInfoState>();
    let _ = use_resource(move || async move {
        let data_serialized = serde_json::to_string(&get_session_state()).unwrap();
        clear_subjects_info_state.send(ClearSubjectInfoState {});
        let addr = format!("{ADDR_BACKEND}/app/v1/subjects_info");
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
                            // DM: find a better way to give feedback to the user
                            // content.write().push_str(format!("There was a error parsing SyncCurrentSubjectInfoState {err}.").as_str());
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        let num_rows = if let Some(Value::Number(val)) = row.get("num_rows") {
                            val.as_u64().unwrap().try_into().unwrap()
                        } else {
                            0
                        };
                        sync_current_subjects_info_state.send(SyncCurrentSubjectInfoState {
                            subject_schema_name: row
                                .get("subject_names")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_column: row
                                .get("column_names")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_type: row
                                .get("type_names")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_row: num_rows,
                        });
                    }
                }
            }
            Err(_err) => (), //content.write().push_str(format!("There was a error getting subjects info {err}.").as_str()),
        }
    });

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut subject_shown = use_signal(|| String::new());

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &SUBJECT_SCHEMA_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    let mut schema_columns = Vec::new();
    let mut schema_types = Vec::new();
    let mut schema_rows = Vec::new();
    if schema_columns.is_empty() {
        (schema_columns, schema_types, schema_rows) =
            get_subject_schema_col_type_rows_by_subject_name(
                subject_shown.read().as_str(),
                &SUBJECT_SCHEMA_NAMES
                    .read()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &SUBJECT_SCHEMA_COLUMNS
                    .read()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &SUBJECT_SCHEMA_TYPES
                    .read()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &SUBJECT_SCHEMA_ROWS.read().iter().collect::<Vec<_>>(),
            );
    }

    // File upload signals
    #[allow(unused_mut)]
    let mut enable_directory_upload = use_signal(|| false);
    let mut files_uploaded = use_signal(|| Vec::new() as Vec<PutSessionState>);
    let file_names = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &files_uploaded
                .read()
                .iter()
                .map(|s| s.document_name.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut content = use_signal(|| String::new());

    let read_files = move |file_engine: Arc<dyn FileEngine>| async move {
        let files = file_engine.files();
        for file_name in &files {
            if let Some(contents) = file_engine.read_file_to_string(file_name).await {
                files_uploaded.write().push(PutSessionState {
                    session_name: create_session_name(
                        EMAIL.read().as_str(),
                        ACTIVE_SESSION_NAME.read().as_str(),
                    ),
                    subject_name: subject_shown.read().to_string(),
                    document_name: file_name.clone(),
                    text: contents,
                });
                // DM: for new we assume the documents are already chunked appropriately...
                // let chunks = chunk_document(contents, 256);
                // for chunk in chunks {
                //     files_uploaded.write().push(UploadedFile {
                //         name: file_name.clone(),
                //         contents: chunk,
                //     });
                // }
            }
        }
    };

    let upload_files = move |evt: FormEvent| async move {
        if let Some(file_engine) = evt.files() {
            read_files(file_engine).await;
        }
    };

    // File download signals
    let mut files_downloaded = use_signal(|| Vec::new() as Vec<DownloadSubject>);

    rsx! {
        // Check for sign-in
        if JWT.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please sign-in before searching subjects." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please activate a session before searching subjects." },
            }
        } else if SUBJECT_SCHEMA_NAMES.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Waiting to retrieve session plan subject schemas..." },
            }
        } else {
            // Search subjects
            div {
                class: "messaging_list",
                div {
                    class: "dropdown_form",
                    form {
                        class: "dropdown_form_input",
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
                        class: "dropdown_form_button",
                        onclick: move |_evt| async move {
                            subject_shown.set(subject_dropdown.to_string());
                            subject_dropdown.set(String::new());
                        },
                        svg { dangerous_inner_html: search_icon_svg() },
                    },
                }

                // Dynamic dropdown of subjects
                if show_subject_dropdown() {
                    div {
                        class: "dropdown_list",
                        ul {
                            id: "search_subjects_dropdown",
                            {subjects_vec().iter().filter(|s| subject_shown.to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
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

                // Table of the subject schema
                div {
                    class: "output_table",
                    table {
                        caption { "Schema for subject {subject_shown.to_string()}."},
                        tr {
                            {SUBJECT_SCHEMA_HEADERS.iter().map(|header| {
                                rsx! {
                                    th { "{header}" }
                                }
                            })}
                        },
                        {(0..schema_columns.len()).map(|i| {
                            let subject_col = schema_columns.get(i).unwrap().to_string();
                            let subject_type = schema_types.get(i).unwrap().to_string();
                            let subject_rows = schema_rows.get(i).unwrap().to_string();
                            rsx! {
                                tr {
                                    td { "{subject_col}" },
                                    td { "{subject_type}" },
                                    td { "{subject_rows}" },
                                }
                            }
                        })}
                    }
                }

                // File upload and download
                // DM: based on https://github.com/DioxusLabs/dioxus/blob/main/examples/file_upload.rs
                if !subject_shown.read().is_empty() {
                    div {
                        class: "file_upload_form",
                        div {
                            id: "file_upload_form",
                            h2 { "Add data to subject {subject_shown}" },
                            div {
                                class: "drop_box",
                                p { "CSV (comma delimiter with headers)" },
                                label { r#for: "textreader", svg { dangerous_inner_html: arrow_add_icon_svg() } }
                                input {
                                    r#type: "file",
                                    accept: ".csv,",
                                    multiple: true,
                                    id: "textreader",
                                    directory: enable_directory_upload,
                                    onchange: upload_files,
                                },
                            }
                        }
                        div {
                            id: "file_download_form",
                            h2 { "Download data from subject {subject_shown}" },
                            div {
                                class: "drop_box",
                                p { "CSV (comma delimiter with headers)" },
                                button {
                                    class: "dropdown_form_button",
                                    onclick: move |_evt| async move {
                                        // Get csv file from the server
                                        files_downloaded.write().clear();
                                        let data = GetSessionState {
                                            session_name: create_session_name(EMAIL.read().as_str(), ACTIVE_SESSION_NAME.read().as_str()),
                                            subject_name: subject_shown.read().to_string(),
                                        };
                                        let data_serialized = serde_json::to_string(&data).unwrap();
                                        let addr = format!("{ADDR_BACKEND}/app/v1/get_state");
                                        match reqwest::Client::new()
                                            .post(addr)
                                            .bearer_auth(JWT.read().to_string())
                                            .header(CONTENT_TYPE, "application/json")
                                            .body(data_serialized)
                                            .send()
                                            .await {
                                            Ok(stream) => {
                                                let mut stream = stream.bytes_stream();
                                                let mut csv_chunks = Vec::new();
                                                while let Some(Ok(bytes)) = stream.next().await {
                                                    let csv_chunk = String::from_utf8_lossy(bytes.as_ref()).into_owned();
                                                    csv_chunks.push(csv_chunk);
                                                }
                                                let data = DownloadSubject {
                                                    download: format!("{}.csv", subject_shown.read().as_str()),
                                                    href: format!("data:text/plain,{}", csv_chunks.join("").as_str()),
                                                };
                                                files_downloaded.write().push(data);
                                            },
                                            Err(err) => content.write().push_str(format!("There was a error downloading subject {err}.").as_str()),
                                        }
                                    },
                                    svg { dangerous_inner_html: arrow_down_icon_svg() },
                                },
                            }
                        }
                    }
                }

                if !files_uploaded.read().is_empty() {
                    // Show uploaded files and their upload status
                    div {
                        class: "files",
                        p { "Files to upload" },
                        ul {
                            id: "uploaded_subject_files",
                            class: "file_list",
                            {file_names.iter().enumerate().map(|(i, f)| {
                                rsx! {
                                    li {
                                        key: "{i}",
                                        div {
                                            class: "files",
                                            svg { dangerous_inner_html: table_icon_svg() }, //color red if failure with error message
                                            h3 { "{f}" },
                                            // div { class: "loader" },
                                        }
                                    }
                                }
                            })}
                        },
                        button {
                            id: "submit_files",
                            onclick: move |_| async move {
                                // Send files to the server
                                for file in files_uploaded.read().iter() {
                                    let data_serialized = serde_json::to_string(file).expect("Failed to serialize data!");
                                    let addr = format!("{ADDR_BACKEND}/app/v1/put_state");
                                    match reqwest::Client::new()
                                        .post(addr)
                                        .bearer_auth(JWT.read().to_string())
                                        .header(CONTENT_TYPE, "application/json")
                                        .body(data_serialized)
                                        .send()
                                        .await {
                                        Ok(response) => match response.text().await {
                                            // DM: Find a better way to give feedback to the user on success and error
                                            Ok(_text) => (),
                                            Err(_err) => (),
                                        },
                                        Err(_err) => (),
                                    }
                                }

                                // Clean up the files
                                files_uploaded.write().clear()
                            },
                            "Submit files"
                        },
                        button {
                            id: "clear_uploaded_files",
                            onclick: move |_| files_uploaded.write().clear(),
                            "Clear files"
                        },
                    }
                }

                if !files_downloaded.read().is_empty() {
                    div {
                        class: "files",
                        p { "Files to download" },
                        ul {
                            id: "download_subject_files",
                            class: "file_list",
                            {files_downloaded.read().iter().enumerate().map(|(i, f)| {
                                rsx! {
                                    li {
                                        key: "{i}",
                                        div {
                                            class: "files",
                                            svg { dangerous_inner_html: table_icon_svg() }, //color red if failure with error message
                                            a {
                                                href: f.href.to_owned(),
                                                download: f.download.to_owned(),
                                                "{f.download.as_str()}"
                                            },
                                        }
                                    }
                                }
                            })}
                        },
                        button {
                            id: "clear_downloaded_files",
                            onclick: move |_| files_downloaded.write().clear(),
                            "Clear files"
                        },
                    }
                }

                // File icon and loader
                // see https://www.w3schools.com/howto/howto_css_loader.asp

                // Draggable processor to task
                // see https://www.w3schools.com/howto/howto_js_draggable.asp
                // see https://www.w3schools.com/HTML/html5_draganddrop.asp
            
            }
        }
    }
}
