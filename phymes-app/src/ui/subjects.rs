use bytes::Bytes;
use dioxus::prelude::*;
use futures::StreamExt;
use phymes_core::{
    session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait}, session_context::SessionContextTableNames}, table::{data_format::DataFormat, table_trait::TableTrait, table_publish::TablePublish}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

// File upload imports
use dioxus::prelude::dioxus_elements::FileEngine;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;

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
        apps::{get_non_duplicated_sorted_subjects, ACTIVE_SESSION_NAME},
        sign_in::{EMAIL, JWT},
        subjects::{
            clear_files_downloaded_state, clear_files_uploaded_state, clear_subject_num_rows_state, clear_subject_schema_state, get_subject_num_rows_by_subject_name, get_subject_schema_col_type_by_subject_name, sync_current_active_subject_state, sync_current_files_downloaded_state, sync_current_files_uploaded_state, sync_current_subject_num_rows_state, sync_current_subject_schema_state, ClearFilesDownloadedState, ClearFilesUploadedState, ClearSubjectNumRowsState, ClearSubjectSchemaState, DownloadSubject, SyncCurrentActiveSubjectState, SyncCurrentSubjectNumRowsState, SyncCurrentSubjectSchemaState, SyncFilesDownloadedState, SyncFilesUploadedState, ACTIVE_SUBJECT_NAME, FILENAMES_DOWNLOADED, FILENAMES_UPLOADED, FILES_DOWNLOADED, FILES_UPLOADED, SUBJECT_NAMES, SUBJECT_NUM_ROWS, SUBJECT_SCHEMA_COLUMNS, SUBJECT_SCHEMA_HEADERS, SUBJECT_SCHEMA_NAMES, SUBJECT_SCHEMA_TYPES
        },
    },
    ui::svg_icons::{aws_table_icon_svg, ms_cloud_add_icon_svg, ms_cloud_arrow_down_icon_svg, ms_cloud_arrow_up_icon_svg, ms_search_icon_svg},
};

/// View to display the subject tables for the session
/// and to allow for easier upload by the user
#[component]
pub fn subjects_interface_view() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_subject_schema_state);
    let sync_current_subjects_schema_state = use_coroutine_handle::<SyncCurrentSubjectSchemaState>();
    use_coroutine(clear_subject_schema_state);
    let clear_subjects_schema_state = use_coroutine_handle::<ClearSubjectSchemaState>();
    use_coroutine(sync_current_subject_num_rows_state);
    let sync_current_subjects_rows_state = use_coroutine_handle::<SyncCurrentSubjectNumRowsState>();
    use_coroutine(clear_subject_num_rows_state);
    let clear_subjects_num_rows_state = use_coroutine_handle::<ClearSubjectNumRowsState>();
    use_coroutine(sync_current_files_uploaded_state);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || SessionInterfaceMessage::get_builder()
        .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_format(&DataFormat::Bytes)
        .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_update(&TablePublish::None)
        .with_stream(false)
    );

    // Get the active session schema for the subject view and
    // Get the active session row counts for the subject view
    // DM: these are combined into a single async block to prevent concurrent mutable borrows of the same user state
    let _ = use_resource(move || async move {
        // Get the active session schema for the subject view
        clear_subjects_schema_state.send(ClearSubjectSchemaState {});
        let route = "/app/v1/get_state";
        let data_serialized = serde_json::to_string(&get_session_state()
            .with_subject(SessionContextTableNames::Subjects.get_name())
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
                                "There was a error parsing SyncCurrentSubjectSchemaState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        sync_current_subjects_schema_state.send(SyncCurrentSubjectSchemaState {
                            subject_schema_name: row
                                .get("subject_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_column: row
                                .get("column_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_type: row
                                .get("type_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                        });
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
                        sync_current_subjects_schema_state.send(SyncCurrentSubjectSchemaState {
                            subject_schema_name: row
                                .get("subject_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_column: row
                                .get("column_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_schema_type: row
                                .get("type_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                        });
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }

        // Get the active session row counts for the subject view
        clear_subjects_num_rows_state.send(ClearSubjectNumRowsState {});
        let route = "/app/v1/get_state";
        let data_serialized = serde_json::to_string(&get_session_state()
            .with_subject(SessionContextTableNames::SubjectsNumRows.get_name())
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
                        sync_current_subjects_rows_state.send(SyncCurrentSubjectNumRowsState {
                            subject_name: row
                                .get("subject_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_num_row: num_rows,
                        });
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
                        let num_rows = if let Some(Value::Number(val)) = row.get("num_rows") {
                            val.as_u64().unwrap().try_into().unwrap()
                        } else {
                            0
                        };
                        sync_current_subjects_rows_state.send(SyncCurrentSubjectNumRowsState {
                            subject_name: row
                                .get("subject_name")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            subject_num_row: num_rows,
                        });
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

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
            div {
                class: "messaging_list",
                subjects_dropdown_menu {}
                subjects_schema_table {}

                if !ACTIVE_SUBJECT_NAME.read().is_empty() {
                    div {
                        class: "file_upload_form",
                        div {
                            id: "file_upload_extend_form",
                            h2 { "Add data to subject {ACTIVE_SUBJECT_NAME}" },
                            attach_files_button {}
                        }
                        div {
                            id: "file_download_form",
                            h2 { "Download data from subject {ACTIVE_SUBJECT_NAME}" },
                            div {
                                class: "drop_box",
                                p { "CSV (comma delimiter with headers)" },
                                download_files_button {}
                            }
                        }
                    }
                }

                if !FILES_UPLOADED.read().is_empty() {
                    upload_files_list {}
                }

                if !FILES_DOWNLOADED.read().is_empty() {
                    download_files_list {}
                }
            }
        }
    }
}

#[component]
pub fn subjects_dropdown_menu() -> Element {
    use_coroutine(sync_current_active_subject_state);
    let sync_current_active_subject_state = use_coroutine_handle::<SyncCurrentActiveSubjectState>();

    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());

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

    rsx! {
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
                    sync_current_active_subject_state.send(SyncCurrentActiveSubjectState {
                        name: subject_dropdown.read().to_string(),
                    });
                    subject_dropdown.set(String::new());
                },
                svg { dangerous_inner_html: ms_search_icon_svg() },
            },
        }

        // Dynamic dropdown of subjects
        if show_subject_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "search_subjects_dropdown",
                    {subjects_vec().iter().filter(|s| ACTIVE_SUBJECT_NAME.to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
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
    }
}

#[component]
pub fn subjects_schema_table() -> Element {
    let schema_columns_types = use_memo(move || get_subject_schema_col_type_by_subject_name(
        ACTIVE_SUBJECT_NAME.read().as_str(),
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
    ));
    
    let num_rows = use_memo(move || get_subject_num_rows_by_subject_name(
        ACTIVE_SUBJECT_NAME.read().as_str(),
        &SUBJECT_NAMES
            .read()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        &SUBJECT_NUM_ROWS.read().iter().collect::<Vec<_>>(),
    ));

    rsx! {
        div {
            class: "output_table",
            table {
                if ACTIVE_SUBJECT_NAME().is_empty() {
                    caption { "No subject selected." },
                } else if num_rows().is_empty() {
                    caption { "Schema for {ACTIVE_SUBJECT_NAME.to_string()}." },
                } else {
                    caption { "{ACTIVE_SUBJECT_NAME.to_string()}: {num_rows().first().unwrap()} rows." },
                }
                tr {
                    {SUBJECT_SCHEMA_HEADERS.iter().map(|header| {
                        rsx! {
                            th { "{header}" }
                        }
                    })}
                },
                {(0..schema_columns_types().0.len()).map(|i| {
                    let subject_col = schema_columns_types().0.get(i).unwrap().to_string();
                    let subject_type = schema_columns_types().1.get(i).unwrap().to_string();
                    rsx! {
                        tr {
                            td { "{subject_col}" },
                            td { "{subject_type}" },
                        }
                    }
                })}
            }
        }
    }
}

#[component]
pub fn attach_files_button() -> Element {
    use_coroutine(sync_current_files_uploaded_state);
    let sync_current_files_uploaded_state = use_coroutine_handle::<SyncFilesUploadedState>();

    #[allow(unused_mut)]
    let mut enable_directory_upload = use_signal(|| false);
    let extend_publish = true;
    let except_files = ".csv,.pdf,.json";

    let read_files = move |file_engine: Arc<dyn FileEngine>, publish: TablePublish| async move {
        let files = file_engine.files();
        for file_name in &files {
            // Determine the file type
            let file_path = std::path::Path::new(file_name);
            match file_path.extension() {
                None => tracing::error!("File {file_name} has no extension."),
                Some(ext) => match DataFormat::from_extension(ext.to_str().unwrap()) {
                    Ok(data_format) => {
                        if let Some(contents) = file_engine.read_file_to_string(file_name).await {
                            let data = SessionInterfaceMessage::get_builder()
                                .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_format(&data_format)
                                .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_update(&publish)
                                .with_stream(false)
                                .with_subject(&ACTIVE_SUBJECT_NAME.read())
                                .with_message(Bytes::from(contents).to_vec())
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap();
                            sync_current_files_uploaded_state.send(SyncFilesUploadedState {
                                files: data,
                                filenames: file_name.to_string()
                            });
                        }
                    }
                    Err(err) => tracing::error!("{err:?}"),
                },
            }
        }
    };

    let upload_files_extend = move |evt: FormEvent| async move {
        if let Some(file_engine) = evt.files() {
            read_files(
                file_engine,
                TablePublish::Extend {
                    table_name: ACTIVE_SUBJECT_NAME.read().to_string(),
                },
            )
            .await;
        }
    };

    let upload_files_replace = move |evt: FormEvent| async move {
        if let Some(file_engine) = evt.files() {
            read_files(
                file_engine,
                TablePublish::Replace {
                    table_name: ACTIVE_SUBJECT_NAME.read().to_string(),
                },
            )
            .await;
        }
    };

    rsx! {
        div {
            class: "drop_box",
            p { "CSV (comma delimiter with headers)" },
            label { r#for: "textread_extend", svg { dangerous_inner_html: ms_cloud_add_icon_svg() } }
            if extend_publish {
                input {
                    r#type: "file",
                    accept: "{except_files}",
                    multiple: true,
                    id: "textread_extend",
                    directory: enable_directory_upload,
                    onchange: upload_files_extend,                
                },
            } else {
                input {
                    r#type: "file",
                    accept: "{except_files}",
                    multiple: true,
                    id: "textread_extend",
                    directory: enable_directory_upload,
                    onchange: upload_files_replace,
                }                
            },
        }
    }
}

#[component]
pub fn upload_files_button() -> Element {    
    use_coroutine(clear_files_uploaded_state);
    let clear_files_uploaded_state = use_coroutine_handle::<ClearFilesUploadedState>();
    rsx! {
        button {
            id: "submit_files",
            onclick: move |_| async move {
                // Send files to the server
                for file in FILES_UPLOADED.read().iter() {
                    let data_serialized = serde_json::to_string(file).unwrap();
                    let route = "/app/v1/put_state";

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
                            // DM: Find a better way to give feedback to the user on success and error
                            Ok(text) => tracing::debug!("Put response {text}"),
                            Err(err) => tracing::error!("Put err {err:?}"),
                        },
                        Err(err) => tracing::error!("Put err {err:?}"),
                    }

                    #[cfg(feature = "serverless")]
                    let config = ServerlessConfig {
                        route: route.to_string(),
                        basic_auth: None,
                        bearer_auth: Some(JWT.read().to_string()),
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
                            let _text = String::from_utf8_lossy(bytes.first().unwrap()).into_owned();
                        }
                        Err(err) => tracing::error!("{err:?}"),
                    }
                }

                // Clean up the files
                clear_files_uploaded_state.send(ClearFilesUploadedState{});
            },
            "Submit files"
        },
        button {
            id: "clear_uploaded_files",
            onclick: move |_| {
                clear_files_uploaded_state.send(ClearFilesUploadedState{});
            },
            "Clear files"
        },
    }
}

#[component]
pub fn upload_files_list() -> Element {
    rsx! {
        div {
            class: "files",
            p { "Files to upload" },
            ul {
                id: "uploaded_subject_files",
                class: "file_list",
                {FILENAMES_UPLOADED.read().iter().enumerate().map(|(i, f)| {
                    rsx! {
                        li {
                            key: "{i}",
                            div {
                                class: "files",
                                svg { dangerous_inner_html: aws_table_icon_svg() }, // color red if failure with error message
                                h3 { "{f}" },
                                // div { class: "loader" },
                            }
                        }
                    }
                })}
            },
            upload_files_button {}
        }
    }
}

#[component]
pub fn download_files_button() -> Element {
    use_coroutine(sync_current_files_downloaded_state);
    let sync_current_files_downloaded_state = use_coroutine_handle::<SyncFilesDownloadedState>();
    use_coroutine(clear_files_downloaded_state);
    let clear_files_downloaded_state = use_coroutine_handle::<ClearFilesDownloadedState>();

    rsx! {
        button {
            class: "dropdown_form_button",
            onclick: move |_evt| async move {
                // Get csv file from the server
                clear_files_downloaded_state.send(ClearFilesDownloadedState{});

                let data = SessionInterfaceMessage::get_builder()
                    .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                    .with_format(&DataFormat::CsvDefault)
                    .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                    .with_update(&TablePublish::None)
                    .with_stream(false)
                    .with_subject(&ACTIVE_SUBJECT_NAME.read())
                    .make_name()
                    .unwrap()
                    .build()
                    .unwrap();
                let data_serialized = serde_json::to_string(&data).unwrap();
                let route = "/app/v1/get_state";

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
                    Ok(stream) => {
                        let mut stream = stream.bytes_stream();
                        let mut csv_chunks = Vec::new();
                        while let Some(Ok(bytes)) = stream.next().await {
                            let csv_chunk = String::from_utf8_lossy(bytes.as_ref()).into_owned();
                            csv_chunks.push(csv_chunk);
                        }
                        sync_current_files_downloaded_state.send(SyncFilesDownloadedState {
                            files: csv_chunks.join(""),
                            filenames: ACTIVE_SUBJECT_NAME.read().as_str().to_string()
                        });
                    },
                    Err(err) => tracing::error!("There was a error downloading subject {err}."),
                }

                #[cfg(feature = "serverless")]
                let config = ServerlessConfig {
                    route: route.to_string(),
                    basic_auth: None,
                    bearer_auth: Some(JWT.read().to_string()),
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
                        let csv_chunks: Vec<String> = bytes
                            .iter()
                            .map(|byte| String::from_utf8_lossy(byte).into_owned())
                            .collect();
                        sync_current_files_downloaded_state.send(SyncFilesDownloadedState {
                            files: csv_chunks.join(""),
                            filenames: ACTIVE_SUBJECT_NAME.read().as_str().to_string()
                        });
                    }
                    Err(err) => tracing::error!("There was a error downloading subject {err}."),
                }
            },
            svg { dangerous_inner_html: ms_cloud_arrow_down_icon_svg() },
        },
    }
}


#[component]
pub fn download_files_list() -> Element {
    use_coroutine(clear_files_downloaded_state);
    let clear_files_downloaded_state = use_coroutine_handle::<ClearFilesDownloadedState>();

    rsx! {
        div {
            class: "files",
            p { "Files to download" },
            ul {
                id: "download_subject_files",
                class: "file_list",
                {(0..FILES_DOWNLOADED.len()).map(|i| {
                    let f_download = format!("{}.csv", FILENAMES_DOWNLOADED().get(i).unwrap());
                    let f_href = format!("data:text/plain,{}", FILES_DOWNLOADED().get(i).unwrap());
                    rsx! {
                        li {
                            key: "{i}",
                            div {
                                class: "files",
                                svg { dangerous_inner_html: aws_table_icon_svg() }, //color red if failure with error message
                                a {
                                    href: f_href.to_owned(),
                                    download: f_download.to_owned(),
                                    "{f_download}"
                                },
                            }
                        }
                    }
                })}
            },
            button {
                id: "clear_downloaded_files",
                onclick: move |_| async move {
                    clear_files_downloaded_state.send(ClearFilesDownloadedState{});
                },
                "Clear files"
            },
        }
    }
}