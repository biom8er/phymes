use anyhow::{anyhow, Result};
use bytes::Bytes;
use dioxus::prelude::*;
use futures::StreamExt;
use phymes_agents::session_plans::available_interface_subjects::AvailableInterfaceSubjects;
use phymes_core::{
    schemas::{available_subjects::create_timestamp_micros, blob::create_blob_batch}, session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait}, session_context::SessionContextTableNames}, table::{data_format::DataFormat, table_publish::TablePublish, table_trait::{Table, TableBuilderTrait, TableTrait}}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

// File upload imports
use dioxus::prelude::dioxus_elements::FileEngine;
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
            get_subject_num_rows_by_subject_name, get_subject_schema_col_type_by_subject_name, SUBJECT_SCHEMA_HEADERS
        },
    },
    ui::svg_icons::{aws_table_icon_svg, b8_microphone_icon_svg, b8_send_icon_svg, fa_trash_icon_svg, ms_attachment_icon_svg, ms_cloud_add_icon_svg, ms_cloud_arrow_down_icon_svg, ms_cloud_arrow_up_icon_svg, ms_code_icon_svg, ms_document_icon_svg, ms_document_text_icon_svg, ms_search_icon_svg, ms_video_icon_svg},
};

pub fn extension_to_icon_svg(extension: &str) -> String {
    match extension.to_lowercase().as_str() {
        "pdf" => ms_document_icon_svg(),
        "mp3" | "wav" | "aac" => b8_microphone_icon_svg(),
        "mp4" | "avi" => ms_video_icon_svg(),
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => ms_search_icon_svg(),
        "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go" | "rs" | "json" | "svg" | "html" => ms_code_icon_svg(),
        "csv" | "tsv" => aws_table_icon_svg(),
        _ => ms_attachment_icon_svg(),
    }
}

pub fn extension_to_subject(extension: &str) -> Result<AvailableInterfaceSubjects> {
    let subject = match extension.to_lowercase().as_str() {
        "pdf" => AvailableInterfaceSubjects::UserPdf,
        "mp3" | "wav" | "aac" => AvailableInterfaceSubjects::UserAudio,
        "mp4" | "avi" => AvailableInterfaceSubjects::UserVideo,
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => AvailableInterfaceSubjects::UserImage,
        "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go" | "rs" | "json" | "svg" | "html" => AvailableInterfaceSubjects::UserScript,
        "csv" | "tsv" => AvailableInterfaceSubjects::UserCsv,
        _ => return Err(anyhow!("Conversion to subject is not supported for extension {extension}")),
    };
    Ok(subject)
}

/// Based on https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
pub fn extension_to_mime_type(extension: &str) -> Result<&str> {
    let mime_type = match extension.to_lowercase().as_str() {
        "txt" => "text/plain",
        "svg" => "image/svg+xml",
        "html" => "text/html",
        "pdf" => "application/pdf",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "aac" => "audio/aac",
        "mp4" => "video/mp4",
        "avi" => "video/x-msvideo",
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tiff" => "image/tiff",
        "js" => "text/javascrip",
        "jar" => "application/java-archive",
        "json" => "application/json",
        "csv" | "tsv" => "text/csv",
        _ => return Err(anyhow!("Conversion to MIME type is not supported for extension {extension}")),
    };
    Ok(mime_type)
}

/// Based on https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Schemes/data
pub fn extension_and_file_to_href(extension: &str, bytes: &[u8]) -> Result<String> {
    let mime_type = extension_to_mime_type(extension)?;
    let href = match extension {
        "txt" | "csv" | "tsv" | "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go" | "rs" | "json" | "svg" | "html" => {
            let data = String::from_utf8_lossy(bytes.as_ref()).into_owned();
            format!{"data:{mime_type},{data}"}
        },
        _ => {           
            let data = String::from_utf8_lossy(bytes.as_ref()).into_owned();
            format!{"data:{mime_type},{data}"}
        },
    };
    Ok(href)
}

pub fn filename_and_extension_to_download(filename: &str, extension: &str) -> String {
    format!("{filename}.{extension}")
}

/// View to display the subject tables for the session
/// and to allow for easier upload by the user
#[component]
pub fn subjects_interface_view() -> Element {
    // Global signals
    let mut subject_schema_names = use_signal(Vec::<String>::new);
    let mut subject_schema_columns = use_signal(Vec::<String>::new);
    let mut subject_schema_types = use_signal(Vec::<String>::new);
    let mut subject_names = use_signal(Vec::<String>::new);
    let mut subject_num_rows = use_signal(Vec::<usize>::new);
    let active_subject_name = use_signal(String::new);
    let files_uploaded = use_signal(Vec::<SessionInterfaceMessage>::new);
    let filenames_uploaded = use_signal(Vec::<String>::new);
    let extensions_uploaded = use_signal(Vec::<String>::new);
    let files_downloaded = use_signal(Vec::<Vec<u8>>::new);
    let filenames_downloaded = use_signal(Vec::<String>::new);
    let extensions_downloaded = use_signal(Vec::<String>::new);

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
        subject_schema_names.set(Vec::new());
        subject_schema_columns.set(Vec::new());
        subject_schema_types.set(Vec::new());
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
                        subject_schema_names.push(row
                            .get("subject_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        subject_schema_columns.push(row
                            .get("column_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        subject_schema_types.push(row
                            .get("type_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
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
                        subject_schema_names.push(row
                            .get("subject_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        subject_schema_columns.push(row
                            .get("column_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        subject_schema_types.push(row
                            .get("type_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }

        // Get the active session row counts for the subject view
        subject_names.set(Vec::new());
        subject_num_rows.set(Vec::new());
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
                        subject_names.push(row
                            .get("subject_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        subject_num_rows.push(num_rows);
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
                        subject_names.push(row
                            .get("subject_name")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string());
                        subject_num_rows.push(num_rows);
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

    rsx! {
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
        } else if subject_schema_names.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Waiting to retrieve session plan subject schemas..." },
            }
        } else {
            div {
                class: "messaging_list",
                subjects_dropdown_menu { active_subject_name, subject_schema_names },
                subjects_schema_table { active_subject_name, subject_schema_names, subject_schema_columns, subject_schema_types, subject_names, subject_num_rows }

                if !active_subject_name().is_empty() {
                    div {
                        class: "file_upload_form",
                        div {
                            id: "file_upload_extend_form",
                            h2 { "Upload data to subject {active_subject_name}" },
                            attach_files_dropbox {active_subject_name, filenames_uploaded, files_uploaded, extensions_uploaded}
                        }
                        div {
                            id: "file_download_form",
                            h2 { "Download data from subject {active_subject_name}" },
                            div {
                                class: "drop_box",
                                p { "CSV (comma delimiter with headers)" },
                                download_files_button { data_format: use_signal(|| DataFormat::CsvDefault), active_subject_name, filenames_downloaded, files_downloaded, extensions_downloaded}
                            }
                        }
                    }

                    div {
                        class: "file_upload_form",
                        if !files_uploaded.read().is_empty() {
                            div {
                                upload_files_list {filenames_uploaded, files_uploaded, extensions_uploaded}
                            }
                            div {
                                div {
                                    upload_files_button {filenames_uploaded, files_uploaded, extensions_uploaded}
                                    clear_upload_files_button {filenames_uploaded, files_uploaded, extensions_uploaded}
                                }
                            }
                        }                        
                        if !files_downloaded.read().is_empty() {
                            div {
                                download_files_list {filenames_downloaded, files_downloaded, extensions_downloaded}                        
                            }
                            div {
                                div {
                                    clear_download_files_button {files_downloaded, filenames_downloaded, extensions_downloaded}
                                }
                            }
                        }
                    }                  
                }
            }
        }
    }
}

#[component]
pub fn subjects_dropdown_menu(mut active_subject_name: Signal<String>, subject_schema_names: Signal<Vec<String>>) -> Element {
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &subject_schema_names
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
                onclick: move |_evt| {
                    active_subject_name.set(subject_dropdown.read().to_string());
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
                    {subjects_vec().iter().filter(|s| active_subject_name.to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
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
pub fn subjects_schema_table(active_subject_name: Signal<String>, 
    subject_schema_names: Signal<Vec<String>>, subject_schema_columns: Signal<Vec<String>>, subject_schema_types: Signal<Vec<String>>, 
    subject_names: Signal<Vec<String>>, subject_num_rows: Signal<Vec<usize>>) -> Element {
    let schema_columns_types = use_memo(move || get_subject_schema_col_type_by_subject_name(
        active_subject_name.read().as_str(),
        &subject_schema_names
            .read()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        &subject_schema_columns
            .read()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        &subject_schema_types
            .read()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
    ));
    
    let num_rows = use_memo(move || get_subject_num_rows_by_subject_name(
        active_subject_name.read().as_str(),
        &subject_names
            .read()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        &subject_num_rows.read().iter().collect::<Vec<_>>(),
    ));

    rsx! {
        div {
            class: "output_table",
            table {
                if active_subject_name().is_empty() {
                    caption { "No subject selected." },
                } else if num_rows().is_empty() {
                    caption { "Schema for {active_subject_name.to_string()}." },
                } else {
                    caption { "{active_subject_name.to_string()}: {num_rows().first().unwrap()} rows." },
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
pub fn attach_files_dropbox(active_subject_name: Signal<String>, mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>, mut filenames_uploaded: Signal<Vec<String>>, mut extensions_uploaded: Signal<Vec<String>>) -> Element {
    rsx! {
        div {
            class: "drop_box",
            p { "CSV (comma delimiter with headers)" },
            attach_files_input { extend_publish: use_signal(|| true), except_files: use_signal(||".csv,.json".to_string()), active_subject_name: Some(active_subject_name), files_uploaded, filenames_uploaded, extensions_uploaded },
            attach_files_input { extend_publish: use_signal(|| false), except_files: use_signal(||".csv,.json".to_string()), active_subject_name: Some(active_subject_name), files_uploaded, filenames_uploaded, extensions_uploaded },
        }
    }
}

#[component]
pub fn attach_files_input(extend_publish: Signal<bool>, except_files: Signal<String>, active_subject_name: Option<Signal<String>>, mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>, mut filenames_uploaded: Signal<Vec<String>>, mut extensions_uploaded: Signal<Vec<String>>) -> Element {
    let enable_directory_upload = use_signal(|| false);

    let read_files = move |file_engine: Arc<dyn FileEngine>, publish: TablePublish| async move {
        let files = file_engine.files();
        for file_name in &files {

            // Determine the file type
            let file_path = std::path::Path::new(file_name);
            match file_path.extension() {
                None => tracing::error!("File {file_name} has no extension."),
                Some(ext) => match DataFormat::from_extension(ext.to_str().unwrap()) {
                    Ok(data_format) => {
                        // if let Some(contents) = file_engine.read_file_to_string(file_name).await {
                        if let Some(contents) = file_engine.read_file(file_name).await {

                            // Determine the subject based on the file extension if no active subject is set
                            let extension = ext.to_str().unwrap();
                            let subject_name = if let Some(name) = &active_subject_name {
                                name.read().to_string()
                            } else {
                                extension_to_subject(extension).unwrap().to_string()
                            };

                            // Wrap the contents into a blob batch if no active subject is set
                            let (message, format) = match active_subject_name {
                                Some(_) => (contents, data_format),
                                None => {
                                    let file_stem = file_path.file_stem().unwrap().to_str().unwrap();
                                    let batch = create_blob_batch(
                                        vec![file_stem.to_string()],
                                        vec![extension.to_string()],
                                        vec![contents],
                                        vec!["user".to_string()],
                                        vec![create_timestamp_micros()],
                                    ).unwrap();
                                    let message = Table::get_builder()
                                        .with_name(subject_name.as_str())
                                        .with_record_batches(vec![batch]).unwrap()
                                        .build().unwrap()
                                        .to_ipc_stream().unwrap();
                                    (message, DataFormat::Ipc)
                                }
                            };

                            // Update the publish method
                            let publish = match publish {
                                TablePublish::Extend {..} => TablePublish::Extend { table_name: subject_name.clone() },
                                TablePublish::Replace {..} => TablePublish::Replace { table_name: subject_name.clone() },
                                _ => TablePublish::None,
                            };

                            // Create the message to upload
                            let data = SessionInterfaceMessage::get_builder()
                                .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_format(&format)
                                .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_update(&publish)
                                .with_stream(false)
                                .with_subject(&subject_name)
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap();
                            files_uploaded.push(data);
                            filenames_uploaded.push(file_name.to_string());
                            extensions_uploaded.push(extension.to_string());
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
                TablePublish::Extend { table_name: "".to_string() }
            )
            .await;
        }
    };

    let upload_files_replace = move |evt: FormEvent| async move {
        if let Some(file_engine) = evt.files() {
            read_files(
                file_engine,
                TablePublish::Replace { table_name: "".to_string() }
            )
            .await;
        }
    };

    rsx! {
        if extend_publish() {
            label { r#for: "textread_extend", svg { dangerous_inner_html: ms_cloud_add_icon_svg() } }
            input {
                r#type: "file",
                accept: "{except_files}",
                multiple: true,
                id: "textread_extend",
                directory: enable_directory_upload,
                onchange: upload_files_extend,                
            },
        } else {
            label { r#for: "textread_add", svg { dangerous_inner_html: ms_cloud_arrow_up_icon_svg() } }
            input {
                r#type: "file",
                accept: "{except_files}",
                multiple: true,
                id: "textread_add",
                directory: enable_directory_upload,
                onchange: upload_files_replace,
            }                
        },
    }
}

#[component]
pub fn attach_textfiles_input(except_files: Signal<String>, mut content: Signal<String>) -> Element {
    let enable_directory_upload = use_signal(|| false);

    let read_files = move |file_engine: Arc<dyn FileEngine>| async move {
        let files = file_engine.files();
        for file_name in &files {
            if let Some(contents) = file_engine.read_file_to_string(file_name).await {
                content.set([content(), contents].join(""));
            }
        }
    };

    let upload_files = move |evt: FormEvent| async move {
        if let Some(file_engine) = evt.files() {
            read_files(file_engine).await;
        }
    };

    rsx! {
        label { r#for: "textread", svg { dangerous_inner_html: ms_document_text_icon_svg() } }
        input {
            r#type: "file",
            accept: "{except_files}",
            multiple: true,
            id: "textread",
            directory: enable_directory_upload,
            onchange: upload_files,                
        },
    }
}

#[component]
pub fn upload_files_list(mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>, mut filenames_uploaded: Signal<Vec<String>>, mut extensions_uploaded: Signal<Vec<String>>) -> Element {
    rsx! {
        div {
            class: "files",
            p { "Files to upload" },
            ul {
                class: "file_list",
                {filenames_uploaded().iter().enumerate().map(|(i, f)| {
                    rsx! {
                        li {
                            key: "{i}",
                            div {
                                class: "files",
                                svg { dangerous_inner_html: extension_to_icon_svg(extensions_uploaded().get(i).unwrap()) },
                                h3 { "{f}" },
                                // div { class: "loader" },
                            }
                        }
                    }
                })}
            },
        }
    }
}

#[component]
pub fn upload_files_button(mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>, mut filenames_uploaded: Signal<Vec<String>>, mut extensions_uploaded: Signal<Vec<String>>) -> Element {
    rsx! {
        button {
            onclick: move |_| async move {
                // Send files to the server
                for file in files_uploaded.read().iter() {
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
                files_uploaded.set(Vec::new());
                filenames_uploaded.set(Vec::new());
                extensions_uploaded.set(Vec::new());
            },
            svg { dangerous_inner_html: b8_send_icon_svg() }
        },
    }
}

#[component]
pub fn clear_upload_files_button(mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>, mut filenames_uploaded: Signal<Vec<String>>, mut extensions_uploaded: Signal<Vec<String>>) -> Element {
    rsx! {
        button {
            onclick: move |_| {
                files_uploaded.set(Vec::new());
                filenames_uploaded.set(Vec::new());
                extensions_uploaded.set(Vec::new());
            },
            svg { dangerous_inner_html: fa_trash_icon_svg() }
        },
    }
}

#[component]
pub fn download_files_button(data_format: Signal<DataFormat>, active_subject_name: Signal<String>, mut files_downloaded: Signal<Vec<Vec<u8>>>, mut filenames_downloaded: Signal<Vec<String>>, mut extensions_downloaded: Signal<Vec<String>>) -> Element {
    rsx! {
        button {
            class: "dropdown_form_button",
            onclick: move |_evt| async move {
                files_downloaded.set(Vec::new());
                filenames_downloaded.set(Vec::new());
                extensions_downloaded.set(Vec::new());

                let data = SessionInterfaceMessage::get_builder()
                    .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                    .with_format(&data_format())
                    .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                    .with_update(&TablePublish::None)
                    .with_stream(false)
                    .with_subject(&active_subject_name.read())
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
                        let mut bytes_vec = Vec::<u8>::new();
                        while let Some(Ok(bytes)) = stream.next().await {
                            bytes_vec.extend(bytes.to_vec());
                        }
                        files_downloaded.push(bytes_vec);
                        filenames_downloaded.push(active_subject_name.read().as_str().to_string());
                        extensions_downloaded.push(data_format().to_extension().to_string());
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
                            file: csv_chunks.join(""),
                            filename: active_subject_name.read().as_str().to_string(),
                            extension: data_format.to_extension().to_string()
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
pub fn download_files_list(mut files_downloaded: Signal<Vec<Vec<u8>>>, mut filenames_downloaded: Signal<Vec<String>>, mut extensions_downloaded: Signal<Vec<String>>) -> Element {
    rsx! {
        div {
            class: "files",
            p { "Files to download" },
            ul {
                class: "file_list",
                {(0..files_downloaded().len()).map(|i| {
                    let f_download = filename_and_extension_to_download(filenames_downloaded().get(i).unwrap(), extensions_downloaded().get(i).unwrap());
                    let f_href = extension_and_file_to_href(extensions_downloaded().get(i).unwrap() ,files_downloaded().get(i).unwrap()).unwrap();
                    rsx! {
                        li {
                            // key: "{i}",
                            div {
                                class: "files",
                                svg { dangerous_inner_html: extension_to_icon_svg(extensions_downloaded().get(i).unwrap()) },
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
        }
    }
}

#[component]
pub fn clear_download_files_button(mut files_downloaded: Signal<Vec<Vec<u8>>>, mut filenames_downloaded: Signal<Vec<String>>, mut extensions_downloaded: Signal<Vec<String>>) -> Element {

    rsx! {
        button {
            onclick: move |_| {
                files_downloaded.set(Vec::new());
                filenames_downloaded.set(Vec::new());
                extensions_downloaded.set(Vec::new());
            },
            svg { dangerous_inner_html: fa_trash_icon_svg() }
        },
    }
}