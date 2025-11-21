use dioxus::prelude::*;
use phymes_core::{
    create_blob_batch, BuildableTrait, BuilderTrait, DataFormat, MessageBuilderTrait,
    SessionInterfaceMessage, SessionInterfaceMessageBuilderTrait, Table, TableBuilderTrait,
    TablePublication, TableTrait,
};
use phymes_diagnostics::create_timestamp_micros;
use phymes_server::create_session_name;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

// File upload imports
use dioxus::prelude::dioxus_elements::FileEngine;
use std::sync::Arc;

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

use crate::state::{
    extension_and_file_to_data_href, extension_to_icon_svg, extension_to_subject,
    filename_and_extension_to_download,
    svg_icons::{
        b8_send_icon_svg, fa_trash_icon_svg, ms_cloud_add_icon_svg, ms_cloud_arrow_down_icon_svg,
        ms_cloud_arrow_up_icon_svg, ms_document_text_icon_svg,
    },
    ACTIVE_SESSION_NAME, EMAIL, JWT,
};

/// Attach files input component
///
/// # Arguments
/// `extend_publish` - whether to extend the subject with the attachment data
/// `except_files` - what files to except
/// `active_subject_name` - Optional, the active subject
/// `subject_names` - Optional, the list of all available subjects
/// `files_uploaded` - the file data to upload
/// `filenames_uploaded` - the filenames associated with each file
/// `extensions_uploaded` - the file extensions associated with each file
#[component]
pub fn attach_files_input(
    extend_publish: Signal<bool>,
    except_files: Signal<String>,
    active_subject_name: Option<Signal<String>>,
    subject_names: Option<Signal<Vec<String>>>,
    mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>,
    mut filenames_uploaded: Signal<Vec<String>>,
    mut extensions_uploaded: Signal<Vec<String>>,
) -> Element {
    let enable_directory_upload = use_signal(|| false);

    let read_files = move |file_engine: Arc<dyn FileEngine>, publish: TablePublication| async move {
        let files = file_engine.files();
        for file_name in &files {
            // Determine the file type
            let file_path = std::path::Path::new(file_name);
            match file_path.extension() {
                None => tracing::error!("File {file_name} has no extension."),
                Some(ext) => match DataFormat::from_extension(ext.to_str().unwrap()) {
                    Ok(data_format) => {
                        if let Some(contents) = file_engine.read_file(file_name).await {
                            let extension = ext.to_str().unwrap();
                            let file_stem = file_path.file_stem().unwrap().to_str().unwrap();
                            // 1. Use the active subject to determine the target subject of the file
                            let (subject_name, is_blob) = if let Some(name) = &active_subject_name {
                                if name().is_empty() {
                                    // 2. If no active subject, use the file_stem and extension to determine the target subject
                                    if let Some(names) = &subject_names {
                                        let subject_name = file_stem.to_string();
                                        if names.read().contains(&subject_name)
                                            && extension == "csv"
                                        {
                                            (subject_name, false)
                                        } else {
                                            match extension_to_subject(extension) {
                                                Ok(subject_name) => {
                                                    (subject_name.to_string(), true)
                                                }
                                                Err(err) => {
                                                    tracing::error!("{err}");
                                                    (file_stem.to_string(), true)
                                                }
                                            }
                                        }
                                    // 3. If no active subject, use the extension only to determine the target interface subject
                                    } else {
                                        match extension_to_subject(extension) {
                                            Ok(subject_name) => (subject_name.to_string(), true),
                                            Err(err) => {
                                                tracing::error!("{err}");
                                                (file_stem.to_string(), true)
                                            }
                                        }
                                    }
                                } else {
                                    (name.read().to_string(), false)
                                }
                            } else {
                                match extension_to_subject(extension) {
                                    Ok(subject_name) => (subject_name.to_string(), true),
                                    Err(err) => {
                                        tracing::error!("{err}");
                                        (file_stem.to_string(), true)
                                    }
                                }
                            };

                            // Wrap the contents into a blob batch if no active subject is set
                            let (message, format) = if is_blob {
                                let batch = create_blob_batch(
                                    vec![file_stem.to_string()],
                                    vec![extension.to_string()],
                                    vec![contents],
                                    vec!["user".to_string()],
                                    vec![create_timestamp_micros()],
                                )
                                .unwrap();
                                let message = Table::get_builder()
                                    .with_name(subject_name.as_str())
                                    .with_record_batches(vec![batch])
                                    .unwrap()
                                    .build()
                                    .unwrap()
                                    .to_ipc_stream()
                                    .unwrap();
                                (message, DataFormat::Ipc)
                            } else {
                                (contents, data_format)
                            };

                            // Update the publish method
                            let publish = match publish {
                                TablePublication::Extend { .. } => TablePublication::Extend {
                                    table_name: subject_name.clone(),
                                },
                                TablePublication::Replace { .. } => TablePublication::Replace {
                                    table_name: subject_name.clone(),
                                },
                                _ => TablePublication::None,
                            };

                            // Create the message to upload
                            let data = SessionInterfaceMessage::get_builder()
                                .with_session_name(&create_session_name(
                                    EMAIL().as_str(),
                                    ACTIVE_SESSION_NAME().as_str(),
                                ))
                                .with_format(&format)
                                .with_publisher(&create_session_name(
                                    EMAIL().as_str(),
                                    ACTIVE_SESSION_NAME().as_str(),
                                ))
                                .with_update(&publish)
                                .with_stream(false)
                                .with_subject(&subject_name)
                                .with_message(message)
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap();
                            files_uploaded.push(data);
                            filenames_uploaded.push(file_stem.to_string());
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
                TablePublication::Extend {
                    table_name: "".to_string(),
                },
            )
            .await;
        }
    };

    let upload_files_replace = move |evt: FormEvent| async move {
        if let Some(file_engine) = evt.files() {
            read_files(
                file_engine,
                TablePublication::Replace {
                    table_name: "".to_string(),
                },
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
pub fn attach_textfiles_input(
    except_files: Signal<String>,
    mut content: Signal<String>,
) -> Element {
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
pub fn upload_files_list(
    mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>,
    mut filenames_uploaded: Signal<Vec<String>>,
    mut extensions_uploaded: Signal<Vec<String>>,
) -> Element {
    rsx! {
        div {
            p { "Files to upload" },
            ul {
                class: "file_list",
                {(0..filenames_uploaded.len()).map(|i| {
                    let filename = filenames_uploaded.get(i).unwrap();
                    let extension = extensions_uploaded.get(i).unwrap();
                    let download = filename_and_extension_to_download(&filename, &extension);
                    rsx! {
                        li {
                            key: "{i}",
                            div {
                                class: "files",
                                svg { dangerous_inner_html: extension_to_icon_svg(&extension) },
                                h3 { "{download}" },
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
pub fn upload_files_button(
    mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>,
    mut filenames_uploaded: Signal<Vec<String>>,
    mut extensions_uploaded: Signal<Vec<String>>,
) -> Element {
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
pub fn clear_upload_files_button(
    mut files_uploaded: Signal<Vec<SessionInterfaceMessage>>,
    mut filenames_uploaded: Signal<Vec<String>>,
    mut extensions_uploaded: Signal<Vec<String>>,
) -> Element {
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
pub fn download_files_button(
    data_format: Signal<DataFormat>,
    active_subject_name: Signal<String>,
    mut files_downloaded: Signal<Vec<Vec<u8>>>,
    mut filenames_downloaded: Signal<Vec<String>>,
    mut extensions_downloaded: Signal<Vec<String>>,
) -> Element {
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
                    .with_update(&TablePublication::None)
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
                        let bytes_vec: Vec<u8> = bytes.into_iter().flat_map(|b| b.to_vec()).collect();
                        files_downloaded.push(bytes_vec);
                        filenames_downloaded.push(active_subject_name.read().as_str().to_string());
                        extensions_downloaded.push(data_format().to_extension().to_string());
                    }
                    Err(err) => tracing::error!("There was a error downloading subject {err}."),
                }
            },
            svg { dangerous_inner_html: ms_cloud_arrow_down_icon_svg() },
        },
    }
}

#[component]
pub fn download_files_list(
    mut files_downloaded: Signal<Vec<Vec<u8>>>,
    mut filenames_downloaded: Signal<Vec<String>>,
    mut extensions_downloaded: Signal<Vec<String>>,
) -> Element {
    rsx! {
        div {
            class: "files",
            p { "Files to download" },
            ul {
                class: "file_list",
                {(0..files_downloaded().len()).map(|i| {
                    let f_download = filename_and_extension_to_download(filenames_downloaded().get(i).unwrap(), extensions_downloaded().get(i).unwrap());
                    let f_href = extension_and_file_to_data_href(extensions_downloaded().get(i).unwrap() ,files_downloaded().get(i).unwrap()).unwrap();
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
pub fn clear_download_files_button(
    mut files_downloaded: Signal<Vec<Vec<u8>>>,
    mut filenames_downloaded: Signal<Vec<String>>,
    mut extensions_downloaded: Signal<Vec<String>>,
) -> Element {
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
