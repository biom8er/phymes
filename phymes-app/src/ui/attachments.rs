// Dioxus imports
use dioxus::prelude::*;

// General imports
use futures::StreamExt;
use phymes_agents::session_plans::available_interface_subjects::AvailableInterfaceSubjects;
use serde_json::{self, Map, Value};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

// Phymes imports
use phymes_core::{
    schemas::available_subjects::convert_timestamp_micros_to_str, session::{common_traits::{BuildableTrait, BuilderTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait}}, table::{data_format::DataFormat, table_publish::TablePublish, table_trait::TableTrait}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;

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

// mod imports
use crate::{
    state::{
        apps::ACTIVE_SESSION_NAME, attachments::update_attachments_state, sign_in::{EMAIL, JWT}
    },
    ui::{subjects::{attach_files_input, clear_upload_files_button, extension_and_file_to_href, extension_to_icon_svg, filename_and_extension_to_download, upload_files_button, upload_files_list}, svg_icons::{aws_assistant_icon_svg, aws_user_icon_svg, fa_trash_icon_svg, ms_arrow_download_icon_svg}},
};

/// View for attachments between the user and AI assistant
#[component]
pub fn attachments_interface_view() -> Element {
    // Global signals
    let attachments_roles = use_signal(Vec::<String>::new);
    let attachments_contents = use_signal(Vec::<Option<Vec<u8>>>::new);
    let attachments_indices = use_signal(Vec::<usize>::new);
    let attachments_timestamps = use_signal(Vec::<i64>::new);
    let attachments_filenames = use_signal(Vec::<String>::new);
    let attachments_extensions = use_signal(Vec::<String>::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || SessionInterfaceMessage::get_builder()
        .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_format(&DataFormat::Bytes)
        .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_update(&TablePublish::None)
        .with_stream(false)
    );

    // Get the last 25 attachments (without the actual blob content) for the attachments view
    let got_attachments = use_memo(move || !attachments_roles().is_empty());
    let _ = use_resource(move || async move {
        // Prevent re-fetching attachments if we already have some
        if got_attachments() {
            return;
        }

        let data = get_session_state()
            .with_subject(AvailableInterfaceSubjects::AggregatedAttachments.to_string().as_str())
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
                                "There was a error parsing SyncCurrentAttachmentsState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        let bytes: Vec<u8> = serde_json::from_value(row.get("bytes").unwrap().to_owned()).unwrap();
                        if row.get("metadata").is_some() {
                            update_attachments_state(attachments_roles, attachments_contents, attachments_indices, attachments_timestamps, attachments_filenames, attachments_extensions,
                                row.get("metadata").unwrap().as_str().unwrap(),
                                Some(bytes),
                                // None,
                                row.get("timestamp").unwrap().as_i64().unwrap(),
                                row.get("filename").unwrap().as_str().unwrap(),
                                row.get("extension").unwrap().as_str().unwrap(),
                            );
                        }
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
                        if row.get("metadata").is_some() {
                            update_attachments_state(attachments_roles, attachments_contents, attachments_indices, attachments_timestamps, attachments_filenames, attachments_extensions,
                                row.get("metadata").unwrap().as_str().unwrap(),
                                None,
                                row.get("timestamp").unwrap().as_i64().unwrap(),
                                row.get("filename").unwrap().as_str().unwrap(),
                                row.get("extension").unwrap().as_str().unwrap(),
                            );
                        }
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
                p { "Please sign-in before attachments." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please activate a session before attachments." },
            }
        } else {
            ul {
                class: "messaging_list",
                {(0..attachments_roles.len()).map(|i| {
                    let role = attachments_roles.get(i).unwrap();
                    let index = attachments_indices.get(i).unwrap();
                    let timestamp = convert_timestamp_micros_to_str(*attachments_timestamps.get(i).unwrap());
                    let content = attachments_contents.get(i).unwrap();
                    let filename = attachments_filenames.get(i).unwrap();
                    let extension = attachments_extensions.get(i).unwrap();
                    rsx! {
                        li {
                            key: "{index}",
                            class: "assistant", // we borrow the assistant class for styling
                            div {
                                class: "entete",
                                if role.as_str() == "assistant" {
                                    svg { dangerous_inner_html: aws_assistant_icon_svg() }
                                    h2 { "AI Assistant" }
                                } else {
                                    svg { dangerous_inner_html: aws_user_icon_svg() }
                                    h2 { "User" }
                                }
                                h3 { "{timestamp}" }
                                svg { dangerous_inner_html: extension_to_icon_svg(&extension) }
                                if let Some(f) = content.as_ref() {
                                    a {
                                        href: extension_and_file_to_href(&extension, f).unwrap(),
                                        download: filename_and_extension_to_download(&filename, &extension),
                                        "{filename_and_extension_to_download(&filename, &extension)}"
                                    },
                                    button {
                                        svg { dangerous_inner_html: fa_trash_icon_svg() }
                                        // TODO: delete the attachment
                                    }
                                } else {
                                    h3 { "{filename}.{extension}" },
                                    button {
                                        svg { dangerous_inner_html: ms_arrow_download_icon_svg() }
                                        // TODO: download the attachment
                                    }
                                }                                
                            }
                        }
                    }
                })}
            }
            attachments_interface_footer { attachments_roles, attachments_contents, attachments_indices, attachments_timestamps, attachments_filenames, attachments_extensions }
        }
    }
}

#[component]
pub fn attachments_interface_footer(mut attachments_roles: Signal<Vec<String>>, mut attachments_contents: Signal<Vec<Option<Vec<u8>>>>, mut attachments_indices: Signal<Vec<usize>>, mut attachments_timestamps: Signal<Vec<i64>>, mut attachments_filenames: Signal<Vec<String>>, mut attachments_extensions: Signal<Vec<String>>) -> Element {
    let files_uploaded = use_signal(Vec::<SessionInterfaceMessage>::new);
    let filenames_uploaded = use_signal(Vec::<String>::new);
    let extensions_uploaded = use_signal(Vec::<String>::new);
    
    // let _ = use_resource(move || async move {

    // });

    rsx! {
        footer {
            div {
                class: "attach_button",
                attach_files_input { extend_publish: use_signal(|| true), except_files: use_signal(||".csv,.pdf,.json".to_string()), active_subject_name: None, files_uploaded, filenames_uploaded, extensions_uploaded }
            }

            div {
                class: "file_upload_form",
                if !files_uploaded.read().is_empty() {
                    div {
                        upload_files_list {files_uploaded, filenames_uploaded, extensions_uploaded}
                    }
                    div {
                        div {
                            upload_files_button {files_uploaded, filenames_uploaded, extensions_uploaded}
                            clear_upload_files_button {files_uploaded, filenames_uploaded, extensions_uploaded}
                        }
                    }
                }
            }
        }
    }
}