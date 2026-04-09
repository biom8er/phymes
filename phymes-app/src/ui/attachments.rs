// Dioxus imports
use dioxus::prelude::*;

use phymes_network::AvailableInterfaceSubjects;
use phymes_diagnostics::convert_timestamp_micros_to_str;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

use phymes_network::{
    SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait,
};
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, MessageBuilderTrait, Publication, SubjectBuilder,
    SubjectBuilderTrait, SubjectTrait,
};
use phymes_server::create_session_name;

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(not(feature = "serverless"))]
use futures::StreamExt;

#[cfg(feature = "serverless")]
use crate::state::RUNTIME_ENV;
#[cfg(feature = "serverless")]
use bytes::Bytes;
#[cfg(feature = "serverless")]
use futures::TryStreamExt;
#[cfg(feature = "serverless")]
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

// mod imports
use crate::{
    state::{
        extension_and_file_to_data_href, extension_to_icon_svg, filename_and_extension_to_download,
        svg_icons::{
            aws_assistant_icon_svg, aws_user_icon_svg, fa_trash_icon_svg,
            ms_arrow_download_icon_svg,
        },
        ACTIVE_SESSION_NAME, EMAIL, JWT,
    },
    ui::{
        attach_files_input, clear_upload_files_button, main_window::split_panel,
        upload_files_button,
    },
};

/// View for attachments between the user and AI assistant
#[component]
pub fn attachments_interface_view() -> Element {
    // Global signals
    let mut attachments_roles = use_signal(Vec::<String>::new);
    let mut attachments_contents = use_signal(Vec::<Option<Vec<u8>>>::new);
    let mut attachments_indices = use_signal(Vec::<usize>::new);
    let mut attachments_timestamps = use_signal(Vec::<i64>::new);
    let mut attachments_filenames = use_signal(Vec::<String>::new);
    let mut attachments_extensions = use_signal(Vec::<String>::new);

    // Update the index in a different scope
    let current_index: Memo<usize> = use_memo(move || {
        if attachments_indices.len() == 0 {
            0
        } else {
            *attachments_indices.last().unwrap()
        }
    });

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

    // Get the last 25 attachments (without the actual blob content) for the attachments view
    let got_attachments = use_memo(move || !attachments_roles().is_empty());
    use_resource(move || async move {
        // Prevent re-fetching attachments if we already have some
        if got_attachments() {
            return;
        }

        let data = get_session_state()
            .with_subject(
                AvailableInterfaceSubjects::AggregatedAttachments
                    .to_string()
                    .as_str(),
            )
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
                let mut bytes = Vec::new();
                while let Some(Ok(b)) = stream.next().await {
                    bytes.extend(b);
                }
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        let combined = table
                            .get_column_as_vec_nonprimitive::<String>("metadata")
                            .unwrap()
                            .into_iter()
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("filename")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("extension")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_primitive::<i64>("timestamp")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_nested_primitive::<u8>("bytes")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .enumerate()
                            .filter_map(|(i, ((((m, f), e), t), b))| {
                                if m.is_empty() {
                                    None
                                } else {
                                    let index = current_index() + i + 1;
                                    Some((m, f, e, t, b, index))
                                }
                            })
                            .collect::<Vec<_>>();
                        for (m, f, e, t, b, index) in combined {
                            attachments_roles.push(m);
                            attachments_filenames.push(f);
                            attachments_extensions.push(e);
                            attachments_timestamps.push(t);
                            attachments_contents.push(Some(b));
                            attachments_indices.push(index);
                        }
                    }
                    Err(err) => tracing::error!("{err:?}"),
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
            object_store_backend: None,
            object_store_bucket: None,
            object_store_config: None,
        };
        #[cfg(feature = "serverless")]
        let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
        #[cfg(feature = "serverless")]
        match serverless_app(config, &mut serverless).await {
            Ok(response) => {
                let bytes: Vec<Bytes> = response
                    .into_body()
                    .into_data_stream()
                    .try_collect()
                    .await
                    .unwrap();
                let bytes = bytes.into_iter().flatten().collect::<Vec<_>>();
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        use phymes_core::SubjectTrait;

                        let table = builder.with_name("").build().unwrap();
                        let combined = table
                            .get_column_as_vec_nonprimitive::<String>("metadata")
                            .unwrap()
                            .into_iter()
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("filename")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("extension")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_primitive::<i64>("timestamp")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_nested_primitive::<u8>("bytes")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .enumerate()
                            .filter_map(|(i, ((((m, f), e), t), b))| {
                                if m.is_empty() {
                                    None
                                } else {
                                    let index = current_index() + i + 1;
                                    Some((m, f, e, t, b, index))
                                }
                            })
                            .collect::<Vec<_>>();
                        for (m, f, e, t, b, index) in combined {
                            attachments_roles.push(m);
                            attachments_filenames.push(f);
                            attachments_extensions.push(e);
                            attachments_timestamps.push(t);
                            attachments_contents.push(Some(b));
                            attachments_indices.push(index);
                        }
                    }
                    Err(err) => tracing::error!("{err:?}"),
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

    rsx! {
        if JWT.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please sign-in before attachments." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please activate a session before attachments." },
            }
        } else {
            split_panel {
                top: rsx! {
                    div {
                        class: "h-full w-full overflow-auto",
                        ul {
                            class: "p-2 flex flex-col list-none",
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
                                        class: "flex flex-col flex-content-start gap-1 my-2", // we borrow the assistant class for styling
                                        div {
                                            class: "flex items-center gap-2",
                                            if role.as_str() == "assistant" {
                                                svg {
                                                    class: "max-w-[48px] max-h-[48px]",
                                                    dangerous_inner_html: aws_assistant_icon_svg()
                                                }
                                                h2 {
                                                    class: "font-bold",
                                                    "AI Assistant"
                                                }
                                            } else {
                                                svg {
                                                    class: "max-w-[48px] max-h-[48px]",
                                                    dangerous_inner_html: aws_user_icon_svg()
                                                }
                                                h2 {
                                                    class: "font-bold",
                                                    "User"
                                                }
                                            }
                                            h3 { "{timestamp}" }
                                            svg {
                                                class: "max-w-[48px] max-h-[48px]",
                                                dangerous_inner_html: extension_to_icon_svg(&extension)
                                            }
                                            if let Some(f) = content.as_ref() {
                                                a {
                                                    href: extension_and_file_to_data_href(&extension, f).unwrap(),
                                                    download: filename_and_extension_to_download(&filename, &extension),
                                                    "{filename_and_extension_to_download(&filename, &extension)}"
                                                },
                                                button {
                                                    class: "p-2 rounded hover:bg-neutral-700 bg-neutral-800 cursor-pointer",
                                                    onclick: move |_| async move {
                                                        *attachments_contents.get_mut(i).unwrap() = None;
                                                    },
                                                    svg {
                                                        class: "max-w-[48px] max-h-[48px]",
                                                        dangerous_inner_html: fa_trash_icon_svg()
                                                    }
                                                }
                                            } else {
                                                h3 { "{filename}.{extension}" },
                                                button {
                                                    class: "p-2 rounded hover:bg-neutral-700 bg-neutral-800 cursor-pointer",
                                                    svg {
                                                        class: "max-w-[48px] max-h-[48px]",
                                                        dangerous_inner_html: ms_arrow_download_icon_svg()
                                                    }
                                                    // TODO: download the attachment
                                                }
                                            }
                                        }
                                    }
                                }
                            })}
                        }
                    }
                },
                bottom: rsx! {
                    attachments_interface_footer { extend_input: use_signal(|| true), add_input: use_signal(|| false), except_files: use_signal(||".csv,.pdf,.json".to_string()), active_subject_name: None, subject_names: None }
                }
            }
        }
    }
}

/// Attach files interface component
///
/// # Arguments
/// `extend_input` - whether to extend the subject with the attachment data
/// `add_input` - whether to replace the subject with the attachment data
/// `except_files` - what files to except
/// `active_subject_name` - Optional, the active subject
/// `subject_names` - Optional, the list of all available subjects
#[component]
pub fn attachments_interface_footer(
    extend_input: Signal<bool>,
    add_input: Signal<bool>,
    except_files: Signal<String>,
    active_subject_name: Option<Signal<String>>,
    subject_names: Option<Signal<Vec<String>>>,
) -> Element {
    let files_uploaded = use_signal(Vec::<SessionInterfaceMessage>::new);
    let filenames_uploaded = use_signal(Vec::<String>::new);
    let extensions_uploaded = use_signal(Vec::<String>::new);

    let filenames = use_memo(move || {
        let mut filenames_vec = Vec::new();
        for i in 0..files_uploaded.len() {
            let download = filename_and_extension_to_download(
                &filenames_uploaded.get(i).unwrap(),
                &extensions_uploaded.get(i).unwrap(),
            );
            filenames_vec.push(download);
        }
        filenames_vec.join(", ")
    });

    let styles = if extend_input() && add_input() {
        "row-span-2 col-span-1 row-start-1 col-start-1 flex flex-col"
    } else {
        "row-span-1 col-span-1 row-start-1 col-start-1 flex flex-col"
    };

    rsx! {
        footer {
            class: "h-full grid grid-rows-[auto_auto_1fr] grid-cols-[auto_1fr_auto] items-center p-2",
            div {
                class: styles,
                if extend_input() {
                    div {
                        class: "p-2 rounded hover:bg-neutral-700 bg-neutral-800 cursor-pointer",
                        attach_files_input { extend_publish: use_signal(|| true), except_files, active_subject_name, subject_names, files_uploaded, filenames_uploaded, extensions_uploaded }
                    }
                }
                if add_input() {
                    div {
                        class: "p-2 rounded hover:bg-neutral-700 bg-neutral-800 cursor-pointer",
                        attach_files_input { extend_publish: use_signal(|| false), except_files, active_subject_name, subject_names, files_uploaded, filenames_uploaded, extensions_uploaded }
                    }
                }
            }

            div {
                class: "w-full h-full flex row-span-3 col-span-1 row-start-1 col-start-2",
                form {
                    class: "w-full h-full",
                    textarea {
                        placeholder: "Staged files",
                        value: "{filenames}",
                        class: "w-full h-full grow p-2 gap-2 rounded bg-neutral-800 text-gray-200 resize-none overflow-auto focus:outline-none",
                    }
                }
            }

            div {
                class: "row-span-2 col-span-1 row-start-1 col-start-3 flex flex-col",
                if !files_uploaded.read().is_empty() {
                    upload_files_button {files_uploaded, filenames_uploaded, extensions_uploaded}
                    clear_upload_files_button {files_uploaded, filenames_uploaded, extensions_uploaded}
                }
            }
        }
    }
}
