use crate::state::svg_icons::{
    aws_table_icon_svg, b8_microphone_icon_svg, ms_attachment_icon_svg, ms_code_icon_svg,
    ms_document_icon_svg, ms_search_icon_svg, ms_video_icon_svg,
};
use anyhow::{anyhow, Result};
use phymes_agents::AvailableInterfaceSubjects;

pub fn extension_to_icon_svg(extension: &str) -> String {
    match extension.to_lowercase().as_str() {
        "pdf" => ms_document_icon_svg(),
        "mp3" | "wav" | "aac" => b8_microphone_icon_svg(),
        "mp4" | "avi" => ms_video_icon_svg(),
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => ms_search_icon_svg(),
        "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go" | "rs" | "json" | "svg"
        | "html" => ms_code_icon_svg(),
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
        "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go" | "rs" | "json" | "svg"
        | "html" => AvailableInterfaceSubjects::UserScript,
        "csv" | "tsv" => AvailableInterfaceSubjects::UserCsv,
        _ => {
            return Err(anyhow!(
                "Conversion to subject is not supported for extension {extension}"
            ))
        }
    };
    Ok(subject)
}

/// Based on <https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types>
fn extension_to_mime_type(extension: &str) -> Result<&str> {
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
        _ => {
            return Err(anyhow!(
                "Conversion to MIME type is not supported for extension {extension}"
            ))
        }
    };
    Ok(mime_type)
}

/// Based on <https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Schemes/data>
///
/// Follows the syntax `data:[<media-type>][;base64],<data>`
///
/// # Notes
///
/// While Data and Blob URLs both allow representing in-memory resources as URLs;
///   the difference is that data URLs embed resources in themselves and have severe size limitations,
///   whereas blob URLs require a backing Blob or MediaSource and can represent larger resources.
///
/// See <https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Schemes/blob>
pub fn extension_and_file_to_data_href(extension: &str, bytes: &[u8]) -> Result<String> {
    let mime_type = extension_to_mime_type(extension)?;
    let href = match extension {
        "txt" | "csv" | "tsv" | "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go"
        | "rs" | "json" | "svg" | "html" => {
            let data = String::from_utf8_lossy(bytes).into_owned();
            format! {"data:{mime_type},{data}"}
        }
        _ => {
            let data = String::from_utf8_lossy(bytes).into_owned();
            format! {"data:{mime_type},{data}"}
        }
    };
    Ok(href)
}

/// Based on <https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Schemes/blob>
///
/// Follows the syntax `blob:<origin>/<uuid>>`
#[allow(dead_code)]
pub fn extension_and_file_to_blob_href(extension: &str, bytes: &[u8]) -> Result<String> {
    let mime_type = extension_to_mime_type(extension)?;
    let href = match extension {
        "txt" | "csv" | "tsv" | "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go"
        | "rs" | "json" | "svg" | "html" => {
            let data = String::from_utf8_lossy(bytes).into_owned();
            format! {"data:{mime_type},{data}"}
        }
        _ => {
            let data = String::from_utf8_lossy(bytes).into_owned();
            format! {"data:{mime_type},{data}"}
        }
    };
    Ok(href)
}

pub fn filename_and_extension_to_download(filename: &str, extension: &str) -> String {
    format!("{filename}.{extension}")
}
