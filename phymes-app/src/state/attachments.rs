// Dioxus imports
use dioxus::prelude::*;

/// Update the attachments content state by adding to the option at the specified index
#[allow(dead_code)]
pub fn update_attachments_content_state(mut attachments_contents: Signal<Vec<Option<Vec<u8>>>>, attachments_content_update: &[u8], index: usize) {
    if let Some(mut existing_content) = attachments_contents.get_mut(index) {
        if let Some(content) = existing_content.as_mut() {
            content.extend(attachments_content_update);
        } else {
            *existing_content = Some(attachments_content_update.to_owned());
        }
    }
}

/// Clear the attachments state
#[allow(dead_code)]
pub fn clear_attachments_state(mut attachments_roles: Signal<Vec<String>>, mut attachments_contents: Signal<Vec<Option<Vec<u8>>>>, mut attachments_indices: Signal<Vec<usize>>, mut attachments_timestamps: Signal<Vec<i64>>, mut attachments_filenames: Signal<Vec<String>>, mut attachments_extensions: Signal<Vec<String>>) {
    attachments_roles.set(Vec::new());
    attachments_contents.set(Vec::new());
    attachments_indices.set(Vec::new());
    attachments_timestamps.set(Vec::new());
    attachments_filenames.set(Vec::new());
    attachments_extensions.set(Vec::new());
}

/// Update the attachments state by appending to it
pub fn update_attachments_state(mut attachments_roles: Signal<Vec<String>>, mut attachments_contents: Signal<Vec<Option<Vec<u8>>>>, mut attachments_indices: Signal<Vec<usize>>, mut attachments_timestamps: Signal<Vec<i64>>, mut attachments_filenames: Signal<Vec<String>>, mut attachments_extensions: Signal<Vec<String>>,
    attachments_role_update: &str, attachments_content_update: Option<Vec<u8>>, attachments_timestamp_update: i64, attachments_filename_update: &str, attachments_extension_update: &str) {
    attachments_roles.push(attachments_role_update.to_owned());
    attachments_contents.push(attachments_content_update);
    attachments_timestamps.push(attachments_timestamp_update);
    attachments_filenames.push(attachments_filename_update.to_owned());
    attachments_extensions.push(attachments_extension_update.to_owned());

    // Update the index in a different scope
    let index = use_memo(move || {
        if attachments_indices.len() == 0 {
            0
        } else {
            let mut index: usize = *attachments_indices.last().unwrap();
            index += 1;
            index
        }
    });
    attachments_indices.push(index());
}