// Dioxus imports
use dioxus::prelude::*;

/// Update the message content state by either replacing the last message or appending to it.
pub fn update_message_content_state(mut messaging_contents: Signal<Vec<String>>, messaging_content_update: &str, replace_last: bool) {
    let mut tmp: String = messaging_contents.pop().unwrap().to_string();
    if replace_last {
        messaging_contents.push(messaging_content_update.to_owned());
    } else {
        tmp.push_str(messaging_content_update);
        messaging_contents.push(tmp);
    }
}

/// Clear the message state
pub fn clear_message_state(mut messaging_roles: Signal<Vec<String>>, mut messaging_contents: Signal<Vec<String>>, mut messaging_indices: Signal<Vec<usize>>, mut messaging_timestamps: Signal<Vec<i64>>) {
    messaging_roles.set(Vec::new());
    messaging_contents.set(Vec::new());
    messaging_indices.set(Vec::new());
    messaging_timestamps.set(Vec::new());
}

/// Update the message state by appending to it
pub fn update_message_state(mut messaging_roles: Signal<Vec<String>>, mut messaging_contents: Signal<Vec<String>>, mut messaging_indices: Signal<Vec<usize>>, mut messaging_timestamps: Signal<Vec<i64>>,
    messaging_role_update: &str, messaging_content_update: &str, messaging_timestamp_update: i64) {
    messaging_roles.push(messaging_role_update.to_owned());
    messaging_contents.push(messaging_content_update.to_owned());
    messaging_timestamps.push(messaging_timestamp_update);

    // Update the index in a different scope
    let index = use_memo(move || {
        if messaging_indices.len() == 0 {
            0
        } else {
            let mut index: usize = *messaging_indices.last().unwrap();
            index += 1;
            index
        }
    });
    messaging_indices.push(index());
}