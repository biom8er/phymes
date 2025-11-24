mod main_window;

mod backend;

mod apps;
mod attachments;
mod builds;
mod files;
mod messaging;
mod metrics;
mod sign_in;
mod subjects;

pub use main_window::{main_window_view, split_panel};

pub use apps::{apps_interface_view, mermaid_view};
pub use attachments::{attachments_interface_footer, attachments_interface_view};
pub use builds::{builds_dropdown_view, builds_interface_footer};
pub use files::{
    attach_files_input, attach_textfiles_input, clear_download_files_button,
    clear_upload_files_button, download_files_button, download_files_list, upload_files_button,
};
pub use messaging::messaging_interface_view;
pub use metrics::metrics_interface_view;
pub use sign_in::sign_in_view;
pub use subjects::subjects_interface_view;
