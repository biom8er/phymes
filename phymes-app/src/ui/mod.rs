mod main_window;

mod backend;

mod messaging;
mod metrics;
mod apps;
mod builds;
mod sign_in;
mod subjects;
mod attachments;
mod files;

pub use main_window::{main_window_view, split_panel_drag_handle};

pub use messaging::messaging_interface_view;
pub use metrics::metrics_interface_view;
pub use apps::{apps_interface_view, mermaid_view};
pub use builds::{builds_dropdown_view, builds_interface_footer};
pub use sign_in::sign_in_view;
pub use subjects::subjects_interface_view;
pub use attachments::{attachments_interface_footer, attachments_interface_view};
pub use files::{attach_files_input, clear_upload_files_button, upload_files_button, clear_download_files_button, download_files_button, download_files_list, attach_textfiles_input};