use anyhow::{Error, Result};
use phymes_diagnostics::create_timestamp_micros;
use phymes_subject::{BuilderTrait, Subject, SubjectBuilder, SubjectBuilderTrait};

use crate::{AvailableSubjects, create_chat_record_batch};

/// Create the error table
///
/// # Arguments
/// `err` - [anyhow::Error]
/// `with_display` - whether to show the full backtrace or not
///
/// # Notes
/// - use :? and not .to_string() with Anyhow::Error to see full backtrace if available
pub fn create_error_subject(err: &Error, with_display: bool) -> Result<Subject> {
    let error_str = if with_display {
        format! {"{err:?}"}
    } else {
        format! {"{err}"}
    };
    let batch = create_chat_record_batch(
        vec!["tool".to_string()],
        vec![error_str],
        vec![create_timestamp_micros()],
    )?;
    SubjectBuilder::new()
        .with_name(AvailableSubjects::NetworkErrors.to_string().as_str())
        .with_record_batches(vec![batch])?
        .build()
}
