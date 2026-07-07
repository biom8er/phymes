use std::str::FromStr;

use anyhow::Result;
use futures::TryStreamExt;
use phymes_diagnostics::create_timestamp_micros;
use phymes_event::Subscription;
use phymes_schemas::AvailableSubjects;
use phymes_subject::{
    BuildableTrait, BuilderTrait, RuntimeEnv, Subject, SubjectBuilderTrait, SubjectTrait,
};
use phymes_task::SubscriptionTrait;

/// Default diagnostic subjects for Errors, Events, Traces, and Metrics
pub fn default_diagnostic_subjects() -> Vec<String> {
    vec![
        AvailableSubjects::NetworkErrors.to_string(),
        AvailableSubjects::NetworkEvents.to_string(),
        AvailableSubjects::NetworkTraces.to_string(),
        AvailableSubjects::NetworkMetrics.to_string(),
    ]
}

/// Extended diagnostic subjects including task and subject change logs
pub fn extended_diagnostic_subjects() -> Vec<String> {
    let mut subjects = default_diagnostic_subjects();
    let extension = vec![
        AvailableSubjects::NetworkTasksRunLog.to_string(),
        AvailableSubjects::SubjectsChangeLog.to_string(),
    ];
    subjects.extend(extension);
    subjects
}

/// Writes diagnostic (and any other network) subjects to `HOME` as CSV
pub async fn write_diagnostic_subjects_to_csv(
    subject_names: &[&str],
    runtime_env: &std::sync::Arc<RuntimeEnv>,
    network_name: &str,
) -> Result<()> {
    // Create directory and subdirectories
    let tmp_dir_str = "../target";
    // let tmp_dir_str = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let timestamp = create_timestamp_micros();
    let diagnostic_dir_str = format!("{tmp_dir_str}/network={network_name}/timestamp={timestamp}");
    let diagnostic_path = std::path::PathBuf::from_str(&diagnostic_dir_str)?;
    let _ = std::fs::create_dir_all(&diagnostic_path);

    for subject_name in subject_names {
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(runtime_env, network_name)?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            // Create the diagnostics subject file
            let diagnostics_file_path = format!(
                "{}/{subject_name}.csv",
                diagnostic_path.as_path().to_str().unwrap()
            );
            let mut diagnostics_file = std::fs::File::create(&diagnostics_file_path)?;

            // Write the subject file to disk
            let subject = Subject::get_builder()
                .with_name(subject_name)
                .with_record_batches(batches)?
                .build()?;
            subject.to_csv_file(&mut diagnostics_file, b',', true)?;
        }
    }

    Ok(())
}
