use anyhow::{anyhow, Result};
use crate::patch::apply_diff::{apply_diff, ApplyDiffMode};

pub fn apply_v4a_patch(
    original: &str,
    diff: &str,
    create: bool,
) -> Result<String> {
    let mode = if create {
        ApplyDiffMode::Create
    } else {
        ApplyDiffMode::Default
    };

    apply_diff(original, diff, mode).map_err(|e| anyhow!("{e:?}"))
}