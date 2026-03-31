use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use diff_match_patch_rs::{DiffMatchPatch, Efficient, PatchInput};
use phymes_core::{ApplyDiffMode, apply_v4a_diff};

fn sample_original() -> String {
    (0..1000).map(|i| format!("line {i}\n")).collect::<String>()
}

fn sample_diff_dmp(original: &str, modified: &str) -> String {
    let dmp = DiffMatchPatch::new();
    let diffs = dmp.diff_main::<Efficient>(original, modified).unwrap();
    let patches = dmp.patch_make(PatchInput::new_diffs(&diffs)).unwrap();
    dmp.patch_to_text(&patches)
}

fn sample_diff_v4a(original: &str, modified: &str) -> String {
    // For now, reuse DMP text as input to apply_diff; swap in real V4A generator when available.
    sample_diff_dmp(original, modified)
}

fn bench_dmp_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_diffs_dmp_vs_v4a");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("diff-match-patch apply", |b| {
        b.iter_batched(
            || {
                let original = sample_original();
                let modified = original.replace("line 500", "LINE 500");
                let diff = sample_diff_dmp(&original, &modified);
                (original, diff)
            },
            |(original, diff)| {
                let dmp = DiffMatchPatch::new();
                let patches = dmp.patch_from_text::<Efficient>(&diff).unwrap();
                let (_new_content, _results) = dmp.patch_apply(&patches, &original).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("v4a apply_diff", |b| {
        b.iter_batched(
            || {
                let original = sample_original();
                let modified = original.replace("line 500", "LINE 500");
                let diff = sample_diff_v4a(&original, &modified);
                (original, diff)
            },
            |(original, diff)| {
                let _new_content = apply_v4a_diff(&original, &diff, ApplyDiffMode::Default).ok();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_dmp_apply);
criterion_main!(benches);
