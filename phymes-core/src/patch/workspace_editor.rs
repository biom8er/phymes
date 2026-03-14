use anyhow::{Result, anyhow};
use crate::WorkspaceSubject;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use crate::patch::apply_patch::{PatchOperation, PatchOperator, apply_patch_auto};

pub struct WorkspaceEditor {
    root: PathBuf,
}

impl WorkspaceEditor {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Recursively walk the workspace starting at the root accumulating the paths and contents of files
    fn walk_workspace(
        &self,
        dir: &Path,
        paths: &mut Vec<PathBuf>,
        contents: &mut Vec<String>,
    ) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.walk_workspace(&path, paths, contents)?;
            } else {
                let rel = path.strip_prefix(&self.root).unwrap_or(&path).to_path_buf();
                contents.push(self.read_file(&rel)?);
                paths.push(rel);
            }
        }
        Ok(())
    }

    fn abs_path(&self, rel: &Path) -> PathBuf {
        self.root.join(rel)
    }

    pub fn create_file(&self, path: &Path, content: &str) -> Result<()> {
        let abs = self.abs_path(path);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(&abs)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn read_file(&self, path: &Path) -> Result<String> {
        let abs = self.abs_path(path);
        if !abs.exists() {
            return Err(anyhow!("Missing file {}", abs.display()));
        }
        let mut file = fs::File::open(&abs)?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;
        Ok(buf)
    }

    pub fn delete_file(&self, path: &Path) -> Result<()> {
        let abs = self.abs_path(path);
        if abs.exists() {
            fs::remove_file(abs)?;
        }
        Ok(())
    }

    pub fn apply_operation(&self, op: &PatchOperation) -> Result<()> {
        match op.operator {
            PatchOperator::Create => self.apply_create(&op.path, &op.diff),
            PatchOperator::Update => self.apply_update(&op.path, &op.diff),
            PatchOperator::Delete => self.apply_delete(&op.path),
        }
    }

    fn apply_create(&self, path: &Path, diff: &str) -> Result<()> {
        let new_content = apply_patch_auto("", diff, true)?;

        self.create_file(path, &new_content)
    }

    fn apply_update(&self, path: &Path, diff: &str) -> Result<()> {
        let original = self.read_file(path)?;

        let new_content = apply_patch_auto(&original, diff, false)?;

        let abs = self.abs_path(path);
        let mut file = fs::File::create(abs)?;
        file.write_all(new_content.as_bytes())?;
        Ok(())
    }

    fn apply_delete(&self, path: &Path) -> Result<()> {
        self.delete_file(path)
    }

    /// Convert the workspace into a list of [WorkspaceSubject]s
    pub fn to_workspace_subject(&self) -> Result<Vec<WorkspaceSubject>> {
        let mut paths = Vec::<PathBuf>::new();
        let mut contents = Vec::<String>::new();
        self.walk_workspace(&self.root, &mut paths, &mut contents)?;
        let workspace_subjects = paths
            .into_iter()
            .zip(contents)
            .map(|(p, c)| WorkspaceSubject {
                path: p.to_str().unwrap().to_string(),
                content: c,
            })
            .collect::<Vec<_>>();
        Ok(workspace_subjects)
    }

    /// Build a new [WorkspaceEditor] after populating the workspace with files
    pub fn from_workspace_subject(
        root: impl Into<PathBuf>,
        workspace: &[WorkspaceSubject],
    ) -> Result<Self> {
        let workspace_editor = WorkspaceEditor::new(root);
        for (path, content) in workspace.iter().map(|w| (&w.path, &w.content)) {
            workspace_editor.create_file(std::path::Path::new(path), content)?;
        }
        Ok(workspace_editor)
    }
}

#[cfg(test)]
pub mod tests {
    use diff_match_patch_rs::{DiffMatchPatch, Efficient, PatchInput};

    use super::*;

    fn temp_root() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn read_file(root: &tempfile::TempDir, rel: &str) -> String {
        std::fs::read_to_string(root.path().join(rel)).unwrap()
    }

    /// End-to-end: V4A diff through WorkspaceEditor.
    #[test]
    fn test_workspace_editor_applies_v4a_diff_end_to_end() {
        let root = temp_root();
        let editor = WorkspaceEditor::new(root.path());

        // Create file via V4A create diff.
        let create_diff = "+hello\n+world\n*** End Patch\n";
        let op = PatchOperation {
            path: std::path::PathBuf::from("foo.txt"),
            diff: create_diff.to_string(),
            operator: PatchOperator::Create,
        };
        editor.apply_operation(&op).unwrap();

        assert_eq!(read_file(&root, "foo.txt"), "hello\nworld");

        // Update via V4A update diff.
        let update_diff = "@@\n hello\n-world\n+WORLD\n*** End Patch\n";
        let op2 = PatchOperation {
            path: std::path::PathBuf::from("foo.txt"),
            diff: update_diff.to_string(),
            operator: PatchOperator::Update,
        };
        editor.apply_operation(&op2).unwrap();

        assert_eq!(read_file(&root, "foo.txt"), "hello\nWORLD");
    }

    /// End-to-end: DMP diff through WorkspaceEditor.
    #[test]
    fn test_workspace_editor_applies_dmp_diff_end_to_end() {
        let root = temp_root();
        let editor = WorkspaceEditor::new(root.path());

        // Seed file directly.
        editor
            .create_file(std::path::Path::new("bar.txt"), "a\nb\nc\n")
            .unwrap();

        let original = "a\nb\nc\n";
        let modified = "a\nB\nc\n";

        let dmp = DiffMatchPatch::new();
        let diffs = dmp.diff_main::<Efficient>(original, modified).unwrap();
        let patches = dmp.patch_make(PatchInput::new_diffs(&diffs)).unwrap();
        let diff_text = dmp.patch_to_text(&patches);

        let op = PatchOperation {
            path: PathBuf::from("bar.txt"),
            diff: diff_text,
            operator: PatchOperator::Update,
        };
        editor.apply_operation(&op).unwrap();

        assert_eq!(read_file(&root, "bar.txt"), "a\nB\nc\n");
    }

    #[test]
    fn test_workspace_editor_to_from_workspace_subjects() -> Result<()> {
        // Mock workspace
        let path = ["install.sh", "src/main.sh"];
        let content = [
            r#"#!/usr/bin/env bash

[[ -n "$1" ]] && echo "$1"
[[ -n "$2" ]] && echo "$2"
[[ -n "$3" ]] && echo "$3""#,
            r#"#!/bin/sh
apk add --no-cache bash
chmod +x ./src/main.sh"#,
        ];
        let workspace_subjects = path
            .into_iter()
            .zip(content)
            .map(|(p, c)| WorkspaceSubject {
                path: p.to_string(),
                content: c.to_string(),
            })
            .collect::<Vec<_>>();

        // Build the workspace
        let root = temp_root();
        let workspace_editor =
            WorkspaceEditor::from_workspace_subject(root.path(), &workspace_subjects)?;

        // Test the conversion
        let test = workspace_editor.to_workspace_subject()?;
        assert_eq!(test, workspace_subjects);

        Ok(())
    }
}
