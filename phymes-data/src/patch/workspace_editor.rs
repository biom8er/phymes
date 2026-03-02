use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};
use anyhow::{anyhow, Result};

use crate::patch::apply_patch::{PatchOperator, PatchOperation, apply_patch_auto};

pub struct WorkspaceEditor {
    root: PathBuf,
}

impl WorkspaceEditor {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
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
            return Err(anyhow!("Missing file {}", abs.display().to_string()));
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

    pub fn list_files(&self) -> Result<Vec<PathBuf>> {
        fn walk(dir: &Path, acc: &mut Vec<PathBuf>, root: &Path) -> io::Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, acc, root)?;
                } else {
                    let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
                    acc.push(rel);
                }
            }
            Ok(())
        }

        let mut files = Vec::new();
        walk(&self.root, &mut files, &self.root)?;
        Ok(files)
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
}

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
}