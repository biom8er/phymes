/// # Search and Replace using diff-fenced
/// 
/// # Example
/// ```
/// FILENAME
/// <<<<<<< SEARCH
/// Code section containing /* MIDDLE CODE TO COMPLETE */
/// =======
/// Same code section with ONLY the middle code implemented
/// >>>>>>> REPLACE
/// ```

use crate::extract_tool_calls_str;

const SECTION_OUTPUT: &str = "```";
const BEGIN_SEARCH: &str = "<<<<<<< SEARCH\n";
const END_SEARCH_BEGIN_REPLACE: &str = "=======\n";
const END_REPLACE: &str = ">>>>>>> REPLACE\n";

#[derive(Debug)]
pub struct SearchAndReplaceDiff {
    filename: String,
    diff: String,
}

impl SearchAndReplaceDiff {
    pub fn new(filename: &str, diff: &str) -> Self {
        Self { filename: filename.to_string(), diff: diff.to_string() }
    }
}

/// Parse a search and replace output diff generated from a fill-in-the-middle coding LLM
pub fn parse_search_and_replace_output(input: &str) -> SearchAndReplaceDiff {
    // Extract the filename
    let filename = extract_tool_calls_str(input, Some(SECTION_OUTPUT), Some(BEGIN_SEARCH)).trim();

    // Extract the diff
    let diff = extract_tool_calls_str(input, Some(BEGIN_SEARCH), Some(END_REPLACE));
    let diff = format!("{BEGIN_SEARCH}{diff}{END_REPLACE}");
    let diff = diff.trim();

    SearchAndReplaceDiff::new(filename, diff)
}

pub fn apply_search_and_replace_patch(input: &str, diff: &str) -> String {
    // Extract the original and new text
    let original_snippet = extract_tool_calls_str(diff, Some(BEGIN_SEARCH), Some(END_SEARCH_BEGIN_REPLACE));
    let new_snippet = extract_tool_calls_str(diff, Some(END_SEARCH_BEGIN_REPLACE), Some(END_REPLACE));

    // Patch the input
    if input.is_empty() {
        new_snippet.to_string()
    } else {
        input.replace(original_snippet, new_snippet)
    }    
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_parse_search_and_replace_output() {
        let input = r#"```
test.rs
<<<<<<< SEARCH
Old text
=======
New text
>>>>>>> REPLACE
```"#;
        let diff = parse_search_and_replace_output(input);
        assert_eq!(diff.filename, "test.rs");
        assert_eq!(diff.diff, r#"<<<<<<< SEARCH
Old text
=======
New text
>>>>>>> REPLACE"#);
    }

    #[test]
    fn test_apply_search_and_replace_patch_create() {
        let original = "";
        let diff = r#"<<<<<<< SEARCH
=======
New text
>>>>>>> REPLACE
```"#;
        let result = apply_search_and_replace_patch(original, diff);
        assert_eq!(result, "New text\n");
    }

    #[test]
    fn test_apply_search_and_replace_patch_update_success() {
        let original = "Old text\n";
        let diff = r#"<<<<<<< SEARCH
Old text
=======
New text
>>>>>>> REPLACE
```"#;
        let result = apply_search_and_replace_patch(original, diff);
        assert_eq!(result, "New text\n");
    }

    #[test]
    fn test_apply_search_and_replace_patch_update_fail() {
        let original = "Old text 1";
        let diff = r#"<<<<<<< SEARCH
Old text
=======
New text
>>>>>>> REPLACE
```"#;
        let result = apply_search_and_replace_patch(original, diff);
        assert_eq!(result, "Old text 1");
    }
}
