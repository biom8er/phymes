/// Extract out the tool_calls from a text message
pub fn extract_tool_calls_str<'a>(
    content: &'a str,
    start: Option<&'a str>,
    end: Option<&'a str>,
) -> &'a str {
    let start = start.unwrap_or("<tool_call>\n");
    let end = end.unwrap_or("\n</tool_call>");
    let start_bytes = content.find(start);
    let end_bytes = content.find(end);

    // check if a match was found
    if start_bytes.is_some() || end_bytes.is_some() {
        let start_bytes = start_bytes.unwrap_or(0);
        let end_bytes = end_bytes.unwrap_or(content.len());
        &content[start_bytes + start.len()..end_bytes]
    } else {
        &content[0..content.len()]
    }
}
/// Format the expected tool calls as valid Vec<serde_json::Value>
pub fn format_tool_calls_str(content: &str) -> String {
    if content.starts_with("{") && content.ends_with("}") {
        let new_content = format!("[{content}]");
        new_content
    } else {
        content.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_calls() {
        let content = r#"
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "celsius"}, "name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "celsius"}}
</tool_call><|im_end|>
"#;
        let extracted = extract_tool_calls_str(content, None, None);
        assert_eq!(
            extracted,
            r#"{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "celsius"}, "name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "celsius"}}"#
        )
    }
}
