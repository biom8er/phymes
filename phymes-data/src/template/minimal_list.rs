/// HTML5 table jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/ul>
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/ol>
pub static MINIMAL_LIST_TEMPLATE: &str = r#"
{%- if ordered %}
<ol{%- if ul_class %} class="{{ ul_class }}"{%- endif %}{%- if ul_style %} style="{{ ul_style }}"{%- endif %}>
{%- else %}
<ul{%- if ul_class %} class="{{ ul_class }}"{%- endif %}{%- if ul_style %} style="{{ ul_style }}"{%- endif %}>
{%- endif %}
{%- for row in rows %}
    <li{%- if li_class %} class="{{ li_class }}"{%- endif %}{%- if li_style %} style="{{ li_style }}"{%- endif %}>{{ row.item }}</li>
{%- endfor %}
{%- if ordered %}
</ol>
{%- else %}
</ul>
{%- endif %}"#;

/// HTML table input jinja2 template
pub static MINIMAL_LIST_INPUT: &str = r#"{
"ordered": "{{ ordered }}",
"ul_class": "{{ ul_class }}",
"ul_style": "{{ ul_style }}",
"li_class": "{{ li_class }}",
"li_style": "{{ li_style }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        SubjectScript, TEMPLATE_TABLE_EXPRESSION,
        template::{MINIMAL_HTML_POST, MINIMAL_HTML_PRE},
    };
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use serde_json::{Map, Value};

    use super::*;

    #[test]
    fn test_minimal_list_html() -> Result<()> {
        // Create the dummy data for the table
        let item_vec = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let item_arr: ArrayRef = Arc::new(StringArray::from(item_vec));
        let batch = RecordBatch::try_from_iter(vec![("item", item_arr)])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the input for the template
        let inputs = serde_json::json!({
            "ordered": "",
            "ul_class": "p-2 flex flex-col list-disc",
            "ul_style": "",
            "li_class": "flex flex-col flex-content-start gap-1 my-2",
            "li_style": ""
        });
        let input_string = SubjectScript::new_from_template(MINIMAL_LIST_INPUT.to_string())
            .apply_template(&inputs)?
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        // Update the input with the dummy chart data
        let mut input_object = serde_json::from_str::<Map<String, Value>>(&input_string)?;
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [MINIMAL_HTML_PRE, MINIMAL_LIST_TEMPLATE, MINIMAL_HTML_POST].join("");
        let script_string =
            SubjectScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n        <script src=\"https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4\"></script>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n    </head>\n    <body>\n<ul class=\"p-2 flex flex-col list-disc\">\n    <li class=\"flex flex-col flex-content-start gap-1 my-2\">Item 1</li>\n    <li class=\"flex flex-col flex-content-start gap-1 my-2\">Item 2</li>\n    <li class=\"flex flex-col flex-content-start gap-1 my-2\">Item 3</li>\n    <li class=\"flex flex-col flex-content-start gap-1 my-2\">Item 4</li>\n    <li class=\"flex flex-col flex-content-start gap-1 my-2\">Item 5</li>\n    <li class=\"flex flex-col flex-content-start gap-1 my-2\">Item 6</li>\n</ul>\n    </body>\n</html>"
        );
        Ok(())
    }
}
