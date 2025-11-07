/// HTML5 table jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/ul>
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/ol>
pub static MINIMAL_LIST_TEMPLATE: &str = r#"
{%- if ordered %}
<ol>
{%- else %}
<ul>
{%- endif %}
{%- for row in rows %}
    <li>{{ row.item }}</li>
{%- endfor %}
{%- if ordered %}
</ol>
{%- else %}
</ul>
{%- endif %}"#;

/// HTML table input jinja2 template
pub static MINIMAL_LIST_INPUT: &str = r#"{
"ordered": "{{ ordered }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::jinja2_templates::{TEMPLATE_TABLE_EXPRESSION, minimal_html::{MINIMAL_HTML_POST, MINIMAL_HTML_PRE}};
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_core::{
        BuildableTrait, BuilderTrait, MappableTrait, Table, TableBuilderTrait, TableScript,
        TableTrait,
    };
    use serde_json::{Map, Value};

    use super::*;

    #[test]
    fn test_minimal_list_html() -> Result<()> {
        // Create the dummy data for the table
        let item_vec = [
            "Item 1",
            "Item 2",
            "Item 3",
            "Item 4",
            "Item 5",
            "Item 6",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        let item_arr: ArrayRef = Arc::new(StringArray::from(item_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("item", item_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the input for the template
        let inputs = serde_json::json!({
            "ordered": "false",
        });
        let input_string = TableScript::new_from_template(MINIMAL_LIST_INPUT.to_string())
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
        let template = [
            MINIMAL_HTML_PRE,
            MINIMAL_LIST_TEMPLATE,
            MINIMAL_HTML_POST,
        ]
        .join("");
        let script_string =
            TableScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n  </head>\n  <body>\n<ol>\n    <li>Item 1</li>\n    <li>Item 2</li>\n    <li>Item 3</li>\n    <li>Item 4</li>\n    <li>Item 5</li>\n    <li>Item 6</li>\n</ol>\n  </body>\n</html>"
        );
        Ok(())
    }
}
