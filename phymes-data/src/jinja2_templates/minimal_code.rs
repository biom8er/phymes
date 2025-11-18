/// HTML5 table jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/code>
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/pre>
/// 
/// # Notes:
/// - only `code` element is supported
/// - `samp` element is not yet fully supported
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/samp>
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/kbd>
pub static MINIMAL_PRECODE_TEMPLATE: &str = r#"
{%- if samp %}
<pre><samp>
{%- else %}
<pre><code>
{%- endif %}
{%- for row in rows %}
{{ row.item }}
{%- endfor %}
{%- if samp %}
</samp></pre>
{%- else %}
</code></pre>
{%- endif %}"#;

/// HTML table input jinja2 template
pub static MINIMAL_PRECODE_INPUT: &str = r#"{
"samp": "{{ samp }}"
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
    fn test_minimal_code_html() -> Result<()> {
        // Create the dummy data for the table
        let item_vec = [
            "y1 = m*x1 + b1;",
            "y2 = m*x2 + b2;",
            "y3 = m*x3 + b3;",
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
            "samp": "",
        });
        let input_string = TableScript::new_from_template(MINIMAL_PRECODE_INPUT.to_string())
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
            MINIMAL_PRECODE_TEMPLATE,
            MINIMAL_HTML_POST,
        ]
        .join("");
        let script_string =
            TableScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n  </head>\n  <body>\n<pre><code>\ny1 = m*x1 + b1;\ny2 = m*x2 + b2;\ny3 = m*x3 + b3;\n</code></pre>\n  </body>\n</html>"
        );
        Ok(())
    }
}
