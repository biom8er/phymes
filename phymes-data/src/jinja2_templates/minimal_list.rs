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

/// The `table_expression` variable name in `DataConfig`
pub static MINIMAL_LIST_EXPRESSION: &str = "rows";

/// HTML table input jinja2 template
pub static MINIMAL_LIST_INPUT: &str = r#"{
"ordered": "{{ ordered }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::jinja2_templates::minimal_html::{MINIMAL_HTML_PRE, MINIMAL_HTML_POST};
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
            .with_name(MINIMAL_LIST_EXPRESSION)
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
            ""
        );
        Ok(())
    }
}
