/// HTML5 table jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/table>
///
/// # Notes
/// - thead and tbody sections are support, but tfoot is not yet supported
/// - href e.g., <td><a href="{{ ontology.URL }}">{{ ontology.name }}</a></td>, is not yet supported
pub static MINIMAL_TABLE_TEMPLATE: &str = r#"
<table>
    <caption>
        {% raw %}{{ caption }}{% endraw %}
    </caption>
    <thead>
        <tr>
{%- for header in headers %}
            <th>{{ header }}</th>
{%- endfor %}
        </tr>
    </thead>
    <tbody>
{% raw %}{%- for row in rows %}{% endraw %}
        <tr>
{%- for header in headers %}
            <td>{% raw %}{{{% endraw %}row.{{ header }}{% raw %}}}{% endraw %}</td>
{%- endfor %}
        </tr>
{% raw %}{%- endfor %}{% endraw %}
    </tbody>
</table>"#;

/// HTML table input jinja2 template
pub static MINIMAL_TABLE_INPUT: &str = r#"{
"caption": "{{ caption }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::jinja2_templates::{
        TEMPLATE_HEADER_EXPRESSION, TEMPLATE_TABLE_EXPRESSION,
        minimal_html::{MINIMAL_HTML_POST, MINIMAL_HTML_PRE},
    };
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray, UInt32Array};
    use phymes_core::{
        BuildableTrait, BuilderTrait, MappableTrait, Table, TableBuilderTrait, TableScript,
        TableTrait,
    };
    use serde_json::{Map, Value};

    use super::*;

    #[test]
    fn test_minimal_table_html() -> Result<()> {
        // Create the dummy data for the table
        let section_vec = [
            "Section 1",
            "Section 1",
            "Section 1",
            "Section 2",
            "Section 2",
            "Section 2",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let task_vec = ["A", "B", "C", "D", "E", "F"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let start_vec = [0, 1, 2, 0, 1, 2];
        let end_vec = [7, 8, 9, 10, 11, 12];

        let section_arr: ArrayRef = Arc::new(StringArray::from(section_vec));
        let task_arr: ArrayRef = Arc::new(StringArray::from(task_vec));
        let start_arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(start_vec));
        let end_arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(end_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("section", section_arr),
            ("task", task_arr),
            ("start", start_arr),
            ("end", end_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // 1. make the thead
        let headers = table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        let mut input_object = Map::new();
        let _ = input_object.insert(TEMPLATE_HEADER_EXPRESSION.to_string(), headers.into());
        let template_inputs = serde_json::to_value(input_object)?;
        let rendered_template = TableScript::new_from_template(MINIMAL_TABLE_TEMPLATE.to_string())
            .apply_template(&template_inputs)?;

        assert_eq!(
            rendered_template,
            "\n<table>\n    <caption>\n        {{ caption }}\n    </caption>\n    <thead>\n        <tr>\n            <th>section</th>\n            <th>task</th>\n            <th>start</th>\n            <th>end</th>\n        </tr>\n    </thead>\n    <tbody>\n{%- for row in rows %}\n        <tr>\n            <td>{{row.section}}</td>\n            <td>{{row.task}}</td>\n            <td>{{row.start}}</td>\n            <td>{{row.end}}</td>\n        </tr>\n{%- endfor %}\n    </tbody>\n</table>"
        );

        // Render the final table
        let inputs = serde_json::json!({
            "caption": "Table caption"
        });
        let mut input_object: Map<String, Value> = serde_json::from_value(inputs)?;
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [
            MINIMAL_HTML_PRE,
            rendered_template.as_str(),
            MINIMAL_HTML_POST,
        ]
        .join("");
        let script_string =
            TableScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n  </head>\n  <body>\n<table>\n    <caption>\n        Table caption\n    </caption>\n    <thead>\n        <tr>\n            <th>section</th>\n            <th>task</th>\n            <th>start</th>\n            <th>end</th>\n        </tr>\n    </thead>\n    <tbody>\n        <tr>\n            <td>Section 1</td>\n            <td>A</td>\n            <td>0</td>\n            <td>7</td>\n        </tr>\n        <tr>\n            <td>Section 1</td>\n            <td>B</td>\n            <td>1</td>\n            <td>8</td>\n        </tr>\n        <tr>\n            <td>Section 1</td>\n            <td>C</td>\n            <td>2</td>\n            <td>9</td>\n        </tr>\n        <tr>\n            <td>Section 2</td>\n            <td>D</td>\n            <td>0</td>\n            <td>10</td>\n        </tr>\n        <tr>\n            <td>Section 2</td>\n            <td>E</td>\n            <td>1</td>\n            <td>11</td>\n        </tr>\n        <tr>\n            <td>Section 2</td>\n            <td>F</td>\n            <td>2</td>\n            <td>12</td>\n        </tr>\n    </tbody>\n</table>\n  </body>\n</html>"
        );
        Ok(())
    }
}
