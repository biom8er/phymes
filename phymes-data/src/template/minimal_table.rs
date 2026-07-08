/// HTML5 table jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/table>
///
/// # Notes
/// - thead and tbody sections are supported, but tfoot is not yet supported
/// - href e.g., <td><a href="{{ ontology.URL }}">{{ ontology.name }}</a></td>, is not yet supported
/// - Styling can be specified via TailwindCSS classes or style attributes
pub static MINIMAL_TABLE_TEMPLATE: &str = r#"
<table{% raw %}{%- if table_class %} class="{{ table_class }}"{%- endif %}{%- if table_style %} style="{{ table_style }}"{%- endif %}{% endraw %}>
    <caption{% raw %}{%- if caption_class %} class="{{ caption_class }}"{%- endif %}{%- if caption_style %} style="{{ caption_style }}"{%- endif %}{% endraw %}>
        {% raw %}{{ caption }}{% endraw %}
    </caption>
    <thead{% raw %}{%- if thead_class %} class="{{ thead_class }}"{%- endif %}{%- if thead_style %} style="{{ thead_style }}"{%- endif %}{% endraw %}>
        <tr>
{%- for header in headers %}
            <th{% raw %}{%- if th_class %} class="{{ th_class }}"{%- endif %}{%- if th_style %} style="{{ th_style }}"{%- endif %}{% endraw %}>{{ header }}</th>
{%- endfor %}
        </tr>
    </thead>
    <tbody{% raw %}{%- if tbody_class %} class="{{ tbody_class }}"{%- endif %}{%- if tbody_style %} style="{{ tbody_style }}"{%- endif %}{% endraw %}>
{% raw %}{%- for row in rows %}{% endraw %}
        <tr{% raw %}{%- if tr_class %} class="{{ tr_class }}"{%- endif %}{%- if tr_style %} style="{{ tr_style }}"{%- endif %}{% endraw %}>
{%- for header in headers %}
            <td{% raw %}{%- if td_class %} class="{{ td_class }}"{%- endif %}{%- if td_style %} style="{{ td_style }}"{%- endif %}{% endraw %}>{% raw %}{{{% endraw %}row.{{ header }}{% raw %}}}{% endraw %}</td>
{%- endfor %}
        </tr>
{% raw %}{%- endfor %}{% endraw %}
    </tbody>
</table>"#;

/// HTML table input jinja2 template
pub static MINIMAL_TABLE_INPUT: &str = r#"{
"caption": "{{ caption }}",
"table_class": "{{ table_class }}",
"table_style": "{{ table_style }}",
"caption_class": "{{ caption_class }}",
"caption_style": "{{ caption_style }}",
"thead_class": "{{ thead_class }}",
"thead_style": "{{ thead_style }}",
"tbody_class": "{{ tbody_class }}",
"tbody_style": "{{ tbody_style }}",
"tr_class": "{{ tr_class }}",
"tr_style": "{{ tr_style }}",
"th_class": "{{ th_class }}",
"th_style": "{{ th_style }}",
"td_class": "{{ tr_class }}",
"td_style": "{{ td_style }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        SubjectScript, TEMPLATE_TABLE_EXPRESSION,
        template::{MINIMAL_HTML_POST, MINIMAL_HTML_PRE, TEMPLATE_HEADER_EXPRESSION},
    };
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray, UInt32Array};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
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
        let table = Subject::get_builder()
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
        let rendered_template =
            SubjectScript::new_from_template(MINIMAL_TABLE_TEMPLATE.to_string())
                .apply_template(&template_inputs)?;

        assert_eq!(
            rendered_template,
            "\n<table{%- if table_class %} class=\"{{ table_class }}\"{%- endif %}{%- if table_style %} style=\"{{ table_style }}\"{%- endif %}>\n    <caption{%- if caption_class %} class=\"{{ caption_class }}\"{%- endif %}{%- if caption_style %} style=\"{{ caption_style }}\"{%- endif %}>\n        {{ caption }}\n    </caption>\n    <thead{%- if thead_class %} class=\"{{ thead_class }}\"{%- endif %}{%- if thead_style %} style=\"{{ thead_style }}\"{%- endif %}>\n        <tr>\n            <th{%- if th_class %} class=\"{{ th_class }}\"{%- endif %}{%- if th_style %} style=\"{{ th_style }}\"{%- endif %}>section</th>\n            <th{%- if th_class %} class=\"{{ th_class }}\"{%- endif %}{%- if th_style %} style=\"{{ th_style }}\"{%- endif %}>task</th>\n            <th{%- if th_class %} class=\"{{ th_class }}\"{%- endif %}{%- if th_style %} style=\"{{ th_style }}\"{%- endif %}>start</th>\n            <th{%- if th_class %} class=\"{{ th_class }}\"{%- endif %}{%- if th_style %} style=\"{{ th_style }}\"{%- endif %}>end</th>\n        </tr>\n    </thead>\n    <tbody{%- if tbody_class %} class=\"{{ tbody_class }}\"{%- endif %}{%- if tbody_style %} style=\"{{ tbody_style }}\"{%- endif %}>\n{%- for row in rows %}\n        <tr{%- if tr_class %} class=\"{{ tr_class }}\"{%- endif %}{%- if tr_style %} style=\"{{ tr_style }}\"{%- endif %}>\n            <td{%- if td_class %} class=\"{{ td_class }}\"{%- endif %}{%- if td_style %} style=\"{{ td_style }}\"{%- endif %}>{{row.section}}</td>\n            <td{%- if td_class %} class=\"{{ td_class }}\"{%- endif %}{%- if td_style %} style=\"{{ td_style }}\"{%- endif %}>{{row.task}}</td>\n            <td{%- if td_class %} class=\"{{ td_class }}\"{%- endif %}{%- if td_style %} style=\"{{ td_style }}\"{%- endif %}>{{row.start}}</td>\n            <td{%- if td_class %} class=\"{{ td_class }}\"{%- endif %}{%- if td_style %} style=\"{{ td_style }}\"{%- endif %}>{{row.end}}</td>\n        </tr>\n{%- endfor %}\n    </tbody>\n</table>"
        );

        // --- alternating row table style ---

        // Render the final table
        let inputs = serde_json::json!({
            "caption": "Table caption",
            "table_class": "table-auto rounded bg-gray-200 text-gray-800",
            "table_style": "",
            "caption_class": "italic",
            "caption_style": "",
            "thead_class": "bg-gray-300",
            "thead_style": "",
            "tbody_class": "table-auto text-gray-800",
            "tbody_style": "",
            "tr_class": "odd:bg-gray-200 even:bg-gray-100",
            "tr_style": "",
            "th_class": "",
            "th_style": "",
            "td_class": "",
            "td_style": ""
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
            SubjectScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
"<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n        <script src=\"https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4\"></script>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';\n            mermaid.registerLayoutLoaders(elkLayouts);\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n    </head>\n    <body>\n<table class=\"table-auto rounded bg-gray-200 text-gray-800\">\n    <caption class=\"italic\">\n        Table caption\n    </caption>\n    <thead class=\"bg-gray-300\">\n        <tr>\n            <th>section</th>\n            <th>task</th>\n            <th>start</th>\n            <th>end</th>\n        </tr>\n    </thead>\n    <tbody class=\"table-auto text-gray-800\">\n        <tr class=\"odd:bg-gray-200 even:bg-gray-100\">\n            <td>Section 1</td>\n            <td>A</td>\n            <td>0</td>\n            <td>7</td>\n        </tr>\n        <tr class=\"odd:bg-gray-200 even:bg-gray-100\">\n            <td>Section 1</td>\n            <td>B</td>\n            <td>1</td>\n            <td>8</td>\n        </tr>\n        <tr class=\"odd:bg-gray-200 even:bg-gray-100\">\n            <td>Section 1</td>\n            <td>C</td>\n            <td>2</td>\n            <td>9</td>\n        </tr>\n        <tr class=\"odd:bg-gray-200 even:bg-gray-100\">\n            <td>Section 2</td>\n            <td>D</td>\n            <td>0</td>\n            <td>10</td>\n        </tr>\n        <tr class=\"odd:bg-gray-200 even:bg-gray-100\">\n            <td>Section 2</td>\n            <td>E</td>\n            <td>1</td>\n            <td>11</td>\n        </tr>\n        <tr class=\"odd:bg-gray-200 even:bg-gray-100\">\n            <td>Section 2</td>\n            <td>F</td>\n            <td>2</td>\n            <td>12</td>\n        </tr>\n    </tbody>\n</table>\n    </body>\n</html>"        );

        // --- dark header style ---

        // Render the final table
        let inputs = serde_json::json!({
            "caption": "Table caption",
            "table_class": "table-auto w-full bg-white border border-gray-300",
            "table_style": "",
            "caption_class": "italic",
            "caption_style": "",
            "thead_class": "bg-gray-800 text-white",
            "thead_style": "",
            "tbody_class": "table-auto text-gray-800",
            "tbody_style": "",
            "tr_class": "",
            "tr_style": "",
            "th_class": "border border-gray-300 px-4 py-2",
            "th_style": "",
            "td_class": "border border-gray-300 px-4 py-2",
            "td_style": ""
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
            SubjectScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
"<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n        <script src=\"https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4\"></script>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';\n            mermaid.registerLayoutLoaders(elkLayouts);\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n    </head>\n    <body>\n<table class=\"table-auto w-full bg-white border border-gray-300\">\n    <caption class=\"italic\">\n        Table caption\n    </caption>\n    <thead class=\"bg-gray-800 text-white\">\n        <tr>\n            <th class=\"border border-gray-300 px-4 py-2\">section</th>\n            <th class=\"border border-gray-300 px-4 py-2\">task</th>\n            <th class=\"border border-gray-300 px-4 py-2\">start</th>\n            <th class=\"border border-gray-300 px-4 py-2\">end</th>\n        </tr>\n    </thead>\n    <tbody class=\"table-auto text-gray-800\">\n        <tr>\n            <td class=\"border border-gray-300 px-4 py-2\">Section 1</td>\n            <td class=\"border border-gray-300 px-4 py-2\">A</td>\n            <td class=\"border border-gray-300 px-4 py-2\">0</td>\n            <td class=\"border border-gray-300 px-4 py-2\">7</td>\n        </tr>\n        <tr>\n            <td class=\"border border-gray-300 px-4 py-2\">Section 1</td>\n            <td class=\"border border-gray-300 px-4 py-2\">B</td>\n            <td class=\"border border-gray-300 px-4 py-2\">1</td>\n            <td class=\"border border-gray-300 px-4 py-2\">8</td>\n        </tr>\n        <tr>\n            <td class=\"border border-gray-300 px-4 py-2\">Section 1</td>\n            <td class=\"border border-gray-300 px-4 py-2\">C</td>\n            <td class=\"border border-gray-300 px-4 py-2\">2</td>\n            <td class=\"border border-gray-300 px-4 py-2\">9</td>\n        </tr>\n        <tr>\n            <td class=\"border border-gray-300 px-4 py-2\">Section 2</td>\n            <td class=\"border border-gray-300 px-4 py-2\">D</td>\n            <td class=\"border border-gray-300 px-4 py-2\">0</td>\n            <td class=\"border border-gray-300 px-4 py-2\">10</td>\n        </tr>\n        <tr>\n            <td class=\"border border-gray-300 px-4 py-2\">Section 2</td>\n            <td class=\"border border-gray-300 px-4 py-2\">E</td>\n            <td class=\"border border-gray-300 px-4 py-2\">1</td>\n            <td class=\"border border-gray-300 px-4 py-2\">11</td>\n        </tr>\n        <tr>\n            <td class=\"border border-gray-300 px-4 py-2\">Section 2</td>\n            <td class=\"border border-gray-300 px-4 py-2\">F</td>\n            <td class=\"border border-gray-300 px-4 py-2\">2</td>\n            <td class=\"border border-gray-300 px-4 py-2\">12</td>\n        </tr>\n    </tbody>\n</table>\n    </body>\n</html>"        );

        Ok(())
    }
}
