/// Mermaid.js xy chart jinja2 template
///
/// see <https://mermaid.js.org/syntax/xyChart.html>
pub static MERMAID_XYCHART_TEMPLATE: &str = r#"
        xychart
            title "{{ title }}"
            x-axis "{{ x_title }}" [{%- for row in rows %}{{ row.x }}{% if not loop.last %}, {% endif %}{%- endfor %}]
            y-axis "{{ y_title }}"
            line [{%- for row in rows %}{{ row.y }}{% if not loop.last %}, {% endif %}{%- endfor %}]"#;

/// Mermaid.js xy chart input jinja2 template
///
/// # Example
///
/// ```rust
/// use phymes_data::SubjectScript;
/// use phymes_data::MERMAID_XYCHART_INPUT;
/// let inputs = serde_json::json!({
///     "title": "chart title",
///     "x_title": "x title",
///     "y_title": "y title"
/// });
///
/// let input_string = SubjectScript::new_from_template(MERMAID_XYCHART_INPUT.to_string()).apply_template(&inputs).unwrap()
///     .lines()
///     .map(|line| line.trim())
///     .collect::<Vec<&str>>()
///     .join("");
/// ```
pub static MERMAID_XYCHART_INPUT: &str = r#"{
"title": "{{ title }}",
"x_title": "{{ x_title }}",
"y_title": "{{ y_title }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{MERMAID_HTML_POST, MERMAID_HTML_PRE, SubjectScript, TEMPLATE_TABLE_EXPRESSION};
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray, UInt32Array};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use serde_json::{Map, Value};

    use super::*;

    #[test]
    fn test_mermaid_xychart_html() -> Result<()> {
        // Create the dummy data for the chart
        let x_vec = [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let y_vec = [
            5000, 6000, 7500, 8200, 9500, 10500, 11000, 10200, 9200, 8500, 7000, 6000,
        ];

        let x_arr: ArrayRef = Arc::new(StringArray::from(x_vec));
        let y_arr: ArrayRef = Arc::new(UInt32Array::from_iter_values(y_vec));
        let batch = RecordBatch::try_from_iter(vec![("x", x_arr), ("y", y_arr)])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the input for the template
        let inputs = serde_json::json!({
            "title": "chart title",
            "x_title": "x title",
            "y_title": "y title",
        });
        let input_string = SubjectScript::new_from_template(MERMAID_XYCHART_INPUT.to_string())
            .apply_template(&inputs)?
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        // Update the input with the dummy chart data
        let mut input_object = serde_json::from_str::<Map<String, Value>>(&input_string)?;
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let xychart_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [
            MERMAID_HTML_PRE,
            MERMAID_XYCHART_TEMPLATE,
            MERMAID_HTML_POST,
        ]
        .join("");
        let script_string =
            SubjectScript::new_from_template(template).apply_template(&xychart_template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';\n            mermaid.registerLayoutLoaders(elkLayouts);\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n  </head>\n  <body>\n    <pre class=\"mermaid\">\n        xychart\n            title \"chart title\"\n            x-axis \"x title\" [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]\n            y-axis \"y title\"\n            line [5000, 6000, 7500, 8200, 9500, 10500, 11000, 10200, 9200, 8500, 7000, 6000]\n    </pre>\n  </body>\n</html>"
        );
        Ok(())
    }
}
