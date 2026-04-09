/// Mermaid.js kanban jinja2 template
///
/// see <https://mermaid.js.org/syntax/kanban.html>
///
/// # Notes
/// * The kanban table MUST be sorted by column_name!
/// * `config` section is not yet supported
/// * The `priority` metadata attribute is not included currently
pub static MERMAID_KANBAN_TEMPLATE: &str = r#"
        kanban
{%- for row in rows %}
    {%- if loop.changed(row.column_name) %}
            {{ row.column_name }}[{{ row.column_label }}]
    {%- endif %}
                {{ row.task_name }}[{{ row.task_description }}]@{ ticket: {{ row.task_ticket }}, assigned: {{ row.task_assigned }}}
{%- endfor %}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{MERMAID_HTML_POST, MERMAID_HTML_PRE, SubjectScript, TEMPLATE_TABLE_EXPRESSION};
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use serde_json::Map;

    use super::*;

    #[test]
    fn test_mermaid_kanban_html() -> Result<()> {
        // Create the dummy data for the messages
        let column_name_vec = ["i", "i", "d", "d", "w", "w", "e", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let column_label_vec = [
            "Info", "Info", "Debug", "Debug", "Warn", "Warn", "Error", "Error",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let task_name_vec = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let task_desc_vec = [
            "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let task_assigned_vec = ["p1", "p2", "p3", "p4", "p1", "p2", "p3", "p4"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let task_ticket_vec = [
            "id-1",
            "id-2",
            "id-3",
            "id-4",
            "DataAnalysis",
            "DataAnalysis",
            "DataAnalysis",
            "DataAnalysis",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let task_priority_vec = ["Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let column_name_arr: ArrayRef = Arc::new(StringArray::from(column_name_vec));
        let column_label_arr: ArrayRef = Arc::new(StringArray::from(column_label_vec));
        let task_name_arr: ArrayRef = Arc::new(StringArray::from(task_name_vec));
        let task_desc_arr: ArrayRef = Arc::new(StringArray::from(task_desc_vec));
        let task_assigned_arr: ArrayRef = Arc::new(StringArray::from(task_assigned_vec));
        let task_ticket_arr: ArrayRef = Arc::new(StringArray::from(task_ticket_vec));
        let task_priority_arr: ArrayRef = Arc::new(StringArray::from(task_priority_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("column_name", column_name_arr),
            ("column_label", column_label_arr),
            ("task_name", task_name_arr),
            ("task_description", task_desc_arr),
            ("task_assigned", task_assigned_arr),
            ("task_ticket", task_ticket_arr),
            ("task_priority", task_priority_arr),
        ])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let kanban_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [MERMAID_HTML_PRE, MERMAID_KANBAN_TEMPLATE, MERMAID_HTML_POST].join("");
        let script_string =
            SubjectScript::new_from_template(template).apply_template(&kanban_template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n  </head>\n  <body>\n    <pre class=\"mermaid\">\n        kanban\n            i[Info]\n                t1[Task 1]@{ ticket: id-1, assigned: p1}\n                t2[Task 2]@{ ticket: id-2, assigned: p2}\n            d[Debug]\n                t3[Task 3]@{ ticket: id-3, assigned: p3}\n                t4[Task 4]@{ ticket: id-4, assigned: p4}\n            w[Warn]\n                t5[Task 5]@{ ticket: DataAnalysis, assigned: p1}\n                t6[Task 6]@{ ticket: DataAnalysis, assigned: p2}\n            e[Error]\n                t7[Task 7]@{ ticket: DataAnalysis, assigned: p3}\n                t8[Task 8]@{ ticket: DataAnalysis, assigned: p4}\n    </pre>\n  </body>\n</html>"
        );
        Ok(())
    }
}
