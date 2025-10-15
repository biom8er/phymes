/// Mermaid.js flowchart jinja2 template
/// 
/// see <https://mermaid.js.org/syntax/flowchart.html>
/// 
/// # Limitations
/// * A section for defining link styles <linkStyle> and node styles <classDef> is not yet implemented
/// * Link IDs is not yet supported
/// * Nested subgraphs is not yet supported
pub static MERMAID_FLOWCHART_TEMPLATE: &'static str = r#"
        flowchart
{%- for row in rows %}
            {{ row.content }}
{%- endfor %}"#;

/// Mermaid.js flowchart nodes section jinja2 template
pub static MERMAID_FLOWCHART_NODES_TEMPLATE: &'static str = r#"{%- for row in rows %}
            {{ row.node_name }}@{ shape: {{ row.node_shape }}, label: '{{ row.node_label }}' }
{%- endfor %}"#;

/// Mermaid.js flowchart links section jinja2 template
pub static MERMAID_FLOWCHART_LINKS_TEMPLATE: &'static str = r#"
{%- for row in rows %}
    {%- if row.link_text %}
            {{ row.subject_name }}{{ row.link_type }}|{{ row.link_text }}|{{ row.object_name }}
    {%- else %}
            {{ row.subject_name }}{{ row.link_type }}{{ row.object_name }}
    {%- endif %}
{%- endfor %}"#;

/// The `table_expression` variable name in `DataConfig`
pub static MERMAID_FLOWCHART_TABLE_EXPRESSION: &'static str = "rows";

/// Mermaid.js gantt input jinja2 template
/// 
/// # Example
/// 
/// ```rust
/// use phymes_core::table::table_script::TableScript;
/// use phymes_data::jinja2_templates::mermaid_gantt::MERMAID_GANTT_INPUT;
/// let inputs = serde_json::json!({
///     "direction": "TD",
/// });
/// 
/// let input_string = TableScript::new_from_template(MERMAID_GANTT_INPUT.to_string()).apply_template(&inputs).unwrap()
///     .lines()
///     .map(|line| line.trim())
///     .collect::<Vec<&str>>()
///     .join("");
/// ```
pub static MERMAID_FLOWCHART_INPUT: &'static str = r#"{
"direction": "{{ direction }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_core::{session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::{table_script::TableScript, table_trait::{Table, TableBuilderTrait, TableTrait}}};
    use serde_json::{Map, Value};
    use crate::jinja2_templates::mermaid_html::{MERMAID_HTML_POST, MERMAID_HTML_PRE};

    use super::*;

    #[test]
    fn test_mermaid_flowchart_html() -> Result<()> {
        // Create the dummy data for the nodes
        let node_vec = ["Age", "Gender", "Ethnicity", "characteristic", "statins", "Cognitive", "RFFT", "VAT", "Q"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let node_shape_vec = ["lean-r", "lean-r", "lean-r", "lean-r", "manual-input", "rounded", "lean-l", "lean-l", "lean-l"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let node_label_vec = ["age", "gender", "ethnic group", "🧬 genetics and ⛅ exposures", "💊 statins", "Cognitive disorder", "📏 Ruff Figural Fluency Test", "📏 Visual Association Test", "📏 questionnaire"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let node_arr: ArrayRef = Arc::new(StringArray::from(node_vec));
        let node_shape_arr: ArrayRef = Arc::new(StringArray::from(node_shape_vec));
        let node_label_arr: ArrayRef = Arc::new(StringArray::from(node_label_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("node_name", node_arr),
            ("node_shape", node_shape_arr),
            ("node_label", node_label_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(MERMAID_FLOWCHART_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ =  input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs  
        let nodes_string = TableScript::new_from_template(MERMAID_FLOWCHART_NODES_TEMPLATE.to_string())
            .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(nodes_string, "\n            Age@{ shape: lean-r, label: 'age' }\n            Gender@{ shape: lean-r, label: 'gender' }\n            Ethnicity@{ shape: lean-r, label: 'ethnic group' }\n            characteristic@{ shape: lean-r, label: '🧬 genetics and ⛅ exposures' }\n            statins@{ shape: manual-input, label: '💊 statins' }\n            Cognitive@{ shape: rounded, label: 'Cognitive disorder' }\n            RFFT@{ shape: lean-l, label: '📏 Ruff Figural Fluency Test' }\n            VAT@{ shape: lean-l, label: '📏 Visual Association Test' }\n            Q@{ shape: lean-l, label: '📏 questionnaire' }");

        // Create the dummy data for the links
        let subject_vec = ["Age", "Gender", "Ethnicity", "characteristic", "statins", "Cognitive", "Cognitive", "characteristic"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let object_vec = ["characteristic", "characteristic", "characteristic", "Cognitive", "Cognitive", "RFFT", "VAT", "Q"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let link_type_vec = ["-->", "-->", "-->", "-->", "-.->", "-->", "-->", "-->"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let link_vec = ["", "", "", "contributes to condition", "🔨 ameliorates condition", "characteristic measured by assay", "characteristic measured by assay", "characteristic measured by assay"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let subject_arr: ArrayRef = Arc::new(StringArray::from(subject_vec));
        let object_arr: ArrayRef = Arc::new(StringArray::from(object_vec));
        let link_type_arr: ArrayRef = Arc::new(StringArray::from(link_type_vec));
        let link_arr: ArrayRef = Arc::new(StringArray::from(link_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("subject_name", subject_arr),
            ("object_name", object_arr),
            ("link_type", link_type_arr),
            ("link_text", link_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(MERMAID_FLOWCHART_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ =  input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs  
        let links_string = TableScript::new_from_template(MERMAID_FLOWCHART_LINKS_TEMPLATE.to_string())
            .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(links_string, "\n            Age-->characteristic\n            Gender-->characteristic\n            Ethnicity-->characteristic\n            characteristic-->|contributes to condition|Cognitive\n            statins-.->|🔨 ameliorates condition|Cognitive\n            Cognitive-->|characteristic measured by assay|RFFT\n            Cognitive-->|characteristic measured by assay|VAT\n            characteristic-->|characteristic measured by assay|Q");

        // Combine the nodes and links        
        let content_vec = vec![nodes_string, links_string];

        let content_arr: ArrayRef = Arc::new(StringArray::from(content_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("content", content_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(MERMAID_FLOWCHART_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the input for the template
        let inputs = serde_json::json!({
            "direction": "TD",
        });
        let input_string = TableScript::new_from_template(MERMAID_FLOWCHART_INPUT.to_string()).apply_template(&inputs)?
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        // Update the input with the dummy chart data
        let mut input_object = serde_json::from_str::<Map<String, Value>>(&input_string)?;
        let _ =  input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [MERMAID_HTML_PRE, MERMAID_FLOWCHART_TEMPLATE, MERMAID_HTML_POST].join("");   
        let script_string = TableScript::new_from_template(template).apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n  </head>\n  <body>\n    <pre class=\"mermaid\">\n        flowchart\n            \n            Age@{ shape: lean-r, label: 'age' }\n            Gender@{ shape: lean-r, label: 'gender' }\n            Ethnicity@{ shape: lean-r, label: 'ethnic group' }\n            characteristic@{ shape: lean-r, label: '🧬 genetics and ⛅ exposures' }\n            statins@{ shape: manual-input, label: '💊 statins' }\n            Cognitive@{ shape: rounded, label: 'Cognitive disorder' }\n            RFFT@{ shape: lean-l, label: '📏 Ruff Figural Fluency Test' }\n            VAT@{ shape: lean-l, label: '📏 Visual Association Test' }\n            Q@{ shape: lean-l, label: '📏 questionnaire' }\n            \n            Age-->characteristic\n            Gender-->characteristic\n            Ethnicity-->characteristic\n            characteristic-->|contributes to condition|Cognitive\n            statins-.->|🔨 ameliorates condition|Cognitive\n            Cognitive-->|characteristic measured by assay|RFFT\n            Cognitive-->|characteristic measured by assay|VAT\n            characteristic-->|characteristic measured by assay|Q\n    </pre>\n    <script type=\"module\">\n        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n        mermaid.initialize({theme: \"dark\", startOnLoad: true });\n    </script>\n  </body>\n</html>"
        );
        Ok(())
    }
}