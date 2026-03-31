/// Mermaid.js erDiagram jinja2 template
///
/// see <https://mermaid.js.org/syntax/erDiagram.html>
pub static MERMAID_ER_DIAGRAM_TEMPLATE: &str = r#"
erDiagram
{%- for row in rows %}
    {{ row.content }}
{%- endfor %}"#;

/// Mermaid.js erDiagram entities section jinja2 template
pub static MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE: &str = r#"{%- for row in rows %}
    {%- if loop.changed(row.entity_name) %}
        {%- if not loop.first %}
    }
        {%- endif %}
    {{ row.entity_name }}["{{ row.entity_alias }}"] {
    {%- endif %}
        {{ row.attribute_type }} {{ row.attribute_name }}{%- if row.attribute_key %} {{ row.attribute_key }}{%- endif %}{%- if row.attribute_comment %} "{{ row.attribute_comment }}"{%- endif %}
{%- endfor %}
    }"#;

/// Mermaid.js erDiagram relations section jinja2 template
pub static MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE: &str = r#"
{%- for row in rows %}
    {{ row.subject_name }} {{ row.relation_type }} {{ row.object_name }}: "{{ row.relation_content }}"
{%- endfor %}"#;

/// Input schema for the erDiagram
pub static MERMAID_ER_DIAGRAM_INPUT: &str = r#"{
"direction": "{{ direction }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::template::{
        TEMPLATE_TABLE_EXPRESSION,
        mermaid_html::{MERMAID_HTML_POST, MERMAID_HTML_PRE},
    };
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_core::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectScript,
        SubjectTrait,
    };
    use serde_json::{Map, Value};

    use super::*;

    #[test]
    fn test_mermaid_er_diagram_html() -> Result<()> {
        // Create the dummy data for the entities
        let attribute_name_vec = ["Measurement", "DataAnalysis", "Group1", "Group2", "Group3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let attribute_type_vec = ["float", "enum", "int", "string", "string"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let attribute_key_vec = ["", "PK", "FK", "", "UK"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let attribute_comment_vec = ["", "e.g., 1, 2, 3", "", "Detailed group description", ""]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let entity_name_vec = ["c", "c", "e", "e", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let entity_alias_vec = ["collections", "collections", "entity", "entity", "entity"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let attribute_name_arr: ArrayRef = Arc::new(StringArray::from(attribute_name_vec));
        let attribute_type_arr: ArrayRef = Arc::new(StringArray::from(attribute_type_vec));
        let attribute_key_arr: ArrayRef = Arc::new(StringArray::from(attribute_key_vec));
        let entity_name_arr: ArrayRef = Arc::new(StringArray::from(entity_name_vec));
        let entity_alias_arr: ArrayRef = Arc::new(StringArray::from(entity_alias_vec));
        let attribute_comment_arr: ArrayRef = Arc::new(StringArray::from(attribute_comment_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("entity_name", entity_name_arr),
            ("entity_alias", entity_alias_arr),
            ("attribute_name", attribute_name_arr),
            ("attribute_type", attribute_type_arr),
            ("attribute_key", attribute_key_arr),
            ("attribute_comment", attribute_comment_arr),
        ])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let entities_string =
            SubjectScript::new_from_template(MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE.to_string())
                .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(
            entities_string,
            "\n    c[\"collections\"] {\n        float Measurement\n        enum DataAnalysis PK \"e.g., 1, 2, 3\"\n    }\n    e[\"entity\"] {\n        int Group1 FK\n        string Group2 \"Detailed group description\"\n        string Group3 UK\n    }"
        );

        // Create the dummy data for the relations
        let subject_vec = ["c"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let object_vec = ["e"].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let relation_type_vec = ["||--o{"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let relation_vec = ["needed for"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let subject_arr: ArrayRef = Arc::new(StringArray::from(subject_vec));
        let object_arr: ArrayRef = Arc::new(StringArray::from(object_vec));
        let relation_type_arr: ArrayRef = Arc::new(StringArray::from(relation_type_vec));
        let relation_arr: ArrayRef = Arc::new(StringArray::from(relation_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("subject_name", subject_arr),
            ("object_name", object_arr),
            ("relation_type", relation_type_arr),
            ("relation_content", relation_arr),
        ])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let relations_string =
            SubjectScript::new_from_template(MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE.to_string())
                .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(relations_string, "\n    c ||--o{ e: \"needed for\"");

        // Combine the entities and relations
        let content_vec = vec![entities_string, relations_string];

        let content_arr: ArrayRef = Arc::new(StringArray::from(content_vec));
        let batch = RecordBatch::try_from_iter(vec![("content", content_arr)])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the input for the template
        let inputs = serde_json::json!({
            "direction": "TB",
        });
        let input_string = SubjectScript::new_from_template(MERMAID_ER_DIAGRAM_INPUT.to_string())
            .apply_template(&inputs)?
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        // Update the input with the dummy chart data
        let mut input_object = serde_json::from_str::<Map<String, Value>>(&input_string)?;
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [
            MERMAID_HTML_PRE,
            MERMAID_ER_DIAGRAM_TEMPLATE,
            MERMAID_HTML_POST,
        ]
        .join("");
        let script_string = SubjectScript::new_from_template(template)
            .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n  </head>\n  <body>\n    <pre class=\"mermaid\">\nerDiagram\n    \n    c[\"collections\"] {\n        float Measurement\n        enum DataAnalysis PK \"e.g., 1, 2, 3\"\n    }\n    e[\"entity\"] {\n        int Group1 FK\n        string Group2 \"Detailed group description\"\n        string Group3 UK\n    }\n    \n    c ||--o{ e: \"needed for\"\n    </pre>\n  </body>\n</html>"
        );
        Ok(())
    }
}
