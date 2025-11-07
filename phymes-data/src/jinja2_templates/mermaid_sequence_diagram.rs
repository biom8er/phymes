/// Mermaid.js sequenceDiagram jinja2 template
///
/// see <https://mermaid.js.org/syntax/sequenceDiagram.html>
pub static MERMAID_SEQUENCE_DIAGRAM_TEMPLATE: &str = r#"
        sequenceDiagram
{%- for row in rows %}
            {{ row.content }}
{%- endfor %}"#;

/// Mermaid.js sequenceDiagram participants section jinja2 template
pub static MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE: &str = r#"{%- for row in rows %}
            participant {{ row.participant_name }}@{ 'type': '{{ row.participant_type }}' }
{%- endfor %}"#;

/// Mermaid.js sequenceDiagram messages section jinja2 template
pub static MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE: &str = r#"
{%- for row in rows %}
    {%- if row.subject_name %}
            {{ row.subject_name }}{{ row.message_type }}{{ row.activation_type }}{{ row.object_name }}: {{ row.message_content }}
    {%- endif %}
    {%- if row.note_content %}
            note {{ row.note_location }} {{ row.object_name }}: {{ row.note_content }}
    {%- endif %}
{%- endfor %}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::jinja2_templates::{TEMPLATE_TABLE_EXPRESSION, mermaid_html::{MERMAID_HTML_POST, MERMAID_HTML_PRE}};
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_core::{
        BuildableTrait, BuilderTrait, MappableTrait, Table, TableBuilderTrait, TableScript,
        TableTrait,
    };
    use serde_json::Map;

    use super::*;

    #[test]
    fn test_mermaid_sequence_diagram_html() -> Result<()> {
        // Create the dummy data for the participants
        let participant_vec = ["Measurement", "DataAnalysis", "Group1", "Group2", "Group3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let participant_type_vec = [
            "collections",
            "collections",
            "participant",
            "participant",
            "participant",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        let participant_arr: ArrayRef = Arc::new(StringArray::from(participant_vec));
        let participant_type_arr: ArrayRef = Arc::new(StringArray::from(participant_type_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("participant_name", participant_arr),
            ("participant_type", participant_type_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let participants_string = TableScript::new_from_template(
            MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE.to_string(),
        )
        .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(
            participants_string,
            "\n            participant Measurement@{ 'type': 'collections' }\n            participant DataAnalysis@{ 'type': 'collections' }\n            participant Group1@{ 'type': 'participant' }\n            participant Group2@{ 'type': 'participant' }\n            participant Group3@{ 'type': 'participant' }"
        );

        // Create the dummy data for the messages
        let subject_vec = [
            "",
            "Measurement",
            "DataAnalysis",
            "DataAnalysis",
            "DataAnalysis",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let object_vec = ["Measurement", "DataAnalysis", "Group1", "Group2", "Group3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let message_type_vec = ["->>", "->>", "->>", "->>", "->>"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let activation_vec = ["", "+", "", "", "-"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let message_vec = [
            "",
            "n=4095<br/>n=904 statin users<br/>n=3191 non-statin users",
            "n=1808<br/>n=904 statin users<br/>n=904 non-statin users",
            "n=1232<br/>n=616 statin users<br/>n=616 non-statin users",
            "n=3609<br/>n=762 statin users<br/>n=2845 non-statin users",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let note_vec = ["RFFT<br/>VAT<br/>Statin use<br/>Other variables<br/>FRS<br/>Propensity score", 
            "two sample t-test with equal variance<br/>Mann-Whitney U-test<br/>one-way ANOVA<br/>ANCOVA<br/>regression analysis", 
            "matched on age, sex, education", 
            "matched on FRS score", 
            "comparison based on propensity score"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let note_location_vec = ["left of", "left of", "left of", "left of", "left of"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let subject_arr: ArrayRef = Arc::new(StringArray::from(subject_vec));
        let object_arr: ArrayRef = Arc::new(StringArray::from(object_vec));
        let message_type_arr: ArrayRef = Arc::new(StringArray::from(message_type_vec));
        let activation_arr: ArrayRef = Arc::new(StringArray::from(activation_vec));
        let message_arr: ArrayRef = Arc::new(StringArray::from(message_vec));
        let note_arr: ArrayRef = Arc::new(StringArray::from(note_vec));
        let note_location_arr: ArrayRef = Arc::new(StringArray::from(note_location_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("subject_name", subject_arr),
            ("object_name", object_arr),
            ("message_type", message_type_arr),
            ("activation_type", activation_arr),
            ("message_content", message_arr),
            ("note_content", note_arr),
            ("note_location", note_location_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let messages_string =
            TableScript::new_from_template(MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE.to_string())
                .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(
            messages_string,
            "\n            note left of Measurement: RFFT<br/>VAT<br/>Statin use<br/>Other variables<br/>FRS<br/>Propensity score\n            Measurement->>+DataAnalysis: n=4095<br/>n=904 statin users<br/>n=3191 non-statin users\n            note left of DataAnalysis: two sample t-test with equal variance<br/>Mann-Whitney U-test<br/>one-way ANOVA<br/>ANCOVA<br/>regression analysis\n            DataAnalysis->>Group1: n=1808<br/>n=904 statin users<br/>n=904 non-statin users\n            note left of Group1: matched on age, sex, education\n            DataAnalysis->>Group2: n=1232<br/>n=616 statin users<br/>n=616 non-statin users\n            note left of Group2: matched on FRS score\n            DataAnalysis->>-Group3: n=3609<br/>n=762 statin users<br/>n=2845 non-statin users\n            note left of Group3: comparison based on propensity score"
        );

        // Combine the participants and messages
        let content_vec = vec![participants_string, messages_string];

        let content_arr: ArrayRef = Arc::new(StringArray::from(content_vec));
        let batch = RecordBatch::try_from_iter(vec![("content", content_arr)])?;
        let table = Table::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Update the input with the dummy chart data
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let sequence_diagram_template_inputs = serde_json::to_value(input_object)?;

        // Create and render the template with the inputs
        let template = [
            MERMAID_HTML_PRE,
            MERMAID_SEQUENCE_DIAGRAM_TEMPLATE,
            MERMAID_HTML_POST,
        ]
        .join("");
        let script_string = TableScript::new_from_template(template)
            .apply_template(&sequence_diagram_template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n  </head>\n  <body>\n    <pre class=\"mermaid\">\n        sequenceDiagram\n            \n            participant Measurement@{ 'type': 'collections' }\n            participant DataAnalysis@{ 'type': 'collections' }\n            participant Group1@{ 'type': 'participant' }\n            participant Group2@{ 'type': 'participant' }\n            participant Group3@{ 'type': 'participant' }\n            \n            note left of Measurement: RFFT<br/>VAT<br/>Statin use<br/>Other variables<br/>FRS<br/>Propensity score\n            Measurement->>+DataAnalysis: n=4095<br/>n=904 statin users<br/>n=3191 non-statin users\n            note left of DataAnalysis: two sample t-test with equal variance<br/>Mann-Whitney U-test<br/>one-way ANOVA<br/>ANCOVA<br/>regression analysis\n            DataAnalysis->>Group1: n=1808<br/>n=904 statin users<br/>n=904 non-statin users\n            note left of Group1: matched on age, sex, education\n            DataAnalysis->>Group2: n=1232<br/>n=616 statin users<br/>n=616 non-statin users\n            note left of Group2: matched on FRS score\n            DataAnalysis->>-Group3: n=3609<br/>n=762 statin users<br/>n=2845 non-statin users\n            note left of Group3: comparison based on propensity score\n    </pre>\n    <script type=\"module\">\n        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n        mermaid.initialize({theme: \"dark\", startOnLoad: true });\n    </script>\n  </body>\n</html>"
        );
        Ok(())
    }
}
