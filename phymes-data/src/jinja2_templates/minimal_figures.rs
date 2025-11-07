/// HTML5 figure jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/figure>
/// 
/// # Notes
/// - 
pub static MINIMAL_FIGURE_TEMPLATE: &str = r#"
{%- for row in rows %}
<div>
    <figure>
        <img src="{{ row.src }}" alt="{{ row.alt }}" style="{{ row.style }}">
        <figcaption>{{ row.caption }}</figcaption>
    </figure>
</div>
{%- endfor %}"#;

/// The `table_expression` variable name in `DataConfig`
pub static MINIMAL_FIGURE_EXPRESSION: &str = "rows";

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
    use serde_json::Map;

    use super::*;

    #[test]
    fn test_minimal_figures_html() -> Result<()> {
        // Create the dummy data for the table
        let src_vec = [
            "https://upload.wikimedia.org/wikipedia/commons/4/4c/DNA_Structure%2BKey%2BLabelled.pn_NoBB.png",
            "https://upload.wikimedia.org/wikipedia/commons/1/19/Phosphate_backbone.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/e/e4/DNA_chemical_structure.svg",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let alt_vec = ["DNA double helix", "DNA simplified", "DNA base pairing"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let style_vec = [
            "width:auto;height:auto;",
            "width:auto;height:auto;",
            "width:auto;height:auto;",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let caption_vec = [
            "The structure of the DNA double helix (type B-DNA). The atoms in the structure are colour-coded by element and the detailed structures of two base pairs are shown in the bottom right.",
            "Simplified diagram",
            "Chemical structure of DNA; hydrogen bonds shown as dotted lines. Each end of the double helix has an exposed 5' phosphate on one strand and an exposed 3′ hydroxyl group (—OH) on the other.",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        let src_arr: ArrayRef = Arc::new(StringArray::from(src_vec));
        let alt_arr: ArrayRef = Arc::new(StringArray::from(alt_vec));
        let style_arr: ArrayRef = Arc::new(StringArray::from(style_vec));
        let caption_arr: ArrayRef = Arc::new(StringArray::from(caption_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("src", src_arr),
            ("alt", alt_arr),
            ("style", style_arr),
            ("caption", caption_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(MINIMAL_FIGURE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Make the inputs
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let template_inputs = serde_json::to_value(input_object)?;

        // Create and rcaptioner the template with the inputs
        let template = [MINIMAL_HTML_PRE, MINIMAL_FIGURE_TEMPLATE, MINIMAL_HTML_POST].join("");
        let script_string =
            TableScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"color-scheme\" content=\"dark light\">\n        <style>\n            @media (prefers-color-scheme: dark) {\n                body {\n                    background-color: black;\n                    color: white;\n                }\n            }\n            @media (prefers-color-scheme: light) {\n                body {\n                    background-color: white;\n                    color: black;\n                }\n            }\n        </style>\n  </head>\n  <body>\n<div>\n    <figure>\n        <img src=\"https://upload.wikimedia.org/wikipedia/commons/4/4c/DNA_Structure%2BKey%2BLabelled.pn_NoBB.png\" alt=\"DNA double helix\" style=\"width:auto;height:auto;\">\n        <figcaption>The structure of the DNA double helix (type B-DNA). The atoms in the structure are colour-coded by element and the detailed structures of two base pairs are shown in the bottom right.</figcaption>\n    </figure>\n</div>\n<div>\n    <figure>\n        <img src=\"https://upload.wikimedia.org/wikipedia/commons/1/19/Phosphate_backbone.jpg\" alt=\"DNA simplified\" style=\"width:auto;height:auto;\">\n        <figcaption>Simplified diagram</figcaption>\n    </figure>\n</div>\n<div>\n    <figure>\n        <img src=\"https://upload.wikimedia.org/wikipedia/commons/e/e4/DNA_chemical_structure.svg\" alt=\"DNA base pairing\" style=\"width:auto;height:auto;\">\n        <figcaption>Chemical structure of DNA; hydrogen bonds shown as dotted lines. Each end of the double helix has an exposed 5' phosphate on one strand and an exposed 3′ hydroxyl group (—OH) on the other.</figcaption>\n    </figure>\n</div>\n  </body>\n</html>"
        );
        Ok(())
    }
}
