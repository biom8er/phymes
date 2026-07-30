/// HTML5 figure jinja2 template
///
/// see <https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/img>
///
/// # Notes
/// - The design is based on a product list template where each image includes a title and caption on their own card wrapped in a flex grid
pub static MINIMAL_FIGURE_TEMPLATE: &str = r#"<div class="mt-6 grid grid-cols-1 gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-4 xl:gap-x-8">
{%- for row in rows %}
    <div class="group relative">
        <img src='{{ row.src }}' alt='{{ row.alt }}' style='{{ row.style }}'{%- if img_class %} class="{{ img_class }}"{%- endif %}/>
        <h3{%- if h3_class %} class="{{ h3_class }}"{%- endif %}{%- if h3_style %} style="{{ h3_style }}"{%- endif %}>{{ row.title }}</h3>
        <p{%- if p_class %} class="{{ p_class }}"{%- endif %}{%- if p_style %} style="{{ p_style }}"{%- endif %}>{{ row.caption }}</p>
    </div>
{%- endfor %}
</div>"#;

/// HTML table input jinja2 template
pub static MINIMAL_FIGURE_INPUT: &str = r#"{
    "img_class": "{{ img_class }}",
    "h3_class": "{{ h3_class }}",
    "h3_style": "{{ h3_style }}",
    "p_class": "{{ p_class }}",
    "p_style": "{{ p_style }}"
}"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        SubjectScript, TEMPLATE_TABLE_EXPRESSION,
        template::{MINIMAL_HTML_POST, MINIMAL_HTML_PRE},
    };
    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use serde_json::{Map, Value};

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
            "Chemical structure of DNA; hydrogen bonds shown as dotted lines. Each end of the double helix has an exposed 5' phosphate on one strand and an exposed 3 hydroxyl group (—OH) on the other.",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let title_vec = ["DNA double helix", "DNA simplified", "DNA base pairing"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let src_arr: ArrayRef = Arc::new(StringArray::from(src_vec));
        let alt_arr: ArrayRef = Arc::new(StringArray::from(alt_vec));
        let style_arr: ArrayRef = Arc::new(StringArray::from(style_vec));
        let caption_arr: ArrayRef = Arc::new(StringArray::from(caption_vec));
        let title_arr: ArrayRef = Arc::new(StringArray::from(title_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("src", src_arr),
            ("alt", alt_arr),
            ("style", style_arr),
            ("caption", caption_arr),
            ("title", title_arr),
        ])?;
        let table = Subject::get_builder()
            .with_name(TEMPLATE_TABLE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Create the input for the template
        let inputs = serde_json::json!({
            "img_class": "aspect-square w-full rounded-md bg-gray-200 object-cover group-hover:opacity-75 lg:aspect-auto lg:h-80",
            "h3_class": "text-lg text-gray-700",
            "h3_style": "",
            "p_class": "mt-1 text-sm text-gray-500",
            "p_style": ""
        });
        let input_string = SubjectScript::new_from_template(MINIMAL_FIGURE_INPUT.to_string())
            .apply_template(&inputs)?
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        // Make the inputs
        let mut input_object = serde_json::from_str::<Map<String, Value>>(&input_string)?;
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let template_inputs = serde_json::to_value(input_object)?;

        // Create and rcaptioner the template with the inputs
        let template = [MINIMAL_HTML_PRE, MINIMAL_FIGURE_TEMPLATE, MINIMAL_HTML_POST].join("");
        let script_string =
            SubjectScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            "<!DOCTYPE html>\n<html>    \n    <head>\n        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n        <script src=\"https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4\"></script>\n        <script type=\"module\">\n            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n            import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';\n            mermaid.registerLayoutLoaders(elkLayouts);\n            mermaid.initialize({theme: \"dark\", startOnLoad: true });\n        </script>\n    </head>\n    <body><div class=\"mt-6 grid grid-cols-1 gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-4 xl:gap-x-8\">\n    <div class=\"group relative\">\n        <img src='https://upload.wikimedia.org/wikipedia/commons/4/4c/DNA_Structure%2BKey%2BLabelled.pn_NoBB.png' alt='DNA double helix' style='width:auto;height:auto;' class=\"aspect-square w-full rounded-md bg-gray-200 object-cover group-hover:opacity-75 lg:aspect-auto lg:h-80\"/>\n        <h3 class=\"text-lg text-gray-700\">DNA double helix</h3>\n        <p class=\"mt-1 text-sm text-gray-500\">The structure of the DNA double helix (type B-DNA). The atoms in the structure are colour-coded by element and the detailed structures of two base pairs are shown in the bottom right.</p>\n    </div>\n    <div class=\"group relative\">\n        <img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Phosphate_backbone.jpg' alt='DNA simplified' style='width:auto;height:auto;' class=\"aspect-square w-full rounded-md bg-gray-200 object-cover group-hover:opacity-75 lg:aspect-auto lg:h-80\"/>\n        <h3 class=\"text-lg text-gray-700\">DNA simplified</h3>\n        <p class=\"mt-1 text-sm text-gray-500\">Simplified diagram</p>\n    </div>\n    <div class=\"group relative\">\n        <img src='https://upload.wikimedia.org/wikipedia/commons/e/e4/DNA_chemical_structure.svg' alt='DNA base pairing' style='width:auto;height:auto;' class=\"aspect-square w-full rounded-md bg-gray-200 object-cover group-hover:opacity-75 lg:aspect-auto lg:h-80\"/>\n        <h3 class=\"text-lg text-gray-700\">DNA base pairing</h3>\n        <p class=\"mt-1 text-sm text-gray-500\">Chemical structure of DNA; hydrogen bonds shown as dotted lines. Each end of the double helix has an exposed 5' phosphate on one strand and an exposed 3 hydroxyl group (—OH) on the other.</p>\n    </div>\n</div>\n    </body>\n</html>"
        );
        Ok(())
    }
}
