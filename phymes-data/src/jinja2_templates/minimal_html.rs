/// HTML template for rendering HTML reports
///
/// The template is split into a `pre` and `post` static str
///   whereby the HTML content string should be inserted in between
///
///
/// ```html
/// <!-- MINIMAL_HTML_PRE -->
/// <!DOCTYPE html>
/// <html>    
///     <head>
///         <meta http-equiv="Content-type" content="text/html;charset=UTF-8">
///         <meta name="color-scheme" content="dark light">
///         <style>
///             @media (prefers-color-scheme: dark) {
///                 body {
///                     background-color: black;
///                     color: white;
///                 }
///             }
///             @media (prefers-color-scheme: light) {
///                 body {
///                     background-color: white;
///                     color: black;
///                 }
///             }
///         </style>
///   </head>
///   <body>
///     <pre class="mermaid">
///
/// <!-- Insert HTML content here -->
///
///             TODO
///
/// <!-- MINIMAL_HTML_POST -->
///
///   </body>
/// </html>
/// ```
/// Part 1 of the minimal html jinja2 template
pub static MINIMAL_HTML_PRE: &str = r#"<!DOCTYPE html>
<html>    
    <head>
        <meta http-equiv="Content-type" content="text/html;charset=UTF-8">
        <meta name="color-scheme" content="dark light">
        <style>
            @media (prefers-color-scheme: dark) {
                body {
                    background-color: black;
                    color: white;
                }
            }
            @media (prefers-color-scheme: light) {
                body {
                    background-color: white;
                    color: black;
                }
            }
        </style>
  </head>
  <body>"#;

/// Part 2 of the minimal html jinja2 template
pub static MINIMAL_HTML_POST: &str = r#"
  </body>
</html>"#;

/// Template for rendering a minimal html jinja2 template with specified HTML tag elements
/// 
/// # Notes
/// - Use the `start_tag`, `header`, and `end_tag` to generate HTML lines like the following
///   <h1>{{ title }}</h1> where tag_state = <h1>, header = title, and end_tag = </h1>
/// - Use only `start_tag`to generate HTML lines without templates like the following
///   <h1>My title</h1> where tag_state = <h1>My title</h1>
pub static MINIMAL_HTML_BODY_TEMPLATE: &str = r#"
{% raw %}{%- for row in rows %}{% endraw %}
{%- for header in headers %}
    {%- if header.header %}
{{ header.start_tag }}{% raw %}{{{% endraw %}row.{{ header.header }}{% raw %}}}{% endraw %}{{ header.end_tag }}
    {%- else %}
{{ header.start_tag }}
    {%- endif %}
{%- endfor %}
{% raw %}{%- endfor %}{% endraw %}"#;

/// The entry point for the HTML body template headers
pub static MINIMAL_HTML_BODY_TEMPLATE_EXPRESSION: &str = "headers";

/// The entry point for the rendered HTML body template rows
pub static MINIMAL_HTML_BODY_EXPRESSION: &str = "rows";

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_core::{
        BuildableTrait, BuilderTrait, MappableTrait, Table, TableBuilderTrait, TableScript,
        TableTrait,
    };
    use serde_json::Map;

    use super::*;

    #[test]
    fn test_minimal_body_html() -> Result<()> {
        // Create the dummy data for the headers
        let start_tag_vec = [
            "<h1>",
            "<p>",
            "<p>",
            "<h2> Background</h2>",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let header_vec = [
            "title",
            "version",
            "description",
            "",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let end_tag_vec = [
            "</h1>",
            "</p>",
            "</p>",
            "",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        let start_tag_arr: ArrayRef = Arc::new(StringArray::from(start_tag_vec));
        let header_arr: ArrayRef = Arc::new(StringArray::from(header_vec));
        let end_tag_arr: ArrayRef = Arc::new(StringArray::from(end_tag_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("start_tag", start_tag_arr),
            ("header", header_arr),
            ("end_tag", end_tag_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(MINIMAL_HTML_BODY_TEMPLATE_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Render the html template
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let template_inputs = serde_json::to_value(input_object)?;
        let rendered_template = TableScript::new_from_template(MINIMAL_HTML_BODY_TEMPLATE.to_string())
            .apply_template(&template_inputs)?;

        assert_eq!(
            rendered_template,
            "\n{%- for row in rows %}\n<h1>{{row.title}}</h1>\n<p>{{row.version}}</p>\n<p>{{row.description}}</p>\n<h2> Background</h2>\n{%- endfor %}"
        );

        // Create the dummy data for the html data
        let title_vec = [
            "Title 1",
            "Title 2",
            "Title 3",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let version_vec = [
            "Version 1",
            "Version 2",
            "Version 3",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let description_vec = [
            "Description 1",
            "Description 2",
            "Description 3",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        let title_arr: ArrayRef = Arc::new(StringArray::from(title_vec));
        let version_arr: ArrayRef = Arc::new(StringArray::from(version_vec));
        let description_arr: ArrayRef = Arc::new(StringArray::from(description_vec));
        let batch = RecordBatch::try_from_iter(vec![
            ("title", title_arr),
            ("version", version_arr),
            ("description", description_arr),
        ])?;
        let table = Table::get_builder()
            .with_name(MINIMAL_HTML_BODY_EXPRESSION)
            .with_record_batches(vec![batch])?
            .build()?;

        // Render the template with actual data matching the headers in the template
        let mut input_object = Map::new();
        let _ = input_object.insert(table.get_name().to_string(), table.to_json_object()?.into());
        let template_inputs = serde_json::to_value(input_object)?;
        let template = [MINIMAL_HTML_PRE, rendered_template.as_str(), MINIMAL_HTML_POST].join("");
        let script_string =
            TableScript::new_from_template(template).apply_template(&template_inputs)?;

        assert_eq!(
            script_string,
            ""
        );
        Ok(())
    }
}