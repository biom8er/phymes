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
