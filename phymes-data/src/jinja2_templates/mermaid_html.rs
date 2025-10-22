/// HTML template for creating mermaid.js diagrams
///
/// The template is split into a `pre` and `post` static str
///   whereby the mermaid.js diagram string should be inserted in between
///
///
/// ```html
/// <!-- MERMAID_HTML_PRE -->
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
/// <!-- Insert mermaid.js diagram string here, e.g.,: -->
///             xychart
///                 title "Sales Revenue"
///                 x-axis [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
///                 y-axis "Revenue (in $)" 4000 --> 11000
///                 line [5000, 6000, 7500, 8200, 9500, 10500, 11000, 10200, 9200, 8500, 7000, 6000]
///
/// <!-- MERMAID_HTML_POST -->
///
///     </pre>
///     <script type="module">
///         import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
///         mermaid.initialize({theme: "dark", startOnLoad: true });
///     </script>
///   </body>
/// </html>
/// ```
/// Part 1 of the Mermaid.js html jinja2 template
pub static MERMAID_HTML_PRE: &str = r#"<!DOCTYPE html>
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
  <body>
    <pre class="mermaid">"#;

/// Part 2 of the Mermaid.js html jinja2 template
pub static MERMAID_HTML_POST: &str = r#"
    </pre>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({theme: "dark", startOnLoad: true });
    </script>
  </body>
</html>"#;
