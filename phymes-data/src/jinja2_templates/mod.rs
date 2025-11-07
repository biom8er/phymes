/// The entry point for the template headers
pub static TEMPLATE_HEADER_EXPRESSION: &str = "headers";

/// The entry point for the template rows
pub static TEMPLATE_TABLE_EXPRESSION: &str = "rows";

mod available_jinja2_templates;
mod mermaid_er_diagram;
mod mermaid_flowchart;
mod mermaid_gantt;
mod mermaid_html;
mod mermaid_kanban;
mod mermaid_sequence_diagram;
mod mermaid_xychart;
mod minimal_html;
mod minimal_figures;
mod minimal_list;
mod minimal_table;

pub use available_jinja2_templates::AvailableJinja2Templates;

pub use mermaid_er_diagram::{
    MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE, MERMAID_ER_DIAGRAM_INPUT,
    MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE, MERMAID_ER_DIAGRAM_TEMPLATE,
};
pub use mermaid_flowchart::{
    MERMAID_FLOWCHART_INPUT, MERMAID_FLOWCHART_LINKS_TEMPLATE, MERMAID_FLOWCHART_NODES_TEMPLATE,
    MERMAID_FLOWCHART_TEMPLATE,
};
pub use mermaid_gantt::{
    MERMAID_GANTT_INPUT, MERMAID_GANTT_TEMPLATE,
};
pub use mermaid_html::{MERMAID_HTML_POST, MERMAID_HTML_PRE};
pub use mermaid_kanban::MERMAID_KANBAN_TEMPLATE;
pub use mermaid_sequence_diagram::{
    MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE, MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE,
    MERMAID_SEQUENCE_DIAGRAM_TEMPLATE,
};
pub use mermaid_xychart::{
    MERMAID_XYCHART_INPUT, MERMAID_XYCHART_TEMPLATE,
};
pub(crate) use minimal_html::test_minimal_html;
pub use minimal_html::{MINIMAL_HTML_BODY_TEMPLATE, MINIMAL_HTML_POST, MINIMAL_HTML_PRE};
pub use minimal_table::{MINIMAL_TABLE_INPUT, MINIMAL_TABLE_TEMPLATE};
pub use minimal_list::{MINIMAL_LIST_INPUT, MINIMAL_LIST_TEMPLATE};
pub use minimal_figures::MINIMAL_FIGURE_TEMPLATE;