mod available_jinja2_templates;
mod mermaid_er_diagram;
mod mermaid_flowchart;
mod mermaid_gantt;
mod mermaid_html;
mod mermaid_kanban;
mod mermaid_sequence_diagram;
mod mermaid_xychart;
mod minimal_html;
mod minimal_table;

pub use available_jinja2_templates::AvailableJinja2Templates;

pub use mermaid_er_diagram::{
    MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE, MERMAID_ER_DIAGRAM_INPUT,
    MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE, MERMAID_ER_DIAGRAM_TABLE_EXPRESSION,
    MERMAID_ER_DIAGRAM_TEMPLATE,
};
pub use mermaid_flowchart::{
    MERMAID_FLOWCHART_INPUT, MERMAID_FLOWCHART_LINKS_TEMPLATE, MERMAID_FLOWCHART_NODES_TEMPLATE,
    MERMAID_FLOWCHART_TABLE_EXPRESSION, MERMAID_FLOWCHART_TEMPLATE,
};
pub use mermaid_gantt::{
    MERMAID_GANTT_INPUT, MERMAID_GANTT_TABLE_EXPRESSION, MERMAID_GANTT_TEMPLATE,
};
pub use mermaid_html::{MERMAID_HTML_POST, MERMAID_HTML_PRE};
pub use mermaid_kanban::{MERMAID_KANBAN_TABLE_EXPRESSION, MERMAID_KANBAN_TEMPLATE};
pub use mermaid_sequence_diagram::{
    MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE, MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE,
    MERMAID_SEQUENCE_DIAGRAM_TABLE_EXPRESSION, MERMAID_SEQUENCE_DIAGRAM_TEMPLATE,
};
pub use mermaid_xychart::{
    MERMAID_XYCHART_INPUT, MERMAID_XYCHART_TABLE_EXPRESSION, MERMAID_XYCHART_TEMPLATE,
};
