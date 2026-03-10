use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{
    MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE, MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE,
    MERMAID_ER_DIAGRAM_TEMPLATE, MERMAID_FLOWCHART_LINKS_TEMPLATE,
    MERMAID_FLOWCHART_NODES_TEMPLATE, MERMAID_FLOWCHART_TEMPLATE, MERMAID_GANTT_TEMPLATE,
    MERMAID_HTML_POST, MERMAID_HTML_PRE, MERMAID_KANBAN_TEMPLATE,
    MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE, MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE,
    MERMAID_SEQUENCE_DIAGRAM_TEMPLATE, MERMAID_XYCHART_TEMPLATE, MINIMAL_CODE_TEMPLATE,
    template::{
        MINIMAL_FIGURE_TEMPLATE, MINIMAL_HTML_BODY_TEMPLATE, MINIMAL_HTML_POST, MINIMAL_HTML_PRE,
        MINIMAL_LIST_TEMPLATE, MINIMAL_TABLE_TEMPLATE,
    },
};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableJinja2Templates {
    #[value(name = "MermaidERDiagramEntities")]
    MermaidERDiagramEntitiesTemplate,
    #[value(name = "MermaidERDiagramRelations")]
    MermaidERDiagramRelationsTemplate,
    #[default]
    #[value(name = "MermaidERDiagramTemplate")]
    MermaidERDiagramTemplate,
    #[value(name = "ChunkDocuments")]
    MermaidERDiagramHTML,
    #[value(name = "MermaidFlowchartNodesTemplate")]
    MermaidFlowchartNodesTemplate,
    #[value(name = "MermaidFlowchartLinksTemplate")]
    MermaidFlowchartLinksTemplate,
    #[value(name = "MermaidFlowchartTemplate")]
    MermaidFlowchartTemplate,
    #[value(name = "MermaidFlowchartHTML")]
    MermaidFlowchartHTML,
    #[value(name = "MermaidGanttTemplate")]
    MermaidGanttTemplate,
    #[value(name = "MermaidGanttHTML")]
    MermaidGanttHTML,
    #[value(name = "MermaidKanbanTemplate")]
    MermaidKanbanTemplate,
    #[value(name = "MermaidKanbanHTML")]
    MermaidKanbanHTML,
    #[value(name = "MermaidSequenceDiagramParticipantsTemplate")]
    MermaidSequenceDiagramParticipantsTemplate,
    #[value(name = "MermaidSequenceDiagramMessagesTemplate")]
    MermaidSequenceDiagramMessagesTemplate,
    #[value(name = "MermaidSequenceDiagramTemplate")]
    MermaidSequenceDiagramTemplate,
    #[value(name = "MermaidSequenceDiagramHTML")]
    MermaidSequenceDiagramHTML,
    #[value(name = "MermaidXYChartTemplate")]
    MermaidXYChartTemplate,
    #[value(name = "MermaidXYChartHTML")]
    MermaidXYChartHTML,
    #[value(name = "MinimalHTMLBodyTemplate")]
    MinimalHTMLBodyTemplate,
    #[value(name = "MinimalHTMLBodyHTML")]
    MinimalHTMLBodyHTML,
    #[value(name = "MinimalHTMLTableTemplate")]
    MinimalHTMLTableTemplate,
    #[value(name = "MinimalHTMLTableHTML")]
    MinimalHTMLTableHTML,
    #[value(name = "MinimalHTMLListTemplate")]
    MinimalHTMLListTemplate,
    #[value(name = "MinimalHTMLListHTML")]
    MinimalHTMLListHTML,
    #[value(name = "MinimalHTMLFiguresTemplate")]
    MinimalHTMLFiguresTemplate,
    #[value(name = "MinimalHTMLFiguresHTML")]
    MinimalHTMLFiguresHTML,
    #[value(name = "MinimalHTMLCodeTemplate")]
    MinimalHTMLCodeTemplate,
    #[value(name = "MinimalHTMLCodeHTML")]
    MinimalHTMLCodeHTML,
    #[value(skip)]
    Custom(String),
}

impl AvailableJinja2Templates {
    /// Access the jinja2 template [String]
    pub fn to_template(&self) -> String {
        match self {
            Self::MermaidERDiagramEntitiesTemplate => {
                MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE.to_string()
            }
            Self::MermaidERDiagramRelationsTemplate => {
                MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE.to_string()
            }
            Self::MermaidERDiagramTemplate => MERMAID_ER_DIAGRAM_TEMPLATE.to_string(),
            Self::MermaidERDiagramHTML => [
                MERMAID_HTML_PRE,
                MERMAID_XYCHART_TEMPLATE,
                MERMAID_HTML_POST,
            ]
            .join(""),
            Self::MermaidFlowchartNodesTemplate => MERMAID_FLOWCHART_NODES_TEMPLATE.to_string(),
            Self::MermaidFlowchartLinksTemplate => MERMAID_FLOWCHART_LINKS_TEMPLATE.to_string(),
            Self::MermaidFlowchartTemplate => MERMAID_FLOWCHART_TEMPLATE.to_string(),
            Self::MermaidFlowchartHTML => [
                MERMAID_HTML_PRE,
                MERMAID_FLOWCHART_TEMPLATE,
                MERMAID_HTML_POST,
            ]
            .join(""),
            Self::MermaidGanttTemplate => MERMAID_GANTT_TEMPLATE.to_string(),
            Self::MermaidGanttHTML => {
                [MERMAID_HTML_PRE, MERMAID_GANTT_TEMPLATE, MERMAID_HTML_POST].join("")
            }
            Self::MermaidKanbanTemplate => MERMAID_KANBAN_TEMPLATE.to_string(),
            Self::MermaidKanbanHTML => {
                [MERMAID_HTML_PRE, MERMAID_KANBAN_TEMPLATE, MERMAID_HTML_POST].join("")
            }
            Self::MermaidSequenceDiagramParticipantsTemplate => {
                MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE.to_string()
            }
            Self::MermaidSequenceDiagramMessagesTemplate => {
                MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE.to_string()
            }
            Self::MermaidSequenceDiagramTemplate => MERMAID_SEQUENCE_DIAGRAM_TEMPLATE.to_string(),
            Self::MermaidSequenceDiagramHTML => [
                MERMAID_HTML_PRE,
                MERMAID_SEQUENCE_DIAGRAM_TEMPLATE,
                MERMAID_HTML_POST,
            ]
            .join(""),
            Self::MermaidXYChartTemplate => MERMAID_XYCHART_TEMPLATE.to_string(),
            Self::MermaidXYChartHTML => [
                MERMAID_HTML_PRE,
                MERMAID_XYCHART_TEMPLATE,
                MERMAID_HTML_POST,
            ]
            .join(""),
            Self::MinimalHTMLBodyTemplate => MINIMAL_HTML_BODY_TEMPLATE.to_string(),
            Self::MinimalHTMLBodyHTML => [
                MINIMAL_HTML_PRE,
                MINIMAL_HTML_BODY_TEMPLATE,
                MINIMAL_HTML_POST,
            ]
            .join(""),
            Self::MinimalHTMLTableTemplate => MINIMAL_TABLE_TEMPLATE.to_string(),
            Self::MinimalHTMLTableHTML => {
                [MINIMAL_HTML_PRE, MINIMAL_TABLE_TEMPLATE, MINIMAL_HTML_POST].join("")
            }
            Self::MinimalHTMLListTemplate => MINIMAL_LIST_TEMPLATE.to_string(),
            Self::MinimalHTMLListHTML => {
                [MINIMAL_HTML_PRE, MINIMAL_LIST_TEMPLATE, MINIMAL_HTML_POST].join("")
            }
            Self::MinimalHTMLFiguresTemplate => MINIMAL_FIGURE_TEMPLATE.to_string(),
            Self::MinimalHTMLFiguresHTML => {
                [MINIMAL_HTML_PRE, MINIMAL_FIGURE_TEMPLATE, MINIMAL_HTML_POST].join("")
            }
            Self::MinimalHTMLCodeTemplate => MINIMAL_CODE_TEMPLATE.to_string(),
            Self::MinimalHTMLCodeHTML => {
                [MINIMAL_HTML_PRE, MINIMAL_CODE_TEMPLATE, MINIMAL_HTML_POST].join("")
            }
            Self::Custom(s) => s.to_string(),
        }
    }
    /// Whether a two-stage rendering is required based on the table headers e.g., for table templates
    pub fn has_headers(&self) -> bool {
        match self {
            Self::MermaidERDiagramEntitiesTemplate
            | Self::MermaidERDiagramRelationsTemplate
            | Self::MermaidERDiagramTemplate
            | Self::MermaidERDiagramHTML
            | Self::MermaidFlowchartNodesTemplate
            | Self::MermaidFlowchartLinksTemplate
            | Self::MermaidFlowchartTemplate
            | Self::MermaidFlowchartHTML
            | Self::MermaidGanttTemplate
            | Self::MermaidGanttHTML
            | Self::MermaidKanbanTemplate
            | Self::MermaidKanbanHTML
            | Self::MermaidSequenceDiagramParticipantsTemplate
            | Self::MermaidSequenceDiagramMessagesTemplate
            | Self::MermaidSequenceDiagramTemplate
            | Self::MermaidSequenceDiagramHTML
            | Self::MermaidXYChartTemplate
            | Self::MermaidXYChartHTML
            | Self::MinimalHTMLListTemplate
            | Self::MinimalHTMLListHTML
            | Self::MinimalHTMLFiguresTemplate
            | Self::MinimalHTMLFiguresHTML
            | Self::MinimalHTMLCodeTemplate
            | Self::MinimalHTMLCodeHTML
            | Self::Custom(_) => false,
            Self::MinimalHTMLTableTemplate
            | Self::MinimalHTMLTableHTML
            | Self::MinimalHTMLBodyTemplate
            | Self::MinimalHTMLBodyHTML => true,
        }
    }
}

impl Display for AvailableJinja2Templates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MermaidERDiagramEntitiesTemplate => write!(f, "MermaidERDiagramEntitiesTemplate"),
            Self::MermaidERDiagramRelationsTemplate => {
                write!(f, "MermaidERDiagramRelationsTemplate")
            }
            Self::MermaidERDiagramTemplate => write!(f, "MermaidERDiagramTemplate"),
            Self::MermaidERDiagramHTML => write!(f, "MermaidERDiagramHTML"),
            Self::MermaidFlowchartNodesTemplate => write!(f, "MermaidFlowchartNodesTemplate"),
            Self::MermaidFlowchartLinksTemplate => write!(f, "MermaidFlowchartLinksTemplate"),
            Self::MermaidFlowchartTemplate => write!(f, "MermaidFlowchartTemplate"),
            Self::MermaidFlowchartHTML => write!(f, "MermaidFlowchartHTML"),
            Self::MermaidGanttTemplate => write!(f, "MermaidGanttTemplate"),
            Self::MermaidGanttHTML => write!(f, "MermaidGanttHTML"),
            Self::MermaidKanbanTemplate => write!(f, "MermaidKanbanTemplate"),
            Self::MermaidKanbanHTML => write!(f, "MermaidKanbanHTML"),
            Self::MermaidSequenceDiagramParticipantsTemplate => {
                write!(f, "MermaidSequenceDiagramParticipantsTemplate")
            }
            Self::MermaidSequenceDiagramMessagesTemplate => {
                write!(f, "MermaidSequenceDiagramMessagesTemplate")
            }
            Self::MermaidSequenceDiagramTemplate => write!(f, "MermaidSequenceDiagramTemplate"),
            Self::MermaidSequenceDiagramHTML => write!(f, "MermaidSequenceDiagramHTML"),
            Self::MermaidXYChartTemplate => write!(f, "MermaidXYChartTemplate"),
            Self::MermaidXYChartHTML => write!(f, "MermaidXYChartHTML"),
            Self::MinimalHTMLBodyTemplate => write!(f, "MinimalHTMLBodyTemplate"),
            Self::MinimalHTMLBodyHTML => write!(f, "MinimalHTMLBodyHTML"),
            Self::MinimalHTMLTableTemplate => write!(f, "MinimalHTMLTableTemplate"),
            Self::MinimalHTMLTableHTML => write!(f, "MinimalHTMLTableHTML"),
            Self::MinimalHTMLListTemplate => write!(f, "MinimalHTMLListTemplate"),
            Self::MinimalHTMLListHTML => write!(f, "MinimalHTMLListHTML"),
            Self::MinimalHTMLFiguresTemplate => write!(f, "MinimalHTMLFiguresTemplate"),
            Self::MinimalHTMLFiguresHTML => write!(f, "MinimalHTMLFiguresHTML"),
            Self::MinimalHTMLCodeTemplate => write!(f, "MinimalHTMLCodeTemplate"),
            Self::MinimalHTMLCodeHTML => write!(f, "MinimalHTMLCodeHTML"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}
