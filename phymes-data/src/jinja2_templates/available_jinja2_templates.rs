use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE, MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE, MERMAID_ER_DIAGRAM_TEMPLATE, MERMAID_FLOWCHART_LINKS_TEMPLATE, MERMAID_FLOWCHART_NODES_TEMPLATE, MERMAID_FLOWCHART_TEMPLATE, MERMAID_GANTT_TEMPLATE, MERMAID_HTML_POST, MERMAID_HTML_PRE, MERMAID_KANBAN_TEMPLATE, MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE, MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE, MERMAID_SEQUENCE_DIAGRAM_TEMPLATE, MERMAID_XYCHART_TEMPLATE};

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
    #[value(skip)]
    Custom(String), 
}

impl AvailableJinja2Templates {
    /// Access the jinja2 template [String]
    pub fn to_template(&self) -> String{
        match self {
            Self::MermaidERDiagramEntitiesTemplate => MERMAID_ER_DIAGRAM_ENTITIES_TEMPLATE.to_string(),
            Self::MermaidERDiagramRelationsTemplate => MERMAID_ER_DIAGRAM_RELATIONS_TEMPLATE.to_string(),
            Self::MermaidERDiagramTemplate => MERMAID_ER_DIAGRAM_TEMPLATE.to_string(),
            Self::MermaidERDiagramHTML => [MERMAID_HTML_PRE, MERMAID_XYCHART_TEMPLATE, MERMAID_HTML_POST].join(""),
            Self::MermaidFlowchartNodesTemplate => MERMAID_FLOWCHART_NODES_TEMPLATE.to_string(),
            Self::MermaidFlowchartLinksTemplate => MERMAID_FLOWCHART_LINKS_TEMPLATE.to_string(),
            Self::MermaidFlowchartTemplate => MERMAID_FLOWCHART_TEMPLATE.to_string(),
            Self::MermaidFlowchartHTML => [MERMAID_HTML_PRE, MERMAID_FLOWCHART_TEMPLATE, MERMAID_HTML_POST].join(""),
            Self::MermaidGanttTemplate => MERMAID_GANTT_TEMPLATE.to_string(),
            Self::MermaidGanttHTML => [MERMAID_HTML_PRE, MERMAID_GANTT_TEMPLATE, MERMAID_HTML_POST].join(""),
            Self::MermaidKanbanTemplate => MERMAID_KANBAN_TEMPLATE.to_string(),
            Self::MermaidKanbanHTML => [MERMAID_HTML_PRE, MERMAID_KANBAN_TEMPLATE, MERMAID_HTML_POST].join(""),
            Self::MermaidSequenceDiagramParticipantsTemplate => MERMAID_SEQUENCE_DIAGRAM_PARTICIPANTS_TEMPLATE.to_string(),
            Self::MermaidSequenceDiagramMessagesTemplate => MERMAID_SEQUENCE_DIAGRAM_MESSAGES_TEMPLATE.to_string(),
            Self::MermaidSequenceDiagramTemplate => MERMAID_SEQUENCE_DIAGRAM_TEMPLATE.to_string(),
            Self::MermaidSequenceDiagramHTML => [MERMAID_HTML_PRE, MERMAID_SEQUENCE_DIAGRAM_TEMPLATE, MERMAID_HTML_POST].join(""),
            Self::MermaidXYChartTemplate => MERMAID_XYCHART_TEMPLATE.to_string(),
            Self::MermaidXYChartHTML => [MERMAID_HTML_PRE, MERMAID_XYCHART_TEMPLATE, MERMAID_HTML_POST].join(""),
            Self::Custom(s) => s.to_string(),
        }
    }
}

impl Display for AvailableJinja2Templates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MermaidERDiagramEntitiesTemplate => write!(f, "MermaidERDiagramEntitiesTemplate"),
            Self::MermaidERDiagramRelationsTemplate => write!(f, "MermaidERDiagramRelationsTemplate"),
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
            Self::MermaidSequenceDiagramParticipantsTemplate => write!(f, "MermaidSequenceDiagramParticipantsTemplate"),
            Self::MermaidSequenceDiagramMessagesTemplate => write!(f, "MermaidSequenceDiagramMessagesTemplate"),
            Self::MermaidSequenceDiagramTemplate => write!(f, "MermaidSequenceDiagramTemplate"),
            Self::MermaidSequenceDiagramHTML => write!(f, "MermaidSequenceDiagramHTML"),
            Self::MermaidXYChartTemplate => write!(f, "MermaidXYChartTemplate"),
            Self::MermaidXYChartHTML => write!(f, "MermaidXYChartHTML"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}