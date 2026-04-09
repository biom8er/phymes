use std::{fmt::Display, sync::Arc};

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use clap::ValueEnum;
use phymes_core::{BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait};
use serde::{Deserialize, Serialize};

use crate::{
    Diff, ExtractXML, PackTabular, Patch, ToolTrait,
    operators::{
        ApplyTemplate, ChunkDocuments, ExtractPDF, ExtractTabular, Filter, FromTasksToParticipants,
        FromTracesToMessages, GroupBy, HumanInTheLoop, Join, Melt, NormalizeTime, Pivot, Select,
        Sort, VectorDistance,
    },
    tensor::{DataConfig, DataOperatorTrait},
};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableOperators {
    #[value(name = "VectorDistance")]
    #[serde(alias = "vector-distance")]
    VectorDistance,
    #[value(name = "Sort")]
    #[serde(alias = "sort")]
    Sort,
    #[default]
    #[value(name = "HumanInTheLoop")]
    #[serde(alias = "human-in-the-loop")]
    HumanInTheLoop,
    #[value(name = "ChunkDocuments")]
    #[serde(alias = "chunk-documents")]
    ChunkDocuments,
    #[value(name = "Join")]
    #[serde(alias = "join")]
    Join,
    #[value(name = "ExtractPDF")]
    #[serde(alias = "extract-pdf")]
    ExtractPDF,
    #[value(name = "GroupBy")]
    #[serde(alias = "group-by")]
    GroupBy,
    #[value(name = "Filter")]
    #[serde(alias = "filter")]
    Filter,
    #[value(name = "ExtractTabular")]
    #[serde(alias = "extract-tabular")]
    ExtractTabular,
    #[value(name = "PackTabular")]
    #[serde(alias = "pack-tabular")]
    PackTabular,
    #[value(name = "Select")]
    #[serde(alias = "select")]
    Select,
    #[value(name = "ApplyTemplate")]
    #[serde(alias = "apply-template")]
    ApplyTemplate,
    #[value(name = "Pivot")]
    #[serde(alias = "pivot")]
    Pivot,
    #[value(name = "ExtractXML")]
    #[serde(alias = "extract-xml")]
    ExtractXML,
    #[value(name = "Melt")]
    #[serde(alias = "melt")]
    Melt,
    #[value(name = "Patch")]
    #[serde(alias = "patch")]
    Patch,
    #[value(name = "Diff")]
    #[serde(alias = "diff")]
    Diff,
    #[value(name = "NormalizeTime")]
    #[serde(alias = "NormalizeTime")]
    NormalizeTime,
    #[value(name = "FromTasksToParticipants")]
    #[serde(alias = "FromTasksToParticipants")]
    FromTasksToParticipants,
    #[value(name = "FromTracesToMessages")]
    #[serde(alias = "FromTracesToMessages")]
    FromTracesToMessages,
}

impl Display for AvailableOperators {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VectorDistance => write!(f, "{}", VectorDistance::get_static_name()),
            Self::Sort => write!(f, "{}", Sort::get_static_name()),
            Self::HumanInTheLoop => write!(f, "{}", HumanInTheLoop::get_static_name()),
            Self::ChunkDocuments => write!(f, "{}", ChunkDocuments::get_static_name()),
            Self::Join => write!(f, "{}", Join::get_static_name()),
            Self::ExtractPDF => write!(f, "{}", ExtractPDF::get_static_name()),
            Self::GroupBy => write!(f, "{}", GroupBy::get_static_name()),
            Self::Filter => {
                write!(f, "{}", Filter::get_static_name())
            }
            Self::ExtractTabular => write!(f, "{}", ExtractTabular::get_static_name()),
            Self::PackTabular => write!(f, "{}", PackTabular::get_static_name()),
            Self::Select => write!(f, "{}", Select::get_static_name()),
            Self::ApplyTemplate => write!(f, "{}", ApplyTemplate::get_static_name()),
            Self::Pivot => write!(f, "{}", Pivot::get_static_name()),
            Self::ExtractXML => write!(f, "{}", ExtractXML::get_static_name()),
            Self::Melt => write!(f, "{}", Melt::get_static_name()),
            Self::Patch => write!(f, "{}", Patch::get_static_name()),
            Self::Diff => write!(f, "{}", Diff::get_static_name()),
            Self::NormalizeTime => write!(f, "{}", NormalizeTime::get_static_name()),
            Self::FromTasksToParticipants => {
                write!(f, "{}", FromTasksToParticipants::get_static_name())
            }
            Self::FromTracesToMessages => write!(f, "{}", FromTracesToMessages::get_static_name()),
        }
    }
}

impl ToolTrait for AvailableOperators {
    fn to_json_tool_schema(&self) -> String {
        match self {
            Self::VectorDistance => VectorDistance::default().to_json_tool_schema(),
            Self::Sort => Sort::default().to_json_tool_schema(),
            Self::HumanInTheLoop => HumanInTheLoop.to_json_tool_schema(),
            Self::ChunkDocuments => ChunkDocuments::default().to_json_tool_schema(),
            Self::Join => Join::default().to_json_tool_schema(),
            Self::ExtractPDF => ExtractPDF::default().to_json_tool_schema(),
            Self::GroupBy => GroupBy::default().to_json_tool_schema(),
            Self::Filter => Filter::default().to_json_tool_schema(),
            Self::ExtractTabular => ExtractTabular::default().to_json_tool_schema(),
            Self::PackTabular => PackTabular::default().to_json_tool_schema(),
            Self::Select => Select::default().to_json_tool_schema(),
            Self::ApplyTemplate => ApplyTemplate::default().to_json_tool_schema(),
            Self::Pivot => Pivot::default().to_json_tool_schema(),
            Self::ExtractXML => ExtractXML::default().to_json_tool_schema(),
            Self::Melt => Melt::default().to_json_tool_schema(),
            Self::Patch => Patch::default().to_json_tool_schema(),
            Self::Diff => Diff::default().to_json_tool_schema(),
            Self::NormalizeTime => NormalizeTime::default().to_json_tool_schema(),
            Self::FromTasksToParticipants => String::new(),
            Self::FromTracesToMessages => String::new(),
        }
    }
    fn get_description(&self) -> String {
        match self {
            Self::VectorDistance => VectorDistance::default().get_description(),
            Self::Sort => Sort::default().get_description(),
            Self::HumanInTheLoop => HumanInTheLoop.get_description(),
            Self::ChunkDocuments => ChunkDocuments::default().get_description(),
            Self::Join => Join::default().get_description(),
            Self::ExtractPDF => ExtractPDF::default().get_description(),
            Self::GroupBy => GroupBy::default().get_description(),
            Self::Filter => Filter::default().get_description(),
            Self::ExtractTabular => ExtractTabular::default().get_description(),
            Self::PackTabular => PackTabular::default().get_description(),
            Self::Select => Select::default().get_description(),
            Self::ApplyTemplate => ApplyTemplate::default().get_description(),
            Self::Pivot => Pivot::default().get_description(),
            Self::ExtractXML => ExtractXML::default().get_description(),
            Self::Melt => Melt::default().get_description(),
            Self::Patch => Patch::default().get_description(),
            Self::Diff => Diff::default().get_description(),
            Self::NormalizeTime => NormalizeTime::default().get_description(),
            Self::FromTasksToParticipants => String::new(),
            Self::FromTracesToMessages => String::new(),
        }
    }
}

impl AvailableOperators {
    pub fn all_varient_names() -> Vec<String> {
        let processor_names = [
            Self::VectorDistance.to_string(),
            Self::Sort.to_string(),
            Self::HumanInTheLoop.to_string(),
            Self::ChunkDocuments.to_string(),
            Self::Join.to_string(),
            Self::ExtractPDF.to_string(),
            Self::GroupBy.to_string(),
            Self::Filter.to_string(),
            Self::ExtractTabular.to_string(),
            Self::PackTabular.to_string(),
            Self::Select.to_string(),
            Self::Pivot.to_string(),
            Self::ExtractXML.to_string(),
            Self::Melt.to_string(),
            Self::Patch.to_string(),
            Self::Diff.to_string(),
            Self::NormalizeTime.to_string(),
        ];
        processor_names
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }
    /// Build the actual operator
    pub fn build(&self, config: &DataConfig) -> Result<Box<dyn DataOperatorTrait>> {
        match self {
            Self::VectorDistance => Ok(Box::new(VectorDistance::new(config)?)),
            Self::Sort => Ok(Box::new(Sort::new(config)?)),
            Self::HumanInTheLoop => Ok(Box::new(HumanInTheLoop::new(config)?)),
            Self::ChunkDocuments => Ok(Box::new(ChunkDocuments::new(config)?)),
            Self::Join => Ok(Box::new(Join::new(config)?)),
            Self::ExtractPDF => Ok(Box::new(ExtractPDF::new(config)?)),
            Self::GroupBy => Ok(Box::new(GroupBy::new(config)?)),
            Self::Filter => Ok(Box::new(Filter::new(config)?)),
            Self::ExtractTabular => Ok(Box::new(ExtractTabular::new(config)?)),
            Self::PackTabular => Ok(Box::new(PackTabular::new(config)?)),
            Self::Select => Ok(Box::new(Select::new(config)?)),
            Self::ApplyTemplate => Ok(Box::new(ApplyTemplate::new(config)?)),
            Self::Pivot => Ok(Box::new(Pivot::new(config)?)),
            Self::ExtractXML => Ok(Box::new(ExtractXML::new(config)?)),
            Self::Melt => Ok(Box::new(Melt::new(config)?)),
            Self::Patch => Ok(Box::new(Patch::new(config)?)),
            Self::Diff => Ok(Box::new(Diff::new(config)?)),
            Self::NormalizeTime => Ok(Box::new(NormalizeTime::new(config)?)),
            Self::FromTasksToParticipants => Ok(Box::new(FromTasksToParticipants::new(config)?)),
            Self::FromTracesToMessages => Ok(Box::new(FromTracesToMessages::new(config)?)),
        }
    }
}

pub fn convert_destinations_to_tools(name: &str, destinations: &[String]) -> Option<Subject> {
    let mut tool_id_vec = Vec::new();
    let mut tool_vec = Vec::new();
    for destination in destinations.iter() {
        if let Ok(ops) = AvailableOperators::from_str(destination, false) {
            tool_id_vec.push(ops.to_string());
            tool_vec.push(ops.to_json_tool_schema());
        }
    }
    if tool_id_vec.is_empty() {
        None
    } else {
        let tool_id: ArrayRef = Arc::new(StringArray::from(tool_id_vec));
        let tool: ArrayRef = Arc::new(StringArray::from(tool_vec));
        let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_id), ("tool", tool)]).unwrap();
        let table = SubjectBuilder::new()
            .with_name(name)
            .with_record_batches(vec![batch])
            .unwrap()
            .build()
            .unwrap();
        Some(table)
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::SubjectTrait;

    use super::*;

    #[test]
    fn test_convert_destinations_to_tools_all() {
        let result = convert_destinations_to_tools(
            "test",
            &[
                "VectorDistance".to_string(),
                "Sort".to_string(),
                "ChunkDocuments".to_string(),
                "Join".to_string(),
                "HumanInTheLoop".to_string(),
                "GroupBy".to_string(),
                "Filter".to_string(),
                "ExtractTabular".to_string(),
                "PackTabular".to_string(),
                "Select".to_string(),
                "ApplyTemplate".to_string(),
                "Pivot".to_string(),
                "ExtractXML".to_string(),
                "Melt".to_string(),
                "Patch".to_string(),
                "Diff".to_string(),
                "NormalizeTime".to_string(),
                "FromTasksToParticipants".to_string(),
                "FromTracesToMessages".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(
            result.get_column_as_vec_str("tool_id"),
            &[
                "VectorDistance",
                "Sort",
                "ChunkDocuments",
                "Join",
                "HumanInTheLoop",
                "GroupBy",
                "Filter",
                "ExtractTabular",
                "PackTabular",
                "Select",
                "ApplyTemplate",
                "Pivot",
                "ExtractXML",
                "Melt",
                "Patch",
                "Diff",
                "NormalizeTime",
                "FromTasksToParticipants",
                "FromTracesToMessages",
            ]
        );
    }

    #[test]
    fn test_convert_destinations_to_tools_missing() {
        let result = convert_destinations_to_tools("test", &["missing".to_string()]);
        assert!(result.is_none());
    }
}
