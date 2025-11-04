use std::{fmt::Display, sync::Arc};

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use clap::ValueEnum;
use phymes_core::{BuilderTrait, MappableTrait, Table, TableBuilder, TableBuilderTrait};
use serde::{Deserialize, Serialize};

use crate::{
    ToolTrait,
    candle_data::DataConfig,
    candle_operators::{
        ApplyTemplate, ChunkDocuments, DataOperatorTrait, ExtractPDFText, ExtractTabularData,
        FilterColumnsAndIndices, FromTasksToParticipants, FromTracesToMessages,
        GroupByAndAggregate, HumanInTheLoop, JoinInner, NormalizeTime, Pivot, SelectAndCast,
        SortColumnAndIndices, VectorDistance,
    },
};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableCandleOperators {
    #[value(name = "VectorDistance")]
    #[serde(alias = "vector-distance")]
    VectorDistance,
    #[value(name = "SortColumnAndIndices")]
    #[serde(alias = "sort-column-and-indices")]
    SortColumnAndIndices,
    #[default]
    #[value(name = "HumanInTheLoop")]
    #[serde(alias = "human-in-the-loop")]
    HumanInTheLoop,
    #[value(name = "ChunkDocuments")]
    #[serde(alias = "chunk-documents")]
    ChunkDocuments,
    #[value(name = "JoinInner")]
    #[serde(alias = "join-inner")]
    JoinInner,
    #[value(name = "ExtractPDFText")]
    #[serde(alias = "extract-pdf-text")]
    ExtractPDFText,
    #[value(name = "GroupByAndAggregate")]
    #[serde(alias = "group-by-and-aggregate")]
    GroupByAndAggregate,
    #[value(name = "FilterColumnsAndIndices")]
    #[serde(alias = "filter-columns-and-indices")]
    FilterColumnsAndIndices,
    #[value(name = "ExtractTabularData")]
    #[serde(alias = "extract-tabular-data")]
    ExtractTabularData,
    #[value(name = "SelectAndCast")]
    #[serde(alias = "select-and-cast")]
    SelectAndCast,
    #[value(name = "ApplyTemplate")]
    #[serde(alias = "apply-template")]
    ApplyTemplate,
    #[value(name = "Pivot")]
    #[serde(alias = "pivot")]
    Pivot,
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

impl Display for AvailableCandleOperators {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VectorDistance => write!(f, "{}", VectorDistance::get_static_name()),
            Self::SortColumnAndIndices => write!(f, "{}", SortColumnAndIndices::get_static_name()),
            Self::HumanInTheLoop => write!(f, "{}", HumanInTheLoop::get_static_name()),
            Self::ChunkDocuments => write!(f, "{}", ChunkDocuments::get_static_name()),
            Self::JoinInner => write!(f, "{}", JoinInner::get_static_name()),
            Self::ExtractPDFText => write!(f, "{}", ExtractPDFText::get_static_name()),
            Self::GroupByAndAggregate => write!(f, "{}", GroupByAndAggregate::get_static_name()),
            Self::FilterColumnsAndIndices => {
                write!(f, "{}", FilterColumnsAndIndices::get_static_name())
            }
            Self::ExtractTabularData => write!(f, "{}", ExtractTabularData::get_static_name()),
            Self::SelectAndCast => write!(f, "{}", SelectAndCast::get_static_name()),
            Self::ApplyTemplate => write!(f, "{}", ApplyTemplate::get_static_name()),
            Self::Pivot => write!(f, "{}", Pivot::get_static_name()),
            Self::NormalizeTime => write!(f, "{}", NormalizeTime::get_static_name()),
            Self::FromTasksToParticipants => {
                write!(f, "{}", FromTasksToParticipants::get_static_name())
            }
            Self::FromTracesToMessages => write!(f, "{}", FromTracesToMessages::get_static_name()),
        }
    }
}

impl ToolTrait for AvailableCandleOperators {
    fn to_json_tool_schema(&self) -> String {
        match self {
            Self::VectorDistance => VectorDistance::default().to_json_tool_schema(),
            Self::SortColumnAndIndices => SortColumnAndIndices::default().to_json_tool_schema(),
            Self::HumanInTheLoop => HumanInTheLoop.to_json_tool_schema(),
            Self::ChunkDocuments => ChunkDocuments::default().to_json_tool_schema(),
            Self::JoinInner => JoinInner::default().to_json_tool_schema(),
            Self::ExtractPDFText => ExtractPDFText::default().to_json_tool_schema(),
            Self::GroupByAndAggregate => GroupByAndAggregate::default().to_json_tool_schema(),
            Self::FilterColumnsAndIndices => {
                FilterColumnsAndIndices::default().to_json_tool_schema()
            }
            Self::ExtractTabularData => ExtractTabularData::default().to_json_tool_schema(),
            Self::SelectAndCast => SelectAndCast::default().to_json_tool_schema(),
            Self::ApplyTemplate => ApplyTemplate::default().to_json_tool_schema(),
            Self::Pivot => Pivot::default().to_json_tool_schema(),
            Self::NormalizeTime => NormalizeTime::default().to_json_tool_schema(),
            Self::FromTasksToParticipants => String::new(),
            Self::FromTracesToMessages => String::new(),
        }
    }
    fn get_description(&self) -> String {
        match self {
            Self::VectorDistance => VectorDistance::default().get_description(),
            Self::SortColumnAndIndices => SortColumnAndIndices::default().get_description(),
            Self::HumanInTheLoop => HumanInTheLoop.get_description(),
            Self::ChunkDocuments => ChunkDocuments::default().get_description(),
            Self::JoinInner => JoinInner::default().get_description(),
            Self::ExtractPDFText => ExtractPDFText::default().get_description(),
            Self::GroupByAndAggregate => GroupByAndAggregate::default().get_description(),
            Self::FilterColumnsAndIndices => FilterColumnsAndIndices::default().get_description(),
            Self::ExtractTabularData => ExtractTabularData::default().get_description(),
            Self::SelectAndCast => SelectAndCast::default().get_description(),
            Self::ApplyTemplate => ApplyTemplate::default().get_description(),
            Self::Pivot => Pivot::default().get_description(),
            Self::NormalizeTime => NormalizeTime::default().get_description(),
            Self::FromTasksToParticipants => String::new(),
            Self::FromTracesToMessages => String::new(),
        }
    }
}

impl AvailableCandleOperators {
    pub fn all_varient_names() -> Vec<String> {
        let processor_names = [
            Self::VectorDistance.to_string(),
            Self::SortColumnAndIndices.to_string(),
            Self::HumanInTheLoop.to_string(),
            Self::ChunkDocuments.to_string(),
            Self::JoinInner.to_string(),
            Self::ExtractPDFText.to_string(),
            Self::GroupByAndAggregate.to_string(),
            Self::FilterColumnsAndIndices.to_string(),
            Self::ExtractTabularData.to_string(),
            Self::SelectAndCast.to_string(),
            Self::Pivot.to_string(),
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
            Self::SortColumnAndIndices => Ok(Box::new(SortColumnAndIndices::new(config)?)),
            Self::HumanInTheLoop => Ok(Box::new(HumanInTheLoop::new(config)?)),
            Self::ChunkDocuments => Ok(Box::new(ChunkDocuments::new(config)?)),
            Self::JoinInner => Ok(Box::new(JoinInner::new(config)?)),
            Self::ExtractPDFText => Ok(Box::new(ExtractPDFText::new(config)?)),
            Self::GroupByAndAggregate => Ok(Box::new(GroupByAndAggregate::new(config)?)),
            Self::FilterColumnsAndIndices => Ok(Box::new(FilterColumnsAndIndices::new(config)?)),
            Self::ExtractTabularData => Ok(Box::new(ExtractTabularData::new(config)?)),
            Self::SelectAndCast => Ok(Box::new(SelectAndCast::new(config)?)),
            Self::ApplyTemplate => Ok(Box::new(ApplyTemplate::new(config)?)),
            Self::Pivot => Ok(Box::new(Pivot::new(config)?)),
            Self::NormalizeTime => Ok(Box::new(NormalizeTime::new(config)?)),
            Self::FromTasksToParticipants => Ok(Box::new(FromTasksToParticipants::new(config)?)),
            Self::FromTracesToMessages => Ok(Box::new(FromTracesToMessages::new(config)?)),
        }
    }
}

pub fn convert_destinations_to_tools(name: &str, destinations: &[String]) -> Option<Table> {
    let mut tool_id_vec = Vec::new();
    let mut tool_vec = Vec::new();
    for destination in destinations.iter() {
        if let Ok(ops) = AvailableCandleOperators::from_str(destination, false) {
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
        let table = TableBuilder::new()
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
    use phymes_core::TableTrait;

    use super::*;

    #[test]
    fn test_convert_destinations_to_tools_all() {
        let result = convert_destinations_to_tools(
            "test",
            &[
                "VectorDistance".to_string(),
                "SortColumnAndIndices".to_string(),
                "ChunkDocuments".to_string(),
                "JoinInner".to_string(),
                "HumanInTheLoop".to_string(),
                "GroupByAndAggregate".to_string(),
                "FilterColumnsAndIndices".to_string(),
                "ExtractTabularData".to_string(),
                "SelectAndCast".to_string(),
                "ApplyTemplate".to_string(),
                "Pivot".to_string(),
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
                "SortColumnAndIndices",
                "ChunkDocuments",
                "JoinInner",
                "HumanInTheLoop",
                "GroupByAndAggregate",
                "FilterColumnsAndIndices",
                "ExtractTabularData",
                "SelectAndCast",
                "ApplyTemplate",
                "Pivot",
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
