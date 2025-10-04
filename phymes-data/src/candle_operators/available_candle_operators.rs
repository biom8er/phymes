use std::{fmt::Display, sync::Arc};

use arrow::array::{ArrayRef, RecordBatch, StringArray};

/// General dependencies
use clap::ValueEnum;
use phymes_core::{
    session::common_traits::{BuilderTrait, MappableTrait},
    table::table_trait::{Table, TableBuilder, TableBuilderTrait},
};
use serde::{Deserialize, Serialize};

use crate::{candle_data::data_config::DataConfig, candle_operators::{
    apply_template::ApplyTemplate, chunk_documents::ChunkDocuments, data_operator::DataOperatorTrait, extract_pdf_text::ExtractPDFText, extract_tabular_data::ExtractTabularData, filter_columns_and_indices::FilterColumnsAndIndices, group_by_and_aggregate::GroupByAndAggregate, human_in_the_loop::HumanInTheLoop, join_inner::JoinInner, relative_similarity_score::RelativeSimilarityScore, select_and_cast::SelectAndCast, sort_column_and_indices::SortColumnAndIndices
}};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum AvailableCandleOperators {
    #[value(name = "RelativeSimilarityScore")]
    #[serde(alias = "relative-similarity-score")]
    RelativeSimilarityScore,
    #[value(name = "SortColumnAndIndices")]
    #[serde(alias = "sort-column-and-indices")]
    SortColumnAndIndices,
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
}

impl Default for AvailableCandleOperators {
    fn default() -> Self {
        Self::RelativeSimilarityScore
    }
}

impl Display for AvailableCandleOperators {    
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RelativeSimilarityScore => write!(f, "{}", RelativeSimilarityScore::get_static_name()),
            Self::SortColumnAndIndices => write!(f, "{}", SortColumnAndIndices::get_static_name()),
            Self::HumanInTheLoop => write!(f, "{}", HumanInTheLoop::get_static_name()),
            Self::ChunkDocuments => write!(f, "{}", ChunkDocuments::get_static_name()),
            Self::JoinInner => write!(f, "{}", JoinInner::get_static_name()),
            Self::ExtractPDFText => write!(f, "{}", ExtractPDFText::get_static_name()),
            Self::GroupByAndAggregate => write!(f, "{}", GroupByAndAggregate::get_static_name()),
            Self::FilterColumnsAndIndices => write!(f, "{}", FilterColumnsAndIndices::get_static_name()),
            Self::ExtractTabularData => write!(f, "{}", ExtractTabularData::get_static_name()),
            Self::SelectAndCast => write!(f, "{}", SelectAndCast::get_static_name()),
            Self::ApplyTemplate => write!(f, "{}", ApplyTemplate::get_static_name()),
        }
    }
}

impl AvailableCandleOperators {
    /// Wrapper to return the JSON schema SortColumnAndIndices
    pub fn get_json_tool_schema(&self) -> String {
        match self {
            Self::RelativeSimilarityScore => RelativeSimilarityScore::get_json_tool_schema(),
            Self::SortColumnAndIndices => SortColumnAndIndices::get_json_tool_schema(),
            Self::HumanInTheLoop => HumanInTheLoop::get_json_tool_schema(),
            Self::ChunkDocuments => ChunkDocuments::get_json_tool_schema(),
            Self::JoinInner => JoinInner::get_json_tool_schema(),
            Self::ExtractPDFText => ExtractPDFText::get_json_tool_schema(),
            Self::GroupByAndAggregate => GroupByAndAggregate::get_json_tool_schema(),
            Self::FilterColumnsAndIndices => FilterColumnsAndIndices::get_json_tool_schema(),
            Self::ExtractTabularData => ExtractTabularData::get_json_tool_schema(),
            Self::SelectAndCast => SelectAndCast::get_json_tool_schema(),
            Self::ApplyTemplate => ApplyTemplate::get_json_tool_schema(),
        }
    }

    /// Build the actual operator
    #[allow(clippy::too_many_arguments)]
    pub fn build(&self, config: &DataConfig) -> Box<dyn DataOperatorTrait> {
        match self {
            Self::RelativeSimilarityScore => Box::new(RelativeSimilarityScore::new(config)),
            Self::SortColumnAndIndices => Box::new(SortColumnAndIndices::new(config)),
            Self::HumanInTheLoop => Box::new(HumanInTheLoop::new(config)),
            Self::ChunkDocuments => Box::new(ChunkDocuments::new(config)),
            Self::JoinInner => Box::new(JoinInner::new(config)),
            Self::ExtractPDFText => Box::new(ExtractPDFText::new(config)),
            Self::GroupByAndAggregate => Box::new(GroupByAndAggregate::new(config)),
            Self::FilterColumnsAndIndices => Box::new(FilterColumnsAndIndices::new(config)),
            Self::ExtractTabularData => Box::new(ExtractTabularData::new(config)),
            Self::SelectAndCast => Box::new(SelectAndCast::new(config)),
            Self::ApplyTemplate => Box::new(ApplyTemplate::new(config)),
        }
    }
}

pub fn convert_destinations_to_tools(name: &str, destinations: &[String]) -> Option<Table> {
    let mut tool_id_vec = Vec::new();
    let mut tool_vec = Vec::new();
    for destination in destinations.iter() {
        if let Ok(ops) = AvailableCandleOperators::from_str(destination, false) {
            tool_id_vec.push(ops.to_string());
            tool_vec.push(ops.get_json_tool_schema());
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
    use phymes_core::table::table_trait::TableTrait;

    use super::*;

    #[test]
    fn test_convert_destinations_to_tools_all() {
        let result = convert_destinations_to_tools(
            "test",
            &[
                "RelativeSimilarityScore".to_string(),
                "SortColumnAndIndices".to_string(),
                "ChunkDocuments".to_string(),
                "JoinInner".to_string(),
                "HumanInTheLoop".to_string(),
                "GroupByAndAggregate".to_string(),
                "FilterColumnsAndIndices".to_string(),
                "ExtractTabularData".to_string(),
                "SelectAndCast".to_string(),
                "ApplyTemplate".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(
            result.get_column_as_vec_str("tool_id"),
            &[
                "RelativeSimilarityScore",
                "SortColumnAndIndices",
                "ChunkDocuments",
                "JoinInner",
                "HumanInTheLoop",
                "GroupByAndAggregate",
                "FilterColumnsAndIndices",
                "ExtractTabularData",
                "SelectAndCast",
                "ApplyTemplate",
            ]
        );
    }

    #[test]
    fn test_convert_destinations_to_tools_missing() {
        let result = convert_destinations_to_tools("test", &["missing".to_string()]);
        assert!(result.is_none());
    }
}
