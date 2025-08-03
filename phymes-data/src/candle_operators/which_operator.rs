use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, StringArray};

/// General dependencies
use clap::ValueEnum;
use phymes_core::{
    session::common_traits::BuilderTrait,
    table::arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait},
};
use serde::{Deserialize, Serialize};

use crate::candle_operators::{
    chunk_documents::ChunkDocuments, data_operator::DataOperatorTrait,
    extract_pdf_text::ExtractPDFText, group_by_and_aggregate::GroupByAndAggregate,
    human_in_the_loop::HumanInTheLoop, join_inner::JoinInner,
    relative_similarity_score::RelativeSimilarityScore,
    sort_scores_and_indices::SortScoresAndIndices,
};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum WhichCandleOperator {
    #[value(name = "relative-similarity-score")]
    #[serde(alias = "relative-similarity-score")]
    RelativeSimilarityScore,
    #[value(name = "sort-scores-and-indices")]
    #[serde(alias = "sort-scores-and-indices")]
    SortScoresAndIndices,
    #[value(name = "human-in-the-loop")]
    #[serde(alias = "human-in-the-loop")]
    HumanInTheLoop,
    #[value(name = "chunk-documents")]
    #[serde(alias = "chunk-documents")]
    ChunkDocuments,
    #[value(name = "join-inner")]
    #[serde(alias = "join-inner")]
    JoinInner,
    #[value(name = "extract-pdf-text")]
    #[serde(alias = "extract-pdf-text")]
    ExtractPDFText,
    #[value(name = "group-by-and-aggregate")]
    #[serde(alias = "group-by-and-aggregate")]
    GroupByAndAggregate,
}

impl Default for WhichCandleOperator {
    fn default() -> Self {
        Self::RelativeSimilarityScore
    }
}

impl WhichCandleOperator {
    /// Wrapper to return the name of any operator
    pub fn get_name(&self) -> &str {
        match self {
            Self::RelativeSimilarityScore => RelativeSimilarityScore::get_static_name(),
            Self::SortScoresAndIndices => SortScoresAndIndices::get_static_name(),
            Self::HumanInTheLoop => HumanInTheLoop::get_static_name(),
            Self::ChunkDocuments => ChunkDocuments::get_static_name(),
            Self::JoinInner => JoinInner::get_static_name(),
            Self::ExtractPDFText => ExtractPDFText::get_static_name(),
            Self::GroupByAndAggregate => GroupByAndAggregate::get_static_name(),
        }
    }

    /// Wrapper to return the JSON schema for the tool
    pub fn get_json_tool_schema(&self) -> String {
        match self {
            Self::RelativeSimilarityScore => RelativeSimilarityScore::get_json_tool_schema(),
            Self::SortScoresAndIndices => SortScoresAndIndices::get_json_tool_schema(),
            Self::HumanInTheLoop => HumanInTheLoop::get_json_tool_schema(),
            Self::ChunkDocuments => ChunkDocuments::get_json_tool_schema(),
            Self::JoinInner => JoinInner::get_json_tool_schema(),
            Self::ExtractPDFText => ExtractPDFText::get_json_tool_schema(),
            Self::GroupByAndAggregate => GroupByAndAggregate::get_json_tool_schema(),
        }
    }

    /// Return the operation based on the name
    pub fn new_from_name(name: &str) -> Option<Self> {
        if name == RelativeSimilarityScore::get_name() {
            Some(Self::RelativeSimilarityScore)
        } else if name == SortScoresAndIndices::get_name() {
            Some(Self::SortScoresAndIndices)
        } else if name == HumanInTheLoop::get_name() {
            Some(Self::HumanInTheLoop)
        } else if name == ChunkDocuments::get_name() {
            Some(Self::ChunkDocuments)
        } else if name == JoinInner::get_name() {
            Some(Self::JoinInner)
        } else if name == ExtractPDFText::get_name() {
            Some(Self::ExtractPDFText)
        } else if name == GroupByAndAggregate::get_name() {
            Some(Self::GroupByAndAggregate)
        } else {
            None
        }
    }

    /// Build the actual operator
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_values: &str,
        rhs_pk: Option<&str>,
        rhs_fk: Option<&str>,
        rhs_values: Option<&str>,
        kwargs: Option<&str>,
    ) -> Box<dyn DataOperatorTrait> {
        match self {
            Self::RelativeSimilarityScore => Box::new(RelativeSimilarityScore::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
            Self::SortScoresAndIndices => Box::new(SortScoresAndIndices::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
            Self::HumanInTheLoop => Box::new(HumanInTheLoop::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
            Self::ChunkDocuments => Box::new(ChunkDocuments::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
            Self::JoinInner => Box::new(JoinInner::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
            Self::ExtractPDFText => Box::new(ExtractPDFText::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
            Self::GroupByAndAggregate => Box::new(GroupByAndAggregate::new(
                lhs_pk, lhs_fk, lhs_values, rhs_pk, rhs_fk, rhs_values, kwargs,
            )),
        }
    }
}

pub fn convert_destinations_to_tools(name: &str, destinations: &[String]) -> Option<ArrowTable> {
    let mut tool_id_vec = Vec::new();
    let mut tool_vec = Vec::new();
    for destination in destinations.iter() {
        if let Some(ops) = WhichCandleOperator::new_from_name(destination) {
            tool_id_vec.push(ops.get_name().to_string());
            tool_vec.push(ops.get_json_tool_schema());
        }
    }
    if tool_id_vec.is_empty() {
        None
    } else {
        let tool_id: ArrayRef = Arc::new(StringArray::from(tool_id_vec));
        let tool: ArrayRef = Arc::new(StringArray::from(tool_vec));
        let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_id), ("tool", tool)]).unwrap();
        let table = ArrowTableBuilder::new()
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
    use phymes_core::table::arrow_table::ArrowTableTrait;

    use super::*;

    #[test]
    fn test_convert_destinations_to_tools_all() {
        let result = convert_destinations_to_tools(
            "test",
            &[
                "relative-similarity-score".to_string(),
                "sort-scores-and-indices".to_string(),
                "chunk-documents".to_string(),
                "join-inner".to_string(),
                "human-in-the-loop".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(
            result.get_column_as_vec_str("tool_id"),
            &[
                "relative-similarity-score",
                "sort-scores-and-indices",
                "chunk-documents",
                "join-inner",
                "human-in-the-loop",
            ]
        );
        let functions = result.get_column_as_vec_str("tool");
        assert!(functions.first().unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"relative-similarity-score\",\"description\":\"Compute the relative similarity score between two different lists of embedding vectors\"")
        );
        assert!(
            functions
                .first()
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.first().unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.first().unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.first().unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .first()
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(1).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"sort-scores-and-indices\",\"description\":\"Sort the the list of computed scores in ascending order\"")
        );
        assert!(
            functions
                .get(1)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(1).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(1).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(1).unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .get(1)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(2).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"chunk-documents\",\"description\":\"Chunk documents by splitting the document text\"")
        );
        assert!(
            functions
                .get(2)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(2).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(2).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(2).unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .get(2)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(3).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"join-inner\",\"description\":\"Join two tables on their foreign keys\"")
        );
        assert!(
            functions
                .get(3)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .get(3)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(4).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"human-in-the-loop\",\"description\":\"Ask a question to clarify the user's query, ask a questionn to get additional information that the user did not provide, confirm a choice of tool, confirm arguments for a tool before answering the user's query or calling a tool, or provide the answer to the user's query.\"")
        );
        assert!(
            functions
                .get(4)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(4).unwrap().contains("\"lhs_args\":{\"type\":\"string\",\"description\":\"The question or answer for the user. Format lhs_arg value as JSON according to the schema {\\\"role\\\": \\\"assistant\\\", \\\"content\\\": \\\"`RESPONSE`\\\"} where `RESPONSE` is where you put your question or answer for the user\"")
        );
        assert!(
            functions
                .get(4)
                .unwrap()
                .contains("\"required\":[\"lhs_args\"]}}}")
        );
    }

    #[test]
    fn test_convert_destinations_to_tools_missing() {
        let result = convert_destinations_to_tools("test", &["missing".to_string()]);
        assert!(result.is_none());
    }
}
