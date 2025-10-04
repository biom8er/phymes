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
        let functions = result.get_column_as_vec_str("tool");
        assert!(functions.first().unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"RelativeSimilarityScore\",\"description\":\"Compute the relative similarity score between two different lists of embedding vectors\"")
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
        assert!(functions.first().unwrap().contains("\"rhs_name\":{\"type\":\"string\",\"description\":\"The name of the right hand side table\"")
        );
        assert!(functions.first().unwrap().contains("\"rhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the right hand side table\"")
        );
        assert!(functions.first().unwrap().contains("\"rhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the right hand side table\"")
        );
        assert!(
            functions
                .first()
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\",\"rhs_name\",\"rhs_pk\",\"rhs_values\"]}}}")
        );

        assert!(functions.get(1).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"SortColumnAndIndices\",\"description\":\"Sort the the list of computed scores in ascending order\"")
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

        assert!(functions.get(2).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"ChunkDocuments\",\"description\":\"Chunk documents by splitting the document text\"")
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

        assert!(functions.get(3).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"JoinInner\",\"description\":\"Join two tables on their foreign keys\"")
        );
        assert!(
            functions
                .get(3)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"rhs_name\":{\"type\":\"string\",\"description\":\"The name of the right hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_fk\":{\"type\":\"string\",\"description\":\"The foriegn key column identifier for the left hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"rhs_fk\":{\"type\":\"string\",\"description\":\"The foriegn key column identifier for the right hand side table\"")
        );
        assert!(
            functions
                .get(3)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"rhs_name\",\"lhs_fk\",\"rhs_fk\"]}}}")
        );

        assert!(functions.get(4).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"HumanInTheLoop\",\"description\":\"The response to the user.\"")
        );
        assert!(
            functions
                .get(4)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(4).unwrap().contains("\"lhs_args\":{\"type\":\"string\",\"description\":\"Format lhs_args value according to the schema {\\\"content\\\": \\\"`RESPONSE`\\\"} where `RESPONSE` is where you put your response for the user\"")
        );
        assert!(
            functions
                .get(4)
                .unwrap()
                .contains("\"required\":[\"lhs_args\"]}}}")
        );

        assert!(functions.get(5).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"GroupByAndAggregate\",\"description\":\"Group by user specified columns and aggregate user specified aggregation columns using the user specified aggregation operators.\"")
        );
        assert!(
            functions
                .get(5)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(5).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table in the form of a JSON list of strings\"")
        );
        assert!(
            functions
                .get(5)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(6).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"FilterColumnsAndIndices\",\"description\":\"Filter by specified columns using a specified comparator operator over specified columns.\"")
        );
        assert!(
            functions
                .get(6)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(6).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table in the form of a JSON list of strings\"")
        );
        assert!(
            functions
                .get(6)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(7).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"ExtractTabularData\",\"description\":\"Extract tabular data in either CSV or JSON format from Bytes\"")
        );
        assert!(
            functions
                .get(7)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(7).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(7).unwrap().contains("\"op_kwargs\":{\"type\":\"string\",\"description\":\"DataSummaryFormat object as a String\"")
        );
        assert!(
            functions
                .get(7)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_values\",\"op_kwargs\"]}}}")
        );

        assert!(functions.get(8).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"SelectAndCast\",\"description\":\"Cast specified columns using a specified cast operator and cast data type with optional column renaming and template injection.\"")
        );
        assert!(
            functions
                .get(8)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(8).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table in the form of a JSON list of strings\"")
        );
        assert!(functions.get(8).unwrap().contains("\"op_kwargs\":{\"type\":\"string\",\"description\":\"DataCastOperator and DataType with optional column renaming and template injection in the form of a JSON object\"")
        );
        assert!(
            functions
                .get(8)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_values\",\"op_kwargs\"]}}}")
        );

        assert!(functions.get(9).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"ApplyTemplate\",\"description\":\"Inject a table into a string template.\"")
        );
        assert!(
            functions
                .get(9)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );

        assert!(functions.get(9).unwrap().contains("\"op_kwargs\":{\"type\":\"string\",\"description\":\"template, table_expression, and input_template in the form of a JSON object\"")
        );
        assert!(
            functions
                .get(9)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"op_kwargs\"]}}}")
        );
    }

    #[test]
    fn test_convert_destinations_to_tools_missing() {
        let result = convert_destinations_to_tools("test", &["missing".to_string()]);
        assert!(result.is_none());
    }
}
