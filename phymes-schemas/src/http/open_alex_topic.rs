use phymes_core::MappableTrait;
use crate::{
    AvailableSchemaTrait, create_schema_from_fields, http::WorkTopicTable,
};
use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Topic {
    pub id: String,
    pub display_name: String,
    pub description: Option<String>,
    pub domain: TopicDomain,
    pub field: TopicField,
    pub subfield: TopicSubfield,
    pub ids: Option<TopicIds>,
    pub keywords: Option<Vec<String>>,
    pub updated_date: Option<String>,
    pub works_count: Option<u32>,
}

impl Topic {
    pub fn build_work_topic_table(
        self,
        work_id: &str,
        is_primary: bool,
        score: f32,
    ) -> WorkTopicTable {
        WorkTopicTable {
            work_id: work_id.to_string(),
            topic_id: self.id,
            is_primary: is_primary as u8,
            score,
        }
    }
    pub fn build_tables(
        self,
    ) -> (
        TopicTable,
        TopicDomainTable,
        TopicFieldTable,
        TopicSubfieldTable,
        Option<TopicIdsTable>,
        Vec<TopicKeywordTable>,
    ) {
        let topic_domain = self.domain.build_topic_domain_table(&self.id);
        let topic_field = self.field.build_topic_field_table(&self.id);
        let topic_subfield = self.subfield.build_topic_subfield_table(&self.id);
        let topic_ids = self.ids.map(|t| t.build_topic_ids_table(&self.id));
        let topic_keyword = self
            .keywords
            .unwrap_or_default()
            .into_iter()
            .map(|k| TopicKeywordTable {
                topic_id: self.id.clone(),
                keyword: k,
            })
            .collect::<Vec<_>>();
        let topic = TopicTable {
            topic_id: self.id,
            display_name: self.display_name,
            description: self.description.unwrap_or_default(),
            updated_date: self.updated_date.unwrap_or_default(),
            works_count: self.works_count.unwrap_or_default(),
        };
        (
            topic,
            topic_domain,
            topic_field,
            topic_subfield,
            topic_ids,
            topic_keyword,
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicTable {
    pub topic_id: String,
    pub display_name: String,
    pub description: String,
    pub updated_date: String,
    pub works_count: u32,
}

impl TopicTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "display_name", "description", "updated_date"];
        let mut fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let field_names = ["works_count"];
        fields_vec.extend(
            field_names
                .iter()
                .map(|f| Field::new(*f, DataType::UInt32, false))
                .collect::<Vec<_>>(),
        );
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicIds {
    pub openalex: Option<String>,
    pub wikipedia: Option<String>,
}

impl TopicIds {
    pub fn build_topic_ids_table(self, topic_id: &str) -> TopicIdsTable {
        TopicIdsTable {
            topic_id: topic_id.to_string(),
            openalex: self.openalex.unwrap_or_default(),
            wikipedia: self.wikipedia.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicIdsTable {
    pub topic_id: String,
    pub openalex: String,
    pub wikipedia: String,
}

impl TopicIdsTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "openalex", "wikipedia"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicIdsTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicIdsTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicSubfield {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

impl TopicSubfield {
    pub fn build_topic_subfield_table(self, topic_id: &str) -> TopicSubfieldTable {
        TopicSubfieldTable {
            topic_id: topic_id.to_string(),
            topic_subfield_id: self.id.unwrap_or_default(),
            display_name: self.display_name.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicSubfieldTable {
    pub topic_id: String,
    pub topic_subfield_id: String,
    pub display_name: String,
}

impl TopicSubfieldTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "topic_subfield_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicSubfieldTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicSubfieldTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicField {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

impl TopicField {
    pub fn build_topic_field_table(self, topic_id: &str) -> TopicFieldTable {
        TopicFieldTable {
            topic_id: topic_id.to_string(),
            topic_field_id: self.id.unwrap_or_default(),
            display_name: self.display_name.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicFieldTable {
    pub topic_id: String,
    pub topic_field_id: String,
    pub display_name: String,
}

impl TopicFieldTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "topic_field_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicFieldTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicFieldTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicDomain {
    pub id: Option<String>,
    pub display_name: Option<String>,
}

impl TopicDomain {
    pub fn build_topic_domain_table(self, topic_id: &str) -> TopicDomainTable {
        TopicDomainTable {
            topic_id: topic_id.to_string(),
            topic_domain_id: self.id.unwrap_or_default(),
            display_name: self.display_name.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicDomainTable {
    pub topic_id: String,
    pub topic_domain_id: String,
    pub display_name: String,
}

impl TopicDomainTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "topic_domain_id", "display_name"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicDomainTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicDomainTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopicKeywordTable {
    pub topic_id: String,
    pub keyword: String,
}

impl TopicKeywordTable {
    fn to_fields() -> Fields {
        let field_names = ["topic_id", "keyword"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        Fields::from(fields_vec)
    }
}

impl MappableTrait for TopicKeywordTable {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl AvailableSchemaTrait for TopicKeywordTable {
    fn to_schema(&self) -> SchemaRef {
        create_schema_from_fields(&Self::to_fields)
    }
}
