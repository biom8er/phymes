use std::{collections::VecDeque, sync::Arc};

use arrow::datatypes::{DataType, Field, Fields, Schema};
use object_store::aws::AmazonS3ConfigKey;
use phymes_data::{
    AvailableOperators, DataColumnOperator, DataComparatorOperator, DataComparatorPredicate,
    DataConfig, DataStreamManager,
};
use phymes_event::{Publication, Subscription};
use phymes_network::{
    DynamicTaskNetworkBuilder, DynamicTaskNetworkNames, DynamicTaskNetworkTypes, NetworkBuilder, NetworkBuilderMermaidTrait,
};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{
    AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, DataEncoding, DataFormat,
};
use phymes_streams::{
    HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType, LimitConfig,
    ObjectStoreConfig, ObjectStoreOptsType,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, ObjectStorageBackend, Subject, SubjectBuilder,
    SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait,
};
use serde_json::{Map, Value};

use crate::{
    EmbedTextNetworkBuilder, ExtractOntologyNetworkBuilder, ExtractPDFNetworkBuilder,
    RetrieveTextNetworkBuilder,
};

/// OpenAlex network
pub struct OpenAlexNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
}

impl Default for OpenAlexNetworkBuilder {
    fn default() -> Self {
        // OpenAlex download from AWS
        let open_alex_network_builder = {
            let network_name = "get_object";
            let mut store_config = Map::<String, Value>::new();
            let _ = store_config.insert(
                AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
                Value::String("true".to_string()),
            );
            let _ = store_config.insert(
                AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
                Value::String("https://s3.amazonaws.com".to_string()),
            );
            let config = ObjectStoreConfig {
                timeout: 5,
                ops_type: ObjectStoreOptsType::Get,
                backend: ObjectStorageBackend::Aws,
                bucket: Some("openalex".to_string()),
                backend_config: Some(serde_json::to_string(&store_config).unwrap()),
                subject_name: Some(AvailableSubjects::ObjectStoreMeta.to_string()),
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::ObjectStoreMeta
                .to_subject(None, None)
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::UserObject
                .to_subject(None, None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::ObjectStoreProcessor,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_lhs.get_name().to_string(),
                },
                publication: Publication::Replace {
                    subject_name: AvailableInterfaceSubjects::UserObject.to_string(),
                },
                subject_lhs: Some(subject_lhs),
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };

        // Extract OpenAlex Works tables
        let network_builder = {
            let network_name = "extract_open_alex_aws_bucket";
            let config = DataConfig {
                lhs_name: Some(AvailableInterfaceSubjects::UserObject.to_string()),
                lhs_pk: Some("location".to_string()),
                lhs_values: Some(vec!["bytes".to_string()]),
                encoding: Some(DataEncoding::Gz),
                format: Some(DataFormat::JsonSchema),
                schema: Some(AvailableSubjects::OpenAlexResponseWorks),
                cpu: false,
                operator: AvailableOperators::ExtractTabular,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::UserObject
                .to_subject(None, None)
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::Bytes
                .to_subject(
                    Some(
                        DynamicTaskNetworkNames::Subject(network_name)
                            .to_string()
                            .as_str(),
                    ),
                    None,
                )
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject_routes = [
                "WorkTable",
                "WorkAwardTable",
                "WorkAuthorshipTable",
                "WorkFunderTable",
                "WorkApcInfoTable",
                "WorkLocationTable",
                "WorkOpenAccessTable",
                "WorkBiblioTable",
                "WorkCitationPercentileTable",
                "WorkCitedByPercentileYearTable",
                "WorkCountsByYearTable",
                "WorkConceptTable",
                "WorkTopicTable",
                "WorkKeywordTable",
                "WorkMeshTagTable",
                "WorkSdgTagTable",
                "WorkCorrespondingAuthorTable",
                "WorkCorrespondingInstitutionTable",
                "WorkIndexedInTable",
                "WorkIdsTable",
                "WorkReferencedWorksTable",
                "WorkRelatedWorksTable",
            ]
            .into_iter()
            .map(|s| {
                let subject = AvailableSubjects::Bytes.to_subject(Some(s), None).unwrap();
                SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::ExtractTabular,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_lhs.get_name().to_string(),
                },
                publication: Publication::Replace {
                    subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                },
                subject_lhs: Some(subject_lhs),
                subject_out: Some(subject_out),
                subject_processor,
                subject_routes: Some(subject_routes),
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let open_alex_network_builder = open_alex_network_builder.extend(network_builder).unwrap();

        // OpenAlex search for OpenAccess articles by topic
        let network_builder = {
            let task_name = "filter_work_topic_table";
            let mut tasks = VecDeque::new();
            {
                let network_name = "coalesce_work_topic_table";
                let subject_name_lhs = "WorkTopicTable";
                let config = LimitConfig {
                    fetch: 512,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::CoalesceProcessor,
                    subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                        subject_name: subject_name_lhs.to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "cmp_work_topic_table";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        [
                            "work_id",
                            "topic_id",
                            "is_primary",
                            "score",
                            "cmp_is_primary",
                            "cmp_score",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    as_columns: Some(
                        [
                            "work_id",
                            "topic_id",
                            "is_primary",
                            "score",
                            "cmp_is_primary",
                            "cmp_score",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    cast_templates: Some(
                        ["", "", "", "", "1", "0.5"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cast_datatypes: Some(
                        [
                            DataType::Utf8,
                            DataType::Utf8,
                            DataType::UInt8,
                            DataType::Float32,
                            DataType::UInt8,
                            DataType::Float32,
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    column_operators: Some(
                        [
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::Value,
                            DataColumnOperator::Value,
                        ]
                        .into_iter()
                        .collect::<Vec<_>>(),
                    ),
                    cpu: false,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "filter_work_topic_table";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        ["is_primary", "score"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cmp_columns: Some(
                        ["cmp_is_primary", "cmp_score"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cmp_operators: Some(
                        [
                            DataComparatorOperator::Equals,
                            DataComparatorOperator::GreaterThan,
                        ]
                        .into_iter()
                        .collect::<Vec<_>>(),
                    ),
                    cmp_predicate: Some(DataComparatorPredicate::All),
                    cpu: true,
                    operator: AvailableOperators::Filter,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Filter,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "select_work_topic_table";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        ["work_id", "topic_id", "is_primary", "score"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cpu: false,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "join_work_topic_table";
                let subject_name_rhs = "open_alex_topics_s";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    rhs_name: Some(subject_name_rhs.to_string()),
                    lhs_fk: Some("topic_id".to_string()),
                    rhs_fk: Some("topic_id".to_string()),
                    lhs_pk: Some("topic_id".to_string()),
                    rhs_pk: Some("topic_id".to_string()),
                    join_operators: Some(phymes_data::DataJoinOperator::Inner),
                    cpu: true,
                    operator: AvailableOperators::Join,
                    lhs_stream: DataStreamManager::Stream,
                    rhs_stream: Some(DataStreamManager::Accumulate),
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let fields = Fields::from(vec![Field::new("topic_id", DataType::Utf8, false)]);
                let schema = Arc::new(Schema::new(fields));
                let subject = Subject::get_builder()
                    .with_name(subject_name_rhs)
                    .with_schema(schema)
                    .with_record_batches(Vec::new())
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_rhs = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let fields = Fields::from(vec![
                    Field::new("work_id", DataType::Utf8, false),
                    Field::new("topic_id", DataType::Utf8, false),
                    Field::new("is_primary", DataType::UInt8, false),
                    Field::new("score", DataType::Float32, false),
                ]);
                let schema = Arc::new(Schema::new(fields));
                let subject = Subject::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::Subject(network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_schema(schema)
                    .with_record_batches(Vec::new())
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_out = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Join,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    subscription_rhs: Some(Subscription::AlwaysAllRecordBatches {
                        subject_name: subject_rhs.get_name().to_string(),
                    }),
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_lhs: tasks.iter().last().unwrap().subject_out.clone(),
                    subject_rhs: Some(subject_rhs),
                    subject_out: Some(subject_out),
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            let mut network_builder = tasks.pop_front().unwrap().build_dynamic();
            while let Some(task) = tasks.pop_front() {
                network_builder = network_builder.extend(task.build_dynamic()).unwrap();
            }
            network_builder
        };
        let open_alex_network_builder = open_alex_network_builder.extend(network_builder).unwrap();

        // OpenAlex search for OpenAccess PDF URLs
        let network_builder = {
            let task_name = "select_open_access_pdf_url";
            let mut tasks = VecDeque::new();
            {
                let network_name = "coalesce_work_location_table";
                let subject_name_lhs = "WorkLocationTable";
                let config = LimitConfig {
                    fetch: 512,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::CoalesceProcessor,
                    subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                        subject_name: subject_name_lhs.to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "cmp_work_location_table";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        [
                            "work_id",
                            "landing_page_url",
                            "pdf_url",
                            "source_id",
                            "license",
                            "version",
                            "is_best_oa",
                            "is_primary",
                            "is_oa",
                            "cmp_is_best_oa",
                            "pdf_url",
                            "cmp_pdf_url_len",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    as_columns: Some(
                        [
                            "work_id",
                            "landing_page_url",
                            "pdf_url",
                            "source_id",
                            "license",
                            "version",
                            "is_best_oa",
                            "is_primary",
                            "is_oa",
                            "cmp_is_best_oa",
                            "pdf_url_len",
                            "cmp_pdf_url_len",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    cast_templates: Some(
                        ["", "", "", "", "", "", "", "", "", "1", "", ""]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cast_datatypes: Some(
                        [
                            DataType::Utf8,
                            DataType::Utf8,
                            DataType::Utf8,
                            DataType::Utf8,
                            DataType::Utf8,
                            DataType::Utf8,
                            DataType::UInt8,
                            DataType::UInt8,
                            DataType::UInt8,
                            DataType::UInt8,
                            DataType::UInt32,
                            DataType::UInt32,
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    column_operators: Some(
                        [
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::None,
                            DataColumnOperator::Value,
                            DataColumnOperator::Len,
                            DataColumnOperator::Zeros,
                        ]
                        .into_iter()
                        .collect::<Vec<_>>(),
                    ),
                    cpu: true,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "filter_work_location_table";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        ["is_best_oa", "pdf_url_len"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cmp_columns: Some(
                        ["cmp_is_best_oa", "cmp_pdf_url_len"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cmp_operators: Some(
                        [
                            DataComparatorOperator::Equals,
                            DataComparatorOperator::GreaterThan,
                        ]
                        .into_iter()
                        .collect::<Vec<_>>(),
                    ),
                    cmp_predicate: Some(DataComparatorPredicate::All),
                    cpu: true,
                    operator: AvailableOperators::Filter,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Filter,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "select_work_location_table";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        [
                            "work_id",
                            "landing_page_url",
                            "pdf_url",
                            "source_id",
                            "license",
                            "version",
                            "is_best_oa",
                            "is_primary",
                            "is_oa",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    cpu: true,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "join_work_location_table";
                let subject_name_rhs = "join_work_topic_table_s";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    rhs_name: Some(subject_name_rhs.to_string()),
                    lhs_fk: Some("work_id".to_string()),
                    rhs_fk: Some("work_id".to_string()),
                    lhs_pk: Some("work_id".to_string()),
                    rhs_pk: Some("work_id".to_string()),
                    join_operators: Some(phymes_data::DataJoinOperator::Inner),
                    cpu: true,
                    operator: AvailableOperators::Join,
                    lhs_stream: DataStreamManager::Stream,
                    rhs_stream: Some(DataStreamManager::Accumulate),
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Join,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    subscription_rhs: Some(Subscription::OnUpdateAllRecordBatches {
                        subject_name: subject_name_rhs.to_string(),
                    }),
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "select_open_acces_pdf_url";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    as_columns: Some(
                        ["", "", "", "content", "", ""]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    lhs_values: Some(
                        [
                            "work_id",
                            "topic_id",
                            "score",
                            "pdf_url",
                            "source_id",
                            "version",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    cpu: false,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let fields = Fields::from(vec![
                    Field::new("work_id", DataType::Utf8, false),
                    Field::new("topic_id", DataType::Utf8, false),
                    Field::new("score", DataType::Float32, false),
                    Field::new("content", DataType::Utf8, false),
                    Field::new("source_id", DataType::Utf8, false),
                    Field::new("version", DataType::Utf8, false),
                ]);
                let schema = Arc::new(Schema::new(fields));
                let subject = Subject::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::Subject(network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_schema(schema)
                    .with_record_batches(Vec::new())
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_out = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_out: Some(subject_out),
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            let mut network_builder = tasks.pop_front().unwrap().build_dynamic();
            while let Some(task) = tasks.pop_front() {
                network_builder = network_builder.extend(task.build_dynamic()).unwrap();
            }
            network_builder
        };
        let open_alex_network_builder = open_alex_network_builder.extend(network_builder).unwrap();

        // Get PDF network
        // DM: Update to use `GetPdfNetworkBuilderStaticWSubject`
        // let get_pdf_network_builder = GetPdfNetworkBuilderStaticWSubject::default().inner.build_dynamic();
        // let retrieve_text_pdf_network_builder = retrieve_text_pdf_network_builder
        //     .extend(get_pdf_network_builder)
        //     .unwrap()
        // DM: change "select_open_acces_pdf_url_s" to "http_client_request_pdf_s"
        let network_builder = {
            let network_name = "get_pdf";
            let subject_name_lhs = "select_open_acces_pdf_url_s";
            let config = HTTPClientConfig {
                timeout: 15,
                request_type: HTTPClientRequestType::Get,
                poll_error: false,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: String::new(),
                subject_name: Some(subject_name_lhs.to_string()),
                request_schema: HTTPClientRequestSchemas::Attachments,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::UserPdf
                .to_subject(None, None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::HTTPClientRequestProcessor,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_name_lhs.to_string(),
                },
                publication: Publication::Extend {
                    subject_name: subject_out.get_name().to_string(),
                },
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let open_alex_network_builder = open_alex_network_builder.extend(network_builder).unwrap();

        // Extract PDF session
        let extract_pdf_network_builder = ExtractPDFNetworkBuilder::default().inner.take().unwrap();
        let open_alex_network_builder = open_alex_network_builder
            .extend(extract_pdf_network_builder)
            .unwrap();

        // Get Owl ontology network
        let network_builder = {
            let network_name = "get_owl";
            let subject_name_lhs = "http_request_owl_s";
            // DM: for compatibility with downstream task
            let subject_name_o = AvailableInterfaceSubjects::UserScript.to_string();
            let config = ObjectStoreConfig {
                timeout: 5,
                ops_type: ObjectStoreOptsType::Get,
                backend: ObjectStorageBackend::LocalFs,
                bucket: Some("/mnt/c".to_string()),
                backend_config: None,
                subject_name: Some(subject_name_lhs.to_string()),
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let fields = Fields::from(vec![Field::new("location", DataType::Utf8, false)]);
            let schema = Arc::new(Schema::new(fields));
            let subject = Subject::get_builder()
                .with_name(subject_name_lhs)
                .with_schema(schema)
                .with_record_batches(Vec::new())
                .unwrap()
                .build()
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::UserObject
                .to_subject(Some(subject_name_o.as_str()), None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::ObjectStoreProcessor,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_lhs.get_name().to_string(),
                },
                publication: Publication::Replace {
                    subject_name: subject_out.get_name().to_string(),
                },
                subject_lhs: Some(subject_lhs),
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        // DM: unstable HTTP client from bioportal...
        // let network_builder = {
        //     let network_name = "get_owl";
        //     let subject_name_lhs = "http_request_owl_s";
        //     let config = HTTPClientConfig {
        //         timeout: 15,
        //         request_type: HTTPClientRequestType::Get,
        //         poll_error: true,
        //         user_agent_type: Some("rust-openalex-client/3.0".to_string()),
        //         base_url: String::new(),
        //         subject_name: Some(subject_name_lhs.to_string()),
        //         request_schema: HTTPClientRequestSchemas::Attachments,
        //         ..Default::default()
        //     };
        //     let config_json = serde_json::to_vec(&config).unwrap();
        //     let subject = SubjectBuilder::new()
        //         .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
        //         .with_json(&config_json, 1)
        //         .unwrap()
        //         .build()
        //         .unwrap();
        //     let subject_processor = SubjectPlan::get_builder()
        //         .with_subject(subject)
        //         .build()
        //         .unwrap();
        //     let fields = Fields::from(vec![Field::new("content", DataType::Utf8, false)]);
        //     let schema = Arc::new(Schema::new(fields));
        //     let subject = Subject::get_builder()
        //         .with_name(subject_name_lhs)
        //         .with_schema(schema)
        //         .with_record_batches(Vec::new()).unwrap()
        //         .build().unwrap();
        //     let subject_lhs = SubjectPlan::get_builder()
        //         .with_subject(subject)
        //         .build()
        //         .unwrap();
        //     let subject = AvailableInterfaceSubjects::UserScript
        //         .to_subject(None, None)
        //         .unwrap();
        //     let subject_out = SubjectPlan::get_builder()
        //         .with_subject(subject)
        //         .build()
        //         .unwrap();
        //     let builder = DynamicTaskNetworkBuilder {
        //         network_name: network_name.to_string(),
        //         dynamic_type: DynamicTaskNetworkTypes::Static,
        //         processor: AvailableProcessors::HTTPClientRequestProcessor,
        //         subscription_lhs: Subscription::OnUpdateAllRecordBatches {
        //             subject_name: subject_name_lhs.to_string(),
        //         },
        //         publication: Publication::Extend {
        //             subject_name: subject_out.get_name().to_string(),
        //         },
        //         subject_lhs: Some(subject_lhs),
        //         subject_out: Some(subject_out),
        //         subject_processor,
        //         ..Default::default()
        //     };
        //     builder.build_dynamic()
        // };
        let open_alex_network_builder = open_alex_network_builder.extend(network_builder).unwrap();

        // Extract Owl ontology with filtering of predicates
        let network_builder = {
            let task_name = "extract_owl";
            let mut tasks = VecDeque::new();
            {
                let network_name = "extract_owl";
                let config = DataConfig {
                    lhs_name: Some(AvailableInterfaceSubjects::UserScript.to_string()),
                    // lhs_pk: Some("filename".to_string()),location
                    lhs_pk: Some("location".to_string()), // DM: using ObjectStore schema but UserScript name
                    lhs_values: Some(
                        ["bytes"]
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    format: Some(DataFormat::Owl),
                    doc_filter: Some(phymes_data::DocumentFilterType::Text),
                    doc_extraction: Some(phymes_data::DocumentExtractType::TextEmbeddings),
                    cpu: false,
                    operator: AvailableOperators::ExtractXML,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::ExtractXML,
                    subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableInterfaceSubjects::UserScript.to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "coalesce_extract_owl";
                let config = LimitConfig {
                    fetch: 512,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::CoalesceProcessor,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            let schema_cols = [
                "entity",
                "subject",
                "predicate",
                "object",
                "graph",
                "dataset",
            ];
            let cmp_cols = [
                "hasDbXref",
                "creation_date",
                "id",
                "seeAlso",
                "contributor",
                "OMO_0002000",
                "date",
            ];
            {
                let network_name = "cmp_owl_predicates";
                let lhs_values = schema_cols
                    .iter()
                    .chain(cmp_cols.iter())
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(lhs_values.clone()),
                    as_columns: Some(lhs_values.clone()),
                    cast_templates: Some(
                        [
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                            "http://www.geneontology.org/formats/oboInOwl#creation_date",
                            "http://www.geneontology.org/formats/oboInOwl#id",
                            "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                            "http://purl.org/dc/terms/contributor",
                            "http://purl.obolibrary.org/obo/OMO_0002000",
                            "http://purl.org/dc/terms/date",
                        ]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                    ),
                    cast_datatypes: Some(
                        lhs_values
                            .iter()
                            .map(|_| DataType::Utf8.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    column_operators: Some(
                        schema_cols
                            .iter()
                            .map(|_| DataColumnOperator::None)
                            .chain(cmp_cols.iter().map(|_| DataColumnOperator::Value))
                            .collect::<Vec<_>>(),
                    ),
                    cpu: false,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "filter_owl_predicates";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        cmp_cols
                            .iter()
                            .map(|_| "predicate".to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cmp_columns: Some(cmp_cols.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
                    cmp_operators: Some(
                        cmp_cols
                            .iter()
                            .map(|_| DataComparatorOperator::NotLike)
                            .collect::<Vec<_>>(),
                    ),
                    cmp_predicate: Some(DataComparatorPredicate::All),
                    cpu: false,
                    operator: AvailableOperators::Filter,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Filter,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: DynamicTaskNetworkNames::Subject(network_name).to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            {
                let network_name = "select_owl_predicates";
                let config = DataConfig {
                    lhs_name: Some(
                        tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    ),
                    lhs_values: Some(
                        schema_cols
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    cpu: false,
                    operator: AvailableOperators::Select,
                    lhs_stream: DataStreamManager::Stream,
                    ..Default::default()
                };
                let config_json = serde_json::to_vec(&config).unwrap();
                let subject = SubjectBuilder::new()
                    .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                    .with_json(&config_json, 1)
                    .unwrap()
                    .build()
                    .unwrap();
                let subject_processor = SubjectPlan::get_builder()
                    .with_subject(subject)
                    .build()
                    .unwrap();
                let builder = DynamicTaskNetworkBuilder {
                    network_name: task_name.to_string(),
                    dynamic_type: DynamicTaskNetworkTypes::Static,
                    processor: AvailableProcessors::Select,
                    subscription_lhs: Subscription::AlwaysAllRecordBatches {
                        subject_name: tasks
                            .iter()
                            .last()
                            .unwrap()
                            .publication
                            .subject_name()
                            .to_string(),
                    },
                    publication: Publication::Replace {
                        subject_name: AvailableSubjects::ParseOwl.to_string(),
                    },
                    subject_processor,
                    ..Default::default()
                };
                tasks.push_back(builder);
            }
            let mut network_builder = tasks.pop_front().unwrap().build_dynamic();
            while let Some(task) = tasks.pop_front() {
                network_builder = network_builder.extend(task.build_dynamic()).unwrap();
            }
            network_builder
        };
        let open_alex_network_builder = open_alex_network_builder.extend(network_builder).unwrap();

        // Extract OWL network
        // DM: the previous networks will overwrite the defaults in this network
        let extract_owl_network = ExtractOntologyNetworkBuilder {
            network_name: "extract_ontology_network",
            as_documents: false,
            include_extract_owl: false,
        };
        let extract_owl_network_builder = NetworkBuilder::from_mermaid_flowchart(
            &extract_owl_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &extract_owl_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(extract_owl_network.network_name);
        let open_alex_network_builder = open_alex_network_builder
            .extend(extract_owl_network_builder)
            .unwrap();

        // Embed text session
        let embed_text_network = EmbedTextNetworkBuilder::default();
        let embed_text_network_builder = NetworkBuilder::from_mermaid_flowchart(
            &embed_text_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &embed_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(embed_text_network.network_name);
        let open_alex_network_builder = open_alex_network_builder
            .extend(embed_text_network_builder)
            .unwrap();

        // Retrieve text session
        let retrieve_text_network = RetrieveTextNetworkBuilder::default();
        let retrieve_text_builder = NetworkBuilder::from_mermaid_flowchart(
            retrieve_text_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            retrieve_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(retrieve_text_network.network_name);
        let open_alex_network_builder = open_alex_network_builder
            .extend(retrieve_text_builder)
            .unwrap();

        OpenAlexNetworkBuilder {
            inner: Some(open_alex_network_builder.with_name("open_alex_network")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{
        DynamicTaskNetworkNames, NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, create_object_store_meta_batch,
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use crate::{extended_diagnostic_subjects, write_diagnostic_subjects_to_csv};

    use super::*;

    #[ignore = "In progress... `cargo test -p phymes-templates test_open_alex_network_v_rust --features api,gpu,hf_hub --release -- --nocapture`"]
    #[tokio::test]
    async fn test_open_alex_network_v_rust() -> Result<()> {
        // Initialize the session
        let open_alex_network_builder = OpenAlexNetworkBuilder::default().inner.take().unwrap();
        let network_name = open_alex_network_builder.name.clone().unwrap();
        let (network, session_messages) = open_alex_network_builder
            .with_runtime_env(
                RuntimeEnv::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_max_steps(100)
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test session data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Make the list of paths to Get
        // let location = vec!["data/jsonl/works/updated_date=2018-01-12/part_0000.gz".to_string()];
        let location = vec!["data/jsonl/works/updated_date=2026-03-10/part_0005.gz".to_string()];
        // let location = vec!["data/jsonl/works/manifest".to_string()];
        let bucket = vec!["openalex".to_string()];
        let e_tag = vec![String::new()];
        let version = vec![String::new()];
        let size = vec![0_u32];
        let last_modified = vec![0_i64];
        let batch =
            create_object_store_meta_batch(location, bucket, e_tag, version, size, last_modified)?;
        let subject = Subject::get_builder()
            .with_name(&AvailableSubjects::ObjectStoreMeta.to_string())
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(subject.get_name())
                .with_publisher(network_arc.get_name())
                .with_subject(subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
                .build()?,
        );
        let topic_ids = vec!["https://openalex.org/T10123".to_string()];
        let topic_arr: ArrayRef = Arc::new(StringArray::from(topic_ids));
        let batch = RecordBatch::try_from_iter(vec![("topic_id", topic_arr)])?;
        let subject = Subject::get_builder()
            .with_name("open_alex_topics_s")
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(subject.get_name())
                .with_publisher(network_arc.get_name())
                .with_subject(subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
                .build()?,
        );
        // let cl_urls = vec![
        //     // "Users/dmccl/Downloads/ontologies/core_predicates.owl",
        //     // "http://purl.obolibrary.org/obo/ro.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/eco.owl".to_string(),
        //     "http://purl.obolibrary.org/obo/cl.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/mondo/mondo-simple.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/uberon/releases/2025-08-15/uberon.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/chebi.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/go.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/pr.owl".to_string(),
        //     // "http://purl.obolibrary.org/obo/hp/releases/2026-02-16/hp-international.owl".to_string(),
        //     // "".to_string(),
        //     // "".to_string(),
        //     // "".to_string(),
        //     // "".to_string(),
        //     // "".to_string(),
        //     ];
        let cl_urls = vec![
            // "Users/dmccl/Downloads/ontologies/HumanDO.owl",
            // "Users/dmccl/Downloads/ontologies/core_predicates.owl",
            // "Users/dmccl/Downloads/ontologies/rdfs-dc-skos.owl",
            "Users/dmccl/Downloads/ontologies/ro.owl", // Breaks with HumanDO but works on its own (no relations!)
                                                       // "Users/dmccl/Downloads/ontologies/eco.owl", // Works with HumanDO
                                                       // "Users/dmccl/Downloads/ontologies/cl.owl", // Works with HumanDO
                                                       // "Users/dmccl/Downloads/ontologies/uberon.owl", // Breaks with HumanDO but works on its own
                                                       // "Users/dmccl/Downloads/ontologies/go.owl", // Works with HumanDO
                                                       // "Users/dmccl/Downloads/ontologies/taxslim.owl", // Works with HumanDO
        ];
        let cl_arr: ArrayRef = Arc::new(StringArray::from(cl_urls));
        // let batch = RecordBatch::try_from_iter(vec![("content", cl_arr)])?;
        let batch = RecordBatch::try_from_iter(vec![("location", cl_arr)])?;
        let subject = Subject::get_builder()
            .with_name("http_request_owl_s")
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(subject.get_name())
                .with_publisher(network_arc.get_name())
                .with_subject(subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the session
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        let extended_diagnostic_subjects = extended_diagnostic_subjects();
        let subject_names = extended_diagnostic_subjects
            .iter()
            .map(|s| s.as_str())
            .chain(["EmbeddingScores", "Documents", "UserQueries"])
            .collect::<Vec<_>>();
        write_diagnostic_subjects_to_csv(
            &subject_names,
            network_arc.runtime_env(),
            network_arc.get_name(),
        )
        .await?;

        assert_eq!(response.len(), 0);

        // Test AWS object store GET
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "WorkTable".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("WorkTable")
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 317124);
        let column = subject.get_column_as_vec_str("work_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/W4367307220");

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "WorkTopicTable".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("WorkTopicTable")
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 91321);
        let column = subject.get_column_as_vec_str("work_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/W4367307220");
        let column = subject.get_column_as_vec_str("topic_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/T10123");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/T13802");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &0.9994);
        // assert_eq!(column.last().unwrap(), &0.2251);
        let column = subject.get_column_as_vec_primitive::<u8>("is_primary")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &1);
        // assert_eq!(column.last().unwrap(), &1);

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "WorkLocationTable".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("WorkLocationTable")
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 48958);
        let column = subject.get_column_as_vec_str("work_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/W4367307220");
        let column = subject.get_column_as_vec_primitive::<u8>("is_best_oa")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &0);
        // assert_eq!(column.last().unwrap(), &0);
        let column = subject.get_column_as_vec_primitive::<u8>("is_primary")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &1);
        // assert_eq!(column.last().unwrap(), &1);
        let column = subject.get_column_as_vec_primitive::<u8>("is_oa")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &0);
        // assert_eq!(column.last().unwrap(), &0);
        let column = subject.get_column_as_vec_str("landing_page_url");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"https://doi.org/10.1016/j.str.2014.09.012"
        // );
        // assert_eq!(
        //     column.last().unwrap(),
        //     &"http://dx.doi.org/10.2307/jj.2430693"
        // );
        let column = subject.get_column_as_vec_str("pdf_url");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"");
        // assert_eq!(column.last().unwrap(), &"");
        let column = subject.get_column_as_vec_str("source_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/S7112016");
        // assert_eq!(column.last().unwrap(), &"");
        let column = subject.get_column_as_vec_str("license");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"");
        // assert_eq!(column.last().unwrap(), &"cc-by");
        let column = subject.get_column_as_vec_str("version");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"publishedVersion");
        // assert_eq!(column.last().unwrap(), &"publishedVersion");

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "extract_open_alex_aws_bucket_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        assert!(batches.is_empty());

        // Test join work topic with user defined topics
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "join_work_topic_table_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("join_work_topic_table_s")
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 13);
        let column = subject.get_column_as_vec_str("work_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/W2063148287");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/W2939699114");
        let column = subject.get_column_as_vec_str("topic_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/T10123");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/T10123");
        let column = subject.get_column_as_vec_primitive::<u8>("is_primary")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &1);
        // assert_eq!(column.last().unwrap(), &1);
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &0.9998);
        // assert_eq!(column.last().unwrap(), &0.9998);

        // Test select open access PDF url as content
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "select_open_acces_pdf_url_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("select_open_acces_pdf_url_s")
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 6);
        let column = subject.get_column_as_vec_str("work_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/W2036554147");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/W4408584426");
        let column = subject.get_column_as_vec_str("topic_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/T10123");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/T10123");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &0.9998);
        // assert_eq!(column.last().unwrap(), &0.9688);
        let column = subject.get_column_as_vec_str("content");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"http://www.jidonline.org/article/S0022202X15321485/pdf"
        // );
        // assert_eq!(
        //     column.last().unwrap(),
        //     &"https://doi.org/10.37184/jlnh.2959-1805.3.9"
        // );
        let column = subject.get_column_as_vec_str("source_id");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"https://openalex.org/S28607811");
        // assert_eq!(column.last().unwrap(), &"https://openalex.org/S4387288081");
        let column = subject.get_column_as_vec_str("version");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"publishedVersion");
        // assert_eq!(column.last().unwrap(), &"publishedVersion");

        // Test HTTP request of open access PDF
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            dbg!(subject.count_rows());
            // assert_eq!(subject.count_rows(), 6);
            let column = subject.get_column_as_vec_str("filename");
            dbg!(column.first().unwrap());
            dbg!(column.last().unwrap());
            // assert_eq!(
            //     column.first().unwrap(),
            //     &"http://www.jidonline.org/article/S0022202X15321485/pdf"
            // );
            // assert_eq!(
            //     column.last().unwrap(),
            //     &"https://doi.org/10.37184/jlnh.2959-1805.3.9"
            // );
            let column = subject.get_column_as_vec_str("extension");
            dbg!(column.first().unwrap());
            dbg!(column.last().unwrap());
            // assert_eq!(column.first().unwrap(), &"text/html;charset=UTF-8");
            // assert_eq!(column.last().unwrap(), &"application/pdf");
            let column = subject.get_column_as_vec_str("metadata");
            dbg!(column.first().unwrap());
            dbg!(column.last().unwrap());
            // assert_eq!(column.first().unwrap(), &"tool");
            // assert_eq!(column.last().unwrap(), &"tool");
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = subject
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert!(column.len() > 100);
        }

        // Test HTTP request of ontologies
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableInterfaceSubjects::UserScript.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserScript.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::ParseOwl.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::ParseOwl.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 30);
        let column = subject.get_column_as_vec_str("entity");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"http://www.w3.org/2002/07/owl#AnnotationProperty"
        // );
        // assert_eq!(
        //     column.last().unwrap(),
        //     &"http://www.w3.org/2002/07/owl#Ontology"
        // );
        let column = subject.get_column_as_vec_str("subject");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"http://purl.obolibrary.org/obo/IAO_0000115"
        // );
        // assert_eq!(
        //     column.last().unwrap(),
        //     &"http://purl.obolibrary.org/obo/ro.owl"
        // );
        let column = subject.get_column_as_vec_str("predicate");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"http://purl.obolibrary.org/obo/IAO_0000115"
        // );
        // assert_eq!(column.last().unwrap(), &"http://purl.org/dc/terms/title");
        let column = subject.get_column_as_vec_str("object");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &""
        // );
        // assert_eq!(column.last().unwrap(), &"OBO Relations Ontology");
        let column = subject.get_column_as_vec_str("graph");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &""
        // );
        // assert_eq!(
        //     column.last().unwrap(),
        //     &""
        // );
        let column = subject.get_column_as_vec_str("dataset");
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(column.first().unwrap(), &"UserScript");
        // assert_eq!(column.last().unwrap(), &"UserScript");

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableInterfaceSubjects::UserQueries.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserQueries.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        dbg!(subject.count_rows());
        // assert_eq!(subject.count_rows(), 3);
        let mut column = subject.get_column_as_vec_str("query_id");
        column.sort();
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"http://purl.obolibrary.org/obo/BFO_0000003"
        // );
        // assert_eq!(
        //     column.last().unwrap(),
        //     &"http://purl.obolibrary.org/obo/RO_0002131"
        // );
        let mut column = subject.get_column_as_vec_str("text");
        column.sort();
        dbg!(column.first().unwrap());
        dbg!(column.last().unwrap());
        // assert_eq!(
        //     column.first().unwrap(),
        //     &"**definition** An entity that has temporal parts and that happens, unfolds or develops through time.\n**has exact synonym** has temporal part\n**has exact synonym** through time\n**has exact synonym** unfolds in time\n**label** occurrent"
        // );

        //         // Make the query data
        //         let mut message_map = HashMap::<String, IPCMessage>::new();
        //         let chat = AvailableInterfaceSubjects::UserMessages
        //             .to_subject_builder(None)
        //             .append_new_user_query_str(
        //                 r#"label: Fanconi syndrome
        // definition: A renal tubular transport disease of the proximal renal tubes characterized by glucosuria, phosphaturia, generalized aminoaciduria and HCO3 wasting.
        // has exact synonym: Lignac-Fanconi syndrome
        // has exact synonym: adult Fanconi Anemia
        // has exact synonym: Congenital Fanconi syndrome
        // has exact synonym: Fanconi-de Toni syndrome
        // has exact synonym: Infantile nephropathic cystinosis
        // has exact synonym: Fanconi-de-Toni syndrome
        // has exact synonym: De Toni-Fanconi syndrome
        // has exact synonym: adult Fanconi syndrome
        // has exact synonym: deToni Fanconi syndrome
        // has exact match: MESH:D005198"#,
        //                 "user",
        //             )?
        //             .build()?;
        //         let _ = message_map.insert(
        //             chat.get_name().to_string(),
        //             IPCMessage::get_builder()
        //                 .with_message(chat.to_ipc_stream()?)
        //                 .with_subject(chat.get_name())
        //                 .with_update(&Publication::Extend {
        //                     subject_name: chat.get_name().to_string(),
        //                 })
        //                 .with_publisher(network_arc.get_name())
        //                 .make_name()?
        //                 .build()?,
        //         );

        //         // 2. Run the session
        //         let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        //         let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        //         let subject_names = extended_diagnostic_subjects
        //             .iter()
        //             .map(|s| s.as_str())
        //             .chain(["EmbeddingScores","Documents","UserQueries",
        //                 "pivot_object_property_entity_s","pivot_class_entity_s",
        //                 "select_annotation_property_entity_s","pivot_annotation_property_entity_s"])
        //             .collect::<Vec<_>>();
        //         write_diagnostic_subjects_to_csv(
        //             &subject_names,
        //             network_arc.runtime_env(),
        //             network_arc.get_name())
        //             .await?;

        // let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
        //     subject_name: AvailableSubjects::SessionErrors.to_string(),
        // }
        // .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        // .unwrap()
        // .try_collect()
        // .await?;
        // if !batches.is_empty() {
        //     let subject = Subject::get_builder()
        //         .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
        //         .with_record_batches(batches)?
        //         .build()?;
        //     println!(
        //         "{}\n{}",
        //         AvailableSubjects::SessionErrors,
        //         String::from_utf8(subject.to_csv(b',', true)?)?
        //     );
        // }
        // let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
        //     subject_name: AvailableSubjects::SessionEvents.to_string(),
        // }
        // .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        // .unwrap()
        // .try_collect()
        // .await?;
        // if !batches.is_empty() {
        //     let subject = Subject::get_builder()
        //         .with_name(AvailableSubjects::SessionEvents.to_string().as_str())
        //         .with_record_batches(batches)?
        //         .build()?;
        //     println!(
        //         "{}\n{}",
        //         AvailableSubjects::SessionEvents,
        //         String::from_utf8(subject.to_csv(b',', true)?)?
        //     );
        // }
        // let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
        //     subject_name: AvailableSubjects::SessionMetrics.to_string(),
        // }
        // .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        // .unwrap()
        // .try_collect()
        // .await?;
        // if !batches.is_empty() {
        //     let subject = Subject::get_builder()
        //         .with_name(AvailableSubjects::SessionTraces.to_string().as_str())
        //         .with_record_batches(batches)?
        //         .build()?;
        //     println!(
        //         "{}\n{}",
        //         AvailableSubjects::SessionTraces,
        //         String::from_utf8(subject.to_csv(b',', true)?)?
        //     );
        // }

        // assert_eq!(response.len(), 0);

        // Test PDF extraction, embedding, and retrieval
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::EmbeddingScores.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::EmbeddingScores.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        dbg!(&subject.count_rows());
        // assert_eq!(subject.count_rows(), 5);
        let column = subject.get_column_as_vec_str("chunk_id");
        dbg!(column.first().unwrap());
        // assert_eq!(column.first().unwrap(), &""https://doi.org/10.37184/jlnh.2959-1805.3.9_4_1");
        let column = subject.get_column_as_vec_str("query_id");
        dbg!(column.first().unwrap());
        // assert_eq!(column.first().unwrap(), &"1775410537711065");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        for t in column {
            assert!(t > 0.15); // Threshold used for filtering
        }

        // let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
        //     subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
        // }
        // .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        // .unwrap()
        // .try_collect()
        // .await?;
        // let subject = Subject::get_builder()
        //     .with_name(
        //         AvailableInterfaceSubjects::ToolMessages
        //             .to_string()
        //             .as_str(),
        //     )
        //     .with_record_batches(batches)?
        //     .build()?;
        // dbg!(&subject.count_rows());
        // // assert_eq!(subject.count_rows(), 1);
        // let column = subject.get_column_as_vec_str("role");
        // dbg!(column.first().unwrap());
        // // assert_eq!(column.first().unwrap(), &"tool");
        // let column = subject.get_column_as_vec_str("content");
        // dbg!(column.first().unwrap());
        // // assert_eq!(
        // //     column.first().unwrap(),
        // //     &"[{\"text\":\"Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.The two \"}]"
        // // );
        // let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
        // for t in column {
        //     assert!(t > 0);
        // }

        Ok(())
    }
}
