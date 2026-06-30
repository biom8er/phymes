use std::collections::VecDeque;

use anyhow::anyhow;
use arrow::datatypes::DataType;
use phymes_data::{AvailableOperators, DataColumnOperator, DataConfig, DataStreamManager};
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_network::{DynamicTaskNetworkBuilder, DynamicTaskNetworkNames, NetworkBuilder, NetworkBuilderMermaidTrait, PipelineTaskNetworkBuilder};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, DataEncoding, DataFormat};
use phymes_streams::LimitConfig;
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait};

use crate::{EmbedTextNetworkBuilder, ExtractPDFNetworkBuilder, RetrieveTextNetworkBuilder};

/// Tabular (columnar data) operator network
/// 
/// # Notes
/// * Pre-specified operators than can be called sequentially to build a full SQL-like SELECT command
/// * Results can be exported to CSV, JSON, HTML, TXT, etc.
/// * Function-calling LLM "assist" for operator calling hints
/// * View of all operators calls sorted by timestamp
/// 
/// # Notes
/// ## Interface tasks
/// * ExtractTabular: from CSV (UserCsv) or JSON (UserJson)
/// * PackTabular: as CSV (AssistantCsv) or JSON (AssistantJson)
/// * ApplyTemplate: as HTML or TXT (AssistantScript)
/// 
/// ## Operator support
/// * SELECT: Supports Unary and Binary data operator support with default HTML table view
/// * DESCRIBE: SubjectsNumRows, SubjectsSchema, TasksProcessorsSubscriptionsPublications
/// 
/// ## Operator order
/// ### Pre-operators (optional)
/// 1. `BatchCoalesce` to optimize the batch sizes for downstream operations
/// 2. `Select` (i.e., Project) to limit the columns processed
/// 3. `Limit` (i.e., Fetch) to limit the number of rows processed
/// 
/// ### Operator chain
/// 1. Unary | Binary operators
/// 
/// ### Post-operators (optional)
/// 1. Optional `Select` (i.e., AS) to rename/reorder/transform columns
/// 2. Optional `Limit` to limit the number of rows returned
pub struct TabularDataOperatorNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
}

impl TabularDataOperatorNetworkBuilder {
    /// Helper to create a Dynamic Unary Operator Network
    fn unary_network_builder(processor: AvailableProcessors, subject_name_lhs: &str) -> NetworkBuilder {
        let task_name = processor.to_string();
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&DynamicTaskNetworkNames::Processor(&task_name).to_string()),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let builder = DynamicTaskNetworkBuilder {
            network_name: task_name,
            is_dynamic: true,
            processor: processor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name_lhs.to_string(),
            },
            publication: Publication::Replace {
                subject_name: subject_name_lhs.to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_processor,
            ..Default::default()
        };
        builder.build_dynamic()
    }

    /// Helper to create a Dynamic Binary Operator Network
    fn binary_network_builder(processor: AvailableProcessors, subject_name_lhs: &str, subject_name_rhs: &str) -> NetworkBuilder {
        let task_name = processor.to_string();
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&DynamicTaskNetworkNames::Processor(&task_name).to_string()),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let builder = DynamicTaskNetworkBuilder {
            network_name: task_name,
            is_dynamic: true,
            processor: processor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name_lhs.to_string(),
            },
            subscription_rhs: Some(Subscription::AlwaysAllRecordBatches {
                subject_name: subject_name_rhs.to_string(),
            }),
            publication: Publication::Replace {
                subject_name: subject_name_lhs.to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_processor,
            ..Default::default()
        };
        builder.build_dynamic()
    }
}

impl Default for TabularDataOperatorNetworkBuilder {
    fn default() -> Self {
        // Constant names used throughout the network
        let subject_name_lhs = "LHS";
        let subject_name_rhs = "RHS";

        // --- Attachments ---
        // UserCsv: here

        // --- Unary | Binary: Lhs ---
        // Extract Tabular data

        // --- Binary: Rhs ---
        // Extract Tabular data
        let mut tabular_data_operator_network_builder = NetworkBuilder::default(); 

        // JOIN network builder
        // Instantiate the LHS and RHS subjects ONLY once here
        let network_builder = {
            let task_name = AvailableProcessors::Join.to_string();
            let subject = AvailableSubjects::Bytes
                .to_subject(
                    Some(&DynamicTaskNetworkNames::Processor(&task_name).to_string()),
                    None,
                )
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::Bytes
                .to_subject(Some(subject_name_lhs), None)
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::Bytes
                .to_subject(Some(subject_name_rhs), None)
                .unwrap();
            let subject_rhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::Bytes
                .to_subject(Some(subject_name_lhs), None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: task_name,
                is_dynamic: true,
                processor: AvailableProcessors::Join,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_lhs.get_name().to_string(),
                },
                subscription_rhs: Some(Subscription::AlwaysAllRecordBatches {
                    subject_name: subject_rhs.get_name().to_string(),
                }),
                publication: Publication::Replace {
                    subject_name: subject_out.get_name().to_string(),
                },
                subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
                subject_lhs: Some(subject_lhs),
                subject_rhs: Some(subject_rhs),
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let tabular_data_operator_network_builder = tabular_data_operator_network_builder.extend(network_builder).unwrap();

        // All unary operators (except ApplyTemplate)
        let unary_operators = [AvailableProcessors::Select,
            AvailableProcessors::Sort,
            AvailableProcessors::GroupBy,
            AvailableProcessors::Filter,
            AvailableProcessors::Pivot,
            AvailableProcessors::Melt,
            AvailableProcessors::LimitProcessor,
            AvailableProcessors::CoalesceProcessor,
        ];
        let tabular_data_operator_network_builder = unary_operators
            .into_iter()
            .map(|op| Self::unary_network_builder(op, subject_name_lhs))
            .reduce(|tabular_data_operator_network_builder, e| tabular_data_operator_network_builder.extend(e).unwrap())
            .unwrap();

        // All binary operators (except Join, which was initialized earlier)
        let binary_operators = [AvailableProcessors::Patch,
            AvailableProcessors::Diff,
            AvailableProcessors::AggregatorProcessor,
        ];
        let tabular_data_operator_network_builder = binary_operators
            .into_iter()
            .map(|op| Self::binary_network_builder(op, subject_name_lhs, subject_name_rhs))
            .reduce(|tabular_data_operator_network_builder, e| tabular_data_operator_network_builder.extend(e).unwrap())
            .unwrap();

        // ApplyTemplate
        let network_builder = {
            let task_name = AvailableProcessors::ApplyTemplate.to_string();
            let subject = AvailableSubjects::Bytes
                .to_subject(
                    Some(&DynamicTaskNetworkNames::Processor(&task_name).to_string()),
                    None,
                )
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::Bytes
                .to_subject(Some(subject_name_lhs), None)
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableSubjects::Bytes
                .to_subject(Some(subject_name_rhs), None)
                .unwrap();
            let subject_rhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::UserScript
                .to_subject(None, None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: task_name,
                is_dynamic: true,
                processor: AvailableProcessors::ApplyTemplate,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_lhs.get_name().to_string(),
                },
                publication: Publication::Replace {
                    subject_name: subject_out.get_name().to_string(),
                },
                subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
                subject_lhs: Some(subject_lhs),
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let tabular_data_operator_network_builder = tabular_data_operator_network_builder.extend(network_builder).unwrap();

        // CSV attachment for each operator step
        let network_builder = {
            let network_name = &AvailableProcessors::PackTabular.to_string();
            let config = DataConfig {
                lhs_name: Some(subject_name_lhs.to_string()),
                encoding: Some(DataEncoding::default()),
                format: Some(DataFormat::None),
                doc_name: Some("result".to_string()),
                schema: Some(AvailableSubjects::default()),
                cpu: false,
                operator: AvailableOperators::PackTabular,
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
            let subject = AvailableInterfaceSubjects::AssistantScript
                .to_subject(None, None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                is_dynamic: false,
                processor: AvailableProcessors::PackTabular,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_name_lhs.to_string(),
                },
                publication: Publication::Replace {
                    subject_name: subject_out.get_name().to_string(),
                },
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };

        TabularDataOperatorNetworkBuilder {
            inner: Some(tabular_data_operator_network_builder.with_name("tabular_data_operator_network")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_data::make_pdf_document_page_per_content;
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{DynamicTaskNetworkNames, NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream};
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, create_attachments_batch, create_queries_batch
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test]
    async fn test_tabular_data_operator_network() -> Result<()> {
        // Initialize the session
        let tabular_data_operator_network_builder = TabularDataOperatorNetworkBuilder::default().inner.take().unwrap();
        let network_name = tabular_data_operator_network_builder.name.clone().unwrap();
        let (network, session_messages) = tabular_data_operator_network_builder
            .with_runtime_env(
                RuntimeEnv::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_max_steps(20)
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

        // Create the PDF document
        let document_texts = &[
            "Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.\n\nA linear chain of amino acid residues is called a polypeptide. A protein contains at least one long polypeptide. Short polypeptides, containing less than 20–30 residues, are rarely considered to be proteins and are commonly called peptides. The individual amino acid residues are bonded together by peptide bonds and adjacent amino acid residues. The sequence of amino acid residues in a protein is defined by the sequence of a gene, which is encoded in the genetic code. In general, the genetic code specifies 20 standard amino acids; but in certain organisms the genetic code can include selenocysteine and—in certain archaea—pyrrolysine. Shortly after or even during synthesis, the residues in a protein are often chemically modified by post-translational modification, which alters the physical and chemical properties, folding, stability, activity, and ultimately, the function of the proteins. Some proteins have non-peptide groups attached, which can be called prosthetic groups or cofactors. Proteins can work together to achieve a particular function, and they often associate to form stable protein complexes.\n\nOnce formed, proteins only exist for a certain period and are then degraded and recycled by the cell's machinery through the process of protein turnover. A protein's lifespan is measured in terms of its half-life and covers a wide range. They can exist for minutes or years with an average lifespan of 1-2 days in mammalian cells. Abnormal or misfolded proteins are degraded more rapidly either due to being targeted for destruction or due to being unstable.\n\nLike other biological macromolecules such as polysaccharides and nucleic acids, proteins are essential parts of organisms and participate in virtually every process within cells. Many proteins are enzymes that catalyse biochemical reactions and are vital to metabolism. Some proteins have structural or mechanical functions, such as actin and myosin in muscle, and the cytoskeleton's scaffolding proteins that maintain cell shape. Other proteins are important in cell signaling, immune responses, cell adhesion, and the cell cycle. In animals, proteins are needed in the diet to provide the essential amino acids that cannot be synthesized. Digestion breaks the proteins down for metabolic use.",
            "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.\n\nThe two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.\n\nBoth strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a process called translation.\n\nWithin eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed.",
            "Lipids are a broad group of organic compounds which include fats, waxes, sterols, fat-soluble vitamins (such as vitamins A, D, E and K), monoglycerides, diglycerides, phospholipids, and others. The functions of lipids include storing energy, signaling, and acting as structural components of cell membranes.[3][4] Lipids have applications in the cosmetic and food industries, and in nanotechnology.[5]\n\nLipids may be broadly defined as hydrophobic or amphiphilic small molecules; the amphiphilic nature of some lipids allows them to form structures such as vesicles, multilamellar/unilamellar liposomes, or membranes in an aqueous environment. Biological lipids originate entirely or in part from two distinct types of biochemical subunits or building-blocks: ketoacyl and isoprene groups.[3] Using this approach, lipids may be divided into eight categories: fatty acyls, glycerolipids, glycerophospholipids, sphingolipids, saccharolipids, and polyketides (derived from condensation of ketoacyl subunits); and sterol lipids and prenol lipids (derived from condensation of isoprene subunits).[3]\n\nAlthough the term lipid is sometimes used as a synonym for fats, fats are a subgroup of lipids called triglycerides. Lipids also encompass molecules such as fatty acids and their derivatives (including tri-, di-, monoglycerides, and phospholipids), as well as other sterol-containing metabolites such as cholesterol.[6] Although humans and other mammals use various biosynthetic pathways both to break down and to synthesize lipids, some essential lipids cannot be made this way and must be obtained from the diet.\n\n",
            "The cell is the basic structural and functional unit of all forms of life. Every cell consists of cytoplasm enclosed within a membrane; many cells contain organelles, each with a specific function. The term comes from the Latin word cellula meaning 'small room'. Most cells are only visible under a microscope. Cells emerged on Earth about 4 billion years ago. All cells are capable of replication, protein synthesis, and motility.\n\nCells are broadly categorized into two types: eukaryotic cells, which possess a nucleus, and prokaryotic cells, which lack a nucleus but have a nucleoid region. Prokaryotes are single-celled organisms such as bacteria, whereas eukaryotes can be either single-celled, such as amoebae, or multicellular, such as some algae, plants, animals, and fungi. Eukaryotic cells contain organelles including mitochondria, which provide energy for cell functions, chloroplasts, which in plants create sugars by photosynthesis, and ribosomes, which synthesise proteins.\n\nCells were discovered by Robert Hooke in 1665, who named them after their resemblance to cells inhabited by Christian monks in a monastery. Cell theory, developed in 1839 by Matthias Jakob Schleiden and Theodor Schwann, states that all organisms are composed of one or more cells, that cells are the fundamental unit of structure and function in all living organisms, and that all cells come from pre-existing cells.",
        ];
        let mut pdf = make_pdf_document_page_per_content(document_texts, true);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes)?;
        let filename = vec!["wiki_dna".to_string()];
        let extension = vec!["pdf".to_string()];
        let bytes = vec![bytes];
        let metadata = vec!["user".to_string()];
        let timestamp = vec![0_i64];
        let batch = create_attachments_batch(filename, extension, bytes, metadata, timestamp)?;
        let blob = AvailableInterfaceSubjects::UserPdf
            .to_subject_builder(None)
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            blob.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(blob.get_name())
                .with_subject(blob.get_name())
                .with_update(&Publication::Replace {
                    subject_name: blob.get_name().to_string(),
                })
                .with_message(blob.to_ipc_stream()?)
                .with_publisher(network_arc.get_name())
                .make_name()?
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the session (embed documents)
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            let _response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            // Make the test session data
            let mut message_map = HashMap::<String, IPCMessage>::new();

            // Make the query data
            let query_ids = vec!["0".to_string()];
            let text =vec!["What are the four molecules that compose DNA?".to_string()];
            let batch = create_queries_batch(query_ids, text)?;
            let queries = AvailableInterfaceSubjects::UserQueries
                .to_subject_builder(None)
                .with_record_batches(vec![batch])?
                .build()?;
            let _ = message_map.insert(
                queries.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_message(queries.to_ipc_stream()?)
                    .with_subject(queries.get_name())
                    .with_update(&Publication::Extend {
                        subject_name: queries.get_name().to_string(),
                    })
                    .with_publisher(network_arc.get_name())
                    .make_name()?
                    .build()?,
            );

            // 2. Run the session (embed queries and retrieve)
            let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            assert_eq!(response.len(), 0);

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
            assert_eq!(subject.count_rows(), 1);
            let column = subject.get_column_as_vec_str("chunk_id");
            assert_eq!(column.first().unwrap(), &"wiki_dna2PdfText { op: 4, bt: 0, tm: PdfTm { a: 1.0, b: 0.0, c: 0.0, d: 1.0, x: 0.0, y: 0.0 }, td: PdfTd { x: 0, y: 0 }, font: PdfFont { font_name: \"F1\", font_subtype: \"Courier\", base_font: \"Type1\" }, font_size: 48, text: \"\" }_0");
            let column = subject.get_column_as_vec_str("query_id");
            assert_eq!(column.first().unwrap(), &"0");
            let column = subject.get_column_as_vec_primitive::<f32>("score")?;
            for t in column {
                assert!(t > 0.15); // Threshold used for filtering
            }

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableInterfaceSubjects::ToolMessages
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 1);
            let column = subject.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"tool");
            let column = subject.get_column_as_vec_str("content");
            // dbg!(column.first().unwrap());
            assert!(column.first().unwrap().contains(&"[{\"text\":\"Deoxyribonucleic acid (DNA)"));
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for t in column {
                assert!(t > 0);
            }
        }

        Ok(())
    }
}
