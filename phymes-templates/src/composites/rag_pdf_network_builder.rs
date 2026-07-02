use phymes_network::{NetworkBuilder, NetworkBuilderMermaidTrait};
use phymes_subject::BuilderTrait;

use crate::{GenerateTextNetworkBuilder, RetrieveTextPDFNetworkBuilder};

/// Retrieval Augmented Generation (RAG) PDF network
pub struct RetrievalAugmentedGenerationPDFNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
}

impl Default for RetrievalAugmentedGenerationPDFNetworkBuilder {
    fn default() -> Self {
        // Retrieve text PDF network
        let rag_pdf_network_builder = RetrieveTextPDFNetworkBuilder::default()
            .inner
            .take()
            .unwrap();

        // Generate text network
        let generate_text_network = GenerateTextNetworkBuilder::default();
        let network_builder = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(generate_text_network.network_name);
        let rag_pdf_network_builder = rag_pdf_network_builder.extend(network_builder).unwrap();

        RetrievalAugmentedGenerationPDFNetworkBuilder {
            inner: Some(rag_pdf_network_builder.with_name("rag_pdf_network")),
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
    use phymes_network::{
        DynamicTaskNetworkNames, NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjectsTrait, create_attachments_batch,
        create_chat_record_batch,
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use crate::{extended_diagnostic_subjects, write_diagnostic_subjects_to_csv};

    use super::*;

    #[tokio::test]
    async fn test_rag_pdf_network() -> Result<()> {
        // Initialize the session
        let rag_pdf_network_builder = RetrievalAugmentedGenerationPDFNetworkBuilder::default()
            .inner
            .take()
            .unwrap();
        let network_name = rag_pdf_network_builder.name.clone().unwrap();
        let (network, session_messages) = rag_pdf_network_builder
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

        // Make the query data
        let role = vec!["user".to_string()];
        let content = vec!["What are the four molecules that compose DNA?".to_string()];
        let timestamp = vec![0_i64];
        let batch = create_chat_record_batch(role, content, timestamp)?;
        let queries = AvailableInterfaceSubjects::UserMessages
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

        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the session
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));

        // DM: Skip actually running the tests as they take too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
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

            // Test RAG
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableInterfaceSubjects::AssistantMessages
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            assert!(subject.count_rows() > 0);
            let column = subject.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            let column = subject.get_column_as_vec_str("content");
            // dbg!(column.first().unwrap());
            assert!(column.first().unwrap().contains("Adenine"));
            assert!(column.first().unwrap().contains("Thymine"));
            assert!(column.first().unwrap().contains("Guanine"));
            assert!(column.first().unwrap().contains("Cytosine"));
            // assert_eq!(column.first().unwrap(), &"");
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
        }

        Ok(())
    }
}
