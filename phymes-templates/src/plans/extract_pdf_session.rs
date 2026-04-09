/// A session for extracting and chunking PDF documents
///
/// # Notes
///
/// * Does not yet include image and table extraction
pub struct ExtractPDFSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl<'a> Default for ExtractPDFSession<'a> {
    fn default() -> Self {
        Self {
            session_context_name: "extract_pdf_session",
        }
    }
}

impl<'a> ExtractPDFSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	%% ------------------------------------------------------------------------------
	%% PDF document extraction
	%% ------------------------------------------------------------------------------
	subgraph extract_pdf_t
		UserPdf-subject-.->|LastRecordBatch|extract_pdf_p-subscribe
		extract_pdf_p-subscribe-->extract_pdf_p-processor
		extract_pdf_p-processor-->extract_pdf_p-publish
		extract_pdf_p-publish-->|Extend|extract_pdf_s-subject
		extract_pdf_s-subject-->|AllRecordBatches|chunk_documents_p-subscribe
		chunk_documents_p-subscribe-->chunk_documents_p-processor
		chunk_documents_p-processor-->chunk_documents_p-publish
		chunk_documents_p-publish-->|Extend|Documents-subject
	end
	extract_pdf_r-rt@{shape: subproc, label: extract_pdf_r}
	extract_pdf_r-rt-->extract_pdf_t
	UserPdf-subject@{shape: doc, label: UserPdf}
	extract_pdf_p-processor@{shape: rect, label: ExtractPDF}
	extract_pdf_p-publish@{shape: fork}
	extract_pdf_p-subscribe@{shape: diamond, label: All}
	extract_pdf_s-subject@{shape: doc, label: chunk_documents_task_1}
	chunk_documents_p-processor@{shape: rect, label: ChunkDocuments}
	chunk_documents_p-publish@{shape: fork}
	chunk_documents_p-subscribe@{shape: diamond, label: All}
	Documents-subject@{shape: doc, label: Documents}
	%% ------------------------------------------------------------------------------"#
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    UserPdf["UserPdf"] {
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }
    extract_pdf_p["extract_pdf_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "UserPdf"
        Utf8 lhs_pk "filename"
        List-Utf8 lhs_values "['bytes']"
        Utf8 operator "ExtractPDF"
        Utf8 lhs_stream "Accumulate"
    }
    extract_pdf_s["extract_pdf_s"] {
        Utf8 chunk_id
        Utf8 document_id
        Utf8 text
    }
    chunk_documents_p["chunk_documents_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "document_id"
        Utf8 lhs_name "extract_pdf_s"
        Utf8 lhs_pk "chunk_id"
        List-Utf8 lhs_values "['text']"
        Utf8 operator "ChunkDocuments"
        Utf8 lhs_stream "Accumulate"
    }
	Documents["Documents"] {
        Utf8 chunk_id
        Utf8 document_id
        Utf8 text
	}"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_data::make_pdf_document;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_network::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStreamStep, SessionStreamStepTrait,
    };
    use phymes_schemas::{
        AttachmentBuilderTraitExt, AvailableInterfaceSubjects, AvailableSubjects,
        AvailableSubjectsTrait,
    };
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_extract_pdf_session() -> Result<()> {
        // Initialize the session
        let extract_pdf_session = ExtractPDFSession::default();
        let (session_ctx, session_messages) = SessionContextBuilder::from_mermaid_flowchart(
            extract_pdf_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            extract_pdf_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(extract_pdf_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Create the document message
        let document_texts = &[
            "Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.\n\nA linear chain of amino acid residues is called a polypeptide. A protein contains at least one long polypeptide. Short polypeptides, containing less than 20–30 residues, are rarely considered to be proteins and are commonly called peptides. The individual amino acid residues are bonded together by peptide bonds and adjacent amino acid residues. The sequence of amino acid residues in a protein is defined by the sequence of a gene, which is encoded in the genetic code. In general, the genetic code specifies 20 standard amino acids; but in certain organisms the genetic code can include selenocysteine and—in certain archaea—pyrrolysine. Shortly after or even during synthesis, the residues in a protein are often chemically modified by post-translational modification, which alters the physical and chemical properties, folding, stability, activity, and ultimately, the function of the proteins. Some proteins have non-peptide groups attached, which can be called prosthetic groups or cofactors. Proteins can work together to achieve a particular function, and they often associate to form stable protein complexes.\n\nOnce formed, proteins only exist for a certain period and are then degraded and recycled by the cell's machinery through the process of protein turnover. A protein's lifespan is measured in terms of its half-life and covers a wide range. They can exist for minutes or years with an average lifespan of 1-2 days in mammalian cells. Abnormal or misfolded proteins are degraded more rapidly either due to being targeted for destruction or due to being unstable.\n\nLike other biological macromolecules such as polysaccharides and nucleic acids, proteins are essential parts of organisms and participate in virtually every process within cells. Many proteins are enzymes that catalyse biochemical reactions and are vital to metabolism. Some proteins have structural or mechanical functions, such as actin and myosin in muscle, and the cytoskeleton's scaffolding proteins that maintain cell shape. Other proteins are important in cell signaling, immune responses, cell adhesion, and the cell cycle. In animals, proteins are needed in the diet to provide the essential amino acids that cannot be synthesized. Digestion breaks the proteins down for metabolic use.",
            "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.\n\nThe two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.\n\nBoth strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a process called translation.\n\nWithin eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed.",
            "Lipids are a broad group of organic compounds which include fats, waxes, sterols, fat-soluble vitamins (such as vitamins A, D, E and K), monoglycerides, diglycerides, phospholipids, and others. The functions of lipids include storing energy, signaling, and acting as structural components of cell membranes.[3][4] Lipids have applications in the cosmetic and food industries, and in nanotechnology.[5]\n\nLipids may be broadly defined as hydrophobic or amphiphilic small molecules; the amphiphilic nature of some lipids allows them to form structures such as vesicles, multilamellar/unilamellar liposomes, or membranes in an aqueous environment. Biological lipids originate entirely or in part from two distinct types of biochemical subunits or building-blocks: ketoacyl and isoprene groups.[3] Using this approach, lipids may be divided into eight categories: fatty acyls, glycerolipids, glycerophospholipids, sphingolipids, saccharolipids, and polyketides (derived from condensation of ketoacyl subunits); and sterol lipids and prenol lipids (derived from condensation of isoprene subunits).[3]\n\nAlthough the term lipid is sometimes used as a synonym for fats, fats are a subgroup of lipids called triglycerides. Lipids also encompass molecules such as fatty acids and their derivatives (including tri-, di-, monoglycerides, and phospholipids), as well as other sterol-containing metabolites such as cholesterol.[6] Although humans and other mammals use various biosynthetic pathways both to break down and to synthesize lipids, some essential lipids cannot be made this way and must be obtained from the diet.\n\n",
            "The cell is the basic structural and functional unit of all forms of life. Every cell consists of cytoplasm enclosed within a membrane; many cells contain organelles, each with a specific function. The term comes from the Latin word cellula meaning 'small room'. Most cells are only visible under a microscope. Cells emerged on Earth about 4 billion years ago. All cells are capable of replication, protein synthesis, and motility.\n\nCells are broadly categorized into two types: eukaryotic cells, which possess a nucleus, and prokaryotic cells, which lack a nucleus but have a nucleoid region. Prokaryotes are single-celled organisms such as bacteria, whereas eukaryotes can be either single-celled, such as amoebae, or multicellular, such as some algae, plants, animals, and fungi. Eukaryotic cells contain organelles including mitochondria, which provide energy for cell functions, chloroplasts, which in plants create sugars by photosynthesis, and ribosomes, which synthesise proteins.\n\nCells were discovered by Robert Hooke in 1665, who named them after their resemblance to cells inhabited by Christian monks in a monastery. Cell theory, developed in 1839 by Matthias Jakob Schleiden and Theodor Schwann, states that all organisms are composed of one or more cells, that cells are the fundamental unit of structure and function in all living organisms, and that all cells come from pre-existing cells.",
        ];
        let mut pdf = make_pdf_document(document_texts);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes)?;

        // Wrap into the message
        let blob = AvailableInterfaceSubjects::UserPdf
            .to_subject_builder(None)
            .with_attachment(Some("WikiBioComponents"), Some("pdf"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&Publication::Extend {
                subject_name: blob.get_name().to_string(),
            })
            .with_publisher(extract_pdf_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![blob_message]);
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Run the first superstep
        let response = SessionStreamStep::run_superstep(Arc::clone(&session_ctx_arc), message_map)
            .await?
            .unwrap();

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::Documents.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::Documents.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 21);
            let column = subject.get_column_as_vec_str("chunk_id");
            assert_eq!(column.first().unwrap(), &"WikiBioComponents_1_0");
            assert_eq!(column.last().unwrap(), &"WikiBioComponents_4_2");
            let column = subject.get_column_as_vec_str("document_id");
            assert_eq!(column.first().unwrap(), &"WikiBioComponents");
            assert_eq!(column.last().unwrap(), &"WikiBioComponents");
            let column = subject.get_column_as_vec_str("text");
            assert_eq!(
                column.first().unwrap(),
                &"Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually"
            );
            assert_eq!(
                column.last().unwrap(),
                &"ts, which in plants create sugars by photosynthesis, and ribosomes, which synthesise proteins.Cells were discovered by Robert Hooke in 1665, who named them after their resemblance to cells inhabited by Christian monks in a monastery. Cell theory, developed in 1839 by Matthias Jakob Schleiden and Theodor Schwann, states that all organisms are composed of one or more cells, that cells are the fundamental unit of structure and function in all living organisms, and that all cells come from pre-existing cells. "
            );
        }
        Ok(())
    }
}
