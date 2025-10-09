// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_data::candle_operators::extract_pdf_text::make_pdf_document;
use std::sync::Arc;

use phymes_agents::{
    session_plans::{available_interface_subjects::{create_message_map, AvailableInterfaceSubjects}, document_rag_session::DocumentRAGSession},
    session_traits::agents::{CustomAgentsBuilderTrait, SessionContextBuilderAgentsTrait},
};
use phymes_core::{
    metrics::{SpanMetricsSet, HashMap}, schemas::{available_subjects::AvailableSubjectsTrait, blob::BlobBuilderTraitExt, chat::ChatBuilderTraitExt}, session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, session_stream::SessionStream, session_stream_state::SessionStreamState,
        session_context_builder::SessionContextBuilderTrait,
    }, table::{table_trait::{TableBuilder, TableBuilderTrait, TableTrait}, table_publish::TablePublish}, task::message::{IPCMessage, MessageBuilderTrait, MessageTrait}
};

pub async fn run_main() -> Result<()> {
    // initialize the metrics
    let metrics = SpanMetricsSet::new();

    // initialize the session
    let mut doc_rag_session = DocumentRAGSession::default();
    if cfg!(not(feature = "candle")) {
        doc_rag_session.chat_api_url = Some("http://0.0.0.0:8000/v1");
        doc_rag_session.embed_api_url = Some("http://0.0.0.0:8001/v1");
    }
    let session_ctx = doc_rag_session
        .build()
        .with_metrics(metrics.clone())
        .with_name(doc_rag_session.session_context_name)
        .build_with_tables()?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

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
    let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
        .append_new_user_query_str("What are the four molecules that compose DNA?", "user")?
        .build()?;
    let chat_message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&TablePublish::Extend { table_name: chat.get_name().to_string() })
        .with_publisher(doc_rag_session.session_context_name)
        .make_name()?
        .build()?;
    let blob = AvailableInterfaceSubjects::UserPdf.to_table_builder(None)
        .with_blob(None, Some(".pdf"), &bytes, None)?
        .build()?;
    let blob_message = IPCMessage::get_builder()
        .with_message(blob.to_ipc_stream()?)
        .with_subject(blob.get_name())
        .with_update(&TablePublish::Extend { table_name: blob.get_name().to_string() })
        .with_publisher(doc_rag_session.session_context_name)
        .make_name()?
        .build()?;

    // ----- Query #1 -----
    // Embed the documents
    let message_map = create_message_map(vec![blob_message]);
    let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
    let _response: Vec<HashMap<String, IPCMessage>> =
        session_stream.try_collect().await?;

    // Embed the query and invoke a response
    let message_map = create_message_map(vec![chat_message]);
    let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
    let mut response: Vec<HashMap<String, IPCMessage>> =
        session_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .last_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            doc_rag_session.session_context_name,
            AvailableInterfaceSubjects::AssistantMessages
        ))
        .unwrap()
        .get_message_own();
    let json_data = TableBuilder::new_from_ipc_stream(&bytes)?
        .with_name("")
        .build()?
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    println!(
        "number of rows {}",
        metrics.clone_inner().output_rows().unwrap()
    );
    println!(
        "elasped compute {}",
        metrics.clone_inner().elapsed_compute().unwrap()
    );

    Ok(())
}
