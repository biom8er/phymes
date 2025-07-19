use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use parking_lot::RwLock;
use phymes_agents::session_plans::{
    agent_session_builder::AgentSessionBuilderTrait,
    chat_agent_session::{
        ChatAgentSession,
        test_chat_agent_session::{bench_chat_agent_session_1, bench_chat_agent_session_2},
    },
};
use phymes_core::{
    metrics::ArrowTaskMetricsSet,
    session::session_context::{SessionStreamState, get_metrics_as_table},
    table::arrow_table::ArrowTableTrait,
};

fn benchmark_chat_agent_session(c: &mut Criterion) {
    // Cases for different input/output lengths
    let user_content_vec = vec![
        // Case 1: Short input, Long output
        (
            "Write a python function to count prime numbers up to N with complete docstrings.",
            "Please provide an example using the functions.",
        ),
        // Case 2: Long input, Short output
        (
            r#"What are the four molecules that compose DNA?\nPlease use the following document to answer the questions:\nDeoxyribonucleic acid ([1] DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.\n\nThe two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.\n\nBoth strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a process called translation.\n\nWithin eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed.\n\nProperties\n\nChemical structure of DNA; hydrogen bonds shown as dotted lines. Each end of the double helix has an exposed 5' phosphate on one strand and an exposed 3′ hydroxyl group (—OH) on the other.\nDNA is a long polymer made from repeating units called nucleotides.[6][7] The structure of DNA is dynamic along its length, being capable of coiling into tight loops and other shapes.[8] In all species it is composed of two helical chains, bound to each other by hydrogen bonds. Both chains are coiled around the same axis, and have the same pitch of 34 ångströms (3.4 nm). The pair of chains have a radius of 10 Å (1.0 nm).[9] According to another study, when measured in a different solution, the DNA chain measured 22–26 Å (2.2–2.6 nm) wide, and one nucleotide unit measured 3.3 Å (0.33 nm) long.[10] The buoyant density of most DNA is 1.7g/cm3.[11]\n\nDNA does not usually exist as a single strand, but instead as a pair of strands that are held tightly together.[9][12] These two long strands coil around each other, in the shape of a double helix. The nucleotide contains both a segment of the backbone of the molecule (which holds the chain together) and a nucleobase (which interacts with the other DNA strand in the helix). A nucleobase linked to a sugar is called a nucleoside, and a base linked to a sugar and to one or more phosphate groups is called a nucleotide. A biopolymer comprising multiple linked nucleotides (as in DNA) is called a polynucleotide.[13]\n\nThe backbone of the DNA strand is made from alternating phosphate and sugar groups.[14] The sugar in DNA is 2-deoxyribose, which is a pentose (five-carbon) sugar. The sugars are joined by phosphate groups that form phosphodiester bonds between the third and fifth carbon atoms of adjacent sugar rings. These are known as the 3′-end (three prime end), and 5′-end (five prime end) carbons, the prime symbol being used to distinguish these carbon atoms from those of the base to which the deoxyribose forms a glycosidic bond.[12]\n\nTherefore, any DNA strand normally has one end at which there is a phosphate group attached to the 5′ carbon of a ribose (the 5′ phosphoryl) and another end at which there is a free hydroxyl group attached to the 3′ carbon of a ribose (the 3′ hydroxyl). The orientation of the 3′ and 5′ carbons along the sugar-phosphate backbone confers directionality (sometimes called polarity) to each DNA strand. In a nucleic acid double helix, the direction of the nucleotides in one strand is opposite to their direction in the other strand: the strands are antiparallel. The asymmetric ends of DNA strands are said to have a directionality of five prime end (5′ ), and three prime end (3′), with the 5′ end having a terminal phosphate group and the 3′ end a terminal hydroxyl group. One major difference between DNA and RNA is the sugar, with the 2-deoxyribose in DNA being replaced by the related pentose sugar ribose in RNA.[12]\n\n\nA section of DNA. The bases lie horizontally between the two spiraling strands[15].\nThe DNA double helix is stabilized primarily by two forces: hydrogen bonds between nucleotides and base-stacking interactions among aromatic nucleobases.[16] The four bases found in DNA are adenine (A), cytosine (C), guanine (G) and thymine (T). These four bases are attached to the sugar-phosphate to form the complete nucleotide, as shown for adenosine monophosphate. Adenine pairs with thymine and guanine pairs with cytosine, forming A-T and G-C base pairs.[17][18]\n\nNucleobase classification\nThe nucleobases are classified into two types: the purines, A and G, which are fused five- and six-membered heterocyclic compounds, and the pyrimidines, the six-membered rings C and T.[12] A fifth pyrimidine nucleobase, uracil (U), usually takes the place of thymine in RNA and differs from thymine by lacking a methyl group on its ring. In addition to RNA and DNA, many artificial nucleic acid analogues have been created to study the properties of nucleic acids, or for use in biotechnology.[19]\n\nNon-canonical bases\n\nModified bases occur in DNA. The first of these recognized was 5-methylcytosine, which was found in the genome of Mycobacterium tuberculosis in 1925.[20] The reason for the presence of these noncanonical bases in bacterial viruses (bacteriophages) is to avoid the restriction enzymes present in bacteria. This enzyme system acts at least in part as a molecular immune system protecting bacteria from infection by viruses.[21] Modifications of the bases cytosine and adenine, the more common and modified DNA bases, play vital roles in the epigenetic control of gene expression in plants and animals.[22]\n\nReferences\n1. "deoxyribonucleic acid". Merriam-Webster.com Dictionary. Merriam-Webster.\n2. Alberts B, Johnson A, Lewis J, Raff M, Roberts K, Walter P (2014). Molecular Biology of the Cell (6th ed.). Garland. p. Chapter 4: DNA, Chromosomes and Genomes. ISBN 978-0-8153-4432-2. Archived from the original on 14 July 2014.\n3. Purcell A. "DNA". Basic Biology. Archived from the original on 5 January 2017.\n4. "Uracil". Genome.gov. Retrieved 21 November 2019.\n5. Russell P (2001). iGenetics. New York: Benjamin Cummings. ISBN 0-8053-4553-1.\n6. Saenger W (1984). Principles of Nucleic Acid Structure. New York: Springer-Verlag. ISBN 0-387-90762-9.\n7. Alberts B, Johnson A, Lewis J, Raff M, Roberts K, Peter W (2002). Molecular Biology of the Cell (Fourth ed.). New York and London: Garland Science. ISBN 0-8153-3218-1. OCLC 145080076. Archived from the original on 1 November 2016.\n8. Irobalieva RN, Fogg JM, Catanese DJ, Catanese DJ, Sutthibutpong T, Chen M, Barker AK, Ludtke SJ, Harris SA, Schmid MF, Chiu W, Zechiedrich L (October 2015). "Structural diversity of supercoiled DNA". Nature Communications. 6 (1): 8440. Bibcode:2015NatCo...6.8440I. doi:10.1038/ncomms9440. ISSN 2041-1723. PMC 4608029. PMID 26455586.\n9. Watson JD, Crick FH (April 1953). "Molecular structure of nucleic acids; a structure for deoxyribose nucleic acid" (PDF). Nature. 171 (4356): 737–38. Bibcode:1953Natur.171..737W. doi:10.1038/171737a0. ISSN 0028-0836. PMID 13054692. S2CID 4253007. Archived (PDF) from the original on 4 February 2007.\n10. Mandelkern M, Elias JG, Eden D, Crothers DM (October 1981). "The dimensions of DNA in solution". Journal of Molecular Biology. 152 (1): 153–61. doi:10.1016/0022-2836(81)90099-1. ISSN 0022-2836. PMID 7338906.\n11. Arrighi, Frances E.; Mandel, Manley; Bergendahl, Janet; Hsu, T. C. (June 1970). "Buoyant densities of DNA of mammals". Biochemical Genetics. 4 (3): 367–376. doi:10.1007/BF00485753. ISSN 0006-2928. PMID 4991030. S2CID 27950750.\n12. Berg J, Tymoczko J, Stryer L (2002). Biochemistry. W.H. Freeman and Company. ISBN 0-7167-4955-6.\n13. IUPAC-IUB Commission on Biochemical Nomenclature (CBN) (December 1970). "Abbreviations and Symbols for Nucleic Acids, Polynucleotides and their Constituents. Recommendations 1970". The Biochemical Journal. 120 (3): 449–54. doi:10.1042/bj1200449. ISSN 0306-3283. PMC 1179624. PMID 5499957. Archived from the original on 5 February 2007.\n14. Ghosh A, Bansal M (April 2003). "A glossary of DNA structures from A to Z". Acta Crystallographica Section D. 59 (Pt 4): 620–26. Bibcode:2003AcCrD..59..620G. doi:10.1107/S0907444903003251. ISSN 0907-4449. PMID 12657780.\n15. Edwards KJ, Brown DG, Spink N, Skelly JV, Neidle S. "RCSB PDB – 1D65: Molecular structure of the B-DNA dodecamer d(CGCAAATTTGCG)2. An examination of propeller twist and minor-groove water structure at 2.2 A resolution". www.rcsb.org. Retrieved 27 March 2023.\n16. Yakovchuk P, Protozanova E, Frank-Kamenetskii MD (2006). "Base-stacking and base-pairing contributions into thermal stability of the DNA double helix". Nucleic Acids Research. 34 (2): 564–74. doi:10.1093/nar/gkj454. ISSN 0305-1048. PMC 1360284. PMID 16449200.\n17. Tropp BE (2012). Molecular Biology (4th ed.). Sudbury, Mass.: Jones and Barlett Learning. ISBN 978-0-7637-8663-2.\n18. Carr S (1953). "Watson-Crick Structure of DNA". Memorial University of Newfoundland. Archived from the original on 19 July 2016. Retrieved 13 July 2016.\n19. Verma S, Eckstein F (1998). "Modified oligonucleotides: synthesis and strategy for users". Annual Review of Biochemistry. 67: 99–134. doi:10.1146/annurev.biochem.67.1.99. ISSN 0066-4154. PMID 9759484.\n20. Johnson TB, Coghill RD (1925). "Pyrimidines. CIII. The discovery of 5-methylcytosine in tuberculinic acid, the nucleic acid of the tubercle bacillus". Journal of the American Chemical Society. 47: 2838–44. doi:10.1021/ja01688a030. ISSN 0002-7863.\n21. Weigele P, Raleigh EA (October 2016). "Biosynthesis and Function of Modified Bases in Bacteria and Their Viruses". Chemical Reviews. 116 (20): 12655–12687. doi:10.1021/acs.chemrev.6b00114. ISSN 0009-2665. PMID 27319741.\n22. Kumar S, Chinnusamy V, Mohapatra T (2018). "Epigenetics of Modified DNA Bases: 5-Methylcytosine and Beyond". Frontiers in Genetics. 9: 640. doi:10.3389/fgene.2018.00640. ISSN 1664-8021. PMC 6305559. PMID 30619465."#,
            "What nucleobases bind to each other?",
        ),
    ];

    // Cases for different configurations
    let chat_agent_session_1 = ChatAgentSession {
        session_context_name: "session_1",
        chat_processor_name: "chat_processor_1",
        chat_task_name: "chat_task_1",
        runtime_env_name: "rt_1",
        chat_subscription_name: "messages",
        chat_api_url: Some("http://0.0.0.0:8000/v1"),
    };
    let config_vec = vec![chat_agent_session_1];

    // Get the target and GPU configuration
    let wasm = if cfg!(target_arch = "wasm32") {
        "wasm"
    } else {
        "native"
    };
    let gpu = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };
    let candle = if cfg!(feature = "candle") {
        "candle"
    } else {
        "openai_api"
    };

    // Benchmark each configuration with each user content sequentially
    for config in config_vec {
        for user_content in &user_content_vec {
            let id = format!(
                "chat-agent-session_{}_{}_{}_{}",
                user_content.0.len(),
                wasm,
                gpu,
                candle
            );
            let mut iter = 0;
            c.bench_function(id.as_str(), |b| {
                b.iter(|| {
                    let metrics = ArrowTaskMetricsSet::new();
                    let session_ctx = config.make_session_context(metrics.clone()).unwrap();
                    let session_stream_state =
                        Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));
                    // DM: Cannot use tokio::runtime::Runtime in WASM context
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap();
                    let _messages = rt.block_on(async {
                        bench_chat_agent_session_1(
                            Arc::clone(&session_stream_state),
                            &config,
                            user_content.0,
                        )
                        .await
                    });
                    let _messages = rt.block_on(async {
                        bench_chat_agent_session_2(
                            Arc::clone(&session_stream_state),
                            &config,
                            user_content.0,
                        )
                        .await
                    });

                    // Export the metrics to CSV
                    let metrics_table = get_metrics_as_table(metrics, "metrics").unwrap();
                    let target_dir = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                    let pathname = format!("{target_dir}/.cache/metrics/{id}_{iter}.csv");
                    let path = std::path::Path::new(pathname.as_str());
                    let prefix = path.parent().unwrap();
                    std::fs::create_dir_all(prefix).unwrap();
                    let mut file = std::fs::File::create(pathname).unwrap();
                    metrics_table.to_csv_file(&mut file, b',', true).unwrap();

                    // Increment the iteration counter
                    iter += 1;
                });
            });
        }
    }
}

criterion_group!(benches, benchmark_chat_agent_session);
criterion_main!(benches);
