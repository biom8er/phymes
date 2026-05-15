/// A session for embedding text querries and documents
///
/// # Notes
pub struct EmbedTextNetworkBuilder<'a> {
    /// Session
    pub network_name: &'a str,
    /// The Asset to use for Text Generation and related parameters
    pub candle_asset: Option<String>,
    pub openai_asset: Option<String>,
    pub weights_config_file: Option<String>,
    pub weights_file: Option<String>,
    pub tokenizer_file: Option<String>,
    pub tokenizer_config_file: Option<String>,
    pub api_url: Option<String>,
    /// The processor to use for text generation
    pub embed_processor: &'a str,
}

impl<'a> Default for EmbedTextNetworkBuilder<'a> {
    fn default() -> Self {
        let (
            candle_asset,
            openai_asset,
            weights_config_file,
            weights_file,
            tokenizer_file,
            tokenizer_config_file,
            api_url,
        ) = if cfg!(feature = "hf_hub") {
            (
                Some("QwenV2_1p5bEmbed".to_string()),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        } else if cfg!(all(feature = "api", not(feature = "candle"))) {
            (
                None,
                Some("NvidiaLlamaV3p2NvEmbedQA1BV2".to_string()),
                None,
                None,
                None,
                None,
                Some("http://0.0.0.0:8001/v1".to_string()),
            )
        } else {
            (
                Some("QuantizedBertEmbed".to_string()),
                None,
                Some(format!(
                    "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                Some(format!(
                    "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                Some(format!(
                    "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                Some(format!(
                    "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                None,
            )
        };
        let generate_text_inference = if cfg!(all(feature = "api", not(feature = "candle"))) {
            "OpenAIEmbedProcessor"
        } else {
            "CandleEmbedProcessor"
        };
        Self {
            network_name: "generate_text_network",
            candle_asset,
            openai_asset,
            weights_config_file,
            weights_file,
            tokenizer_config_file,
            tokenizer_file,
            api_url,
            embed_processor: generate_text_inference,
        }
    }
}

impl<'a> EmbedTextNetworkBuilder<'a> {
    fn embed_text_p(&self) -> String {
        let mut lines = Vec::new();
        if let Some(candle_asset) = &self.candle_asset {
            let line = format!(r#"Utf8 candle_asset "{candle_asset}""#);
            lines.push(line);
        }
        if let Some(openai_asset) = &self.openai_asset {
            let line = format!(r#"Utf8 openai_asset "{openai_asset}""#);
            lines.push(line);
        }
        if let Some(tokenizer_config_file) = &self.tokenizer_config_file {
            let line = format!(r#"Utf8 tokenizer_config_file "{tokenizer_config_file}""#);
            lines.push(line);
        }
        if let Some(tokenizer_file) = &self.tokenizer_file {
            let line = format!(r#"Utf8 tokenizer_file "{tokenizer_file}""#);
            lines.push(line);
        }
        if let Some(weights_config_file) = &self.weights_config_file {
            let line = format!(r#"Utf8 weights_config_file "{weights_config_file}""#);
            lines.push(line);
        }
        if let Some(weights_file) = &self.weights_file {
            let line = format!(r#"Utf8 weights_file "{weights_file}""#);
            lines.push(line);
        }
        if let Some(api_url) = &self.api_url {
            let line = format!(r#"Utf8 api_url "{api_url}""#);
            lines.push(line);
        }
        lines.join("\n\t\t")
    }
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let embed_processor = self.embed_processor;
        format!(
            r#"flowchart TD
	%% ------------------------------------------------------------------------------
	%% Messages to Query
	%% ------------------------------------------------------------------------------
	subgraph select_query_t
	    UserMessages-subject-.->|LastRecordBatch|select_query_from_messages_p-subscribe
	    select_query_from_messages_p-subscribe-->select_query_from_messages_p-processor
	    select_query_from_messages_p-processor-->select_query_from_messages_p-publish
	    select_query_from_messages_p-publish-->|Replace|UserQueries-subject
	end
	embed_text_r-rt@{{shape: subproc, label: embed_text_r}}
	embed_text_r-rt-->select_query_t
	UserMessages-subject@{{shape: doc, label: UserMessages}}
	select_query_from_messages_p-processor@{{shape: rect, label: Select}}
	select_query_from_messages_p-publish@{{shape: fork}}
	select_query_from_messages_p-subscribe@{{shape: diamond, label: All}}
	UserQueries-subject@{{shape: doc, label: UserQueries}}
	%% ------------------------------------------------------------------------------
	%% Embed Query
	%% ------------------------------------------------------------------------------
	subgraph embed_query_t
	    UserQueries-subject-.->|AllRecordBatches|coalesce_query_p-subscribe
	    coalesce_query_p-subscribe-->coalesce_query_p-processor
	    coalesce_query_p-processor-->coalesce_query_p-publish
	    coalesce_query_p-publish-->|Extend|coalesce_query_s-subject
	    coalesce_query_s-subject-->|AllRecordBatches|embed_query_p-subscribe
	    embed_query_p-subscribe-->embed_query_p-processor
	    embed_query_p-processor-->embed_query_p-publish
	    embed_query_p-publish-->|Replace|QueryEmbeddings-subject
	end
	embed_text_r-rt-->embed_query_t
	coalesce_query_p-processor@{{shape: rect, label: CoalesceProcessor}}
	coalesce_query_p-publish@{{shape: fork}}
	coalesce_query_p-subscribe@{{shape: diamond, label: All}}
	coalesce_query_s-subject@{{shape: doc, label: coalesce_query_s}}
	embed_query_p-processor@{{shape: rect, label: {embed_processor}}}
	embed_query_p-publish@{{shape: fork}}
	embed_query_p-subscribe@{{shape: diamond, label: All}}
	QueryEmbeddings-subject@{{shape: doc, label: QueryEmbeddings}}
	%% ------------------------------------------------------------------------------
	%% Embed Documents
	%% ------------------------------------------------------------------------------
	subgraph embed_documents_t
	    Documents-subject-.->|AllRecordBatches|coalesce_documents_p-subscribe
	    coalesce_documents_p-subscribe-->coalesce_documents_p-processor
	    coalesce_documents_p-processor-->coalesce_documents_p-publish
	    coalesce_documents_p-publish-->|Extend|coalesce_documents_s-subject
	    coalesce_documents_s-subject-->|AllRecordBatches|embed_documents_p-subscribe
	    embed_documents_p-subscribe-->embed_documents_p-processor
	    embed_documents_p-processor-->embed_documents_p-publish
	    embed_documents_p-publish-->|Extend|DocumentEmbeddings-subject
	end
	embed_text_r-rt-->embed_documents_t
	Documents-subject@{{shape: doc, label: Documents}}
	coalesce_documents_p-processor@{{shape: rect, label: CoalesceProcessor}}
	coalesce_documents_p-publish@{{shape: fork}}
	coalesce_documents_p-subscribe@{{shape: diamond, label: All}}
	coalesce_documents_s-subject@{{shape: doc, label: coalesce_documents_s}}
	embed_documents_p-processor@{{shape: rect, label: {embed_processor}}}
	embed_documents_p-publish@{{shape: fork}}
	embed_documents_p-subscribe@{{shape: diamond, label: All}}
	DocumentEmbeddings-subject@{{shape: doc, label: DocumentEmbeddings}}
	%% ------------------------------------------------------------------------------"#)
    }
    /// Return the Mermaid.js ER diagram representation of the session
    ///
    /// # Note
    /// * for QWEN, the following cast template should be used
    ///   List-Utf8 cast_templates "['','Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {{ content }}']"
    pub fn as_mermaid_erdiagram(&self) -> String {
        let embed_text_p = self.embed_text_p();
        format!(
            r#"erDiagram
    UserMessages["UserMessages"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    select_query_from_messages_p["select_query_from_messages_p"] {{
        List-Utf8 as_columns "['query_id','text']"
        List-Utf8 cast_datatypes "['Utf8','Utf8']"
        List-Utf8 cast_operators "['Cast','None']"
        List-Utf8 cast_templates "['','']"
        List-Utf8 column_operators "['None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "UserMessages"
        List-Utf8 lhs_values "['timestamp','content']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','']"
        Utf8 lhs_stream "Accumulate"
    }}
    UserQueries["UserQueries"] {{
        Utf8 query_id
        Utf8 text
    }}
	coalesce_query_p["coalesce_query_p"] {{
	    Int64 fetch "8"
	    Utf8 summary_format "None"
	}}
	embed_query_p["embed_query_p"] {{
	    Utf8 documents "coalesce_query_s"
	    Boolean cpu "false"
	    Utf8 encoding_format "float"
	    Utf8 input_type "query"
	    Utf8 modality "text"
        {embed_text_p}
	}}
	QueryEmbeddings["QueryEmbeddings"] {{
	    Utf8 query_id
	    List-Float32 embedding
	}}
	Documents["Documents"] {{
        Utf8 chunk_id
        Utf8 document_id
        Utf8 text
	}}
	coalesce_documents_p["coalesce_documents_p"] {{
	    Int64 fetch "4"
	    Utf8 summary_format "None"
	}}
	embed_documents_p["embed_documents_p"] {{
	    Utf8 documents "coalesce_documents_s"
	    Boolean cpu "false"
	    Utf8 encoding_format "float"
	    Utf8 input_type "passage"
	    Utf8 modality "text"
        {embed_text_p}
	}}
	DocumentEmbeddings["DocumentEmbeddings"] {{
	    Utf8 chunk_id
	    Utf8 document_id
	    List-Float32 embedding
	}}"#)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAppsTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait,
        NetworkStreamStep, NetworkStreamStepTrait,
    };
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait,
        create_documents_batch,
    };
    use phymes_streams::ChatBuilderTraitExt;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_embed_text_network() -> Result<()> {
        // Initialize the session
        let embed_text_network = EmbedTextNetworkBuilder::default();
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &embed_text_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &embed_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(embed_text_network.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Documents data
        let chunk_id = [
            "WikiBioComponents_2_0",
            "WikiBioComponents_2_1",
            "WikiBioComponents_2_2",
            "WikiBioComponents_2_3",
            "WikiBioComponents_2_4",
            "WikiBioComponents_2_5",
            "WikiBioComponents_2_6",
            "WikiBioComponents_3_0",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let document_id = [
            "WikiBioComponents",
            "WikiBioComponents",
            "WikiBioComponents",
            "WikiBioComponents",
            "WikiBioComponents",
            "WikiBioComponents",
            "WikiBioComponents",
            "WikiBioComponents",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let text = [
            "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.The two ",
            "olecules that are essential for all known forms of life.The two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and t",
            "hosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; t",
            "ged purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.Both strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of ",
            "us antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a ",
            "trands specify the sequence of amino acids within proteins in a process called translation.Within eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or ",
            "clear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed. ",
            "Lipids are a broad group of organic compounds which include fats, waxes, sterols, fat-soluble vitamins (such as vitamins A, D, E and K), monoglycerides, diglycerides, phospholipids, and others. The functions of lipids include storing energy, signaling, and acting as structural components of cell membranes.[3][4] Lipids have applications in the cosmetic and food industries, and in nanotechnology.[5]Lipids may be broadly defined as hydrophobic or amphiphilic small molecules; the amphiphilic nature of some lip"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let batch = create_documents_batch(chunk_id, document_id, text)?;
        let document = AvailableSubjects::Documents.to_subject(None, Some(vec![batch]))?;
        let document_message = IPCMessage::get_builder()
            .with_message(document.to_ipc_stream()?)
            .with_subject(document.get_name())
            .with_update(&Publication::Extend {
                subject_name: document.get_name().to_string(),
            })
            .with_publisher(embed_text_network.network_name)
            .make_name()?
            .build()?;

        // Chat message
        let chat = AvailableInterfaceSubjects::UserMessages
            .to_subject_builder(None)
            .append_new_user_query_str("What are the four molecules that compose DNA?", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(embed_text_network.network_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![chat_message, document_message]);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // Run the first superstep
            let response = NetworkStreamStep::run_superstep(Arc::clone(&network_arc), message_map)
                .await?
                .unwrap();

            assert_eq!(response.len(), 0);

            {
                // Test supsersteps
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
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("query_id");
                assert!(!column.is_empty());
                let column = subject.get_column_as_vec_str("text");
                assert_eq!(
                    column.first().unwrap(),
                    &"What are the four molecules that compose DNA?"
                );
            }

            // Run the second superstep
            let response = NetworkStreamStep::run_superstep(
                Arc::clone(&network_arc),
                HashMap::<String, IPCMessage>::new(),
            )
            .await?
            .unwrap();

            assert_eq!(response.len(), 0);

            {
                // Test supsersteps
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableSubjects::QueryEmbeddings.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(AvailableSubjects::QueryEmbeddings.to_string().as_str())
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("query_id");
                assert!(!column.is_empty());
                let column = subject.get_column_as_vec_nested_primitive::<f32>("embedding")?;
                assert_eq!(column.len(), 1);
                #[cfg(feature = "hf_hub")]
                assert_eq!(column.first().unwrap().len(), 1536);
                #[cfg(not(feature = "hf_hub"))]
                assert_eq!(column.first().unwrap().len(), 384);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableSubjects::DocumentEmbeddings.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(AvailableSubjects::DocumentEmbeddings.to_string().as_str())
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 8);
                let column = subject.get_column_as_vec_str("chunk_id");
                assert_eq!(column.first().unwrap(), &"WikiBioComponents_2_0");
                assert_eq!(column.last().unwrap(), &"WikiBioComponents_3_0");
                let column = subject.get_column_as_vec_str("document_id");
                assert_eq!(column.first().unwrap(), &"WikiBioComponents");
                assert_eq!(column.last().unwrap(), &"WikiBioComponents");
                let column = subject.get_column_as_vec_nested_primitive::<f32>("embedding")?;
                assert_eq!(column.len(), 8);
                #[cfg(feature = "hf_hub")]
                assert_eq!(column.first().unwrap().len(), 1536);
                #[cfg(not(feature = "hf_hub"))]
                assert_eq!(column.first().unwrap().len(), 384);
            }
        }
        Ok(())
    }
}
