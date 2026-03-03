mod blob;
mod documents;
mod queries;
// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
mod openai_embedding;

pub use blob::{AttachmentBuilderTraitExt, AttachmentsSubject, create_attachments_batch, create_attachments_fields, create_blob_batch, create_blob_fields, create_diff_batch, create_diff_fields,
    create_patch_fields, create_patch_batch, create_workspace_fields, create_workspace_batch, create_workspace_patch_fields, create_workspace_patch_batch, WorkspaceSubject};
pub use documents::{
    create_document_embeddings_fields, create_documents_batch, create_documents_embeddings_batch,
    create_documents_fields, create_embeddings_scores_fields, create_join_chunks_scores_fields,
};
pub use openai_embedding::{EmbeddingRequest, EmbeddingResponse, EncodingFormat};
pub use queries::{
    create_queries_batch, create_queries_fields, create_query_embeddings_batch,
    create_query_embeddings_fields,
};
