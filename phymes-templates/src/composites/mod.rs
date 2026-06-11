#[cfg(feature = "api")]
mod generate_code_network_builder;
#[cfg(feature = "api")]
mod open_alex_network_builder;
#[cfg(feature = "api")]
mod retrieve_text_pdf_network_builder;

#[cfg(feature = "api")]
pub use generate_code_network_builder::GenerateCodeNetworkBuilder;
#[cfg(feature = "api")]
pub use open_alex_network_builder::OpenAlexNetworkBuilder;
#[cfg(feature = "api")]
pub use retrieve_text_pdf_network_builder::RetrieveTextPDFNetworkBuilder;

