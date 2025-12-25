mod available_table_subscribe_policies;
mod data_format;
mod data_types;
mod stream;
mod stream_adapter;
mod table_builder;
mod table_publication;
mod table_script;
mod table_subscribe_policy;
mod table_subscription;
mod table_trait;

pub use available_table_subscribe_policies::AvailableTableSubscribePolicies;
pub use data_format::{CsvFormat, DataFormat, JsonFormat, OwlFormat};
pub use data_types::{from_data_type_to_str, from_str_to_data_type, parse_str_to_data_type};
pub use stream::{
    IPCRecordBatchStream, RecordBatchStream, SendableIPCRecordBatchStream,
    SendableRecordBatchStream,
};
pub use stream_adapter::{
    EmptyRecordBatchStream, RecordBatchReceiverStream, RecordBatchReceiverStreamBuilder,
    RecordBatchStreamAdapter,
};
pub use table_builder::{TableBuilder, TableBuilderTrait};
pub use table_publication::{TablePublication, TablePublicationTrait};
pub use table_script::TableScript;
pub use table_subscribe_policy::TableSubscribePolicyTrait;
pub(crate) use table_subscribe_policy::{
    AllTableNamesSubscribe, AllTableSchemasSubscribe, AlwaysSubscribe, AnyTableNameSubscribe,
    AnyTableSchemaSubscribe, ChatContentSubscribe,
};
pub use table_subscription::{TableSubscription, TableSubscriptionTrait};
pub use table_trait::{Table, TableTrait, test_table};

use parking_lot::RwLock;
use phymes_diagnostics::HashMap;
use std::sync::Arc;

/// Table HashMap with Arc/RwLock for thread-safe multiple reads
pub type StateMap = HashMap<String, Arc<RwLock<Table>>>;
