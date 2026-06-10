mod error;
mod session_interface_message;

pub use error::{create_error_message_map, create_error_message_map_stream};
use phymes_diagnostics::HashMap;
use phymes_subject::MappableTrait;
pub use session_interface_message::{
    SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait,
    SessionInterfaceMessageTrait,
};

/// Helper function to create the message map from a vector of messages
pub fn create_message_map<T>(messages: Vec<T>) -> HashMap<String, T>
where
    T: MappableTrait,
{
    let mut incoming_message_map = HashMap::<String, T>::new();
    for message in messages {
        incoming_message_map.insert(message.get_name().to_string(), message);
    }
    incoming_message_map
}
