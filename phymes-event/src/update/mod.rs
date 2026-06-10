mod available_update_events;
mod update_event;

pub use available_update_events::AvailableUpdateEvents;
pub use update_event::{
    SubjectChangedSinceLastRunUpdate, SubjectExistsUpdate, SubjectHasBatchesUpdate,
    UpdateEventTrait,
};
