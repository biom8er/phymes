use std::{sync::Arc, fmt::Debug};

use arrow::array::{ArrayRef, ArrowPrimitiveType, PrimitiveBuilder};
use clap::ValueEnum;
use num_traits::{ PrimInt, Signed};
use serde::{Deserialize, Serialize};

/// A fully generic integer range iterator (exclusive end)
#[derive(Debug, Clone)]
struct IntRange<T> {
    current: T,
    end: T,
    step: T,
}

impl<T> IntRange<T>
where
    T: PrimInt + Signed, // Works for signed & unsigned integers
{
    /// Create a new range from `start` to `end` with a given `step`
    pub fn new(start: T, end: T, step: T) -> Self {
        assert!(step != T::zero(), "Step cannot be zero");
        Self { current: start, end, step }
    }
}

impl<T> Iterator for IntRange<T>
where
    T: PrimInt + Signed + Debug,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if (self.step > T::zero() && self.current < self.end)
            || (self.step < T::zero() && self.current > self.end)
        {
            let val = self.current;
            self.current = self.current + self.step;
            Some(val)
        } else {
            None
        }
    }
}

/// Build an apache arrow array sequence from `start` to `end` with a given `step`
pub fn create_arrow_array_sequence<T: ArrowPrimitiveType<Native = T> + PrimInt + Signed + Debug>(start: T, end: T, step: T) -> ArrayRef {
    let mut builder = PrimitiveBuilder::<T>::new();
    for v in IntRange::new(start, end, step) {
        builder.append_value(v);
    }
    Arc::new(builder.finish())
}

/// SubjectSequence
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize, Default, ValueEnum)]
pub enum SubjectSequenceType {
    #[default]
    #[value(name = "UInt32Sequence")]
    UInt32Sequence,
    #[value(name = "Int64Sequence")]
    Int64Sequence,
}