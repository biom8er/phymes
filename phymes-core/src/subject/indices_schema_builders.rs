use std::sync::Arc;

use arrow::{array::{
    ArrayRef, ArrowPrimitiveType, BooleanBuilder, ListBuilder, PrimitiveBuilder, RecordBatch, UInt64Builder
}, datatypes::SchemaRef};

pub struct BTreeIndexBuilder<K: ArrowPrimitiveType, V: ArrowPrimitiveType> {
    node_id: UInt64Builder,
    is_leaf: BooleanBuilder,
    keys: ListBuilder<PrimitiveBuilder<K>>,
    values: ListBuilder<PrimitiveBuilder<V>>,
    children: ListBuilder<UInt64Builder>,
    next_leaf: UInt64Builder,
    schema: SchemaRef,
}

impl<K: ArrowPrimitiveType, V: ArrowPrimitiveType> BTreeIndexBuilder<K, V> {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            node_id: UInt64Builder::new(),
            is_leaf: BooleanBuilder::new(),
            keys: ListBuilder::new(PrimitiveBuilder::<K>::new()),
            values: ListBuilder::new(PrimitiveBuilder::<V>::new()),
            children: ListBuilder::new(UInt64Builder::new()),
            next_leaf: UInt64Builder::new(),
            schema,
        }
    }

    pub fn append_node(
        &mut self,
        id: u64,
        is_leaf: bool,
        keys: &[K::Native],
        values: Option<&[V::Native]>,
        children: Option<&[u64]>,
        next_leaf: Option<u64>,
    ) {
        self.node_id.append_value(id);
        self.is_leaf.append_value(is_leaf);

        // keys
        for k in keys {
            self.keys.values().append_value(*k);
        }
        self.keys.append(true);

        // values
        if let Some(vals) = values {
            for v in vals {
                self.values.values().append_value(*v);
            }
            self.values.append(true);
        } else {
            self.values.append(false);
        }

        // children
        if let Some(ch) = children {
            for c in ch {
                self.children.values().append_value(*c);
            }
            self.children.append(true);
        } else {
            self.children.append(false);
        }

        // next_leaf
        if let Some(n) = next_leaf {
            self.next_leaf.append_value(n);
        } else {
            self.next_leaf.append_null();
        }
    }

    pub fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(self.node_id.finish()) as ArrayRef,
                Arc::new(self.is_leaf.finish()),
                Arc::new(self.keys.finish()),
                Arc::new(self.values.finish()),
                Arc::new(self.children.finish()),
                Arc::new(self.next_leaf.finish()),
            ],
        )
        .unwrap()
    }
}
pub struct HashIndexBuilder<K: ArrowPrimitiveType, V: ArrowPrimitiveType> {
    bucket_id: UInt64Builder,
    hash: UInt64Builder,
    key: PrimitiveBuilder<K>,
    value: PrimitiveBuilder<V>,
    schema: SchemaRef,
}

impl<K: ArrowPrimitiveType, V: ArrowPrimitiveType> HashIndexBuilder<K, V> {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            bucket_id: UInt64Builder::new(),
            hash: UInt64Builder::new(),
            key: PrimitiveBuilder::<K>::new(),
            value: PrimitiveBuilder::<V>::new(),
            schema,
        }
    }

    pub fn append_entry(&mut self, bucket: u64, hash: u64, key: K::Native, value: V::Native) {
        self.bucket_id.append_value(bucket);
        self.hash.append_value(hash);
        self.key.append_value(key);
        self.value.append_value(value);
    }

    pub fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(self.bucket_id.finish()),
                Arc::new(self.hash.finish()),
                Arc::new(self.key.finish()),
                Arc::new(self.value.finish()),
            ],
        )
        .unwrap()
    }
}
pub struct GiSTIndexBuilder<P: ArrowPrimitiveType, T: ArrowPrimitiveType> {
    node_id: UInt64Builder,
    is_leaf: BooleanBuilder,
    predicate: PrimitiveBuilder<P>,
    child_id: UInt64Builder,
    tuple: PrimitiveBuilder<T>,
    schema: SchemaRef,
}

impl<P: ArrowPrimitiveType, T: ArrowPrimitiveType> GiSTIndexBuilder<P, T> {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            node_id: UInt64Builder::new(),
            is_leaf: BooleanBuilder::new(),
            predicate: PrimitiveBuilder::<P>::new(),
            child_id: UInt64Builder::new(),
            tuple: PrimitiveBuilder::<T>::new(),
            schema,
        }
    }

    pub fn append_internal(&mut self, id: u64, predicate: P::Native, child: u64) {
        self.node_id.append_value(id);
        self.is_leaf.append_value(false);
        self.predicate.append_value(predicate);
        self.child_id.append_value(child);
        self.tuple.append_null();
    }

    pub fn append_leaf(&mut self, id: u64, predicate: P::Native, tuple: T::Native) {
        self.node_id.append_value(id);
        self.is_leaf.append_value(true);
        self.predicate.append_value(predicate);
        self.child_id.append_null();
        self.tuple.append_value(tuple);
    }

    pub fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(self.node_id.finish()),
                Arc::new(self.is_leaf.finish()),
                Arc::new(self.predicate.finish()),
                Arc::new(self.child_id.finish()),
                Arc::new(self.tuple.finish()),
            ],
        )
        .unwrap()
    }
}
pub struct SPGiSTIndexBuilder<K: ArrowPrimitiveType, V: ArrowPrimitiveType> {
    node_id: UInt64Builder,
    is_leaf: BooleanBuilder,
    label: PrimitiveBuilder<K>,
    child_ids: ListBuilder<UInt64Builder>,
    key: PrimitiveBuilder<K>,
    value: PrimitiveBuilder<V>,
    schema: SchemaRef,
}

impl<K: ArrowPrimitiveType, V: ArrowPrimitiveType> SPGiSTIndexBuilder<K, V> {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            node_id: UInt64Builder::new(),
            is_leaf: BooleanBuilder::new(),
            label: PrimitiveBuilder::<K>::new(),
            child_ids: ListBuilder::new(UInt64Builder::new()),
            key: PrimitiveBuilder::<K>::new(),
            value: PrimitiveBuilder::<V>::new(),
            schema,
        }
    }

    pub fn append_inner(&mut self, id: u64, label: K::Native, children: &[u64]) {
        self.node_id.append_value(id);
        self.is_leaf.append_value(false);
        self.label.append_value(label);

        for c in children {
            self.child_ids.values().append_value(*c);
        }
        self.child_ids.append(true);

        self.key.append_null();
        self.value.append_null();
    }

    pub fn append_leaf(&mut self, id: u64, key: K::Native, value: V::Native) {
        self.node_id.append_value(id);
        self.is_leaf.append_value(true);
        self.label.append_null();
        self.child_ids.append(false);
        self.key.append_value(key);
        self.value.append_value(value);
    }

    pub fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(self.node_id.finish()),
                Arc::new(self.is_leaf.finish()),
                Arc::new(self.label.finish()),
                Arc::new(self.child_ids.finish()),
                Arc::new(self.key.finish()),
                Arc::new(self.value.finish()),
            ],
        )
        .unwrap()
    }
}
pub struct GINIndexBuilder<K: ArrowPrimitiveType> {
    key: PrimitiveBuilder<K>,
    posting_list: ListBuilder<UInt64Builder>,
    schema: SchemaRef,
}

impl<K: ArrowPrimitiveType> GINIndexBuilder<K> {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            key: PrimitiveBuilder::<K>::new(),
            posting_list: ListBuilder::new(UInt64Builder::new()),
            schema,
        }
    }

    pub fn append_entry(&mut self, key: K::Native, postings: &[u64]) {
        self.key.append_value(key);

        for p in postings {
            self.posting_list.values().append_value(*p);
        }
        self.posting_list.append(true);
    }

    pub fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(self.key.finish()),
                Arc::new(self.posting_list.finish()),
            ],
        )
        .unwrap()
    }
}
pub struct BRINIndexBuilder<S: ArrowPrimitiveType> {
    block_start: UInt64Builder,
    block_end: UInt64Builder,
    summary: PrimitiveBuilder<S>,
    schema: SchemaRef,
}

impl<S: ArrowPrimitiveType> BRINIndexBuilder<S> {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            block_start: UInt64Builder::new(),
            block_end: UInt64Builder::new(),
            summary: PrimitiveBuilder::<S>::new(),
            schema,
        }
    }

    pub fn append_range(&mut self, start: u64, end: u64, summary: S::Native) {
        self.block_start.append_value(start);
        self.block_end.append_value(end);
        self.summary.append_value(summary);
    }

    pub fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(self.block_start.finish()),
                Arc::new(self.block_end.finish()),
                Arc::new(self.summary.finish()),
            ],
        )
        .unwrap()
    }
}