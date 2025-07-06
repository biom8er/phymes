use hashbrown::DefaultHashBuilder;

// The HashMap and HashSet implementations that should be used as the uniform defaults
pub type HashMap<K, V, S = DefaultHashBuilder> = hashbrown::HashMap<K, V, S>;
pub type HashSet<T, S = DefaultHashBuilder> = hashbrown::HashSet<T, S>;
