#[macro_use]
use criterion::Criterion;

fn benchmark_chat_processor(c: &mut Criterion) {
    c.bench_function("chat_processor", |b| {
        b.iter(|| {
            // Simulate processing a chat message
            let message = "Hello, world!";
            let processed_message = message.to_uppercase();
            assert_eq!(processed_message, "HELLO, WORLD!");
        });
    });
}