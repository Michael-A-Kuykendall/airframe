use aho_corasick::AhoCorasick;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use libfse::{FseMap, FseOpcode, FseScanner, Rule};

fn bench_attack_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("Attack Simulation");

    // Scenario 1: Clean Traffic (Sparse/No matches)
    // 64KB of random data, unlikely to match anything
    // This tests the "Search" speed (Prefilter vs Byte-Loop)
    let clean_text = "Safe internet traffic with nothing suspicious.".repeat(500);
    let clean_patterns = vec!["exploit", "malware", "virus", "injection"];

    group.throughput(Throughput::Bytes(clean_text.len() as u64));

    // Aho-Corasick Clean
    let ac_clean = AhoCorasick::new(&clean_patterns).unwrap();
    group.bench_function("clean_traffic_aho_corasick", |b| {
        b.iter(|| {
            // Typical usage: iterate matches, do something trivial
            let mut count = 0;
            for _ in ac_clean.find_iter(black_box(clean_text.as_bytes())) {
                count += 1;
            }
            black_box(count);
        })
    });

    // LibFSE Clean
    let rules_clean: Vec<Rule> = clean_patterns
        .iter()
        .map(|p| Rule::new(p, FseOpcode::Record(1)))
        .collect();
    let map_clean = FseMap::compile(rules_clean).unwrap();
    let mut scanner_clean = FseScanner::new(&map_clean).unwrap();

    group.bench_function("clean_traffic_libfse", |b| {
        b.iter(|| {
            scanner_clean.reset_rule_state();
            scanner_clean.reset_automaton_state().unwrap();
            black_box(scanner_clean.scan(black_box(clean_text.as_bytes()))).unwrap();
        })
    });

    // Scenario 2: DoS / High Density Attack
    // Text is composed ENTIRELY of overlapping matches.
    // Simulates a WAF being hammered by signatures.
    // Pattern: "a"
    // Text: "aaaaaaaa..."
    let attack_text = "a".repeat(100_000); // 100KB
    let attack_patterns = vec!["a"];

    group.throughput(Throughput::Bytes(attack_text.len() as u64));

    // Aho-Corasick Attack
    let ac_attack = AhoCorasick::new(&attack_patterns).unwrap();
    group.bench_function("dos_attack_aho_corasick", |b| {
        b.iter(|| {
            let mut count = 0;
            // Iterate matches. In high density, this creates 100,000 Match structs.
            for _ in ac_attack.find_iter(black_box(attack_text.as_bytes())) {
                count += 1;
            }
            black_box(count);
        })
    });

    // LibFSE Attack
    let rules_attack = vec![Rule::new("a", FseOpcode::Record(1))];
    let map_attack = FseMap::compile(rules_attack).unwrap();
    let mut scanner_attack = FseScanner::new(&map_attack).unwrap();

    group.bench_function("dos_attack_libfse", |b| {
        b.iter(|| {
            scanner_attack.reset_rule_state();
            scanner_attack.reset_automaton_state().unwrap();
            // Should be constant time regardless of match count (just OR instructions)
            black_box(scanner_attack.scan(black_box(attack_text.as_bytes()))).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_attack_scenarios);
criterion_main!(benches);
