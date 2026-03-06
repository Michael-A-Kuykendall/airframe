use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libfse::{FseMap, FseOpcode, FseScanner, Rule};
use aho_corasick::AhoCorasick;

fn bench_scanners(c: &mut Criterion) {
    // 1. Setup Data
    let patterns = vec!["needle", "haystack", "mission_critical", "abort_code"];
    let text = "This is a haystack that might contain a needle or two. ".repeat(1000) 
             + "Sometimes we find a mission_critical error code. "
             + "But mostly just haystack text repeating over and over. ";
    let input = text.as_bytes();

    // 2. Setup LibFSE
    let rules: Vec<Rule> = patterns.iter().map(|p| {
        Rule::new(p, FseOpcode::Record(1))
    }).collect();
    let map = FseMap::compile(rules).unwrap();
    let mut scanner = FseScanner::new(&map).unwrap();

    // 3. Setup Baseline (Aho-Corasick)
    // We try to match configuration: contiguous NFA is standard for Aho 1.1 default
    let ac = AhoCorasick::new(&patterns).unwrap();

    let mut group = c.benchmark_group("Scanner Comparison");

    // Benchmark LibFSE (Our Engine)
    group.bench_function("libfse_scan", |b| {
        b.iter(|| {
            // Reset state to ensure we measure cold-start logic (writes happen)
            scanner.reset_rule_state();
            // Reset automaton allows us to scan from the start of the state machine,
            // correctly benchmarking the "from zero" scan cost each iteration.
            scanner.reset_automaton_state().unwrap();
            
            black_box(scanner.scan(black_box(input))).unwrap();
        })
    });

    // Benchmark Baseline (Raw Aho-Corasick)
    group.bench_function("aho_corasick_find_iter", |b| {
        b.iter(|| {
            // Count matches to force execution
            let count = ac.find_iter(black_box(input)).count();
            black_box(count);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_scanners);
criterion_main!(benches);
