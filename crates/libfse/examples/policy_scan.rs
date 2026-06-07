//! Policy scanning quickstart for libfse.
//!
//! Demonstrates all three opcodes (Reject, Record, Ignore), the borrowless
//! ScanCursor pattern, and the fail-closed error semantics.
//!
//! Run: cargo run --example policy_scan

use libfse::{FseMap, FseOpcode, Rule};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Define your policy rules ────────────────────────────────────────
    let rules = vec![
        // Hard block: any input containing "DROP TABLE" is rejected immediately.
        Rule {
            pattern: b"DROP TABLE".to_vec(),
            opcode: FseOpcode::Reject(0),
        },
        // Audit: record when a SELECT * is seen (rule ID 1).
        Rule {
            pattern: b"SELECT *".to_vec(),
            opcode: FseOpcode::Record(1),
        },
        // Suppress: SQL comments are noise — ignore them.
        Rule {
            pattern: b"--".to_vec(),
            opcode: FseOpcode::Ignore,
        },
    ];

    // ── 2. Compile once — reuse across threads via Arc<FseMap> ─────────────
    //
    //  Compilation builds a fused DFA + precomputed action table.
    //  All rules share a single DFA pass — rule count has O(1) impact on scan.
    let map = FseMap::compile(rules)?;

    // ── 3. Create a ScanCursor — one per stream, no borrow of FseMap ───────
    //
    //  The cursor holds DFA walker state + rule bitset independently of
    //  the map. Store it alongside Arc<FseMap> without lifetime friction.
    let mut cursor = map.new_cursor()?;

    // ── 4. Safe input: SELECT * should record, -- should be ignored ─────────
    println!("Scanning: 'SELECT * FROM users -- safe query'");
    let summary = map.scan_with_cursor(&mut cursor, b"SELECT * FROM users -- safe query")?;
    println!(
        "  → ok | match states: {} | rules fired: {} | distinct rules: {}",
        summary.match_states_seen, summary.pattern_hits, summary.rules_recorded
    );

    // ── 5. Dangerous input: DROP TABLE triggers immediate Err ───────────────
    println!("\nScanning: 'DROP TABLE users'");
    let result = map.scan_with_cursor(&mut cursor, b"DROP TABLE users");
    match result {
        Err(libfse::scanner::Violation::PolicyReject { rule_id, .. }) => {
            println!("  → REJECTED (rule_id={})", rule_id);
        }
        Err(libfse::scanner::Violation::IntegrityError { details, .. }) => {
            println!("  → INTEGRITY ERROR: {}", details);
        }
        Ok(_) => println!("  → ok (unexpected)"),
    }

    // ── 6. Cursor is poisoned after Err — create a fresh one ───────────────
    let mut cursor2 = map.new_cursor()?;
    println!("\nFresh cursor. Scanning safe input again:");
    let summary2 = map.scan_with_cursor(&mut cursor2, b"SELECT * FROM orders")?;
    println!("  → ok | rules fired: {}", summary2.rules_recorded);

    Ok(())
}
