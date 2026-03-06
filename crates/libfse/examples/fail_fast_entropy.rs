//! Fail-Fast Experiment: Entropy Detection via LibFSE
//!
//! This example implements the "Interior Chat" idea:
//! 1. Compute entropy from logits (simulated).
//! 2. Quantize/Flag the entropy state.
//! 3. Feed it to FSE Scanner.
//! 4. FSE executes "Reject" on high entropy (Fail-Fast).

use libfse::{FseMap, FseOpcode, FseScanner, Rule, RuleId};
use libfse::metrics::shannon_entropy_from_logits;

const RULE_HIGH_ENTROPY: RuleId = 1;
const SIGNAL_HIGH_ENTROPY: u8 = 0xFF;
const SIGNAL_OK: u8 = 0x00;
const ENTROPY_THRESHOLD: f32 = 2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fail-Fast Experiment: Entropy Detection ===");

    // 1. Compile FSE Policy
    // Rule: If we see the HIGH_ENTROPY signal, REJECT immediately.
    let rules = vec![
        Rule::new(&[SIGNAL_HIGH_ENTROPY], FseOpcode::Reject(RULE_HIGH_ENTROPY)),
    ];
    let map = FseMap::compile(rules)?;
    let mut scanner = FseScanner::new(&map)?;

    // 2. Simulate Token Stream & Logits
    // Case A: Low Entropy (Confident "The cat is...")
    // [0.9, 0.05, 0.05] -> Low entropy
    let logits_safe = vec![10.0f32, 0.0, 0.0, -5.0, -5.0]; // Highly peaked
    
    // Case B: High Entropy (Confusion "The ??? is...")
    // [0.2, 0.2, 0.2, 0.2, 0.2] -> Flat -> High entropy
    let logits_risky = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // Uniform

    // Run Simulation
    run_step(&mut scanner, "Token 1 (Safe)", &logits_safe)?;
    
    // This should trigger the fail-fast
    match run_step(&mut scanner, "Token 2 (Risky)", &logits_risky) {
        Ok(_) => println!("WARNING: High entropy token was accepted! (Test Failed)"),
        Err(e) => {
            println!("\n!!! FAIL-FAST TRIGGERED !!!");
            println!("  Violation: {:?}", e);
            println!("  Action: Dropping generic note/log as requested.");
            println!("  System State: HALTED.");
        }
    }

    Ok(())
}

fn run_step(scanner: &mut FseScanner, label: &str, logits: &[f32]) -> Result<(), libfse::Violation> {
    // 1. Compute Metric
    let ent = shannon_entropy_from_logits(logits);
    print!("{:<15} | Entropy: {:.4} | ", label, ent);

    // 2. Normalize Signal (The "Fused" Data Element)
    // If entropy exceeds threshold, we emit the HIGH_ENTROPY signal byte.
    let signal = if ent > ENTROPY_THRESHOLD {
        print!("Status: HIGH RISK -> ");
        SIGNAL_HIGH_ENTROPY
    } else {
        print!("Status: OK        -> ");
        SIGNAL_OK
    };

    // 3. FSE Execution (The Policy Kernel)
    // We scan the *signal*, not the text. The policy engine decides functionality.
    // In a real loop, we would scan both text AND signals.
    let summary = scanner.scan(&[signal])?;
    
    println!("FSE: Accepted (Recorded: {})", summary.rules_recorded);
    Ok(())
}
