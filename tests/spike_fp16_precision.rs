// SPIKE 2: FP16 Precision Validation
// Test if F32→F16→F32 roundtrip preserves our parity threshold

use half::f16;

#[test]
fn spike_fp16_precision_real_values() {
    // Real values from our parity test (L0.21 checkpoint)
    let test_values = vec![
        -0.00479799,
        -0.00655676,
        0.02155690,
        0.05804433,
        0.0,
        1.0,
        -1.0,
        3.14159,
        -3.14159,
        1e-5,
        -1e-5,
        1e-6,
        -1e-6, // Small values
        100.0,
        -100.0, // Large values
    ];

    println!("\n=== FP16 PRECISION TEST ===");
    println!("Testing F32→F16→F32 roundtrip on real model outputs");
    println!("Parity threshold: 1e-6 (max allowed error)");
    println!("");

    let mut max_error = 0.0f32;
    let mut max_error_val = 0.0f32;
    let mut errors_over_1e6 = 0;
    let mut errors_over_1e7 = 0;

    for (i, &original) in test_values.iter().enumerate() {
        let fp16 = f16::from_f32(original);
        let roundtrip = fp16.to_f32();
        let error = (original - roundtrip).abs();

        if error > max_error {
            max_error = error;
            max_error_val = original;
        }

        if error > 1e-6 {
            errors_over_1e6 += 1;
        }
        if error > 1e-7 {
            errors_over_1e7 += 1;
        }

        let status = if error < 1e-6 { "✓" } else { "✗ FAIL" };

        println!(
            "[{}] {} F32: {:.8}, F16: {:.8}, Roundtrip: {:.8}, Error: {:.2e}",
            i,
            status,
            original,
            fp16.to_f32(),
            roundtrip,
            error
        );
    }

    println!("\n=== RESULTS ===");
    println!("Max error: {:.2e} (at value {})", max_error, max_error_val);
    println!(
        "Values with error > 1e-6: {} / {}",
        errors_over_1e6,
        test_values.len()
    );
    println!(
        "Values with error > 1e-7: {} / {}",
        errors_over_1e7,
        test_values.len()
    );

    if max_error < 1e-6 {
        println!("\n*** UNEXPECTED: FP16 SUFFICIENT ***");
        println!("FP16 precision sufficient for parity threshold");
        println!("Can use FP16 cache (23 MB vs 46 MB F32)");
    } else {
        println!("\n*** SPIKE 2 RESULT: VALIDATED ***");
        println!("FP16 precision INSUFFICIENT (as expected)!");
        println!("Must use F32 cache (doubles memory to 92 MB)");
    }

    // Assert that FP16 DOES fail (validates our F32 design decision)
    assert!(max_error >= 1e-6, 
        "Test expected FP16 to fail parity threshold, but it passed with error {}. Re-evaluate F32 requirement.", 
        max_error);
}

#[test]
fn spike_fp16_precision_accumulation() {
    // Test if errors accumulate over many operations (like 2048 attention positions)

    println!("\n=== FP16 ACCUMULATION TEST ===");
    println!("Simulating 2048 attention score accumulations");

    let base_value = 0.001f32; // Typical attention score magnitude

    // Test 1: Sum of 2048 small values
    let mut sum_f32 = 0.0f32;
    let mut sum_f16_roundtrip = 0.0f32;

    for _ in 0..2048 {
        sum_f32 += base_value;

        let fp16 = f16::from_f32(base_value);
        sum_f16_roundtrip += fp16.to_f32();
    }

    let accumulation_error = (sum_f32 - sum_f16_roundtrip).abs();

    println!("F32 sum: {:.8}", sum_f32);
    println!("F16 sum: {:.8}", sum_f16_roundtrip);
    println!("Accumulation error: {:.2e}", accumulation_error);

    if accumulation_error < 1e-5 {
        println!("✓ Accumulation error acceptable");
    } else {
        println!("✗ Accumulation error too large!");
    }

    // Test 2: Dot product (Q·K in attention)
    let q: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let k: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();

    let dot_f32: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

    let dot_f16: f32 = q
        .iter()
        .zip(k.iter())
        .map(|(a, b)| {
            let a16 = f16::from_f32(*a);
            let b16 = f16::from_f32(*b);
            let product = a16.to_f32() * b16.to_f32();
            product
        })
        .sum();

    let dot_error = (dot_f32 - dot_f16).abs();

    println!("\nDot product test (Q·K):");
    println!("F32 result: {:.8}", dot_f32);
    println!("F16 result: {:.8}", dot_f16);
    println!("Dot product error: {:.2e}", dot_error);

    if dot_error < 1e-4 {
        println!("✓ Dot product error acceptable");
        println!("\n*** ACCUMULATION TEST: PASS ***");
    } else {
        println!("✗ Dot product error too large!");
        println!("\n*** ACCUMULATION TEST: FAIL ***");
    }
}
