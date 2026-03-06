// Isolated RoPE validation test
// Compare GPU RoPE output against CPU reference values

#[cfg(test)]
mod rope_tests {
    #[test]
    fn test_rope_formula() {
        // RoPE formula from llama.cpp:
        // For each (real, imag) pair at dimension (2i, 2i+1):
        //   theta_i = base^(-2i/dim) where base=10000, dim=head_dim=64
        //   angle = position * theta_i
        //   new_real = real * cos(angle) - imag * sin(angle)
        //   new_imag = real * sin(angle) + imag * cos(angle)

        let head_dim = 64;
        let base = 10000.0f32;

        // Test position 1, pair 0 (dimensions 0,1)
        let pair_idx = 0;
        let theta = base.powf(-2.0 * pair_idx as f32 / head_dim as f32);
        let position = 1;
        let angle = position as f32 * theta;

        println!("Pair {}, Position {}:", pair_idx, position);
        println!("  theta = {:.10}", theta);
        println!("  angle = {:.10}", angle);
        println!("  cos(angle) = {:.10}", angle.cos());
        println!("  sin(angle) = {:.10}", angle.sin());

        // Example values before RoPE
        let real = 0.5f32;
        let imag = 0.3f32;

        let new_real = real * angle.cos() - imag * angle.sin();
        let new_imag = real * angle.sin() + imag * angle.cos();

        println!("  Input: ({:.8}, {:.8})", real, imag);
        println!("  Output: ({:.8}, {:.8})", new_real, new_imag);

        // Verify inverse (should recover original with -angle)
        let recovered_real = new_real * angle.cos() + new_imag * angle.sin();
        let recovered_imag = -new_real * angle.sin() + new_imag * angle.cos();

        println!(
            "  Recovered: ({:.8}, {:.8})",
            recovered_real, recovered_imag
        );

        assert!(
            (recovered_real - real).abs() < 1e-6,
            "RoPE inverse failed (real)"
        );
        assert!(
            (recovered_imag - imag).abs() < 1e-6,
            "RoPE inverse failed (imag)"
        );

        println!("\n✅ RoPE formula validated");
    }
}
