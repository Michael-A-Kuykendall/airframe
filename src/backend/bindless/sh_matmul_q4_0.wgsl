
// IEEE-754 binary16 -> binary32. Float-arithmetic only — NO bitcast of a
// computed integer (unreliable on this driver, where unpack2x16float is also
// broken). Bit-exact to airframe_observe::quant_formula::f16_to_f32 for
// normal/zero/subnormal values; the P2 algebraic_audit gate validates the
// shader element-wise against that reference.
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp  = (bits >> 10u) & 0x1fu;
    let mant = bits & 0x3ffu;
    let sign_f = select(-1.0, 1.0, sign == 0u);
    if (exp == 0u) {
        if (mant == 0u) {
            return sign_f * 0.0;
        }
        // subnormal: (-1)^sign * mant * 2^-24 (exact division by power of two)
        return sign_f * (f32(mant) / f32(1u << 24u));
    }
    if (exp == 0x1fu) {
        // ±inf / NaN. Real GGUF weight scales are always finite, so this branch
        // is unreachable for valid input; return 0.0 to stay parse-clean.
        return 0.0;
    }
    // normal: (-1)^sign * (1 + mant/1024) * 2^(exp-15)
    // exp-15 may be negative, so split on the sign of the shift: every shift
    // count stays non-negative and every power-of-two op is exact, making the
    // result bit-identical to the reference integer-assembled f32.
    let fraction = 1.0 + f32(mant) / 1024.0;
    if (exp >= 15u) {
        let p = f32(1u << (exp - 15u));
        return sign_f * fraction * p;
    } else {
        let p = f32(1u << (15u - exp));
        return sign_f * fraction / p;
    }
}
// MatMul Q4_0: y = x @ W^T
// x: f32 vector [K]
// W: Q4_0 matrix [N, K] (stored row-major, so N rows of K elements)
// Output: y [N]
//
// Each thread computes one element of y (one row of W dot x).

@group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
@group(0) @binding(1) var<storage, read> input_x: array<f32>; // The activation vector
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

struct MatMulParams {
    N: u32, // Number of rows (output size)
    K: u32, // Number of columns (inner dim)
    weights_offset: u32, // Byte offset in GGUF blob where matrix starts
    padding: u32,
}

// Helper: Unpack f16 from u32
fn get_f16(byte_offset: u32) -> f32 {
    let u32_idx = byte_offset / 4u;
    let shift = (byte_offset % 4u) * 8u;
    let u32_val = gguf_blob[u32_idx];
    let u16_val = (u32_val >> shift) & 0xFFFFu;
    return f16_to_f32(u16_val);
}

// Helper: Read u8 from u32 array
fn get_u8(byte_offset: u32) -> u32 {
    let u32_idx = byte_offset / 4u;
    let shift = (byte_offset % 4u) * 8u;
    let u32_val = gguf_blob[u32_idx];
    return (u32_val >> shift) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.N) { return; }

    let K = params.K;
    // Q4_0 block size is 32. 
    // Row bytes = (K / 32) * 18.
    let blocks_per_row = K / 32u;
    let row_bytes = blocks_per_row * 18u;
    
    // Start of this row in GGUF blob
    let row_start_offset = params.weights_offset + (row * row_bytes);
    
    var sum: f32 = 0.0;

    // Loop over blocks
    for (var b = 0u; b < blocks_per_row; b++) {
        let block_offset = row_start_offset + (b * 18u);
        let d = get_f16(block_offset);
        
        // Loop over 16 bytes in block (32 elements total)
        // GGML Q4_0 SPLIT layout:
        // Elements 0-15:  low nibbles from bytes 0-15
        // Elements 16-31: high nibbles from bytes 0-15
        for (var i = 0u; i < 16u; i++) {
            let qs = get_u8(block_offset + 2u + i);
            
            // Low nibble -> element i (0..15)
            let n_low = qs & 0x0Fu;
            let val_low = (f32(n_low) - 8.0) * d;
            let col_low = (b * 32u) + i;
            sum += val_low * input_x[col_low];
            
            // High nibble -> element i + 16 (16..31)
            let n_high = (qs >> 4u) & 0x0Fu;
            let val_high = (f32(n_high) - 8.0) * d;
            let col_high = (b * 32u) + i + 16u;
            sum += val_high * input_x[col_high];
        }
    }

    output_y[row] = sum;
}
