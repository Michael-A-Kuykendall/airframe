// Logit Masking Shader
// This compute shader applies a mask to logits to suppress tokens based on policy.

@group(0) @binding(0)
var<storage, read_write> logits: array<f32>;

@group(0) @binding(1)
var<storage, read> mask: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if (index >= arrayLength(&logits)) {
        return;
    }
    
    // Check mask. If mask value > 0, suppress logit (set to -infinity)
    // Assuming mask is 1-to-1 with logits for now (dense mask).
    // In practice, this might be a sparse list or bitmask, 
    // but starting with dense array for simplicity (Approach #2).
    
    // Note: mask length should match logits length generally, 
    // or we need safe access.
    if (index < arrayLength(&mask)) {
        if (mask[index] != 0u) {
            logits[index] = -3.40282347e+38; // -f32::MAX (approx -Infinity)
        }
    }
}
