// sh_rope_shift.wgsl
// Helical Context Compaction Kernel
// Purpose: Slides KV Cache elements downward in the sequence dimension.
//
// IMPORTANT: This codebase uses RELATIVE RoPE (RoPE(i-j) applied on-the-fly
// during attention). K and V are stored RAW in the cache — no absolute position
// encoding. Therefore the shift is a PURE POSITIONAL COPY: move data from
// old_seq_idx to new_seq_idx = old_seq_idx - shift_amt.  No rotation needed.
//
// Threading model:
// x: dimension index    (0 to head_dim-1, not paired)
// y: kv_head_idx        (0 to n_head_kv-1)
// z: seq_offset         (0 to num_elements_to_shift-1)

struct CompactionParams {
    keep_sink: u32,
    shift_amt: u32,
    old_seq_len: u32,
    n_head_kv: u32,
    head_dim: u32,
    _pad0: u32,       // unused (was rope_dim)
    _pad1: f32,       // unused (was rope_base)
    max_seq_len: u32,
}

// Source buffers (read-only snapshot) — avoids overlap hazard on in-place shift
@group(0) @binding(0) var<storage, read> k_src: array<f32>;
@group(0) @binding(1) var<storage, read> v_src: array<f32>;
// Destination buffers (write target — may alias the live cache)
@group(0) @binding(2) var<storage, read_write> k_dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_dst: array<f32>;
@group(0) @binding(4) var<uniform> params: CompactionParams;

@compute @workgroup_size(64, 4, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let d = global_id.x;      // dimension index (0 to head_dim-1)
    let h = global_id.y;      // KV head index
    let offset = global_id.z; // sequence offset within shifted region
    
    let head_dim = params.head_dim;
    let n_head_kv = params.n_head_kv;
    
    // Bounds check
    if (d >= head_dim) { return; }
    if (h >= n_head_kv) { return; }
    
    let start_shift_seq = params.keep_sink + params.shift_amt;
    let old_seq_idx = start_shift_seq + offset;
    
    if (old_seq_idx >= params.old_seq_len) { return; }
    
    let new_seq_idx = old_seq_idx - params.shift_amt;
    
    // Buffer layout: [max_seq_len, n_head_kv, head_dim] — position-major
    let old_idx = (old_seq_idx * n_head_kv * head_dim) + (h * head_dim) + d;
    let new_idx = (new_seq_idx * n_head_kv * head_dim) + (h * head_dim) + d;
    
    // Pure copy — no RoPE manipulation.
    // This codebase stores K/V RAW; RoPE is applied as RoPE(i-j) during attention.
    // Since both current_pos and cache indices shift by the same amount, the
    // relative distance (i-j) is invariant and no rotation correction is needed.
    k_dst[new_idx] = k_src[old_idx];
    v_dst[new_idx] = v_src[old_idx];
}
