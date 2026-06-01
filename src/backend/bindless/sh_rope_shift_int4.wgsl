// sh_rope_shift_int4.wgsl
// Helical Context Compaction — INT4 packed+scale buffer extension.
// Companion to sh_rope_shift.wgsl (F32 staging buffers).
//
// This kernel shifts the INT4 KV cache packed nibble buffers and their
// per-head scale buffers when the helical compaction window fires.
// Must be dispatched AFTER the F32 shift (sh_rope_shift.wgsl) so that
// the packed+scale state stays in sync with the F32 staging state.
//
// Buffer layouts (same sequence-major order as F32 path):
//   packed: [max_seq_len × n_head_kv × (head_dim/8)]  U32 — 8 nibbles per element
//   scale:  [max_seq_len × n_head_kv]                 F32 — one scale per (pos, head)
//
// Threading model: same workgroup shape as sh_rope_shift.wgsl (64, 4, 1)
//   x: packed dim index  (0 to head_dim/8-1)
//   y: kv_head_idx       (0 to n_head_kv-1)
//   z: seq_offset        (0 to elements_to_shift-1)
// When x==0 the thread also copies the single scale element for (head, position).

struct CompactionParams {
    keep_sink:   u32,
    shift_amt:   u32,
    old_seq_len: u32,
    n_head_kv:   u32,
    head_dim:    u32,
    _pad0:       u32,
    _pad1:       f32,
    max_seq_len: u32,
}

@group(0) @binding(0) var<uniform>            params:        CompactionParams;

// Frozen snapshot sources (scratch copies, read-only)
@group(0) @binding(1) var<storage, read>      packed_k_src:  array<u32>;
@group(0) @binding(2) var<storage, read>      scale_k_src:   array<f32>;
@group(0) @binding(3) var<storage, read>      packed_v_src:  array<u32>;
@group(0) @binding(4) var<storage, read>      scale_v_src:   array<f32>;

// Live cache destinations (write targets — may alias source in GPU memory)
@group(0) @binding(5) var<storage, read_write> packed_k_dst: array<u32>;
@group(0) @binding(6) var<storage, read_write> scale_k_dst:  array<f32>;
@group(0) @binding(7) var<storage, read_write> packed_v_dst: array<u32>;
@group(0) @binding(8) var<storage, read_write> scale_v_dst:  array<f32>;

@compute @workgroup_size(64, 4, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let d8     = global_id.x;  // packed element index (0..head_dim/8-1)
    let h      = global_id.y;  // KV head index
    let offset = global_id.z;  // sequence offset within shifted region

    let n_head_kv   = params.n_head_kv;
    let head_dim    = params.head_dim;
    let hd8         = head_dim / 8u;  // number of U32 elements per head-vector

    if h >= n_head_kv { return; }

    let start_shift_seq = params.keep_sink + params.shift_amt;
    let old_seq_idx     = start_shift_seq + offset;

    if old_seq_idx >= params.old_seq_len { return; }

    let new_seq_idx = old_seq_idx - params.shift_amt;

    // --- Shift packed K/V nibble buffers ---
    // packed layout: [max_seq_len * n_head_kv * hd8], element = (pos*n_head_kv + h)*hd8 + d8
    if d8 < hd8 {
        let old_pk = (old_seq_idx * n_head_kv + h) * hd8 + d8;
        let new_pk = (new_seq_idx * n_head_kv + h) * hd8 + d8;
        packed_k_dst[new_pk] = packed_k_src[old_pk];
        packed_v_dst[new_pk] = packed_v_src[old_pk];
    }

    // --- Shift scale buffers (one element per (pos, head)) ---
    // scale layout: [max_seq_len * n_head_kv], element = pos*n_head_kv + h
    // Use the d8==0 thread to avoid redundant writes.
    if d8 == 0u {
        let old_sc = old_seq_idx * n_head_kv + h;
        let new_sc = new_seq_idx * n_head_kv + h;
        scale_k_dst[new_sc] = scale_k_src[old_sc];
        scale_v_dst[new_sc] = scale_v_src[old_sc];
    }
}
