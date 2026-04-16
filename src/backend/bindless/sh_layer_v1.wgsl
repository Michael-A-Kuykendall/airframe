// sh_layer_v1.wgsl
// Full Transformer Layer (TinyLlama Q4_0) - Split Kernels
// Revised for Bindless Architecture V2.3 (Preflight Fused Resources)

// Constants
const BLOCK_SIZE: u32 = 32u; // Q4_0 Block Size

struct LayerOffsets {
    attn_norm: u32,
    attn_q: u32,
    attn_k: u32,
    attn_v: u32,
    attn_out: u32,
    ffn_norm: u32,
    ffn_gate: u32,
    ffn_down: u32,
    ffn_up: u32,
    layer_idx: u32,     // padding[0] from host = layer index
    pad2: u32,
    pad3: u32,
};

struct LayerParams {
    dim: u32,           // 2048
    head_count: u32,    // 32
    head_count_kv: u32, // 4 (GQA)
    head_dim: u32,      // 64
    rms_eps: f32,       // 1e-5
    ffn_dim: u32,       // Feed-forward intermediate dim (e.g. 5632)
    temp_stride: u32,   // Per-token temp buffer stride in floats (e.g. 16384)
    pad: u32,
};

struct CacheParams {
    current_pos: u32,   // Position to write new K/V (0-based)
    seq_len: u32,       // Total cached positions (current_pos + batch_size)
    max_seq_len: u32,   // Context window (2048)
    batch_size: u32,    // Number of tokens in this dispatch
    logical_pos_base: u32, // Logical base of the compacted sliding window
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

// Bindings
@group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
@group(0) @binding(1) var<storage, read_write> activation_in: array<f32>; // The "Residual" stream
@group(0) @binding(2) var<storage, read_write> temp_state: array<f32>;    // Scratchpad
@group(0) @binding(3) var<uniform> offsets: LayerOffsets;
@group(0) @binding(4) var<uniform> params: LayerParams;
@group(0) @binding(5) var<storage, read> norm_bank: array<f32>;           // [n_layer * dim * 2 + dim]
@group(0) @binding(6) var<storage, read> rope_table: array<f32>;           // [2048 × head_dim/2 × 2] pre-computed (cos, sin)
@group(0) @binding(7) var<storage, read_write> kv_cache_k: array<f32>;    // K cache [max_seq * n_head_kv * head_dim]
@group(0) @binding(8) var<storage, read_write> kv_cache_v: array<f32>;    // V cache [max_seq * n_head_kv * head_dim]
@group(0) @binding(9) var<uniform> cache_params: CacheParams;             // Sequence position tracking

// Helper functions for Q4_0 dequant
fn unpack_q4_0(block_val: u32, idx_in_block: u32) -> f32 {
    let shift = (idx_in_block % 8u) * 4u;
    return f32((block_val >> shift) & 0xFu) - 8.0;
}

// -------------------------------------------------------------------------
// Kernel 0: Attention RMSNorm Provider
// -------------------------------------------------------------------------
// Computes the shared attention-normalized activation stream once per token.
// Writes normalized activations into temp_state[0..dim) for Q/K/V consumers.
@compute @workgroup_size(256, 1, 1)
fn main_attn_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }

    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let val = activation_in[act_base + i];
        sum_sq += val * val;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    let norm_offset_base = offsets.layer_idx * 2u * params.dim;
    let norm_w = norm_bank[norm_offset_base + idx];
    temp_state[temp_base + idx] = activation_in[act_base + idx] * rms * norm_w;
}

// -------------------------------------------------------------------------
// Kernel 1: QKV Generation + RoPE + Cache Update
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_qkv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;
    
    if (token_idx >= cache_params.batch_size) { return; }
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    
    let dim_q = params.head_count * params.head_dim;           
    let dim_k = params.head_count_kv * params.head_dim;        
    let dim_v = params.head_count_kv * params.head_dim;        
    let total_out = dim_q + dim_k + dim_v;                     
    
    if (idx >= total_out) { return; }

    // 1. Select Weight Matrix & Row
    var weight_off_roots: array<u32, 3>;
    weight_off_roots[0] = offsets.attn_q;
    weight_off_roots[1] = offsets.attn_k;
    weight_off_roots[2] = offsets.attn_v;

    var target_stage = 0u; // 0=Q, 1=K, 2=V
    var row_idx = idx;
    
    if (idx >= dim_q) {
        if (idx < dim_q + dim_k) {
            target_stage = 1u;
            row_idx = idx - dim_q;
        } else {
            target_stage = 2u;
            row_idx = idx - (dim_q + dim_k);
        }
    }

    let weight_byte_offset = weight_off_roots[target_stage];
    
    // 2. MatMul (Row `row_idx`) against the staged attention-normalized provider.
    var dot: f32 = 0.0;
    let blocks_per_row = params.dim / 32u;
    // Stride per row: 18 bytes * blocks
    let row_start_byte = weight_byte_offset + (row_idx * blocks_per_row * 18u);


    for (var b = 0u; b < blocks_per_row; b++) {
        let block_base = row_start_byte + (b * 18u);
        
        // Read Scale (F16)
        let scale_idx = block_base / 4u;
        let scale_packed = extractBits(gguf_blob[scale_idx], (block_base % 4u) * 8u, 16u);
        let scale = unpack2x16float(scale_packed).x;

        // Process 32 weights
        // We load Quants (16 bytes = 4 u32s)
        // Q4_0 Layout is quirky:
        // [scale:2] [qs:16]
        let qs_byte_start = block_base + 2u;
        
        for (var i = 0u; i < 32u; i++) {
            let col = b * 32u + i;
            let val_x = temp_state[temp_base + col];

            // Extract nibble - GGML Q4_0 SPLIT layout:
            // Elements 0-15:  low nibbles from bytes 0-15
            // Elements 16-31: high nibbles from bytes 0-15
            let byte_idx = i % 16u;  // which byte (0-15)
            let qs_idx = qs_byte_start + byte_idx;
            let qs_word = gguf_blob[qs_idx / 4u];
            let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
            let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
            
            let val_w = (f32(nib) - 8.0) * scale;
            dot += val_x * val_w;
        }
    }

    // 4. RoPE - REMOVED (Relative RoPE Architecture)
    // Q and K are stored RAW without absolute position encoding.
    // Position-dependent rotation is applied as relative RoPE(i-j)
    // during the Q·K dot product in main_attn_out.
    // This enables infinite context via sliding window:
    // - Only relative distances matter (observer-relative frame)
    // - Max relative distance bounded by window size (always in training range)
    // - No position counter overflow, no extrapolation artifacts

    // 4. Store Output
    // Q -> Temp State (offset dim)
    // K, V -> KV Cache at current position
    // 
    // Cache layout per buffer: [max_seq, n_head_kv, head_dim]
    // Element offset: (pos * n_head_kv * head_dim) + (head * head_dim) + dim
    
    if (target_stage == 0u) {
        // Q goes to temp_state for attention computation
        temp_state[temp_base + params.dim + row_idx] = dot;
    } else if (target_stage == 1u) {
        // K -> K cache
        // row_idx = head * head_dim + dim_in_head (0..255 for 4 heads * 64 dims)
        let head = row_idx / params.head_dim;
        let dim_in_head = row_idx % params.head_dim;
        let cache_idx = ((cache_params.current_pos + token_idx) * params.head_count_kv * params.head_dim)
                      + (head * params.head_dim)
                      + dim_in_head;
        kv_cache_k[cache_idx] = dot; 
    } else {
        // V -> V cache
        let head = row_idx / params.head_dim;
        let dim_in_head = row_idx % params.head_dim;
        let cache_idx = ((cache_params.current_pos + token_idx) * params.head_count_kv * params.head_dim)
                      + (head * params.head_dim)
                      + dim_in_head;
        kv_cache_v[cache_idx] = dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Attention — Flash Attention online softmax (no scores buffer)
// -------------------------------------------------------------------------
// Computes attention output in a single pass using online softmax
// (Flash Attention style — Milakov & Gimelshein 2018).
// Benefits:
//   - O(1) auxiliary state per thread (running_max, running_sum, running_out)
//   - No external scratch buffer required
//   - Handles arbitrary max_seq_len — correct for ctx > 2048
//   - Works for batched prefill (batch_size > 1) and decode (batch_size == 1)
@compute @workgroup_size(256, 1, 1)
fn main_attn_out(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;       // Output dimension index (0..dim-1)
    let token_idx = global_id.y; // Batch token index

    if (token_idx >= cache_params.batch_size) { return; }
    if (idx >= params.dim) { return; }

    let temp_base  = token_idx * params.temp_stride;
    let gqa_ratio  = params.head_count / params.head_count_kv;
    let scale      = 1.0 / sqrt(f32(params.head_dim));
    let head_idx   = idx / params.head_dim;
    let head_offset = idx % params.head_dim;
    let kv_head_idx = head_idx / gqa_ratio;
    let q_base     = temp_base + params.dim + head_idx * params.head_dim;
    let n_pairs    = params.head_dim / 2u;
    let compact_query_pos = cache_params.current_pos + token_idx;
    let logical_query_pos = cache_params.logical_pos_base + compact_query_pos;

    const SINK_COUNT: u32 = 4u;

    // Online softmax accumulators (Flash Attention — O(1) memory)
    var running_max: f32 = -1e10;
    var running_sum: f32 = 0.0;
    var running_out: f32 = 0.0;

    for (var pos = 0u; pos < cache_params.seq_len; pos++) {
        // Causal mask: skip positions the current query cannot attend to
        if (pos > cache_params.current_pos + token_idx) { continue; }

        var rel: u32 = compact_query_pos - pos;
        if (pos < SINK_COUNT) {
            // Sink positions are pinned to absolute slots 0..SINK_COUNT-1 across compactions.
            // Use the logical query position so sink-relative distance survives helical shift.
            rel = logical_query_pos - pos;
            rel = min(rel, cache_params.max_seq_len - 1u);
        } else if (rel >= cache_params.max_seq_len) {
            // Beyond sliding window context horizon: skip
            continue;
        }

        // Compute Q · RoPE(rel) · K[pos]  (grouped-query attention, GQA)
        var dot_qk: f32 = 0.0;
        let k_base = pos * params.head_count_kv * params.head_dim
                   + kv_head_idx * params.head_dim;
        for (var p = 0u; p < n_pairs; p++) {
            let tbl   = rel * n_pairs * 2u + p * 2u;
            let cos_a = rope_table[tbl];
            let sin_a = rope_table[tbl + 1u];
            let doff  = p * 2u;
            let q_re  = temp_state[q_base + doff];
            let q_im  = temp_state[q_base + doff + 1u];
            let k_re  = kv_cache_k[k_base + doff];
            let k_im  = kv_cache_k[k_base + doff + 1u];
            dot_qk += (q_re * k_re + q_im * k_im) * cos_a
                    + (q_re * k_im - q_im * k_re) * sin_a;
        }
        let score = dot_qk * scale;

        // V value for this position / kv-head / output element
        let v_val = kv_cache_v[
            pos * params.head_count_kv * params.head_dim
            + kv_head_idx * params.head_dim
            + head_offset
        ];

        // Online softmax update (numerically stable)
        let m_new     = max(running_max, score);
        let exp_diff  = exp(running_max - m_new);
        let exp_score = exp(score - m_new);
        running_sum = running_sum * exp_diff + exp_score;
        running_out = running_out * exp_diff + exp_score * v_val;
        running_max = m_new;
    }

    // Finalize: divide accumulated output by softmax denominator
    var context_val = 0.0;
    if (running_sum > 0.0) {
        context_val = running_out / running_sum;
    }
    temp_state[temp_base + idx] = context_val;
}

// -------------------------------------------------------------------------
// Kernel 2.5: Output Projection (NEW - Split from attention)
// -------------------------------------------------------------------------
// Apply output projection matrix to attention context
@compute @workgroup_size(256, 1, 1)
fn main_attn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // Output dimension (0..2047)
    let token_idx = global_id.y;
    
    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    
    var dot = 0.0;
    let weight_byte_offset = offsets.attn_out;
    let blocks_per_row = params.dim / 32u;
    let row_start_byte = weight_byte_offset + (idx * blocks_per_row * 18u);
    
    for (var b = 0u; b < blocks_per_row; b++) {
        let block_base = row_start_byte + (b * 18u);
        let scale_idx = block_base / 4u;
        let scale_packed = extractBits(gguf_blob[scale_idx], (block_base % 4u) * 8u, 16u);
        let w_scale = unpack2x16float(scale_packed).x;
        
        let qs_byte_start = block_base + 2u;
        
        for (var i = 0u; i < 32u; i++) {
            let col = b * 32u + i; // Input dimension (0..2047)
            
            // Read context value from temp_state (computed by main_attn_out)
            let val_ctx = temp_state[temp_base + col];
            
            // Decode weight - GGML Q4_0 SPLIT layout:
            // Elements 0-15:  low nibbles from bytes 0-15
            // Elements 16-31: high nibbles from bytes 0-15
            let byte_idx = i % 16u;
            let qs_idx = qs_byte_start + byte_idx;
            let qs_word = gguf_blob[qs_idx / 4u];
            let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
            let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
            let val_w = (f32(nib) - 8.0) * w_scale;
            
            dot += val_ctx * val_w;
        }
    }
    
    // Add residual connection
    let residual = activation_in[act_base + idx];
    activation_in[act_base + idx] = residual + dot;
}

// -------------------------------------------------------------------------
// Kernel 3: FFN Proj (Norm -> Gate/Up -> SiLu)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Output size: FFN Intermed Dim (5632 for TinyLlama).
    // Gate and Up are computed in parallel.
    // temp_state layout: [Gate (0..5631), Up (5632..11263)]
    // We launch 11264 threads?
    // Typically ffn_gate and ffn_up are separate matrices.
    // We can handle them by idx range.
    
    let ffn_dim = params.ffn_dim; 
    let idx = global_id.x;
    let token_idx = global_id.y;
    
    if (idx >= ffn_dim * 2u || token_idx >= cache_params.batch_size) { return; }
    
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    
    // 0. RMS Norm (FFN Norm) - Same Naive implementation
    // Ideally Read-Once.
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let val = activation_in[act_base + i];
        sum_sq += val * val;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);
    
    // Select Matrix
    var weight_off: u32;
    var row_idx = idx;
    
    if (idx < ffn_dim) {
        weight_off = offsets.ffn_gate; // Gate
    } else {
        weight_off = offsets.ffn_up; // Up
        row_idx = idx - ffn_dim;
    }
    
    // MatMul
    var dot = 0.0;
    let blocks_per_row = params.dim / 32u;
    let row_start_byte = weight_off + (row_idx * blocks_per_row * 18u);
    
    // Norm Params Base (Layer * 2 + 1 for FFN)
    // layer_idx comes from offsets.layer_idx
    let norm_offset_base = (offsets.layer_idx * 2u + 1u) * params.dim;

    for (var b = 0u; b < blocks_per_row; b++) {
        let block_base = row_start_byte + (b * 18u);
        let scale_idx = block_base / 4u;
        let scale_packed = extractBits(gguf_blob[scale_idx], (block_base % 4u) * 8u, 16u);
        let scale = unpack2x16float(scale_packed).x;
        let qs_byte_start = block_base + 2u;
        
        for (var i = 0u; i < 32u; i++) {
            let col = b * 32u + i;
            let norm_w = norm_bank[norm_offset_base + col];
            let val_x = activation_in[act_base + col] * rms * norm_w;
            
            // GGML Q4_0 SPLIT layout: Elements 0-15 = low nibbles, 16-31 = high nibbles
            let byte_idx = i % 16u;
            let qs_idx = qs_byte_start + byte_idx;
            let qs_word = gguf_blob[qs_idx / 4u];
            let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
            let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
            let val_w = (f32(nib) - 8.0) * scale;
            
            dot += val_x * val_w;
        }
    }
    
    // If Gate, Apply SiLu
    if (idx < ffn_dim) {
        let silu = dot / (1.0 + exp(-dot));
        temp_state[temp_base + idx] = silu;
        // Optimization: Store directly? We need Up * Gate.
        // We can wait for Up thread? No.
        // Store to temp.
    } else {
        temp_state[temp_base + idx] = dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 4: FFN Down (Multiply Gate*Up -> Down -> Residual)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_down(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // 0..2047 (Dim)
    let token_idx = global_id.y;
    
    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    
    let ffn_dim = params.ffn_dim;
    var dot = 0.0;
    
    let weight_off = offsets.ffn_down;
    // Down matrix is Dim x FFN_Dim.
    // Row  corresponds to Output .
    // Input is FFN_Dim (Gate * Up).
    
    let blocks_per_row = ffn_dim / 32u;
    let row_start_byte = weight_off + (idx * blocks_per_row * 18u);
    
    for (var b = 0u; b < blocks_per_row; b++) {
        let block_base = row_start_byte + (b * 18u);
        let scale_idx = block_base / 4u;
        let scale_packed = extractBits(gguf_blob[scale_idx], (block_base % 4u) * 8u, 16u);
        let scale = unpack2x16float(scale_packed).x;
        let qs_byte_start = block_base + 2u;
        
        for (var i = 0u; i < 32u; i++) {
            let col = b * 32u + i;
            
            // Fused Gate * Up
            let val_gate = temp_state[temp_base + col];          // SiLu(Gate)
            let val_up = temp_state[temp_base + ffn_dim + col];  // Up
            let val_x = val_gate * val_up;
            
            // GGML Q4_0 SPLIT layout: Elements 0-15 = low nibbles, 16-31 = high nibbles
            let byte_idx = i % 16u;
            let qs_idx = qs_byte_start + byte_idx;
            let qs_word = gguf_blob[qs_idx / 4u];
            let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
            let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
            let val_w = (f32(nib) - 8.0) * scale;
            
            dot += val_x * val_w;
        }
    }
    
    // Debug stash: store pure FFN down output before residual
    // temp_state layout uses 0..ffn_dim*2-1; ffn_dim*2..temp_stride is free for dim floats
    temp_state[temp_base + params.ffn_dim * 2u + idx] = dot;

    // Residual Add
    let residual = activation_in[act_base + idx];
    activation_in[act_base + idx] = residual + dot;
}
