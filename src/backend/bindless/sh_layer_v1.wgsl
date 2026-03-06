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
    max_seq_len: u32,   // Context window (8192)
    batch_size: u32,    // Number of tokens in this dispatch
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

    // 1. Calculate RMS Norm of Input (Naive / Per-Thread)
    // Optimization: Should be Workgroup Shared Memory reduction.
    // For now, doing it naively to ensure correctness.
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let val = activation_in[act_base + i];
        sum_sq += val * val;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    // 2. Select Weight Matrix & Row
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
    
    // 3. MatMul (Row `row_idx`)
    var dot: f32 = 0.0;
    let blocks_per_row = params.dim / 32u;
    // Stride per row: 18 bytes * blocks
    let row_start_byte = weight_byte_offset + (row_idx * blocks_per_row * 18u);

    // Norm Weights (Binding 5)
    // Offset in bank: layer_idx * 2 * dim (AttnNorm is 1st)
    // layer_idx now comes from offsets.layer_idx (padding[0] in Rust)
    let layer_idx = offsets.layer_idx;
    let norm_offset_base = layer_idx * 2u * params.dim; // AttnNorm is first


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
            
            // Apply Norm On-Fly
            let norm_w = norm_bank[norm_offset_base + col];
            let val_x = activation_in[act_base + col] * rms * norm_w;

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

    // 5. Store Output
    // Q -> Temp State (offset 0)
    // K, V -> KV Cache at current position
    // 
    // Cache layout per buffer: [max_seq, n_head_kv, head_dim]
    // Element offset: (pos * n_head_kv * head_dim) + (head * head_dim) + dim
    
    if (target_stage == 0u) {
        // Q goes to temp_state for attention computation
        temp_state[temp_base + row_idx] = dot;
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
// Kernel 2: Attention (QK^T + Scale + Softmax + V-Mix + OutProj + Residual)
// -------------------------------------------------------------------------
// Full Self-Attention implementation for autoregressive decode.
// 1. Compute attention scores: Q @ K^T for all cached positions
// 2. Scale by 1/sqrt(head_dim)
// 3. Apply causal mask (can't attend beyond current_pos)
// 4. Softmax across sequence
// 5. Compute context: weighted sum of V
// 6. Output projection + residual
@compute @workgroup_size(256, 1, 1)
fn main_attn_out(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // Output dimension index (0..2047)
    let token_idx = global_id.y;
    
    if (token_idx >= cache_params.batch_size) { return; }
    let temp_base = token_idx * params.temp_stride;
    
    if (idx >= params.dim) { return; }

    // Constants for TinyLlama GQA
    let dim_k = params.head_count_kv * params.head_dim; // 4 * 64 = 256
    let gqa_ratio = params.head_count / params.head_count_kv; // 32 / 4 = 8
    let scale = 1.0 / sqrt(f32(params.head_dim)); // 1/sqrt(64) = 0.125

    // Map output index to (head_idx, head_offset)
    let head_idx = idx / params.head_dim;      // 0..31
    let head_offset = idx % params.head_dim;   // 0..63
    let kv_head_idx = head_idx / gqa_ratio;    // GQA: 0..3
    
    let q_base = temp_base + head_idx * params.head_dim;    // Q vector base in temp_state
    
    // --------------------------------------------------------------------
    // STEP 1: Compute attention scores for all positions in cache
    // scores[pos] = (Q · K[pos]) / sqrt(head_dim)
    // Thread-private local array for attention scores.
    // Maps to NVIDIA local memory (DRAM-backed, L1/L2 cached).
    // 8192 × 4 bytes = 32 KB per thread. Safe on RTX 3060.
    // --------------------------------------------------------------------
    var scores: array<f32, 8192>;
    var max_score = -1e10;
    
    // Attention sink count: first N tokens are ALWAYS visible regardless of distance.
    // These serve as the "origin" of the helical manifold — the model was trained
    // to always see BOS/system tokens and uses them as stable attention sinks
    // (excess probability mass drains here instead of corrupting content tokens).
    // StreamingLLM (Xiao et al. 2023) proved 4 sinks enable infinite generation.
    const SINK_COUNT: u32 = 4u;
    
    for (var pos = 0u; pos < cache_params.seq_len; pos++) {
        // Compute Q · RoPE(i-j) · K[pos] with relative RoPE
        // i = current query position, j = cached key position
        // Only the relative distance (i-j) enters the rotation.
        // Since i >= j (causal), rel_pos >= 0 and bounded by window size.
        var rel_pos = (cache_params.current_pos + token_idx) - pos;
        
        // Attention sinks: first SINK_COUNT tokens stay visible forever.
        // Clamp their distance to max 2047 so the RoPE angle stays in-distribution.
        // The model doesn't use these for positional content — just as stabilizers.
        if (pos < SINK_COUNT) {
            rel_pos = min(rel_pos, 2047u);
        } else if (rel_pos > 2047u) {
            // Regular sliding window: mask out tokens beyond training horizon.
            scores[pos] = -1e10;
            continue;
        }
        
        var dot_qk = 0.0;
        let k_idx_base = (pos * params.head_count_kv * params.head_dim)
                       + (kv_head_idx * params.head_dim);
        
        // Process dimension pairs: dims (2p, 2p+1) form a complex number
        // Formula: score += (q_re*k_re + q_im*k_im)*cos(Δθ)
        //                 + (q_re*k_im - q_im*k_re)*sin(Δθ)
        // where Δ = rel_pos, θ = frequency for this pair
        // FSE optimization: pre-computed cos/sin table eliminates per-thread trig
        let n_pairs = params.head_dim / 2u;
        for (var p = 0u; p < n_pairs; p++) {
            let table_idx = rel_pos * n_pairs * 2u + p * 2u;
            let cos_a = rope_table[table_idx];
            let sin_a = rope_table[table_idx + 1u];
            
            let d = p * 2u;
            let q_re = temp_state[q_base + d];
            let q_im = temp_state[q_base + d + 1u];
            let k_re = kv_cache_k[k_idx_base + d];
            let k_im = kv_cache_k[k_idx_base + d + 1u];
            
            dot_qk += (q_re * k_re + q_im * k_im) * cos_a
                    + (q_re * k_im - q_im * k_re) * sin_a;
        }
        
        // Apply scaling
        let score = dot_qk * scale;
        
        // Apply causal masking: can only attend to pos <= current_pos + token_idx
        if (pos <= cache_params.current_pos + token_idx) {
            scores[pos] = score;
            max_score = max(max_score, score);
        } else {
            scores[pos] = -1e10; // Mask out future positions
        }
    }
    
    // --------------------------------------------------------------------
    // STEP 2: Softmax - numerically stable version
    // exp(score - max) / sum(exp(score - max))
    // --------------------------------------------------------------------
    var sum_exp = 0.0;
    for (var pos = 0u; pos < cache_params.seq_len; pos++) {
        if (pos <= cache_params.current_pos + token_idx) {
            let exp_score = exp(scores[pos] - max_score);
            scores[pos] = exp_score;
            sum_exp += exp_score;
        } else {
            scores[pos] = 0.0;
        }
    }
    
    // Normalize to get attention weights
    for (var pos = 0u; pos < cache_params.seq_len; pos++) {
        scores[pos] = scores[pos] / sum_exp;
    }
    
    // --------------------------------------------------------------------
    // STEP 3: Compute context = weighted sum of V values
    // context[head_offset] = sum over pos of (attn_weight[pos] * V[pos, kv_head, head_offset])
    // --------------------------------------------------------------------
    var context_val = 0.0;
    for (var pos = 0u; pos < cache_params.seq_len; pos++) {
        if (pos <= cache_params.current_pos + token_idx) {
            let attn_weight = scores[pos];
            
            // V cache index: [pos, kv_head, head_offset]
            let v_idx = (pos * params.head_count_kv * params.head_dim)
                      + (kv_head_idx * params.head_dim)
                      + head_offset;
            let v_val = kv_cache_v[v_idx];
            context_val += attn_weight * v_val;
        }
    }
    
    // --------------------------------------------------------------------
    // STEP 4: Output projection - Context @ O_weight
    // This maps [n_head * head_dim] -> [dim]
    // Each thread idx computes one element of the output
    // We need the full context vector [head_idx, head_offset] for all heads
    // --------------------------------------------------------------------
    
    // Store context temporarily (each thread stores its contribution)
    // Problem: We need the full context vector for matmul, but each thread
    // only computed one element (for its head_offset).
    // Solution: Write to temp_state as intermediate, use barrier
    // OR: Compute full context in matmul loop (redundant but correct)
    
    // COMPROMISE: Keep context in temp_state, read from there in matmul
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
