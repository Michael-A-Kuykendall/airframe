// sh_rmsnorm.wgsl
// Root Mean Square Normalization
// y = x * w * rsqrt(mean(x^2) + eps)

struct Params {
    count: u32,
    weight_offset: u32, // Word index (byte_offset / 4) to the start of the weight tensor in GGUF blob
    bias_offset: u32,   // Word index (byte_offset / 4) to bias tensor; 0 disables bias
    eps: f32,
    norm_type: u32, // 0 = RMSNorm, 1 = LayerNorm (mean+variance)
};

@group(0) @binding(0)  var<storage, read> blob_0: array<u32>;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;

const BLOB_SPLIT_0: u32 = 500000000u;  // 2,000,000,000 bytes / 4 = 500M words
const BLOB_SPLIT_1: u32 = 1000000000u; // 4,000,000,000 bytes / 4 = 1B words

fn read_blob(word_idx: u32) -> u32 {
    if word_idx < BLOB_SPLIT_0 {
        return blob_0[word_idx];
    } else if word_idx < BLOB_SPLIT_1 {
        return blob_1[word_idx - BLOB_SPLIT_0];
    } else {
        return blob_2[word_idx - BLOB_SPLIT_1];
    }
}
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256;
var<workgroup> s_sum: array<f32, BLOCK_SIZE>;
var<workgroup> s_sum_sq: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let count = params.count;

    // 1. Accumulate sums needed for RMSNorm/LayerNorm
    var sum = 0.0;
    var sum_sq = 0.0;
    for (var i = tid; i < count; i += BLOCK_SIZE) {
        let val = input[i]; // Assuming single row for now (group_id.x can handle batch later)
        sum += val;
        sum_sq += val * val;
    }

    // 2. Reduce in Shared Memory
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    workgroupBarrier();

    // Tree reduction for 256 threads
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        workgroupBarrier();
    }

    // 3. Compute normalization scale
    let mean = s_sum[0] / f32(count);
    let mean_sq = s_sum_sq[0] / f32(count);
    let variance = max(mean_sq - mean * mean, 0.0);
    let scale = select(
        inverseSqrt(mean_sq + params.eps),
        inverseSqrt(variance + params.eps),
        params.norm_type == 1u,
    );

    // 4. Apply Scale and Weight
    // weight_offset is already a word index (byte_offset / 4), passed from Rust as (byte_offset / 4) as u32.
    let w_u32_start = params.weight_offset;
    
    for (var i = tid; i < count; i += BLOCK_SIZE) {
        let val = input[i];
        
        // Read Weight: it's a simple F32 array in the file
        // Reinterpret u32 bits as f32
        let w_bits = read_blob(w_u32_start + i);
        let w_val = bitcast<f32>(w_bits);
        let b_val = select(0.0, bitcast<f32>(read_blob(params.bias_offset + i)), params.bias_offset != 0u);

        let centered = select(val, val - mean, params.norm_type == 1u);
        output[i] = centered * scale * w_val + b_val;
    }
}
