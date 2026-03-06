// sh_rmsnorm.wgsl
// Root Mean Square Normalization
// y = x * w * rsqrt(mean(x^2) + eps)

struct Params {
    count: u32,
    weight_offset: u32, // Offset in bytes to the start of the weight tensor in GGUF blob
    eps: f32,
    padding: u32,
};

@group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256;
var<workgroup> s_sum: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let count = params.count;

    // 1. Accumulate Sum of Squares
    var sum_sq = 0.0;
    for (var i = tid; i < count; i += BLOCK_SIZE) {
        let val = input[i]; // Assuming single row for now (group_id.x can handle batch later)
        sum_sq += val * val;
    }

    // 2. Reduce in Shared Memory
    s_sum[tid] = sum_sq;
    workgroupBarrier();

    // Tree reduction for 256 threads
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    // 3. Compute Scale
    // Only thread 0 computes the final scale, but we need to broadcast or everyone recomputes?
    // Everyone reads s_sum[0]
    let mean = s_sum[0] / f32(count);
    let scale = inverseSqrt(mean + params.eps);

    // 4. Apply Scale and Weight
    // Helper to read F32 from GGUF Blob (u32 array)
    // weight_offset is in bytes. Align check needed? Usually GGUF F32 is 4-byte aligned.
    let w_u32_start = params.weight_offset / 4u;
    
    for (var i = tid; i < count; i += BLOCK_SIZE) {
        let val = input[i];
        
        // Read Weight: it's a simple F32 array in the file
        // Reinterpret u32 bits as f32
        let w_bits = gguf_blob[w_u32_start + i];
        let w_val = bitcast<f32>(w_bits);

        output[i] = val * scale * w_val;
    }
}
