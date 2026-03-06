// sh_matmul_f32.wgsl
// MatMul F32: y = x @ W^T
// x: f32 vector [K]
// W: f32 matrix [N, K] (Row Major)
// Output: y [N]

struct MatMulParams {
    N: u32,
    K: u32,
    weights_offset: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.N) { return; }

    let K = params.K;
    var sum = 0.0;
    
    let w_row_start = row * K;
    
    // Accumulate dot product
    for (var k = 0u; k < K; k = k + 1u) {
        sum = sum + W[w_row_start + k] * input_x[k];
    }
    
    output_y[row] = sum;
}
