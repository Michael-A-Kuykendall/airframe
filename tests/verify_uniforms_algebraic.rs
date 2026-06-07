//! Algebraic verification: Do uniforms reach the shader?
//!
//! This test writes a simple shader that reads uniforms and writes them to output.
//! If this fails, the uniform binding is broken at the WGPU level.


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TestUniforms {
    value_a: u32,
    value_b: u32,
    value_c: f32,
    padding: u32,
}

#[tokio::test]
#[ignore]
async fn test_uniform_passthrough() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Algebraic Uniform Passthrough Test ===\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("No GPU");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // Create uniform buffer with known values
    let test_data = TestUniforms {
        value_a: 42,
        value_b: 12345,
        value_c: std::f32::consts::PI,
        padding: 0,
    };

    println!("[1/4] Uploading uniforms:");
    println!("      value_a = {}", test_data.value_a);
    println!("      value_b = {}", test_data.value_b);
    println!("      value_c = {:.5}", test_data.value_c);

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Uniforms"),
        size: std::mem::size_of::<TestUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&test_data));

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: 16, // 4 x f32
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Simple shader that copies uniforms to output
    let shader_src = r#"
struct Uniforms {
    value_a: u32,
    value_b: u32,
    value_c: f32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
    output[0] = f32(uniforms.value_a);
    output[1] = f32(uniforms.value_b);
    output[2] = uniforms.value_c;
    output[3] = 999.0; // Sentinel value
}
"#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Test Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Test Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Test Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Test Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    println!("[2/4] Dispatching shader...");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_pipeline(&pipeline);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, 16);
    let idx = queue.submit(Some(encoder.finish()));

    println!("[3/4] Reading back results...");
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();

    let data = slice.get_mapped_range();
    let results: &[f32] = bytemuck::cast_slice(&data);

    println!("      Shader wrote:");
    println!("      output[0] = {:.1} (expected 42.0)", results[0]);
    println!("      output[1] = {:.1} (expected 12345.0)", results[1]);
    println!("      output[2] = {:.5} (expected PI)", results[2]);
    println!("      output[3] = {:.1} (expected 999.0)", results[3]);

    println!("\n[4/4] === VERDICT ===");
    let a_match = (results[0] - 42.0).abs() < 0.1;
    let b_match = (results[1] - 12345.0).abs() < 0.1;
    let c_match = (results[2] - std::f32::consts::PI).abs() < 0.01;
    let sentinel_match = (results[3] - 999.0).abs() < 0.1;

    if a_match && b_match && c_match && sentinel_match {
        println!("✅ PASS: Uniforms reach shader correctly");
    } else {
        println!("❌ FAIL: Uniform binding is broken!");
        if !a_match {
            println!("    value_a mismatch");
        }
        if !b_match {
            println!("    value_b mismatch");
        }
        if !c_match {
            println!("    value_c mismatch");
        }
        if !sentinel_match {
            println!("    Shader didn't execute!");
        }
    }

    assert!(a_match && b_match && c_match && sentinel_match);
    Ok(())
}
