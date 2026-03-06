// SPIKE 3: WGSL Compilation Limits
// Test if we can actually compile the attention shader we proposed

const SHADER_SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read> test_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn test_large_local_array() {
    // Test 1: Can we declare 2048-element local array?
    var scores: array<f32, 2048>;
    
    // Test 2: Can we loop 2048 times?
    for (var i = 0u; i < 2048u; i++) {
        scores[i] = f32(i) * 0.001;
    }
    
    // Test 3: Can we do nested loops? (like attention score computation)
    var sum = 0.0;
    for (var i = 0u; i < 64u; i++) {
        for (var j = 0u; j < 2048u; j++) {
            sum += scores[j] * f32(i);
        }
    }
    
    // Test 4: Softmax-like reduction
    var max_val = -1e9;
    for (var i = 0u; i < 2048u; i++) {
        max_val = max(max_val, scores[i]);
    }
    
    var sum_exp = 0.0;
    for (var i = 0u; i < 2048u; i++) {
        scores[i] = exp(scores[i] - max_val);
        sum_exp += scores[i];
    }
    
    for (var i = 0u; i < 2048u; i++) {
        scores[i] /= sum_exp;
    }
    
    // Write result
    output[0] = scores[0];
    output[1] = scores[2047];
    output[2] = sum;
}
"#;

#[tokio::test]
async fn spike_wgsl_array_limits() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("No device");

    println!("\n=== WGSL COMPILATION TEST ===");
    println!("Testing shader with:");
    println!("  - 2048-element local array (8KB stack)");
    println!("  - Nested loops (64 × 2048 iterations)");
    println!("  - Softmax reduction pattern");

    // Try to compile shader (will panic if WGSL limits exceeded)
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Test Large Array Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });
    println!("\n✓ Shader compiled successfully!");

    // Create pipeline to verify it actually works
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("test_large_local_array"),
        compilation_options: Default::default(),
        cache: None,
    });

    println!("✓ Pipeline created successfully!");

    // Run it to verify it actually executes
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 2048 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 3 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 3 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, 3 * 4);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

    loop {
        device.poll(wgpu::PollType::Poll).unwrap();
        if let Ok(res) = rx.try_recv() {
            res.expect("Buffer map failed");
            break;
        }
    }

    let data = slice.get_mapped_range();
    let results: &[f32] = bytemuck::cast_slice(&data);

    println!("\n=== EXECUTION TEST ===");
    println!("Output[0] (first softmax): {:.8}", results[0]);
    println!("Output[1] (last softmax): {:.8}", results[1]);
    println!("Output[2] (nested sum): {:.8}", results[2]);

    if results[0].is_finite() && results[1].is_finite() && results[2].is_finite() {
        println!("\n✓ Shader executed successfully!");
        println!("\n*** SPIKE 3 RESULT: PASS ***");
        println!("WGSL supports our attention kernel architecture:");
        println!("  - 2048-element local arrays: ✓");
        println!("  - Nested loops: ✓");
        println!("  - Softmax reduction: ✓");
    } else {
        println!("\n✗ Shader produced invalid results!");
        panic!("Execution failed");
    }
}
