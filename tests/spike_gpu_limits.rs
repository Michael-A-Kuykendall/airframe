// SPIKE 1: Query actual GPU limits before assuming anything
// This tells us what we can ACTUALLY do, not what we hope

#[tokio::test]
async fn spike_query_gpu_limits() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter");

    let info = adapter.get_info();
    let limits = adapter.limits();

    println!("\n=== GPU DISCOVERY ===");
    println!("Device: {} ({:?})", info.name, info.device_type);
    println!("Backend: {:?}", info.backend);
    println!("Driver: {}", info.driver);

    println!("\n=== CRITICAL LIMITS FOR KV CACHE ===");
    println!(
        "Max storage buffers per stage: {}",
        limits.max_storage_buffers_per_shader_stage
    );
    println!("  ^ Need: 7 (gguf, activation, k_cache, v_cache, params×3)");

    println!(
        "Max buffer size: {} bytes ({} MB)",
        limits.max_buffer_size,
        limits.max_buffer_size / 1_000_000
    );
    println!("  ^ Need per KV buffer: 1,048,576 bytes (1 MB)");
    println!("  ^ Total KV cache: 44 MB (22 layers × 2 buffers)");

    println!(
        "Max storage buffer binding size: {} bytes ({} MB)",
        limits.max_storage_buffer_binding_size,
        limits.max_storage_buffer_binding_size / 1_000_000
    );

    println!(
        "Max compute workgroup size: {:?}",
        limits.max_compute_workgroup_size_x
    );
    println!("  ^ Need: 64 (for attention heads)");

    println!(
        "Max compute invocations per workgroup: {}",
        limits.max_compute_invocations_per_workgroup
    );
    println!("  ^ Need: 256 (current setting)");

    println!("\n=== VALIDATION ===");

    let needs_7_buffers = limits.max_storage_buffers_per_shader_stage >= 7;
    let needs_1mb_buffer = limits.max_buffer_size >= 1_048_576;
    let needs_44mb_total = limits.max_buffer_size >= 44_000_000;
    let needs_64_workgroup = limits.max_compute_workgroup_size_x >= 64;

    println!("✓ Can bind 7 storage buffers: {}", needs_7_buffers);
    println!("✓ Can allocate 1 MB buffers: {}", needs_1mb_buffer);
    println!("✓ Can allocate 44 MB total: {}", needs_44mb_total);
    println!("✓ Can use workgroup_size(64): {}", needs_64_workgroup);

    if needs_7_buffers && needs_1mb_buffer && needs_44mb_total && needs_64_workgroup {
        println!("\n*** SPIKE 1 RESULT: PASS ***");
        println!("GPU supports our KV cache architecture");
    } else {
        println!("\n*** SPIKE 1 RESULT: FAIL ***");
        println!("GPU does NOT support our requirements");
        println!("Architecture redesign needed!");
    }

    assert!(needs_7_buffers, "Insufficient storage buffer bindings");
    assert!(needs_1mb_buffer, "Buffer size too small");
    assert!(needs_64_workgroup, "Workgroup size too small");
}
