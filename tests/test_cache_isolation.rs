//! Verify each layer binds to its own KV cache buffer
use airframe::backend::bindless::kv_cache::KVCache;

#[tokio::test]
#[ignore]
async fn test_per_layer_cache_isolation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Per-Layer Cache Isolation Test ===\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("No GPU");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    let kv_cache = KVCache::new(&device, 22, 4, 64, 2048);

    // Write unique pattern to each layer's K cache
    println!("[1/3] Writing layer-specific patterns to each cache...");
    for layer_idx in 0..22 {
        let pattern: Vec<f32> = (0..256)
            .map(|i| (layer_idx as f32 * 1000.0) + (i as f32))
            .collect();
        let pattern_bytes = bytemuck::cast_slice(&pattern);

        queue.write_buffer(kv_cache.get_k_buffer(layer_idx), 0, pattern_bytes);
    }

    device.poll(wgpu::PollType::Poll).unwrap();

    // Read back and verify each layer has its own pattern
    println!("[2/3] Reading back patterns...\n");
    let mut all_correct = true;

    for layer_idx in 0..22 {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Staging L{}", layer_idx)),
            size: 256 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(kv_cache.get_k_buffer(layer_idx), 0, &staging, 0, 256 * 4);
        let idx = queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        // Check first 10 values
        let expected_first = layer_idx as f32 * 1000.0;
        let actual_first = vals[0];
        let is_correct = (actual_first - expected_first).abs() < 0.1;

        let status = if is_correct { "✅" } else { "❌" };
        println!(
            "Layer {:2}: {} | Expected first={:.1}, Actual={:.1}",
            layer_idx, status, expected_first, actual_first
        );

        if !is_correct {
            all_correct = false;
            println!(
                "        ❌ CORRUPTION! Layer {} cache contains Layer {}'s data!",
                layer_idx,
                (actual_first / 1000.0) as usize
            );
        }

        drop(data);
        staging.unmap();
    }

    println!("\n[3/3] === VERDICT ===");
    if all_correct {
        println!("✅ PASS: All 22 layers have isolated cache buffers");
    } else {
        println!("❌ FAIL: Cache buffers are shared or mixed up!");
    }

    assert!(all_correct, "Cache isolation violated");
    Ok(())
}
