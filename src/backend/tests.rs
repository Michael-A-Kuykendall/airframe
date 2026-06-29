#[cfg(test)]
mod tests_inner {
    use crate::backend::pipeline::LogitMaskPipeline;
    use crate::backend::wgpu::WgpuContext;
    use wgpu::util::DeviceExt;

    #[tokio::test]
    async fn test_logit_mask_gpu() {
        let ctx = WgpuContext::new().await;
        let pipeline = LogitMaskPipeline::new(&ctx);

        let logits_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask_data: Vec<u32> = vec![0, 1, 0, 0, 1];
        // Expected: [1.0, -inf, 3.0, 4.0, -inf]

        let logits_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Logits Buffer"),
                contents: bytemuck::cast_slice(&logits_data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let mask_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mask Buffer"),
                contents: bytemuck::cast_slice(&mask_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        pipeline.run(&ctx, &logits_buffer, &mask_buffer, logits_data.len() as u32);

        // Read back
        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (logits_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &logits_buffer,
            0,
            &staging_buffer,
            0,
            (logits_data.len() * 4) as u64,
        );

        let idx = ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();
        receiver.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);

        assert_eq!(result[0], 1.0);
        assert!(result[1] < -1000.0); // -infinity
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);
        assert!(result[4] < -1000.0); // -infinity
    }
}
