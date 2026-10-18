//! RoPE pair layout follows GGUF's adjacent (Llama) or split-half (Qwen) convention.
const SHADERS: [&str; 2] = [
    include_str!("../src/backend/bindless/sh_layer_v1.wgsl"),
    include_str!("../src/backend/bindless/sh_layer_v1_int4.wgsl"),
];

#[test]
fn rotary_shaders_parse_and_validate() {
    for source in SHADERS {
        let module = naga::front::wgsl::parse_str(source).expect("production WGSL must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("production WGSL must validate");
    }
}

#[test]
#[ignore = "requires a WebGPU adapter; run with --ignored"]
fn gpu_rotary_pairs_match_reference_layouts() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default().with_env(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("GPU adapter required");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("GPU device required");
        // Match the existing GPU tests: retain handles until process exit to
        // avoid Dozen's device-destruction crash under WSL.
        std::mem::forget((device.clone(), queue.clone()));
        eprintln!("RoPE pairing adapter: {:?}", adapter.get_info());
        use wgpu::util::DeviceExt;
        for production_source in SHADERS {
            // Execute the production selector, rather than a second copy of its logic.
            let normalized = production_source.replace("\r\n", "\n");
            let rest = normalized.split_once("fn rope_pair_indices(").unwrap().1;
            let body = rest.split_once("\n}\n").unwrap().0;
            let helper = format!("fn rope_pair_indices({body}\n}}");
            let source = format!(
                "struct Params {{ rope_dim: u32, rope_neox: u32, pad0: u32, pad1: u32 }};
                 @group(0) @binding(0) var<uniform> params: Params;
                 @group(0) @binding(1) var<storage, read_write> result: array<u32>;
                 {helper}
                 @compute @workgroup_size(1)
                 fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                     let pair = rope_pair_indices(id.x);
                     result[id.x * 2u] = pair.x;
                     result[id.x * 2u + 1u] = pair.y;
                 }}"
            );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("production RoPE pairing"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            for (rope_dim, neox, expected) in [
                (8, 0, [0u32, 1, 2, 3, 4, 5, 6, 7]),
                (8, 1, [0u32, 4, 1, 5, 2, 6, 3, 7]),
                (4, 1, [0u32, 2, 1, 3, 4, 5, 6, 7]),
            ] {
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[rope_dim, neox, 0u32, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let output = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 32,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 32,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output.as_entire_binding(),
                        },
                    ],
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(4, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 32);
                queue.submit(Some(encoder.finish()));
                let (tx, rx) = std::sync::mpsc::channel();
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let mapped = staging.slice(..).get_mapped_range();
                assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), expected);
                drop(mapped);
                staging.unmap();
            }
        }
    });
}
