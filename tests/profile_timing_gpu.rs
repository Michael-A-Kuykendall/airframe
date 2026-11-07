#![cfg(feature = "isf")]
// Compile the private GPU adapter in this test crate without widening its product API.
#[allow(dead_code)]
#[path = "../src/backend/bindless/pool_timestamp.rs"]
mod pool_timestamp;
#[allow(dead_code)]
#[path = "../src/backend/bindless/profiling.rs"]
mod profiling;
use airframe_observe::profiling::ProfileConfig;
use profiling::{generation_contexts, FrameProfile, KernelStage};
use std::io::Write;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
async fn device() -> (wgpu::Device, wgpu::Queue) {
    // Keep the loader alive through test-thread TLS teardown (Mesa/Dozen dlclose callbacks).
    static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
    let instance = INSTANCE.get_or_init(|| {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::default().with_env(),
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        })
    });
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12)
        .into_iter()
        .max_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 3,
            wgpu::DeviceType::IntegratedGpu => 2,
            wgpu::DeviceType::VirtualGpu => 1,
            _ => 0,
        })
        .expect("GPU test requires an adapter");
    eprintln!("GPU control: {:?}", adapter.get_info());
    eprintln!(
        "requesting device, adapter features={:?}",
        adapter.features()
    );
    let result = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: if std::env::var("PROFILE_TEST_HOST").is_ok() {
                wgpu::Features::empty()
            } else {
                adapter.features() & wgpu::Features::TIMESTAMP_QUERY
            },
            ..Default::default()
        })
        .await
        .expect("GPU device");
    eprintln!("device ready");
    result
}
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enabled: bool,
    gpu: bool,
    delay_readback: bool,
) -> (u32, Option<serde_json::Value>) {
    let out = Capture::default();
    let mut session =
        enabled.then(|| ProfileConfig::with_writer(Box::new(out.clone()), gpu).begin_generation());
    let context = session.as_ref().map(|s| generation_contexts(s.handle()).0);
    let mut profile = FrameProfile::new(device, queue, context, 1, 0);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("profile test atomic"),
        contents: bytemuck::cast_slice(&[0u32; 64]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("profiling control"),
        source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> data: array<u32>; @compute @workgroup_size(1) fn main(){data[0]=42u;}".into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (stage, dispatches) in [(KernelStage::AttnNorm, 1), (KernelStage::LmHead, 2)] {
        let mut pass = profile.begin_compute_pass(
            &mut encoder,
            &wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            },
            Some(0),
            stage,
        );
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        for _ in 0..dispatches {
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, 4);
    profile.resolve(&mut encoder);
    profile.submit_final(queue, encoder.finish());
    profile.begin_readback();
    let (tx, rx) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let wait = profile.wait_start();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    profile.wait_end(wait);
    let view = readback.slice(..).get_mapped_range();
    let value = bytemuck::cast_slice::<u8, u32>(&view)[0];
    drop(view);
    readback.unmap();
    let pending = delay_readback.then(|| profile.delay_readback_for_test());
    let started = std::time::Instant::now();
    profile.finish(device);
    if delay_readback {
        assert!(started.elapsed() >= std::time::Duration::from_millis(100));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
    drop(pending);
    if let Some(s) = &mut session {
        s.finish(true);
    }
    let bytes = out.0.lock().unwrap();
    let report = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .last()
        .map(|s| serde_json::from_str(s).unwrap());
    (value, report)
}
#[test]
#[ignore = "requires a real GPU"]
fn gpu_pass_timing_counts_and_math_parity() {
    pollster::block_on(async {
        let (device, queue) = device().await;
        let (off, report) = run(&device, &queue, false, false, false);
        assert_eq!(off, 42);
        assert!(report.is_none());
        let (on, report) = run(&device, &queue, true, true, false);
        assert_eq!(on, off);
        let report = report.unwrap();
        assert_eq!(report["compute_pass_count"], 2);
        assert_eq!(report["dispatch_count"], 3);
        assert_eq!(report["submissions"]["queue_submission_count"], 1);
        assert_eq!(report["prefill_forwards"], 1);
        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            for stage in report["stages"].as_array().unwrap() {
                assert!(
                    stage["totals"]["gpu_ns_sum"].as_u64().unwrap_or(0) > 0,
                    "{report}"
                );
            }
        }
        let (host, report) = run(&device, &queue, true, false, false);
        assert_eq!(host, off);
        assert_eq!(report.unwrap()["gpu_timestamp_enabled"], false);
        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let (value, report) = run(&device, &queue, true, true, true);
            assert_eq!(value, off);
            let report = report.unwrap();
            assert_eq!(report["status"], "complete");
            for stage in report["stages"].as_array().unwrap() {
                assert!(stage["totals"]["gpu_ns_sum"].is_null());
                assert_eq!(stage["totals"]["missing_gpu"]["readback_timeout"], 1);
            }
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}
#[test]
#[ignore = "requires a real GPU"]
fn gpu_query_reservations_capacity_and_invalid_data() {
    pollster::block_on(async {
        let (device, queue) = device().await;
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("GPU timestamp feature unavailable; host-only control exercised separately");
            return;
        }
        let mut pool = pool_timestamp::TimestampPool::for_compute_passes(
            &device,
            2,
            queue.get_timestamp_period() as f64,
        )
        .unwrap();
        {
            let (idx, w) = pool.reserve_pass().unwrap();
            assert_eq!(idx, 0);
            assert_eq!(w.beginning_of_pass_write_index, Some(0));
            assert_eq!(w.end_of_pass_write_index, Some(1));
        }
        {
            let (idx, w) = pool.reserve_pass().unwrap();
            assert_eq!(idx, 1);
            assert_eq!(w.beginning_of_pass_write_index, Some(2));
        }
        assert!(pool.reserve_pass().is_none());
        assert_eq!(pool.used_pairs(), 2);
        assert!(pool.elapsed_ns(0, &[]).is_none());
        assert!(pool.elapsed_ns(0, &[0, 0, 4, 9]).is_none());
        assert!(pool.elapsed_ns(0, &[9, 1, 4, 9]).is_none());
        assert!(pool.elapsed_ns(2, &[1, 9, 4, 9]).is_none());
        assert!(pool.elapsed_ns(1, &[1, 9, 4, 9]).unwrap() > 0.0);
        pool.reset();
        assert_eq!(pool.reserve_pass().unwrap().0, 0);
        assert!(
            pool_timestamp::TimestampPool::new(&device, 2, 1.0).is_none(),
            "device requested only pass-boundary timestamp feature"
        );
        assert!(pool_timestamp::TimestampPool::for_compute_passes(
            &device,
            wgpu::QUERY_SET_MAX_QUERIES,
            1.0
        )
        .is_none());
    });
}
