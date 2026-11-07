//! Q/K RMS normalization ownership contract; no model or quantization rounding involved.
use airframe::backend::bindless::pipeline::{CacheParams, LayerOffsets, LayerParams};
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

fn values(d: usize, seed: usize) -> Vec<f32> {
    (0..d)
        .map(|i| 0.1 + ((i * 17 + seed * 13) % 41) as f32 * 0.01)
        .collect()
}
fn weights(d: usize, k: bool) -> Vec<f32> {
    (0..d)
        .map(|i| {
            if k {
                17.0 + (i % 19) as f32
            } else {
                1.0 + (i % 5) as f32
            }
        })
        .collect()
}
fn oracle(x: &[f32], w: &[f32], eps: f32) -> Vec<f64> {
    let mean = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64;
    x.iter()
        .zip(w)
        .map(|(v, w)| f64::from(*v) / (mean + f64::from(eps)).sqrt() * f64::from(*w))
        .collect()
}
fn bound(expected: f64, d: usize) -> f64 {
    // gamma_(2d+8), conservatively scaled for inverse sqrt and multiply.
    // Absolute floor covers backend inverseSqrt rounding near zero.
    let n = (2 * d + 8) as f64;
    let u = f64::from(f32::EPSILON) / 2.0;
    8.0 * n * u / (1.0 - n * u) * expected.abs() + 2e-5
}
#[test]
fn old_element_ownership_has_legal_wave_schedule_counterexample() {
    let d = 128;
    let original = values(d, 0);
    let w = weights(d, false);
    let eps = 1e-6;
    let expected = oracle(&original, &w, eps);
    let mut inplace = original.clone();
    // Each wave performs all reads, then all writes; no inter-wave barrier.
    // This satisfies each invocation's program order in the original kernel.
    for start in (0..d).step_by(32) {
        let mut sum = 0.0_f32;
        for v in &inplace {
            sum += v * v;
        }
        let inv = (sum / d as f32 + eps).sqrt().recip();
        let pending: Vec<_> = (start..start + 32)
            .map(|i| inplace[i] * inv * w[i])
            .collect();
        inplace[start..start + 32].copy_from_slice(&pending);
    }
    let excess = inplace
        .iter()
        .zip(&expected)
        .map(|(a, b)| (f64::from(*a) - b).abs() / bound(*b, d))
        .fold(0.0_f64, f64::max);
    assert!(
        excess > 100.0,
        "counterexample must exceed rounding envelope: {excess}"
    );
    eprintln!("LEGAL_WAVE_COUNTEREXAMPLE max_error_to_rounding_bound={excess}");
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl Harness {
    async fn new() -> Self {
        static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
        let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
        let instance = INSTANCE.get_or_init(|| {
            wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends,
                flags: wgpu::InstanceFlags::default().with_env(),
                ..Default::default()
            })
        });
        let adapter = instance
            .enumerate_adapters(backends)
            .into_iter()
            .max_by_key(|a| match a.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu => 3,
                wgpu::DeviceType::IntegratedGpu => 2,
                _ => 0,
            })
            .expect("GPU required");
        let info = adapter.get_info();
        assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
        eprintln!("QK_ADAPTER {info:?}");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let entries: Vec<_> = (0..17)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if [3, 4, 9].contains(&binding) {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: ![1, 2, 7, 8].contains(&binding),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("actual V1 QK"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/backend/bindless/sh_layer_v1.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main_qk_norm"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            layout,
            pipeline,
        }
    }
    fn run(&self, d: usize, batch: usize, enabled: bool, seed: usize) -> f64 {
        let qheads = 4;
        let kheads = 2;
        let qlen = qheads * d;
        let klen = kheads * d;
        let hidden = qlen / 2;
        let qstart = qlen;
        let stride = qstart + qlen + 64;
        let qw = weights(d, false);
        let kw = weights(d, true);
        let mut blob = vec![0_u8; 256];
        let qoff = blob.len() as u32 / 2;
        blob.extend(bytemuck::cast_slice(&qw));
        let koff = blob.len() as u32 / 2;
        blob.extend(bytemuck::cast_slice(&kw));
        let eps = 1e-6;
        let params = LayerParams {
            dim: hidden as u32,
            head_count: qheads as u32,
            head_count_kv: kheads as u32,
            head_dim: d as u32,
            temp_stride: stride as u32,
            qk_norm_enabled: u32::from(enabled),
            rms_eps: eps,
            chunk_words: blob.len() as u32 / 4,
            ..LayerParams::zeroed()
        };
        let offsets = LayerOffsets {
            attn_q_norm: qoff,
            attn_k_norm: koff,
            ..LayerOffsets::zeroed()
        };
        let cache = CacheParams {
            current_pos: 1,
            batch_size: batch as u32,
            max_seq_len: (batch + 2) as u32,
            ..CacheParams::zeroed()
        };
        let mut temp = vec![-91_f32; batch * stride];
        let mut keys = vec![-93_f32; (batch + 2) * klen];
        let mut want_temp: Vec<f64> = temp.iter().map(|x| f64::from(*x)).collect();
        let mut want_keys: Vec<f64> = keys.iter().map(|x| f64::from(*x)).collect();
        for t in 0..batch {
            for h in 0..qheads {
                let x = values(d, seed + t * 7 + h);
                let start = t * stride + qstart + h * d;
                temp[start..start + d].copy_from_slice(&x);
                let expected = if enabled {
                    oracle(&x, &qw, eps)
                } else {
                    x.iter().map(|x| f64::from(*x)).collect()
                };
                want_temp[start..start + d].copy_from_slice(&expected);
            }
            for h in 0..kheads {
                let x = values(d, seed + t * 11 + h + 23);
                let start = (t + 1) * klen + h * d;
                keys[start..start + d].copy_from_slice(&x);
                let expected = if enabled {
                    oracle(&x, &kw, eps)
                } else {
                    x.iter().map(|x| f64::from(*x)).collect()
                };
                want_keys[start..start + d].copy_from_slice(&expected);
            }
        }
        let mk = |b: &[u8], uniform| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: b,
                    usage: if uniform {
                        wgpu::BufferUsages::UNIFORM
                    } else {
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    },
                })
        };
        let blob = mk(&blob, false);
        let temp = mk(bytemuck::cast_slice(&temp), false);
        let keys = mk(bytemuck::cast_slice(&keys), false);
        let offsets = mk(bytemuck::bytes_of(&offsets), true);
        let params = mk(bytemuck::bytes_of(&params), true);
        let cache = mk(bytemuck::bytes_of(&cache), true);
        let dummy: Vec<_> = (0..11).map(|_| mk(&[0_u8; 256], false)).collect();
        let buffers = [
            &blob, &dummy[0], &temp, &offsets, &params, &dummy[1], &dummy[2], &keys, &dummy[3],
            &cache, &dummy[4], &dummy[5], &dummy[6], &dummy[7], &dummy[8], &dummy[9], &dummy[10],
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: temp.size() + keys.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(((qlen + klen) as u32).div_ceil(256), batch as u32 + 1, 1);
        }
        enc.copy_buffer_to_buffer(&temp, 0, &rb, 0, temp.size());
        enc.copy_buffer_to_buffer(&keys, 0, &rb, temp.size(), keys.size());
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let view = rb.slice(..).get_mapped_range();
        let actual: &[f32] = bytemuck::cast_slice(&view);
        let mut max_ratio = 0_f64;
        for (i, (a, b)) in actual
            .iter()
            .zip(want_temp.iter().chain(&want_keys))
            .enumerate()
        {
            let active = if i < want_temp.len() {
                let j = i % stride;
                j >= qstart && j < qstart + qlen
            } else {
                let j = i - want_temp.len();
                j >= klen && j < (batch + 1) * klen
            };
            if !active || !enabled {
                assert_eq!(f64::from(*a), *b, "guard/off changed i={i}");
            } else {
                assert!(a.is_finite());
                max_ratio = max_ratio.max((f64::from(*a) - b).abs() / bound(*b, d));
            }
        }
        eprintln!("QK_OPERATOR d={d} batch={batch} enabled={enabled} seed={seed} max_bound_ratio={max_ratio}");
        max_ratio
    }
}
#[tokio::test]
#[ignore = "requires an actual GPU; run sequentially in an isolated process"]
async fn actual_v1_qk_norm_matches_stable_input_contract() {
    let h = Harness::new().await;
    let baseline = std::env::var_os("AIRFRAME_QK_OBSERVE_UNSAFE_BASELINE").is_some();
    let mut maximum = 0_f64;
    for d in [64, 128, 256, 384, 512] {
        for batch in [1, 64] {
            for seed in 0..3 {
                maximum = maximum.max(h.run(d, batch, true, seed));
            }
            assert_eq!(h.run(d, batch, false, 0), 0.0);
        }
    }
    eprintln!("QK_SUMMARY observe_unsafe_baseline={baseline} maximum_bound_ratio={maximum}");
    if !baseline {
        assert!(maximum <= 1.0, "stable-input numerical contract violated");
    }
}
