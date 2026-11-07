//! Actual V1 projection parity on a real GPU, including guarded cache writes.
use airframe::backend::bindless::pipeline::{CacheParams, LayerOffsets, LayerParams};
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;
fn weights(slot: u32, rows: usize, dim: usize) -> Vec<u8> {
    let mut out = Vec::new();
    if slot <= 1 {
        for i in 0..rows * dim {
            let v = ((i * 17 % 31) as f32 - 15.0) * 0.003;
            if slot == 0 {
                out.extend(v.to_le_bytes());
            } else {
                out.extend(half::f16::from_f32(v).to_le_bytes());
            }
        }
    } else {
        let (epb, bpb) = match slot {
            2 => (32, 18),
            3 => (32, 22),
            4 => (32, 34),
            5 => (256, 144),
            6 => (256, 176),
            7 => (256, 210),
            _ => unreachable!(),
        };
        for b in 0..rows * dim / epb {
            let mut block = vec![0u8; bpb];
            match slot {
                2..=4 => {
                    block[..2].copy_from_slice(&half::f16::from_f32(0.025).to_le_bytes());
                    for (j, v) in block[2..].iter_mut().enumerate() {
                        *v = ((j * 13 + b * 7) % 256) as u8;
                    }
                }
                5 | 6 => {
                    block[..2].copy_from_slice(&half::f16::from_f32(0.012).to_le_bytes());
                    block[2..4].copy_from_slice(&half::f16::from_f32(0.004).to_le_bytes());
                    block[4..16].fill(0x11);
                    for (j, v) in block[16..].iter_mut().enumerate() {
                        *v = ((j * 11 + b * 7) % 256) as u8;
                    }
                }
                7 => {
                    for (j, v) in block[..192].iter_mut().enumerate() {
                        *v = ((j * 19 + b * 5) % 256) as u8;
                    }
                    block[192..208].fill(3);
                    block[208..].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
                }
                _ => unreachable!(),
            }
            out.extend(block);
        }
    }
    out
}
struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    reference: wgpu::ComputePipeline,
    cooperative: wgpu::ComputePipeline,
    attn_reference: wgpu::ComputePipeline,
    attn_cooperative: wgpu::ComputePipeline,
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
            .expect("real GPU required");
        let info = adapter.get_info();
        if std::env::var_os("AIRFRAME_TEST_REQUIRE_METAL").is_some() {
            assert_eq!(info.backend, wgpu::Backend::Metal);
        }
        assert_ne!(
            info.device_type,
            wgpu::DeviceType::Cpu,
            "software adapter is not GPU evidence"
        );
        eprintln!("FFN parity adapter: {info:?}");
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
            label: Some("actual V1 projection"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../src/backend/bindless/sh_layer_v1.wgsl"),
                    include_str!("../src/backend/bindless/sh_qkv_cooperative.wgsl"),
                    include_str!("../src/backend/bindless/sh_attn_proj_cooperative.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let attn_reference = pipeline("main_attn_proj");
        let attn_cooperative = pipeline("main_attn_proj_cooperative");
        let reference = pipeline("main_qkv");
        let cooperative = pipeline("main_qkv_cooperative");
        Self {
            device,
            queue,
            layout,
            reference,
            cooperative,
            attn_reference,
            attn_cooperative,
        }
    }

    fn attn(&self, slot: u32, batch: u32, wide: bool, cooperative: bool) -> Vec<f32> {
        let cols = if slot <= 1 { 257usize } else { 256usize };
        let dim = if wide { 8usize } else { 512usize };
        let stride = 2 * dim + cols + 32;
        let weight_bytes = weights(slot, dim, cols);
        let mut blob = vec![0u8; 256];
        blob.extend(&weight_bytes);
        let params = LayerParams {
            dim: dim as u32,
            head_count: 1,
            head_count_kv: 1,
            head_dim: cols as u32,
            temp_stride: stride as u32,
            formula_attn_out: slot,
            chunk_words: blob.len() as u32 / 4,
            ..LayerParams::zeroed()
        };
        let offsets = LayerOffsets {
            attn_out: 128,
            ..LayerOffsets::zeroed()
        };
        let cache = CacheParams {
            batch_size: batch,
            ..CacheParams::zeroed()
        };
        let input: Vec<f32> = (0..batch as usize * dim)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.03)
            .collect();
        let mut temp = vec![-91.0f32; batch as usize * stride];
        for t in 0..batch as usize {
            for j in 0..cols {
                temp[t * stride + j] = ((j * 13 + t * 7) % 31) as f32 * 0.007 - 0.1;
            }
        }
        let initial = temp.clone();
        let buffer = |bytes: &[u8], uniform: bool| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: if uniform {
                        wgpu::BufferUsages::UNIFORM
                    } else {
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    },
                })
        };
        let blob = buffer(&blob, false);
        let act = buffer(bytemuck::cast_slice(&input), false);
        let temp = buffer(bytemuck::cast_slice(&temp), false);
        let off = buffer(bytemuck::bytes_of(&offsets), true);
        let par = buffer(bytemuck::bytes_of(&params), true);
        let cp = buffer(bytemuck::bytes_of(&cache), true);
        let dummy: Vec<_> = (0..11).map(|_| buffer(&[0u8; 256], false)).collect();
        let buffers = [
            &blob, &act, &temp, &off, &par, &dummy[0], &dummy[1], &dummy[2], &dummy[3], &cp,
            &dummy[4], &dummy[5], &dummy[6], &dummy[7], &dummy[8], &dummy[9], &dummy[10],
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: temp.size() + act.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(if cooperative {
                &self.attn_cooperative
            } else {
                &self.attn_reference
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                if cooperative {
                    dim as u32
                } else {
                    (dim as u32).div_ceil(256)
                },
                batch + 1,
                1,
            );
        }
        enc.copy_buffer_to_buffer(&temp, 0, &rb, 0, temp.size());
        enc.copy_buffer_to_buffer(&act, 0, &rb, temp.size(), act.size());
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let view = rb.slice(..).get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        let stash = dim + cols;
        for t in 0..batch as usize {
            for j in 0..stride {
                if j < stash || j >= stash + dim {
                    assert_eq!(
                        out[t * stride + j],
                        initial[t * stride + j],
                        "attn input/scratch changed"
                    );
                }
            }
            for row in 0..dim {
                let dot = out[t * stride + stash + row];
                let residual = out[initial.len() + t * dim + row];
                assert!(
                    dot.is_finite() && (residual - input[t * dim + row] - dot).abs() < 3e-5,
                    "attn residual/stash mismatch"
                );
                if slot == 0 {
                    let expected = (0..cols)
                        .map(|j| {
                            let k = (row * cols + j) * 4;
                            f32::from_le_bytes(weight_bytes[k..k + 4].try_into().unwrap())
                                * initial[t * stride + j]
                        })
                        .sum::<f32>();
                    assert!((dot - expected).abs() < 3e-5, "attention CPU oracle");
                }
            }
        }
        out
    }
    fn qkv(
        &self,
        slot: u32,
        v_slot: u32,
        batch: u32,
        microbatch: (u32, u32),
        bias: bool,
        cooperative: bool,
    ) -> Vec<f32> {
        let (offset, count) = microbatch;
        let dim = if slot <= 1 && v_slot <= 1 {
            257usize
        } else {
            256
        };
        let q = 12usize;
        let kv = 4usize;
        let stride = dim * 2 + q + 64;
        let mut blob = vec![0u8; 256];
        let mut append = |bytes: Vec<u8>| {
            while !blob.len().is_multiple_of(4) {
                blob.push(0);
            }
            let offset = blob.len() as u32 / 2;
            blob.extend(bytes);
            offset
        };
        let qoff = append(weights(slot, q, dim));
        let koff = append(weights(slot, kv, dim));
        let voff = append(weights(v_slot, kv, dim));
        let biases: Vec<Vec<f32>> = [q, kv, kv]
            .into_iter()
            .enumerate()
            .map(|(s, n)| {
                (0..n)
                    .map(|r| (s as f32 + 1.0) * 0.07 + r as f32 * 0.001)
                    .collect()
            })
            .collect();
        let boff: Vec<u32> = biases
            .iter()
            .map(|b| {
                if bias {
                    append(bytemuck::cast_slice(b).to_vec())
                } else {
                    0
                }
            })
            .collect();
        while !blob.len().is_multiple_of(4) {
            blob.push(0);
        }
        let params = LayerParams {
            dim: dim as u32,
            temp_stride: stride as u32,
            head_count: 3,
            head_count_kv: 1,
            head_dim: 4,
            batch_offset: offset,
            batch_count: count,
            formula_qk: slot,
            formula_v: v_slot,
            chunk_words: blob.len() as u32 / 4,
            ..LayerParams::zeroed()
        };
        let offsets = LayerOffsets {
            attn_q: qoff,
            attn_k: koff,
            attn_v: voff,
            attn_q_bias: boff[0],
            attn_k_bias: boff[1],
            attn_v_bias: boff[2],
            ..LayerOffsets::zeroed()
        };
        let cache = CacheParams {
            current_pos: 2,
            batch_size: batch,
            max_seq_len: batch + 3,
            ..CacheParams::zeroed()
        };
        let input: Vec<f32> = (0..batch as usize * dim)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.03)
            .collect();
        let mut temp = vec![-91.0f32; batch as usize * stride];
        for t in 0..batch as usize {
            temp[t * stride..t * stride + dim].copy_from_slice(&input[t * dim..(t + 1) * dim]);
        }
        let initial_temp = temp.clone();
        let initial_kv = vec![-91.0f32; (batch as usize + 3) * kv];
        let buffer = |bytes: &[u8], uniform: bool| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: if uniform {
                        wgpu::BufferUsages::UNIFORM
                    } else {
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    },
                })
        };
        let blob = buffer(&blob, false);
        let act = buffer(bytemuck::cast_slice(&input), false);
        let temp = buffer(bytemuck::cast_slice(&temp), false);
        let off = buffer(bytemuck::bytes_of(&offsets), true);
        let par = buffer(bytemuck::bytes_of(&params), true);
        let cp = buffer(bytemuck::bytes_of(&cache), true);
        let k = buffer(bytemuck::cast_slice(&initial_kv), false);
        let v = buffer(bytemuck::cast_slice(&initial_kv), false);
        let dummy: Vec<_> = (0..10).map(|_| buffer(&[0u8; 256], false)).collect();
        let buffers = [
            &blob, &act, &temp, &off, &par, &dummy[0], &dummy[1], &k, &v, &cp, &dummy[2],
            &dummy[3], &dummy[4], &dummy[5], &dummy[6], &dummy[7], &dummy[8],
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let size = temp.size() + act.size() + k.size() + v.size();
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(if cooperative {
                &self.cooperative
            } else {
                &self.reference
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                if cooperative {
                    (q + 2 * kv) as u32
                } else {
                    ((q + 2 * kv) as u32).div_ceil(256)
                },
                count + 1,
                1,
            );
        }
        let mut end = 0;
        for b in [&temp, &act, &k, &v] {
            enc.copy_buffer_to_buffer(b, 0, &rb, end, b.size());
            end += b.size();
        }
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let view = rb.slice(..).get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        let act_start = initial_temp.len();
        let k_start = act_start + input.len();
        let v_start = k_start + initial_kv.len();
        assert_eq!(&out[act_start..k_start], &input);
        for t in 0..batch as usize {
            let active = t >= offset as usize && t < (offset + count).min(batch) as usize;
            for j in 0..stride {
                let is_q = j >= dim && j < dim + q;
                if !(active && is_q) {
                    assert_eq!(
                        out[t * stride + j],
                        initial_temp[t * stride + j],
                        "QKV scratch overlap t={t} j={j}"
                    );
                }
            }
            for (start, stage, rows) in [
                (0usize, 0usize, q),
                (k_start, 1usize, kv),
                (v_start, 2usize, kv),
            ] {
                let index = if stage == 0 {
                    t * stride + dim
                } else {
                    start + (2 + t) * kv
                };
                for row in 0..rows {
                    if !active {
                        assert_eq!(out[index + row], -91.0);
                        continue;
                    }
                    assert!(out[index + row].is_finite());
                    let kind = if stage == 2 { v_slot } else { slot };
                    if kind == 0 {
                        let w = weights(0, rows, dim);
                        let dot = (0..dim)
                            .map(|j| {
                                let p = (row * dim + j) * 4;
                                let weight = f32::from_le_bytes(w[p..p + 4].try_into().unwrap());
                                input[t * dim + j] * weight
                            })
                            .sum::<f32>()
                            + if bias { biases[stage][row] } else { 0.0 };
                        assert!(
                            (out[index + row] - dot).abs() < 3e-5,
                            "QKV CPU oracle stage={stage} row={row}"
                        );
                    }
                }
            }
        }
        for start in [k_start, v_start] {
            assert!(out[start..start + 2 * kv].iter().all(|x| *x == -91.0));
            assert!(
                out[start + (2 + batch as usize) * kv..start + initial_kv.len()]
                    .iter()
                    .all(|x| *x == -91.0)
            );
        }
        out
    }
}

#[test]
fn cooperative_attention_projection_preserves_width_stash_and_residual() {
    pollster::block_on(async {
        let h = Harness::new().await;
        for slot in 0..=7 {
            for batch in [1, 3] {
                for wide in [false, true] {
                    let a = h.attn(slot, batch, wide, false);
                    let b = h.attn(slot, batch, wide, true);
                    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                        assert!(
                            x.is_finite()
                                && y.is_finite()
                                && (x - y).abs() <= 3e-4 + 3e-4 * x.abs(),
                            "attn slot={slot} batch={batch} wide={wide} idx={i}: {x} != {y}"
                        );
                    }
                }
            }
        }
    });
}
#[test]
fn cooperative_qkv_matches_reference_bias_gqa_microbatch_and_cpu() {
    pollster::block_on(async {
        let h = Harness::new().await;
        for (slot, v_slot) in (0..=7).map(|s| (s, s)).chain([(5, 7), (0, 1)]) {
            for (batch, offset, count) in [(1, 0, 1), (3, 0, 3), (3, 1, 1)] {
                for bias in [false, true] {
                    let a = h.qkv(slot, v_slot, batch, (offset, count), bias, false);
                    let b = h.qkv(slot, v_slot, batch, (offset, count), bias, true);
                    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                        assert!(x.is_finite() && y.is_finite() && (x-y).abs() <= 3e-4+3e-4*x.abs(),"qkv slot={slot}/{v_slot} batch={batch} offset={offset} bias={bias} idx={i}: {x} != {y}");
                    }
                }
            }
        }
    });
}
