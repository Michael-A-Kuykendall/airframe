//! Actual V1 GPU reference/staged parity; no-adapter is a failure.
use airframe::backend::bindless::pipeline::{CacheParams, LayerOffsets, LayerParams};
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

fn vector_weights(slot: u32, rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = weights(slot, rows, cols);
    let block_bytes = if slot == 5 { 144 } else { 210 };
    for (block, data) in bytes.chunks_exact_mut(block_bytes).enumerate() {
        if slot == 5 {
            for (j, value) in data[4..16].iter_mut().enumerate() {
                *value = (j * 59 + block * 73) as u8;
            }
        } else {
            for (j, value) in data[192..208].iter_mut().enumerate() {
                *value = [0, 127, 128, 255, 1, 129, 64, 192][(j + block) % 8];
            }
        }
    }
    bytes
}

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

#[derive(Clone, Copy)]
struct VectorCase {
    cols: usize,
    up_slot: u32,
    misaligned: bool,
    gelu: bool,
    vectorized: bool,
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    norm: wgpu::ComputePipeline,
    reference: wgpu::ComputePipeline,
    staged: wgpu::ComputePipeline,
    vectorized: wgpu::ComputePipeline,
    down_reference: wgpu::ComputePipeline,
    down_cooperative: wgpu::ComputePipeline,
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
            label: Some("actual V1 FFN"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}\n{}",
                    include_str!("../src/backend/bindless/sh_layer_v1.wgsl"),
                    include_str!("../src/backend/bindless/sh_ffn_staged.wgsl"),
                    include_str!("../src/backend/bindless/sh_ffn_down_cooperative.wgsl"),
                    include_str!("../src/backend/bindless/sh_ffn_vectorized.wgsl")
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
        let norm = pipeline("main_ffn_norm");
        let reference = pipeline("main_ffn_proj");
        let staged = pipeline("main_ffn_proj_staged");
        let vectorized = pipeline("main_ffn_proj_vectorized");
        let down_reference = pipeline("main_ffn_down");
        let down_cooperative = pipeline("main_ffn_down_cooperative");
        Self {
            device,
            queue,
            layout,
            norm,
            reference,
            staged,
            vectorized,
            down_reference,
            down_cooperative,
        }
    }

    fn run(
        &self,
        slot: u32,
        batch: u32,
        staged: bool,
        qk_norm: bool,
        down: bool,
        vector_case: Option<VectorCase>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dim = if let Some(case) = vector_case {
            case.cols
        } else if !down && slot <= 1 {
            257usize
        } else {
            256usize
        };
        let ffn = if down {
            if slot <= 1 {
                257usize
            } else {
                256usize
            }
        } else {
            8usize
        };
        let stride = 2 * ffn + dim + 64;
        let up_slot = vector_case.map_or(slot, |c| c.up_slot);
        let weight_bytes = |s| {
            if vector_case.is_some() {
                vector_weights(s, ffn, dim)
            } else {
                weights(
                    s,
                    if down { dim } else { ffn },
                    if down { ffn } else { dim },
                )
            }
        };
        let gate_bytes = weight_bytes(slot);
        let up_bytes = weight_bytes(up_slot);
        let base_words = u32::from(vector_case.is_some());
        let mut blob = vec![
            0u8;
            256 + base_words as usize * 4
                + usize::from(vector_case.is_some_and(|c| c.misaligned)) * 2
        ];
        let gate = blob.len();
        blob.extend_from_slice(&gate_bytes);
        let up = blob.len();
        blob.extend_from_slice(&up_bytes);
        blob.resize(blob.len().next_multiple_of(4), 0);
        let offsets = LayerOffsets {
            ffn_gate: (gate / 2) as u32 - base_words * 2,
            ffn_up: (up / 2) as u32 - base_words * 2,
            ffn_down: (gate / 2) as u32,
            layer_idx: 1,
            ..LayerOffsets::zeroed()
        };
        let params = LayerParams {
            dim: dim as u32,
            ffn_dim: ffn as u32,
            temp_stride: stride as u32,
            rms_eps: 1e-5,
            ffn_kind_policy: 1,
            formula_ffn_gate: slot,
            formula_ffn_up: up_slot,
            attn_logit_softcap: if vector_case.is_some_and(|c| c.gelu) {
                1.0
            } else {
                0.0
            },
            blob_base_words: base_words,
            formula_ffn_down: slot,
            qk_norm_enabled: qk_norm as u32,
            chunk_words: (blob.len() / 4) as u32,
            ..LayerParams::zeroed()
        };
        let cache = CacheParams {
            batch_size: batch,
            ..CacheParams::zeroed()
        };
        let input: Vec<f32> = (0..batch as usize * dim)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.07)
            .collect();
        let slots = if qk_norm { 6 } else { 4 };
        let ffn_slot = if qk_norm { 3 } else { 1 };
        let mut norms = vec![0.25f32; 2 * slots * dim];
        let norm_start = (slots + ffn_slot) * dim;
        for j in 0..dim {
            norms[norm_start + j] = 0.8 + j as f32 * 0.001;
        }
        let mut temp = vec![f32::NAN; batch as usize * stride];
        if down {
            for t in 0..batch as usize {
                for j in 0..ffn {
                    temp[t * stride + j] = ((j * 7 + t * 3) % 23) as f32 * 0.013 - 0.14;
                    temp[t * stride + ffn + j] = ((j * 11 + t * 5) % 19) as f32 * 0.017 - 0.15;
                }
            }
        }
        let initial_temp = temp.clone();
        let buffer = |data: &[u8], uniform: bool| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data,
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
        let norm = buffer(bytemuck::cast_slice(&norms), false);
        let cp = buffer(bytemuck::bytes_of(&cache), true);
        let dummy: Vec<_> = (0..10).map(|_| buffer(&[0u8; 256], false)).collect();
        let buffers = [
            &blob, &act, &temp, &off, &par, &norm, &dummy[0], &dummy[1], &dummy[2], &cp, &dummy[3],
            &dummy[4], &dummy[5], &dummy[6], &dummy[7], &dummy[8], &dummy[9],
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
        let size = temp.size() + act.size();
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if !down {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.norm);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, batch, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            let pipeline = match (down, staged) {
                (true, true) => &self.down_cooperative,
                (true, false) => &self.down_reference,
                (false, true) if vector_case.is_some_and(|c| c.vectorized) => &self.vectorized,
                (false, true) => &self.staged,
                (false, false) => &self.reference,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = if down {
                if staged {
                    dim as u32
                } else {
                    (dim as u32).div_ceil(256)
                }
            } else {
                if staged {
                    ffn as u32 * 2
                } else {
                    (ffn as u32 * 2).div_ceil(256)
                }
            };
            let extra = u32::from(vector_case.is_some_and(|c| c.vectorized) && staged);
            pass.dispatch_workgroups(groups + extra, batch + extra, 1);
        }
        encoder.copy_buffer_to_buffer(&temp, 0, &rb, 0, temp.size());
        encoder.copy_buffer_to_buffer(&act, 0, &rb, temp.size(), act.size());
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let view = rb.slice(..).get_mapped_range();
        let values: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        if down {
            let output = &values[batch as usize * stride..];
            for t in 0..batch as usize {
                assert_eq!(
                    &values[t * stride..t * stride + 2 * ffn],
                    &initial_temp[t * stride..t * stride + 2 * ffn],
                    "FFN down changed gate/up input"
                );
                for row in 0..dim {
                    let stash = values[t * stride + 2 * ffn + row];
                    assert!(stash.is_finite());
                    assert!(
                        (output[t * dim + row] - input[t * dim + row] - stash).abs() < 3e-5,
                        "residual/stash mismatch"
                    );
                }
            }
            return (output.to_vec(), input, Vec::new());
        }
        assert_eq!(
            &values[batch as usize * stride..],
            &input,
            "projection changed residual stream"
        );
        let mut output = Vec::new();
        for t in 0..batch as usize {
            output.extend_from_slice(&values[t * stride..t * stride + 2 * ffn]);
            let x = &input[t * dim..(t + 1) * dim];
            let rms = (x.iter().map(|v| v * v).sum::<f32>() / dim as f32 + 1e-5)
                .sqrt()
                .recip();
            for j in 0..dim {
                let actual = values[t * stride + 2 * ffn + j];
                let expected = x[j] * rms * norms[norm_start + j];
                assert!(
                    (actual - expected).abs() < 3e-5,
                    "scratch overlap/stale norm: {actual} != {expected}"
                );
            }
        }
        if let Some(case) = vector_case {
            for t in 0..batch as usize {
                assert!(
                    values[t * stride + 2 * ffn + dim..(t + 1) * stride]
                        .iter()
                        .all(|x| x.is_nan()),
                    "projection wrote scratch guards"
                );
                if !case.vectorized || !staged {
                    continue;
                }
                let x = &input[t * dim..(t + 1) * dim];
                let rms = (x.iter().map(|v| v * v).sum::<f32>() / dim as f32 + 1e-5)
                    .sqrt()
                    .recip();
                for (is_gate, s, bytes) in [(true, slot, &gate_bytes), (false, up_slot, &up_bytes)]
                {
                    let formula = airframe_observe::quant_formula::QUANT_FORMULAS
                        .iter()
                        .find(|f| f.type_id == if s == 5 { 12 } else { 14 })
                        .unwrap();
                    for row in 0..ffn {
                        let dot = (0..dim)
                            .map(|j| {
                                let block = (row * dim + j) / 256 * formula.block_bytes;
                                let weight = (formula.dequant)(
                                    &bytes[block..block + formula.block_bytes],
                                    j % 256,
                                );
                                (x[j] * rms * norms[norm_start + j]) as f64 * weight as f64
                            })
                            .sum::<f64>() as f32;
                        let expected = if !is_gate {
                            dot
                        } else if case.gelu {
                            0.5 * dot
                                * (1.0 + (0.797_884_6 * (dot + 0.044_715 * dot.powi(3))).tanh())
                        } else {
                            dot / (1.0 + (-dot).exp())
                        };
                        let actual = output[t * 2 * ffn + usize::from(!is_gate) * ffn + row];
                        assert!(actual.is_finite() && (actual - expected).abs() <= 8e-4 + 5e-4 * expected.abs(), "CPU oracle slot={s} cols={dim} batch={batch} row={row}: {actual} != {expected}");
                    }
                }
            }
        }
        (output, input, norms[norm_start..norm_start + dim].to_vec())
    }
}

#[test]
#[ignore = "requires a real GPU; AIRFRAME_TEST_REQUIRE_METAL=1 makes this a strict Metal gate"]
fn staged_ffn_matches_reference_for_quant_types_batches_and_norm_layouts() {
    pollster::block_on(async {
        let h = Harness::new().await;
        for slot in 0..=7 {
            for batch in [1, 3] {
                for qk in [false, true] {
                    let (reference, _, _) = h.run(slot, batch, false, qk, false, None);
                    let (staged, input, norms) = h.run(slot, batch, true, qk, false, None);
                    for (i, (a, b)) in reference.iter().zip(&staged).enumerate() {
                        assert!(
                            a.is_finite()
                                && b.is_finite()
                                && (a - b).abs() <= 2e-4 + 2e-4 * a.abs(),
                            "slot={slot} batch={batch} qk={qk} output={i}: {a} != {b}"
                        );
                    }
                    if slot == 0 {
                        let dim = 257;
                        let bytes = weights(0, 8, dim);
                        let w: Vec<f32> = bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect();
                        for t in 0..batch as usize {
                            let x = &input[t * dim..(t + 1) * dim];
                            let rms = (x.iter().map(|v| v * v).sum::<f32>() / dim as f32 + 1e-5)
                                .sqrt()
                                .recip();
                            for row in 0..8 {
                                let dot = (0..dim)
                                    .map(|j| x[j] * rms * norms[j] * w[row * dim + j])
                                    .sum::<f32>();
                                let gate = dot / (1.0 + (-dot).exp());
                                assert!((staged[t * 16 + row] - gate).abs() < 3e-5);
                                assert!((staged[t * 16 + 8 + row] - dot).abs() < 3e-5);
                            }
                        }
                    }
                }
            }
        }
    });
}
#[test]
#[ignore = "requires a real GPU; use AIRFRAME_TEST_REQUIRE_METAL=1 for strict Metal execution"]
fn cooperative_down_matches_reference_cpu_and_residual_contract() {
    pollster::block_on(async {
        let h = Harness::new().await;
        for slot in 0..=7 {
            for batch in [1, 3, 8] {
                let (reference, _, _) = h.run(slot, batch, false, false, true, None);
                let (cooperative, input, _) = h.run(slot, batch, true, false, true, None);
                for (i, (a, b)) in reference.iter().zip(&cooperative).enumerate() {
                    assert!(
                        a.is_finite() && b.is_finite() && (a - b).abs() <= 3e-4 + 3e-4 * a.abs(),
                        "down slot={slot} batch={batch} idx={i}: {a} != {b}"
                    );
                }
                if slot == 0 {
                    let cols = 257;
                    let bytes = weights(0, 256, cols);
                    let w: Vec<f32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    for t in 0..batch as usize {
                        for row in 0..256 {
                            let dot = (0..cols)
                                .map(|j| {
                                    let gate = ((j * 7 + t * 3) % 23) as f32 * 0.013 - 0.14;
                                    let up = ((j * 11 + t * 5) % 19) as f32 * 0.017 - 0.15;
                                    gate * up * w[row * cols + j]
                                })
                                .sum::<f32>();
                            let expected = input[t * 256 + row] + dot;
                            assert!((cooperative[t * 256 + row] - expected).abs() < 3e-5);
                        }
                    }
                }
            }
        }
    });
}
#[test]
#[ignore = "requires a real GPU; AIRFRAME_TEST_REQUIRE_METAL=1 requires Metal"]
fn vectorized_ffn_matches_scalar_and_observe_oracle() {
    pollster::block_on(async {
        let h = Harness::new().await;
        let mut cases = 0;
        for (gate_slot, up_slot) in [(5, 5), (7, 7), (5, 7), (7, 5)] {
            for cols in [256, 512, 1280, 3072] {
                for batch in [1, 3, 8] {
                    for qk in [false, true] {
                        for gelu in [false, true] {
                            for misaligned in [false, true] {
                                let case = VectorCase {
                                    cols,
                                    up_slot,
                                    misaligned,
                                    gelu,
                                    vectorized: false,
                                };
                                let (reference, _, _) =
                                    h.run(gate_slot, batch, false, qk, false, Some(case));
                                let (staged, _, _) =
                                    h.run(gate_slot, batch, true, qk, false, Some(case));
                                let (vector, _, _) = h.run(
                                    gate_slot,
                                    batch,
                                    true,
                                    qk,
                                    false,
                                    Some(VectorCase {
                                        vectorized: true,
                                        ..case
                                    }),
                                );
                                for (i, (&value, &expected)) in
                                    vector.iter().zip(&reference).enumerate()
                                {
                                    assert!(value.is_finite() && expected.is_finite() && (value-expected).abs() <= 8e-4 + 5e-4 * expected.abs(), "reference parity gate={gate_slot} up={up_slot} cols={cols} batch={batch} gelu={gelu} unaligned={misaligned} index={i}: {value} != {expected}");
                                    assert!(
                                        (value - staged[i]).abs() <= 8e-4 + 5e-4 * staged[i].abs(),
                                        "staged parity index={i}: {value} != {}",
                                        staged[i]
                                    );
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 384);
    });
}
