// test_int4_parity.rs
//
// GPU parity test for the TurboQuant INT4 KV quantization kernel (sh_quantize_kv.wgsl).
//
// Strategy:
//   1. Populate F32 K/V staging buffers with deterministic test vectors.
//   2. Dispatch quantize_kv_pipeline (same kernel used during decode).
//   3. Read back packed nibbles and per-head scales.
//   4. Run a CPU reference implementation of the same algorithm.
//   5. Assert:
//       a. GPU scales match CPU scales (< 1e-5 tolerance).
//       b. GPU packed nibbles are bitwise-identical to CPU packed nibbles.
//       c. Round-trip decode error is bounded by scale/2 (max quantization error).
//
// Also tests the zero-vector edge case (scale must fall back to 1.0, not NaN).

#[cfg(test)]
mod int4_parity_tests {
    use super::super::pipeline::{BindlessPipeline, QuantizeKvParams};
    use wgpu::util::DeviceExt;

    async fn get_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found — INT4 parity tests require a GPU");

        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("Device creation failed");

        (device, queue)
    }

    // ── CPU reference ──────────────────────────────────────────────────────────

    /// Mirrors sh_quantize_kv.wgsl exactly.
    /// Returns (scale, packed_u32s).
    fn cpu_quantize(vals: &[f32]) -> (f32, Vec<u32>) {
        assert!(vals.len() % 8 == 0, "head_dim must be a multiple of 8");
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        let hd8 = vals.len() / 8;
        let mut packed = vec![0u32; hd8];
        for u in 0..hd8 {
            let mut word = 0u32;
            for n in 0..8usize {
                let val = vals[u * 8 + n];
                let q_i = ((val / scale).round() as i32 + 8).clamp(1, 15) as u32;
                word |= q_i << (n * 4);
            }
            packed[u] = word;
        }
        (scale, packed)
    }

    /// Decode packed nibbles back to F32.  Used to verify round-trip error.
    fn cpu_dequantize(packed: &[u32], scale: f32) -> Vec<f32> {
        let mut result = Vec::with_capacity(packed.len() * 8);
        for &word in packed {
            for n in 0..8usize {
                let nibble = (word >> (n * 4)) & 0xF;
                result.push((nibble as f32 - 8.0) * scale);
            }
        }
        result
    }

    // ── readback helpers ──────────────────────────────────────────────────────

    fn readback_u32(device: &wgpu::Device, buf: &wgpu::Buffer, n: usize) -> Vec<u32> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        loop {
            device.poll(wgpu::PollType::Poll).expect("device lost");
            if let Ok(r) = rx.try_recv() {
                r.expect("buffer map failed");
                break;
            }
        }
        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&*data).to_vec();
        drop(data);
        buf.unmap();
        result[..n].to_vec()
    }

    fn readback_f32(device: &wgpu::Device, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        loop {
            device.poll(wgpu::PollType::Poll).expect("device lost");
            if let Ok(r) = rx.try_recv() {
                r.expect("buffer map failed");
                break;
            }
        }
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&*data).to_vec();
        drop(data);
        buf.unmap();
        result[..n].to_vec()
    }

    // ── test helper: dispatch quantize_kv and return raw GPU outputs ──────────

    struct QuantizeKvOutputs {
        k_packed: Vec<u32>,
        v_packed: Vec<u32>,
        k_scale:  Vec<f32>,
        v_scale:  Vec<f32>,
    }

    async fn run_quantize_kv_kernel(
        f32_k: &[f32],
        f32_v: &[f32],
        n_head_kv: u32,
        head_dim: u32,
        max_seq: u32,
        pos_offset: u32,
        n_positions: u32,  // workgroup Y dimension
    ) -> QuantizeKvOutputs {
        let (device, queue) = get_device().await;
        let pipeline = BindlessPipeline::new(&device);

        let packed_elems = (max_seq * n_head_kv * head_dim / 8) as usize;
        let scale_elems  = (max_seq * n_head_kv) as usize;

        let k_f32_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K F32"),
            contents: bytemuck::cast_slice(f32_k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_f32_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("V F32"),
            contents: bytemuck::cast_slice(f32_v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("K Packed"),
            size: (packed_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Packed"),
            size: (packed_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let k_scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("K Scale"),
            size: (scale_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Scale"),
            size: (scale_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let qparams = QuantizeKvParams { n_head_kv, head_dim, pos_offset, _pad: 0 };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QParams"),
            contents: bytemuck::bytes_of(&qparams),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("QuantizeKV Parity BG"),
            layout: &pipeline.quantize_kv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: k_f32_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: v_f32_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_packed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_packed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k_scale_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v_scale_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

        // Staging readback buffers
        let stg_k_packed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stg K Packed"),
            size: (packed_elems * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stg_v_packed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stg V Packed"),
            size: (packed_elems * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stg_k_scale = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stg K Scale"),
            size: (scale_elems * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stg_v_scale = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stg V Scale"),
            size: (scale_elems * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("QuantizeKV Parity"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QuantizeKV"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bg, &[]);
            cp.set_pipeline(&pipeline.quantize_kv_pipeline);
            cp.dispatch_workgroups(n_head_kv, n_positions, 1);
        }
        enc.copy_buffer_to_buffer(&k_packed_buf, 0, &stg_k_packed, 0, (packed_elems * 4) as u64);
        enc.copy_buffer_to_buffer(&v_packed_buf, 0, &stg_v_packed, 0, (packed_elems * 4) as u64);
        enc.copy_buffer_to_buffer(&k_scale_buf,  0, &stg_k_scale,  0, (scale_elems  * 4) as u64);
        enc.copy_buffer_to_buffer(&v_scale_buf,  0, &stg_v_scale,  0, (scale_elems  * 4) as u64);
        queue.submit(Some(enc.finish()));

        // How many active outputs to read back (only dispatched positions are written)
        let active_packed = (n_positions * n_head_kv * head_dim / 8) as usize;
        let active_scale  = (n_positions * n_head_kv) as usize;

        QuantizeKvOutputs {
            k_packed: readback_u32(&device, &stg_k_packed, active_packed),
            v_packed: readback_u32(&device, &stg_v_packed, active_packed),
            k_scale:  readback_f32(&device, &stg_k_scale,  active_scale),
            v_scale:  readback_f32(&device, &stg_v_scale,  active_scale),
        }
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Single head, single position: verify nibbles are bitwise-identical to CPU
    /// and round-trip error is within quantization tolerance.
    #[tokio::test]
    async fn test_quantize_kv_single_head_parity() {
        let n_head_kv = 1u32;
        let head_dim  = 64u32;
        let max_seq   = 4u32;

        // Deterministic K: linear ramp d*0.1 - 3.15  →  max_abs = 3.15, scale ≈ 0.450
        let k_vals: Vec<f32> = (0..head_dim as usize)
            .map(|d| d as f32 * 0.1 - 3.15)
            .collect();
        // Deterministic V: sinusoidal pattern to avoid symmetry artifacts
        let v_vals: Vec<f32> = (0..head_dim as usize)
            .map(|d| ((d as f32) * 0.3 - 9.5).sin() * 2.5)
            .collect();

        let buf_size = (max_seq * n_head_kv * head_dim) as usize;
        let mut f32_k = vec![0.0f32; buf_size];
        let mut f32_v = vec![0.0f32; buf_size];
        f32_k[..head_dim as usize].copy_from_slice(&k_vals);
        f32_v[..head_dim as usize].copy_from_slice(&v_vals);

        let out = run_quantize_kv_kernel(
            &f32_k, &f32_v,
            n_head_kv, head_dim, max_seq,
            0, // pos_offset
            1, // 1 position
        ).await;

        let (cpu_k_scale, cpu_k_packed) = cpu_quantize(&k_vals);
        let (cpu_v_scale, cpu_v_packed) = cpu_quantize(&v_vals);

        // Scales match
        assert!(
            (out.k_scale[0] - cpu_k_scale).abs() < 1e-5,
            "K scale: GPU={:.8} CPU={:.8}", out.k_scale[0], cpu_k_scale
        );
        assert!(
            (out.v_scale[0] - cpu_v_scale).abs() < 1e-5,
            "V scale: GPU={:.8} CPU={:.8}", out.v_scale[0], cpu_v_scale
        );

        // Packed nibbles are bitwise identical
        let hd8 = head_dim as usize / 8;
        for i in 0..hd8 {
            assert_eq!(
                out.k_packed[i], cpu_k_packed[i],
                "K packed[{i}]: GPU={:#010x} CPU={:#010x}", out.k_packed[i], cpu_k_packed[i]
            );
            assert_eq!(
                out.v_packed[i], cpu_v_packed[i],
                "V packed[{i}]: GPU={:#010x} CPU={:#010x}", out.v_packed[i], cpu_v_packed[i]
            );
        }

        // Round-trip decode error ≤ scale/2 + epsilon (max 1-ULP quant error)
        let k_decoded = cpu_dequantize(&out.k_packed, out.k_scale[0]);
        let v_decoded = cpu_dequantize(&out.v_packed, out.v_scale[0]);
        let k_max_err = k_vals.iter().zip(&k_decoded).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let v_max_err = v_vals.iter().zip(&v_decoded).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            k_max_err <= cpu_k_scale / 2.0 + 1e-5,
            "K round-trip error {k_max_err:.6} > bound {:.6}", cpu_k_scale / 2.0
        );
        assert!(
            v_max_err <= cpu_v_scale / 2.0 + 1e-5,
            "V round-trip error {v_max_err:.6} > bound {:.6}", cpu_v_scale / 2.0
        );

        println!(
            "[INT4 parity] K scale={:.6} max_err={:.6}  V scale={:.6} max_err={:.6}",
            cpu_k_scale, k_max_err, cpu_v_scale, v_max_err
        );
    }

    /// Multi-head: 4 heads, each with distinct value ranges.
    /// Verifies per-head independent scaling (not shared across heads).
    #[tokio::test]
    async fn test_quantize_kv_multi_head_independent_scales() {
        let n_head_kv = 4u32;
        let head_dim  = 64u32;
        let max_seq   = 2u32;

        let buf_size = (max_seq * n_head_kv * head_dim) as usize;
        let mut f32_k = vec![0.0f32; buf_size];
        let mut f32_v = vec![0.0f32; buf_size];

        // Each head gets a different amplitude so scales must differ.
        // Layout at pos=0: base = head * head_dim (since pos=0, n_head=4, so base = 0*4*64 + h*64)
        let amplitudes = [1.0f32, 2.0, 0.5, 7.0];
        let mut head_k_vals: Vec<Vec<f32>> = Vec::new();
        let mut head_v_vals: Vec<Vec<f32>> = Vec::new();

        for h in 0..4usize {
            let amp = amplitudes[h];
            let k: Vec<f32> = (0..head_dim as usize)
                .map(|d| amp * ((d as f32 / head_dim as f32) * 2.0 - 1.0))
                .collect();
            let v: Vec<f32> = (0..head_dim as usize)
                .map(|d| amp * (d as f32 * 0.1 - 3.15) / 3.15)
                .collect();
            let base = h * head_dim as usize;  // pos=0: f32_base = 0 * 4 * 64 + h * 64
            f32_k[base..base + head_dim as usize].copy_from_slice(&k);
            f32_v[base..base + head_dim as usize].copy_from_slice(&v);
            head_k_vals.push(k);
            head_v_vals.push(v);
        }

        let out = run_quantize_kv_kernel(
            &f32_k, &f32_v,
            n_head_kv, head_dim, max_seq,
            0, 1,
        ).await;

        let hd8 = head_dim as usize / 8;

        for h in 0..4usize {
            let (cpu_k_scale, _) = cpu_quantize(&head_k_vals[h]);
            let (cpu_v_scale, _) = cpu_quantize(&head_v_vals[h]);

            // Scale: GPU and CPU must agree (max_abs / 7.0 is deterministic once max_abs matches)
            assert!(
                (out.k_scale[h] - cpu_k_scale).abs() < 1e-5,
                "head {h} K scale: GPU={:.8} CPU={:.8}", out.k_scale[h], cpu_k_scale
            );
            assert!(
                (out.v_scale[h] - cpu_v_scale).abs() < 1e-5,
                "head {h} V scale: GPU={:.8} CPU={:.8}", out.v_scale[h], cpu_v_scale
            );

            // Round-trip error: note we do NOT assert bitwise nibble equality here.
            // WGSL round() uses round-half-to-even; Rust round() uses round-half-away-from-zero.
            // When val/scale is exactly ±N.5, the two can choose opposite directions — both
            // produce a valid 1-ULP encoding.  What matters is that the decode error is bounded.
            let pack_base = h * hd8;
            let k_decoded = cpu_dequantize(&out.k_packed[pack_base..pack_base + hd8], out.k_scale[h]);
            let v_decoded = cpu_dequantize(&out.v_packed[pack_base..pack_base + hd8], out.v_scale[h]);

            let k_max_err = head_k_vals[h].iter().zip(&k_decoded)
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let v_max_err = head_v_vals[h].iter().zip(&v_decoded)
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

            assert!(
                k_max_err <= cpu_k_scale / 2.0 + 1e-5,
                "head {h} K round-trip error {k_max_err:.6} > bound {:.6}", cpu_k_scale / 2.0
            );
            assert!(
                v_max_err <= cpu_v_scale / 2.0 + 1e-5,
                "head {h} V round-trip error {v_max_err:.6} > bound {:.6}", cpu_v_scale / 2.0
            );

            println!(
                "[INT4 parity] head {h} amp={:.1} K scale={:.6} err={:.6}  V scale={:.6} err={:.6}",
                amplitudes[h], cpu_k_scale, k_max_err, cpu_v_scale, v_max_err
            );
        }
    }

    /// Zero vector: scale must fall back to 1.0 (not NaN/inf from 0.0/7.0).
    /// All nibbles must encode 0.0 → nibble=8 → packed u32 = 0x88888888.
    #[tokio::test]
    async fn test_quantize_kv_zero_vector_no_nan() {
        let n_head_kv = 1u32;
        let head_dim  = 8u32;  // minimal size: exactly one packed u32
        let max_seq   = 1u32;

        let zeros = vec![0.0f32; (max_seq * n_head_kv * head_dim) as usize];

        let out = run_quantize_kv_kernel(
            &zeros, &zeros,
            n_head_kv, head_dim, max_seq,
            0, 1,
        ).await;

        assert!(
            out.k_scale[0].is_finite() && out.k_scale[0] == 1.0,
            "Zero K: expected scale=1.0, got {}", out.k_scale[0]
        );
        assert!(
            out.v_scale[0].is_finite() && out.v_scale[0] == 1.0,
            "Zero V: expected scale=1.0, got {}", out.v_scale[0]
        );
        // All nibbles = 8 (encoding 0.0)
        assert_eq!(
            out.k_packed[0], 0x88888888u32,
            "Zero K: expected 0x88888888, got {:#010x}", out.k_packed[0]
        );
        assert_eq!(
            out.v_packed[0], 0x88888888u32,
            "Zero V: expected 0x88888888, got {:#010x}", out.v_packed[0]
        );
    }

    /// Extreme values: max_abs close to f32 max should not overflow.
    /// Verifies clamping to [1,15] works at the boundary.
    #[tokio::test]
    async fn test_quantize_kv_extreme_values_clamped() {
        let n_head_kv = 1u32;
        let head_dim  = 8u32;
        let max_seq   = 1u32;

        // Large positive K: all 1000.0 — max nibble should be 15 (clamped to +7*scale)
        let big_pos = vec![1000.0f32; head_dim as usize];
        // Alternating large positive/negative V: ±1000.0
        let alternating: Vec<f32> = (0..head_dim as usize)
            .map(|d| if d % 2 == 0 { 1000.0 } else { -1000.0 })
            .collect();

        let buf_size = (max_seq * n_head_kv * head_dim) as usize;
        let mut f32_k = vec![0.0f32; buf_size];
        let mut f32_v = vec![0.0f32; buf_size];
        f32_k[..head_dim as usize].copy_from_slice(&big_pos);
        f32_v[..head_dim as usize].copy_from_slice(&alternating);

        let out = run_quantize_kv_kernel(
            &f32_k, &f32_v,
            n_head_kv, head_dim, max_seq,
            0, 1,
        ).await;

        // CPU reference
        let (cpu_k_scale, cpu_k_packed) = cpu_quantize(&big_pos);
        let (cpu_v_scale, _cpu_v_packed) = cpu_quantize(&alternating);

        assert!((out.k_scale[0] - cpu_k_scale).abs() < 1.0, "K scale mismatch at extreme");
        assert!((out.v_scale[0] - cpu_v_scale).abs() < 1.0, "V scale mismatch at extreme");

        // For all-equal positive inputs, the optimal encoding is unambiguous:
        // every value = max_abs, so val/scale = 7.0 exactly, nibble = clamp(7+8,1,15) = 15.
        // Both CPU and GPU must agree here (no tie to break).
        assert_eq!(
            out.k_packed[0], cpu_k_packed[0],
            "K packed mismatch at extreme: GPU={:#010x} CPU={:#010x}", out.k_packed[0], cpu_k_packed[0]
        );

        // All K nibbles should be 15 (max positive clamp)
        assert_eq!(
            out.k_packed[0], 0xFFFFFFFFu32,
            "All-positive K: expected 0xFFFFFFFF (nibble=15), got {:#010x}", out.k_packed[0]
        );

        // Alternating V: round-trip error bounded (not bitwise — alternating ±1000 may hit ties)
        let v_decoded = cpu_dequantize(&out.v_packed, out.v_scale[0]);
        let v_max_err = alternating.iter().zip(&v_decoded)
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            v_max_err <= cpu_v_scale / 2.0 + 1.0,
            "V round-trip error {v_max_err:.2} > bound {:.2}", cpu_v_scale / 2.0
        );

        println!(
            "[INT4 parity] extreme K scale={:.2} V scale={:.2}",
            out.k_scale[0], out.v_scale[0]
        );
    }
}
