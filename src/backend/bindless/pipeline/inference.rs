//! Full-model inference dispatch methods for `BindlessPipeline`.
// TODO: break run_full_model_prefill_chunked_with_cache_state into a separate chunking helper once prefill chunking is the default path.
use super::super::loader::BindlessModel;
use super::*;
use crate::backend::tdr::TdrScheduler;
use crate::core::routing::ModelRoutePlan;
use wgpu::util::DeviceExt;

/// Result type for model inference returning three activation vectors
type InferenceResult = Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>;

impl BindlessPipeline {
    pub fn run_full_model(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        spec: &ModelSpec,
    ) -> Vec<f32> {
        self.run_full_model_with_cache(
            device,
            queue,
            model,
            input_embd,
            head_weights_override,
            0,
            1,
            spec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_full_model_with_cache(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        seq_len: u32,
        spec: &ModelSpec,
    ) -> Vec<f32> {
        self.run_full_model_with_cache_state(
            device,
            queue,
            model,
            input_embd,
            head_weights_override,
            current_pos,
            seq_len,
            None,
            spec,
        )
        .expect("GPU forward pass failed")
        .2
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_full_model_prefill_chunked_with_cache_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
        spec: &ModelSpec,
        chunk_tokens: u32,
    ) -> InferenceResult {
        let dim = spec.n_embd;
        assert!(dim > 0, "spec.n_embd must be > 0");
        assert!(
            input_embd.len().is_multiple_of(dim),
            "input_embd must align to token rows"
        );
        assert!(chunk_tokens > 0, "chunk_tokens must be > 0");

        let trace_chunks = std::env::var("AIRFRAME_TRACE_PREFILL_CHUNKS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let total_tokens = input_embd.len() / dim;
        if total_tokens == 0 {
            return self.run_full_model_with_cache_state(
                device,
                queue,
                model,
                input_embd,
                head_weights_override,
                current_pos,
                current_pos,
                kv_state,
                spec,
            );
            // ^ Return type is now Result, so this propagates Ok or Err correctly.
        }

        let chunk_rows = chunk_tokens as usize;
        let mut processed_tokens = 0u32;
        let mut last_result = None;

        for (chunk_idx, chunk) in input_embd.chunks(chunk_rows * dim).enumerate() {
            let chunk_token_count = (chunk.len() / dim) as u32;
            let chunk_current_pos = current_pos + processed_tokens;
            let chunk_seq_len = chunk_current_pos + chunk_token_count;

            if trace_chunks {
                eprintln!(
                    "[PREFILL] chunk={} tokens={} current_pos={} seq_len={}",
                    chunk_idx, chunk_token_count, chunk_current_pos, chunk_seq_len
                );
            }

            last_result = Some(self.run_full_model_with_cache_state(
                device,
                queue,
                model,
                chunk,
                head_weights_override,
                chunk_current_pos,
                chunk_seq_len,
                kv_state,
                spec,
            )?);

            if trace_chunks {
                eprintln!("[PREFILL] chunk={} complete", chunk_idx);
            }

            processed_tokens += chunk_token_count;
        }

        last_result.ok_or_else(|| "chunked prefill produced no chunks".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_full_model_with_cache_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        seq_len: u32,
        kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
        spec: &ModelSpec,
    ) -> InferenceResult {
        // Derive all constants from ModelSpec
        let dim = spec.n_embd as u32;
        let layer_count = spec.n_layer;
        let vocab_size = spec.n_vocab as u32;
        let ffn_dim = spec.ff_dim as u32;
        let temp_stride = spec.temp_buffer_size as u32;

        // Phase 4a escape hatch: set AIRFRAME_PINGPONG_ACTIVATION=1 to enable ping-pong.
        // Default off until Steps 3-4 are verified.
        let use_pingpong = std::env::var("AIRFRAME_PINGPONG_ACTIVATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let weight_quant_type = model
            .metadata
            .get_tensor_type("blk.0.attn_q.weight")
            .unwrap_or(2);
        let qt_v = model
            .metadata
            .get_tensor_type("blk.0.attn_v.weight")
            .unwrap_or(weight_quant_type);
        let qt_ffn_down = model
            .metadata
            .get_tensor_type("blk.0.ffn_down.weight")
            .unwrap_or(weight_quant_type);
        let packed_quant_type = weight_quant_type | (qt_v << 8) | (qt_ffn_down << 16);

        // Q4_K uses different shader pipelines (type 12)
        let use_q4k_pipeline = weight_quant_type == 12;
        let _ = packed_quant_type; // per-layer quant is computed in the loop below

        // 1. Buffers
        let batch_size = (input_embd.len() as u32) / dim;
        // A. Activation (Residual Stream) - Init with Embeddings
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation A"),
            contents: bytemuck::cast_slice(input_embd),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // A2. Activation B (Ping-Pong partner).
        // Only created when ping-pong is active to avoid wasting VRAM on the old path.
        // When use_pingpong=false, activation_buffer_b is a dummy zero-byte buffer
        // that is never actually bound or used.
        let activation_buffer_b = if use_pingpong {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Activation B (Ping-Pong)"),
                contents: bytemuck::cast_slice(input_embd), // same initial residual
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        } else {
            // Dummy 1-byte buffer — never bound, just satisfies the type system.
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Activation B (disabled)"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        // B. Temp Buffer
        // Needs to hold FFN Gate + Up + scratch space per token
        let temp_buffer_size = batch_size as u64 * temp_stride as u64 * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // C. Layer Params (computed per-layer below; placeholder base for struct copy)
        // NOTE: quant_type varies per layer in mixed-quant models (e.g. Q4_K_M).
        //       Per-layer params buffers are created inside the layer loop.
        let use_route_v2_layer_params = std::env::var("SHIMMY_ROUTE_V2_LAYER_PARAMS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let route_plan = use_route_v2_layer_params.then(|| {
            ModelRoutePlan::from_spec_and_tensors(spec, |name| {
                model.metadata.tensor_offsets.contains_key(name)
            })
        });
        let ffn_kind_policy = route_plan
            .as_ref()
            .map(ModelRoutePlan::ffn_kind_policy_code)
            .unwrap_or(ModelRoutePlan::FFN_KIND_INFER);
        let qkv_layout_policy = route_plan
            .as_ref()
            .map(ModelRoutePlan::qkv_layout_policy_code)
            .unwrap_or(ModelRoutePlan::QKV_LAYOUT_INFER);

        let params_base = LayerParams {
            dim,
            head_count: spec.n_head as u32,
            head_count_kv: spec.n_head_kv as u32,
            head_dim: spec.head_dim as u32,
            rope_dim: spec.rope_dim as u32,
            rms_eps: spec.rms_eps,
            ffn_dim,
            temp_stride,
            quant_type: 0, // overridden per-layer below
            attn_logit_softcap: spec.attn_logit_softcap,
            post_norm_enabled: if spec.arch_string() == "gemma2" { 1 } else { 0 },
            qk_norm_enabled: if spec.has_qk_norm { 1 } else { 0 },
            layer_norm_enabled: if spec.uses_layer_norm() { 1 } else { 0 },
            ffn_kind_policy,
            qkv_layout_policy,
            batch_offset: 0,
            batch_count: batch_size,
            q_weight_k: 0,
            k_weight_k: 0,
        };

        // Adaptive QKV micro-batch chunk size.
        // Reads SHIMMY_PREFILL_CHUNK; defaults to 1 (safest — one token per dispatch).
        // Users with fast GPUs can raise this; Q4_K_M on RTX 3060 is safe at 1.
        // A future TIMESTAMP_QUERY calibration pass will auto-tune this at model load.
        let qkv_chunk: u32 = std::env::var("SHIMMY_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .clamp(1, batch_size.max(1));

        // ── TDR Scheduler ────────────────────────────────────────────────────
        // TdrScheduler owns the command encoder and tracks accumulated GPU time.
        // It replaces the scattered tdr_submit_poll! / tdr_yield_if_needed! macros
        // with clean, testable methods. Platform-aware budget (1400ms Windows,
        // 30000ms Linux/macOS). Override with SHIMMY_TDR_BUDGET_MS.
        //
        // Patent Notice: FSE + D0 Saturation Fabric scheduling.
        // Pending patent by Michael A. Kuykendall. All rights reserved.
        let mut tdr = TdrScheduler::new(device, queue, "Full Model");
        let tdr_log = std::env::var("AIRFRAME_LOG_TDR_POLLS")
            .map(|v| v == "1")
            .unwrap_or(false);

        // D. Output Logits
        // Only computed for the LAST token in the sequence (usually).
        // If we want all logits, we'd need batch_size * vocab_size.
        // For now, let's stick to last token logic for compatibility.
        let logits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Logits"),
            size: (vocab_size as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let l21_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L21 Final Norm Output"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pre_norm_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pre-Final-Norm Output"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // E. KV Cache
        // Shader cache indexing is layer-local: [pos, kv_head, head_dim] with no layer axis.
        // Therefore full-model loop must bind a distinct K/V buffer per layer.
        let kv_size_per_buffer = spec.kv_cache_size_per_layer as u64;
        let local_kv_storage_per_layer = if kv_state.is_none() {
            let mut bufs = Vec::with_capacity(layer_count);
            for i in 0..layer_count {
                let kv_buffer_k = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV Cache K L{}", i)),
                    size: kv_size_per_buffer,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let kv_buffer_v = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV Cache V L{}", i)),
                    size: kv_size_per_buffer,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                bufs.push((kv_buffer_k, kv_buffer_v));
            }
            Some(bufs)
        } else {
            None
        };

        // F. Cache Params
        let cache_params = CacheParams {
            current_pos,
            seq_len, // Total cached positions (including this batch)
            max_seq_len: spec.n_ctx as u32,
            batch_size,
            logical_pos_base: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 2. Prepare Layers (Offsets & BindGroups)
        // For ping-pong: two bind group arrays — one with activation_buffer (A) at binding 1,
        // one with activation_buffer_b (B). Layer i uses set_a when i%2==0, set_b when i%2==1.
        // For the old path (use_pingpong=false): only set_a is used; set_b is empty.
        let mut layer_bind_groups = Vec::new();   // set A: activation_buffer at binding 1
        let mut layer_bind_groups_b = Vec::new(); // set B: activation_buffer_b at binding 1
        let mut _offset_buffers = Vec::new(); // Keep alive
        let mut _params_buffers: Vec<wgpu::Buffer> = Vec::new(); // Keep alive
        let mut _layer_params: Vec<LayerParams> = Vec::new(); // Per-layer params for QKV chunking

        for i in 0..layer_count {
            let compiled = &model.metadata.compiled_layers[i];
            let mut layer_params_i = LayerParams {
                quant_type: compiled.quant_type_packed,
                ..params_base
            };
            if spec.arch_string() == "qwen3" {
                let packed_k = 2 * dim;
                layer_params_i.q_weight_k = packed_k;
                layer_params_i.k_weight_k = packed_k;
            }
            let params_buffer_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Params", i)),
                contents: bytemuck::bytes_of(&layer_params_i),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let (kv_buffer_k_ref, kv_buffer_v_ref): (&wgpu::Buffer, &wgpu::Buffer) =
                if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                    (&kv_k_layers[i], &kv_v_layers[i])
                } else {
                    let (local_k, local_v) = &local_kv_storage_per_layer
                        .as_ref()
                        .expect("local KV storage missing")[i];
                    (local_k, local_v)
                };

            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Offsets", i)),
                contents: bytemuck::bytes_of(&compiled.offsets),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Build bind group with a specific activation buffer at binding 1.
            // This closure lets us create both A and B sets without duplicating all entries.
            let make_bg = |act_buf: &wgpu::Buffer, label: &str| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.layer_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: model.blob_binding_0(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: act_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: temp_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buffer_i.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .norm_bank_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .rope_cache_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: kv_buffer_k_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: kv_buffer_v_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: cache_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: model.blob_binding_1(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: model.blob_binding_2(),
                        },
                    ],
                })
            };

            // Set A: activation_buffer (always built — used by old path + even layers in pingpong)
            let bg_a = make_bg(&activation_buffer, &format!("Layer {} BG-A", i));

            // Set B: activation_buffer_b (only built when pingpong is active)
            if use_pingpong {
                let bg_b = make_bg(&activation_buffer_b, &format!("Layer {} BG-B", i));
                layer_bind_groups_b.push(bg_b);
            }

            _offset_buffers.push(buf);
            _params_buffers.push(params_buffer_i);
            _layer_params.push(layer_params_i);
            layer_bind_groups.push(bg_a);
        }

        // 3. Final Norm
        let norm_weight = model
            .metadata
            .get_tensor_offset("output_norm.weight")
            .expect("output_norm missing");
        let norm_bias = model
            .metadata
            .get_tensor_offset("output_norm.bias")
            .map(|off| (off / 4) as u32)
            .unwrap_or(0);
        let norm_params = RMSNormParams {
            count: dim,
            weights_offset: (norm_weight / 4) as u32, // word index (byte_offset / 4); safe: 4.4GB/4 = 1.1B < u32::MAX
            bias_offset: norm_bias,
            eps: spec.rms_eps,
            norm_type: if matches!(spec.arch, crate::core::spec::ModelArch::Phi) {
                1
            } else {
                0
            },
        };
        let norm_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Final Norm Params"),
            contents: bytemuck::bytes_of(&norm_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Offset for the LAST token in the batch
        let last_token_offset = (batch_size as u64 - 1u64) * (dim as u64) * 4u64;
        let token_size = std::num::NonZeroU64::new((dim as u64) * 4u64).unwrap();

        let norm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Final Norm BG"),
            layout: &self.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &activation_buffer,
                        offset: last_token_offset,
                        size: Some(token_size),
                    }),
                },
                // Use temp_buffer for output to avoid read-write aliasing on activation_buffer
                // Output to the BEGINNING of temp_buffer (reusing space)
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &temp_buffer,
                        offset: 0,
                        size: Some(token_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: norm_param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: model.blob_binding_1(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: model.blob_binding_2(),
                },
            ],
        });

        // 4. Output Head
        // When head_weights_override = Some(buf): diagnostic F32 matmul override path.
        // When head_weights_override = None (default): blob-based quantized head — reads
        //   output.weight directly from the GGUF blob, no dequant buffer required.
        let head_tensor_name = if model.metadata.get_tensor_type("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let head_weight_off = (model
            .metadata
            .get_tensor_offset(head_tensor_name)
            .unwrap_or(0)
            / 4) as u32;
        let head_quant_type = model
            .metadata
            .get_tensor_type(head_tensor_name)
            .unwrap_or(2);

        enum HeadBg {
            F32(wgpu::BindGroup),
            Blob(wgpu::BindGroup),
        }

        let head_bg = if let Some(override_buf) = head_weights_override {
            // --- Diagnostic F32 override (kept for shimmy_eval comparison tests) ---
            let head_params = MatMulParams {
                n: vocab_size,
                k: dim,
                weights_offset: head_weight_off,
                padding: 0,
            };
            let head_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Head Params F32"),
                contents: bytemuck::bytes_of(&head_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            HeadBg::F32(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG F32"),
                layout: &self.matmul_f32_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: override_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &temp_buffer,
                            offset: 0,
                            size: Some(token_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: logits_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: head_param_buf.as_entire_binding(),
                    },
                ],
            }))
        } else {
            // --- Default blob-based path: output.weight stays quantized on GPU ---
            let head_params = HeadBlobParams {
                vocab_size,
                dim,
                weight_off: head_weight_off,
                quant_type: head_quant_type,
                softcap: spec.final_logit_softcap,
                _pad: 0,
            };
            let head_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Head Params Blob"),
                contents: bytemuck::bytes_of(&head_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            HeadBg::Blob(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG Blob"),
                layout: &self.lm_head_blob_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.blob_binding_0(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &temp_buffer,
                            offset: 0,
                            size: Some(token_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: logits_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: head_param_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: model.blob_binding_1(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: model.blob_binding_2(),
                    },
                ],
            }))
        };

        // 5. Command Encoding — managed by TdrScheduler (see tdr above).
        // The initial encoder was created by TdrScheduler::new().

        let wg_dim = dim.div_ceil(256);
        let ffn_total = ffn_dim * 2; // Gate + Up need this many threads
        let wg_ffn = ffn_total.div_ceil(256); // Ceil div by workgroup size (256)
        let wg_norm = dim.div_ceil(256);
        // sh_head_blob.wgsl uses @workgroup_size(64, 1, 1); matmul_f32 uses @workgroup_size(256).
        let wg_head_blob = vocab_size.div_ceil(64);
        let wg_head_f32 = vocab_size.div_ceil(256);

        // QKV Dispatch Calculation
        let q_len = params_base.head_count * params_base.head_dim;
        let kv_len = params_base.head_count_kv * params_base.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = total_qkv.div_ceil(256);
        let wg_qknorm = (q_len + kv_len).div_ceil(256); // must cover all Q+K elements, not just head_dim
        let trace_prefill_layers = std::env::var("AIRFRAME_TRACE_PREFILL_LAYERS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let disable_output_norm = std::env::var("SHIMMY_DISABLE_OUTPUT_NORM")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // Loop Layers
        for (i, bg) in layer_bind_groups.iter().enumerate() {
            let params_layer = _layer_params[i]; // per-layer quant_type + base fields
            if trace_prefill_layers {
                eprintln!(
                    "[PREFILL-LAYER] start layer={} batch_size={} current_pos={} seq_len={}",
                    i, batch_size, current_pos, seq_len
                );
            }
            // Each kernel in its own compute pass to guarantee memory barriers
            // (matches layer.rs ordering). PostAttnNorm and PostFfwNorm are no-ops
            // when params.post_norm_enabled == 0 (non-Gemma models).
            // Select pipeline based on quantization type (Q4_K vs Q4_0/F16)
            // Note: Q4_K shader only has qkv, attn_out, attn_proj, post_attn_norm, ffn_proj, ffn_down, post_ffn_norm
            //       It does NOT have: attn_norm, qk_norm, ffn_norm, post_ffw_norm - use V1 for those
            let (
                pipe_attn_norm,
                pipe_qkv,
                pipe_qk_norm,
                pipe_attn_out,
                pipe_attn_proj,
                pipe_post_attn_norm,
                pipe_ffn_norm,
                pipe_ffn_proj,
                pipe_ffn_down,
                pipe_post_ffw_norm,
            ) = if use_q4k_pipeline {
                (
                    &self.layer_pipeline_attn_norm, // Q4K: Use V1 (no main_attn_norm in Q4K shader)
                    &self.layer_pipeline_q4k_qkv,
                    &self.layer_pipeline_qk_norm, // Q4K: now has real main_qk_norm in sh_layer_q4k.wgsl (self-contained for Q4K Q writes + offsets)
                    &self.layer_pipeline_q4k_attn_out,
                    &self.layer_pipeline_q4k_attn_proj,
                    &self.layer_pipeline_post_attn_norm, // Q4K non-Gemma: Use V1 for post_attn_norm (Q4K version is Gemma-only)
                    &self.layer_pipeline_ffn_norm, // Q4K: Use V1 (no main_ffn_norm in Q4K shader)
                    &self.layer_pipeline_q4k_ffn_proj,
                    &self.layer_pipeline_q4k_ffn_down,
                    &self.layer_pipeline_post_ffw_norm, // Q4K: Use V1 (no main_post_ffw_norm in Q4K shader)
                )
            } else {
                (
                    &self.layer_pipeline_attn_norm,
                    &self.layer_pipeline_qkv,
                    &self.layer_pipeline_qk_norm,
                    &self.layer_pipeline_attn_out,
                    &self.layer_pipeline_attn_proj,
                    &self.layer_pipeline_post_attn_norm,
                    &self.layer_pipeline_ffn_norm,
                    &self.layer_pipeline_ffn_proj,
                    &self.layer_pipeline_ffn_down,
                    &self.layer_pipeline_post_ffw_norm,
                )
            };

            {
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - AttnNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_attn_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            // QKV: micro-batched to avoid Windows TDR on Q4_K_M models.
            // Each chunk dispatches `qkv_chunk` tokens, submits, and polls before the next.
            // The params buffer is updated via write_buffer — no new bind group needed.
            {
                let params_buf_i = &_params_buffers[i];
                let mut qkv_offset: u32 = 0;
                while qkv_offset < batch_size {
                    let this_chunk = (batch_size - qkv_offset).min(qkv_chunk);
                    // Patch batch_offset + batch_count into the layer params buffer
                    let params_chunk = LayerParams {
                        batch_offset: qkv_offset,
                        batch_count: this_chunk,
                        ..params_layer
                    };
                    // Submit all pending work before write_buffer (required ordering)
                    let label_pre = format!("Layer {} QKV pre-chunk {}", i, qkv_offset);
                    tdr.force_yield(&label_pre)?;
                    // Update the params buffer in place
                    queue.write_buffer(params_buf_i, 0, bytemuck::bytes_of(&params_chunk));
                    {
                        let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("Layer {} QKV [{}/{}]", i, qkv_offset, batch_size)),
                            timestamp_writes: None,
                        });
                        cpass.set_bind_group(0, bg, &[]);
                        cpass.set_pipeline(pipe_qkv);
                        cpass.dispatch_workgroups(wg_qkv, this_chunk, 1);
                    }
                    let label_chunk = format!("Layer {} QKV chunk {}", i, qkv_offset);
                    tdr.force_yield(&label_chunk)?;
                    // Restore full params for remaining kernels
                    queue.write_buffer(params_buf_i, 0, bytemuck::bytes_of(&params_layer));
                    qkv_offset += this_chunk;
                }
            }
            {
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - QKNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_qk_norm);
                cpass.dispatch_workgroups(wg_qknorm, batch_size, 1);
            }
            {
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - AttnOut", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_attn_out);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            // TDR: yield after attn_out only if accumulated budget exceeded.
            {
                let label = format!("layer-{}-attn_out", i);
                tdr.yield_if_needed(&label)?;
            }
            {
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - AttnProj", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_attn_proj);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            {
                // Post-attention norm correction (Gemma-2 only; no-op for post_norm_enabled==0)
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - PostAttnNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_post_attn_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            if params_layer.quant_type != 12u32 {
                // For Q4K, ffn_norm is inside the Q4K ffn_proj kernel; skip V1 to avoid double norm.
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - FFNNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_ffn_norm);
                cpass.dispatch_workgroups(1, batch_size, 1);
            }
            {
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - FFNProj", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_ffn_proj);
                cpass.dispatch_workgroups(wg_ffn, batch_size, 1);
            }
            {
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - FFNDown", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_ffn_down);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            // TDR: yield after ffn_down only if accumulated budget exceeded.
            {
                let label = format!("layer-{}-ffn_down", i);
                tdr.yield_if_needed(&label)?;
            }
            {
                // Post-FFW norm correction (Gemma-2 only; no-op for post_norm_enabled==0)
                let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - PostFfwNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_post_ffw_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }

            // TDR: conditional yield at layer boundary.
            // NOTE: this yield is still required even with ping-pong because the WGSL shaders
            // do in-place residual adds (read_write activation_in) — the ping-pong swap
            // happens at the bind group level but within a single encoder, wgpu on D3D12
            // may not emit UAV barriers between passes on the same read_write buffer.
            // Step 4 (remove this yield) requires the WGSL to use separate read/write bindings.
            // Until then, keep this yield for correctness on DeepSeek Q4K.
            {
                let label = format!("layer-{}-boundary", i);
                tdr.yield_if_needed(&label)?;
            }

            if trace_prefill_layers {
                // Layer trace readback — uses its own encoder, separate from tdr.
                let mut trace_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("Layer {} Trace Readback", i)),
                    });
                trace_encoder.copy_buffer_to_buffer(
                    &activation_buffer,
                    last_token_offset,
                    &pre_norm_buffer,
                    0,
                    (dim as u64) * 4,
                );
                queue.submit(Some(trace_encoder.finish()));
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|_| {
                        "GPU device lost or TDR timeout during layer trace readback".to_string()
                    })?;
                tdr.reset_accumulator(); // readback did its own submit+poll

                let trace_slice = pre_norm_buffer.slice(..);
                let (tx_trace, rx_trace) = std::sync::mpsc::channel();
                trace_slice.map_async(wgpu::MapMode::Read, move |res| tx_trace.send(res).unwrap());
                loop {
                    device
                        .poll(wgpu::PollType::Poll)
                        .map_err(|_| "GPU device lost during layer trace poll".to_string())?;
                    if let Ok(res) = rx_trace.try_recv() {
                        res.map_err(|_| "Layer trace buffer map failed".to_string())?;
                        break;
                    }
                }
                let mapped = trace_slice.get_mapped_range();
                let trace_vals: &[f32] = bytemuck::cast_slice(&mapped);
                let nan_count = trace_vals.iter().filter(|&&x| x.is_nan()).count();
                let first5: Vec<f32> = trace_vals.iter().take(5).copied().collect();
                eprintln!(
                    "[PREFILL-LAYER-TRACE] layer={} nan={}/{} first5={:?}",
                    i,
                    nan_count,
                    trace_vals.len(),
                    first5
                );
                drop(mapped);
                pre_norm_buffer.unmap();
            }

            if trace_prefill_layers {
                eprintln!("[PREFILL-LAYER] complete layer={}", i);
            }
        }

        // Snapshot h20 (post-layer-loop, pre-final-norm)
        if tdr_log {
            eprintln!("[TDR-STATS] batch_size={} layers={} total_yields={} forced_per_layer_min={}",
                batch_size, layer_count, tdr.yield_count,
                if layer_count > 0 { tdr.yield_count / layer_count as u32 } else { 0 });
        }
        tdr.encoder.copy_buffer_to_buffer(
            &activation_buffer,
            last_token_offset,
            &pre_norm_buffer,
            0,
            (dim as u64) * 4,
        );

        // Final Norm — separate pass so wgpu inserts a memory barrier before the
        // LM Head pass reads from temp_buffer (same region that norm writes).
        if disable_output_norm {
            tdr.encoder.copy_buffer_to_buffer(
                &activation_buffer,
                last_token_offset,
                &temp_buffer,
                0,
                (dim as u64) * 4u64,
            );
        } else {
            let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Final Norm"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &norm_bg, &[]);
            cpass.set_pipeline(&self.rmsnorm_pipeline);
            cpass.dispatch_workgroups(wg_norm, 1, 1);
        }
        // LM Head
        {
            let mut cpass = tdr.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Head"),
                timestamp_writes: None,
            });
            match &head_bg {
                HeadBg::Blob(bg) => {
                    cpass.set_bind_group(0, bg, &[]);
                    cpass.set_pipeline(&self.lm_head_blob_pipeline);
                    cpass.dispatch_workgroups(wg_head_blob, 1, 1);
                }
                HeadBg::F32(bg) => {
                    cpass.set_bind_group(0, bg, &[]);
                    cpass.set_pipeline(&self.matmul_f32_pipeline);
                    cpass.dispatch_workgroups(wg_head_f32, 1, 1);
                }
            }
        }

        // 6. Readback
        let output_size = (vocab_size * 4) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        tdr.encoder.copy_buffer_to_buffer(&temp_buffer, 0, &l21_buffer, 0, (dim as u64) * 4);
        tdr.encoder.copy_buffer_to_buffer(&logits_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(tdr.encoder.finish()));

        let pre_norm_slice = pre_norm_buffer.slice(..);
        let (tx_pre, rx_pre) = std::sync::mpsc::channel();
        pre_norm_slice.map_async(wgpu::MapMode::Read, move |res| tx_pre.send(res).unwrap());

        let l21_slice = l21_buffer.slice(..);
        let (tx_l21, rx_l21) = std::sync::mpsc::channel();
        l21_slice.map_async(wgpu::MapMode::Read, move |res| tx_l21.send(res).unwrap());

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        let mut pre_done = false;
        let mut l21_done = false;
        let mut main_done = false;

        loop {
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|_| "GPU device lost during readback poll".to_string())?;

            if !pre_done {
                if let Ok(res) = rx_pre.try_recv() {
                    res.map_err(|_| {
                        "Pre-norm buffer map failed. Device lost or TDR timeout.".to_string()
                    })?;
                    pre_done = true;
                }
            }
            if !l21_done {
                if let Ok(res) = rx_l21.try_recv() {
                    res.map_err(|_| {
                        "L21 buffer map failed. Device lost or TDR timeout.".to_string()
                    })?;
                    l21_done = true;
                }
            }
            if !main_done {
                if let Ok(res) = rx.try_recv() {
                    res.map_err(|_| "Buffer map failed. Device lost or TDR timeout.".to_string())?;
                    main_done = true;
                }
            }

            if pre_done && l21_done && main_done {
                break;
            }
        }

        let pre_norm_data = pre_norm_slice.get_mapped_range();
        let pre_norm_result: Vec<f32> = bytemuck::cast_slice(&pre_norm_data).to_vec();
        drop(pre_norm_data);
        pre_norm_buffer.unmap();

        let l21_data = l21_slice.get_mapped_range();
        let l21_result: Vec<f32> = bytemuck::cast_slice(&l21_data).to_vec();
        drop(l21_data);
        l21_buffer.unmap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok((pre_norm_result, l21_result, result))
    }
}
