//! Full-model inference dispatch methods for `BindlessPipeline`.
// TODO: break run_full_model_prefill_chunked_with_cache_state into a separate chunking helper once prefill chunking is the default path.
use super::*;
use super::super::loader::BindlessModel;
use wgpu::util::DeviceExt;

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
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let dim = spec.n_embd;
        assert!(dim > 0, "spec.n_embd must be > 0");
        assert!(input_embd.len() % dim == 0, "input_embd must align to token rows");
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
        let mut chunk_idx = 0usize;
        let mut last_result = None;

        for chunk in input_embd.chunks(chunk_rows * dim) {
            let chunk_token_count = (chunk.len() / dim) as u32;
            let chunk_current_pos = current_pos + processed_tokens;
            let chunk_seq_len = chunk_current_pos + chunk_token_count;

            if trace_chunks {
                eprintln!(
                    "[PREFILL] chunk={} tokens={} current_pos={} seq_len={}",
                    chunk_idx,
                    chunk_token_count,
                    chunk_current_pos,
                    chunk_seq_len
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
            chunk_idx += 1;
        }

        last_result.ok_or_else(|| "chunked prefill produced no chunks".to_string())
    }

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
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        // Derive all constants from ModelSpec
        let dim = spec.n_embd as u32;
        let layer_count = spec.n_layer;
        let vocab_size = spec.n_vocab as u32;
        let ffn_dim = spec.ff_dim as u32;
        let temp_stride = spec.temp_buffer_size as u32;

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
        let packed_quant_type =
            weight_quant_type | (qt_v << 8) | (qt_ffn_down << 16);
        let _ = packed_quant_type; // per-layer quant is computed in the loop below

        // 1. Buffers
        let batch_size = (input_embd.len() as u32) / dim;
        // A. Activation (Residual Stream) - Init with Embeddings
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation"),
            contents: bytemuck::cast_slice(input_embd),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // B. Temp Buffer
        // Needs to hold FFN Gate + Up + scratch space per token
        let temp_buffer_size = batch_size as u64 * temp_stride as u64 * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // C. Layer Params (computed per-layer below; placeholder base for struct copy)
        // NOTE: quant_type varies per layer in mixed-quant models (e.g. Q4_K_M).
        //       Per-layer params buffers are created inside the layer loop.
        let params_base = LayerParams {
            dim,
            head_count: spec.n_head as u32,
            head_count_kv: spec.n_head_kv as u32,
            head_dim: spec.head_dim as u32,
            rms_eps: spec.rms_eps,
            ffn_dim,
            temp_stride,
            quant_type: 0, // overridden per-layer below
            attn_logit_softcap: spec.attn_logit_softcap,
            post_norm_enabled: if spec.arch_string() == "gemma2" { 1 } else { 0 },
            qk_norm_enabled: if spec.has_qk_norm { 1 } else { 0 },
        };

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
            let mut bufs = Vec::with_capacity(layer_count as usize);
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
        let mut layer_bind_groups = Vec::new();
        let mut _offset_buffers = Vec::new(); // Keep alive
        let mut _params_buffers: Vec<wgpu::Buffer> = Vec::new(); // Keep alive

        for i in 0..layer_count {
            let compiled = &model.metadata.compiled_layers[i as usize];
            let layer_params_i = LayerParams {
                quant_type: compiled.quant_type_packed,
                ..params_base
            };
            let params_buffer_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Params", i)),
                contents: bytemuck::bytes_of(&layer_params_i),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let (kv_buffer_k_ref, kv_buffer_v_ref): (&wgpu::Buffer, &wgpu::Buffer) =
                if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                    (&kv_k_layers[i as usize], &kv_v_layers[i as usize])
                } else {
                    let (local_k, local_v) = &local_kv_storage_per_layer
                        .as_ref()
                        .expect("local KV storage missing")[i as usize];
                    (local_k, local_v)
                };

            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Offsets", i)),
                contents: bytemuck::bytes_of(&compiled.offsets),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Layer {} BindGroup", i)),
                layout: &self.layer_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.gpu_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: activation_buffer.as_entire_binding(),
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
                ],
            });

            _offset_buffers.push(buf);
            _params_buffers.push(params_buffer_i);
            layer_bind_groups.push(bg);
        }

        // 3. Final Norm
        let norm_weight = model
            .metadata
            .get_tensor_offset("output_norm.weight")
            .expect("output_norm missing");
        let norm_params = RMSNormParams {
            count: dim,
            weights_offset: norm_weight as u32,
            eps: spec.rms_eps,
            padding: 0,
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
                    resource: model.gpu_buffer.as_entire_binding(),
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
            ],
        });

        // 4. Output Head (MatMul)
        let head_weight = model
            .metadata
            .get_tensor_offset("output.weight")
            .or_else(|| model.metadata.get_tensor_offset("token_embd.weight"))
            .unwrap_or(0);

        let head_params = MatMulParams {
            n: vocab_size,
            k: dim,
            weights_offset: head_weight as u32,
            padding: 0,
        };
        let head_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Head Params"),
            contents: bytemuck::bytes_of(&head_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let head_bg = if let Some(override_buf) = head_weights_override {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            })
        } else {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG"),
                layout: &self.matmul_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.gpu_buffer.as_entire_binding(),
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
            })
        };

        // 5. Command Encoding
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Full Model"),
        });

        let wg_dim = (dim + 255) / 256;
        let ffn_total = ffn_dim * 2; // Gate + Up need this many threads
        let wg_ffn = (ffn_total + 255) / 256; // Ceil div by workgroup size (256)
        let wg_norm = (dim + 255) / 256;
        let wg_head = (vocab_size + 255) / 256;

        // QKV Dispatch Calculation
        let q_len = params_base.head_count * params_base.head_dim;
        let kv_len = params_base.head_count_kv * params_base.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = (total_qkv + 255) / 256;
        let trace_prefill_layers = std::env::var("AIRFRAME_TRACE_PREFILL_LAYERS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

// Loop Layers
        for (i, bg) in layer_bind_groups.iter().enumerate() {
            if trace_prefill_layers {
                eprintln!(
                    "[PREFILL-LAYER] start layer={} batch_size={} current_pos={} seq_len={}",
                    i,
                    batch_size,
                    current_pos,
                    seq_len
                );
            }
            // Each kernel in its own compute pass to guarantee memory barriers
            // (matches layer.rs ordering). PostAttnNorm and PostFfwNorm are no-ops
            // when params.post_norm_enabled == 0 (non-Gemma models).
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - AttnNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_attn_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - QKV", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_qkv);
                cpass.dispatch_workgroups(wg_qkv, batch_size, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - AttnOut", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_attn_out);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - AttnProj", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_attn_proj);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            {
                // Post-attention norm correction (Gemma-2 only; no-op for post_norm_enabled==0)
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - PostAttnNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_post_attn_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - FFNProj", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
                cpass.dispatch_workgroups(wg_ffn, batch_size, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - FFNDown", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_ffn_down);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            {
                // Post-FFW norm correction (Gemma-2 only; no-op for post_norm_enabled==0)
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Loop Layer {} - PostFfwNorm", i)),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(&self.layer_pipeline_post_ffw_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            
            if trace_prefill_layers {
                eprintln!("[PREFILL-LAYER] complete layer={}", i);
            }
        }

        // Snapshot h20 (post-layer-loop, pre-final-norm)
        // Copy only the LAST token's state for validation
        encoder.copy_buffer_to_buffer(
            &activation_buffer,
            last_token_offset,
            &pre_norm_buffer,
            0,
            (dim as u64) * 4,
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Norm+Head"),
                timestamp_writes: None,
            });

            // Final Norm
            cpass.set_bind_group(0, &norm_bg, &[]);
            cpass.set_pipeline(&self.rmsnorm_pipeline);
            cpass.dispatch_workgroups(wg_norm, 1, 1);

            // Head
            cpass.set_bind_group(0, &head_bg, &[]);
            if head_weights_override.is_some() {
                cpass.set_pipeline(&self.matmul_f32_pipeline);
            } else {
                cpass.set_pipeline(&self.matmul_pipeline);
            }
            cpass.dispatch_workgroups(wg_head, 1, 1);
        }

        // 6. Readback
        let output_size = (vocab_size * 4) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&temp_buffer, 0, &l21_buffer, 0, (dim as u64) * 4);
        encoder.copy_buffer_to_buffer(&logits_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

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
            device.poll(wgpu::PollType::Poll)
                .map_err(|_| "GPU device lost during readback poll".to_string())?;
            
            if !pre_done {
                if let Ok(res) = rx_pre.try_recv() {
                    res.map_err(|_| "Pre-norm buffer map failed. Device lost or TDR timeout.".to_string())?;
                    pre_done = true;
                }
            }
            if !l21_done {
                if let Ok(res) = rx_l21.try_recv() {
                    res.map_err(|_| "L21 buffer map failed. Device lost or TDR timeout.".to_string())?;
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
