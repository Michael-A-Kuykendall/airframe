//! Single-layer test and cache-enabled layer dispatch methods for `BindlessPipeline`.
// TODO: extract run_layer_with_cache_debug into a conditional-compile debug module once the debug path is stabilised.
use super::*;
use super::super::loader::BindlessModel;
use super::super::kv_cache::KVCache;
use wgpu::util::DeviceExt;

/// Debug return type with 6 activation vectors
type LayerDebugOutput = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

impl BindlessPipeline {
    pub fn run_qkv_only_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        println!(
            "[BindlessPipeline] Running QKV ONLY Test (Dim={})",
            params.dim
        );

        // 1. Activation In (ReadWrite)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation In"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Temp State for Q (ReadWrite)
        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 3. Uniforms
        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // KV Cache - K and V go here
        let kv_size = 2048u64 * (params.head_count_kv as u64) * (params.head_dim as u64) * 4 * 2; // max_seq * n_head_kv * head_dim * sizeof(f32) * 2 (K+V)
        let kv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache Temp"),
            size: kv_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 4. Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("QKV Test BindGroup"),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
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
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
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
                    resource: kv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: model.dummy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: model.dummy_buf.as_entire_binding(),
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

        // 5. Run attention provider, then QKV kernel
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = total_qkv.div_ceil(256);
        let wg_dim = params.dim.div_ceil(256);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AttnNorm Only"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QKV Only"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        }

        // Stage readback buffers
        let q_size = (q_len as u64) * 4; // Q: 2048 floats
        let kv_unit = (kv_len as u64) * 4; // K or V: 256 floats

        let staging_q = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Q"),
            size: q_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging K"),
            size: kv_unit,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging V"),
            size: kv_unit,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Q from temp_state[dim..dim+q_len]
        encoder.copy_buffer_to_buffer(&temp_buffer, params.dim as u64 * 4, &staging_q, 0, q_size);
        // K from kv_cache[0..kv_len]
        encoder.copy_buffer_to_buffer(&kv_buffer, 0, &staging_k, 0, kv_unit);
        // V from kv_cache[kv_len..kv_len*2]
        encoder.copy_buffer_to_buffer(&kv_buffer, kv_unit, &staging_v, 0, kv_unit);

        queue.submit(Some(encoder.finish()));

        // Readback
        let q_vec = self.readback_helper(device, &staging_q);
        let k_vec = self.readback_helper(device, &staging_k);
        let v_vec = self.readback_helper(device, &staging_v);

        (q_vec, k_vec, v_vec)
    }

    /// Run Full Layer Test
    pub fn run_layer_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> Vec<f32> {
        self.run_layer_stepwise_test(device, queue, model, input, offsets, params, false)
            .1
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_stepwise_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
        read_halfway: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        println!(
            "[BindlessPipeline] Running Layer V1 Test Stepwise (Dim={})",
            params.dim
        );

        // 1. Activation In (ReadWrite)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation In"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Temp State (ReadWrite, Scratchpad)
        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 3. Uniforms
        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // KV Caches (Temp for Test - K and V separate)
        let kv_size_per_buffer = (params.dim as u64 / params.head_count as u64)
            * (params.head_count_kv as u64)
            * 2048u64
            * 4; // head_dim * n_head_kv * max_seq * sizeof(f32)
        let kv_buffer_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache K Temp"),
            size: kv_size_per_buffer,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let kv_buffer_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache V Temp"),
            size: kv_size_per_buffer,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Cache params (temporary - position 0)
        let cache_params = CacheParams {
            current_pos: 0,
            seq_len: 1,
            max_seq_len: 2048, // test default
            batch_size: 1,
            logical_pos_base: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params Temp"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 4. Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Layer Test BindGroup"),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
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
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
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
                    resource: kv_buffer_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: kv_buffer_v.as_entire_binding(),
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
        });

        // 5. Dispatch Sequence
        // Part 1: QKV + Attn
        let dim = params.dim as u64;
        let wg_dim = params.dim.div_ceil(256);

        // QKV Needs more threads: Q + K + V
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = total_qkv.div_ceil(256);

        // FFN needs threads for both Gate and Up: ffn_dim * 2
        let ffn_total = params.ffn_dim * 2; // Gate + Up
        let wg_ffn = ffn_total.div_ceil(256); // Ceil div by workgroup size

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Kernel 1: Attention Norm Provider
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AttnNorm"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // Kernel 2: QKV
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QKV"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        }

        // Kernel 2.5: QK Norm (Qwen3; no-op when qk_norm_enabled == 0)
        {
            let wg_qknorm = (q_len + kv_len).div_ceil(256);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QKNorm"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qk_norm);
            cpass.dispatch_workgroups(wg_qknorm, 1, 1);
        }

        // Kernel 3: Attention
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attn"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_out);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // Kernel 4: Attention Projection
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AttnProj"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_proj);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        let staging_mid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Mid"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if read_halfway {
            encoder.copy_buffer_to_buffer(&activation_buffer, 0, &staging_mid, 0, dim * 4);
        }

        // Kernel 5: FFN Norm Broadcast (cooperative reduction)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFNNorm"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_norm);
            // X=1: one workgroup (256 threads) cooperates on the single token.
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // Kernel 6: FFN Projection
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFN"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
            cpass.dispatch_workgroups(wg_ffn, 1, 1);
        }

        // Kernel 7: FFN Down
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFNDown"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_down);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        let staging_final = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Final"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &staging_final, 0, dim * 4);
        queue.submit(Some(encoder.finish()));

        // Readbacks
        let mut mid_vec = Vec::new();
        if read_halfway {
            mid_vec = self.readback_helper(device, &staging_mid);
        }
        let final_vec = self.readback_helper(device, &staging_final);

        (mid_vec, final_vec)
    }

    /// Run a single transformer layer with stateful KV cache
    ///
    /// This method:
    /// 1. Uses the provided KVCache for attention context
    /// 2. Writes new K/V at cache.get_seq_len() position
    /// 3. Reads all cached K/V for attention (with causal masking)
    /// 4. Increments cache position after success
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `queue` - WGPU queue
    /// * `model` - Loaded GGUF model with preflight data
    /// * `kv_cache` - Mutable KVCache for this layer
    /// * `layer_idx` - Layer index (0..21)
    /// * `input` - Input activation vector (dim elements)
    /// * `offsets` - Weight offsets for this layer
    /// * `params` - Layer hyperparameters
    ///
    /// # Returns
    /// Output activation vector (dim elements)
    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_with_cache(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> Vec<f32> {
        let dim = params.dim as u64;

        // 1. Create activation buffer (input/output)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Activation Layer {}", layer_idx)),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Temp state buffer (for Q, attention scores, FFN intermediate)
        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Temp State Layer {}", layer_idx)),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 3. Uniform buffers
        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Cache params uniform
        let cache_params = CacheParams {
            current_pos: kv_cache.get_seq_len(), // Position to write new K/V
            seq_len: kv_cache.get_seq_len() + 1, // After write, this many positions cached
            max_seq_len: kv_cache.max_len(),
            batch_size: 1,
            logical_pos_base: kv_cache.get_window_base(),
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 5. Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} BindGroup", layer_idx)),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
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
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
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
                    resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding(),
                }, // K cache for this layer
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding(),
                }, // V cache for this layer
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
        });

        // 6. Calculate workgroup counts
        let wg_dim = params.dim.div_ceil(256);
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = total_qkv.div_ceil(256);
        let ffn_total = params.ffn_dim * 2;
        let wg_ffn = ffn_total.div_ceil(256);

        // 7. Dispatch compute passes (6 kernels, each in separate pass for synchronization)
        // CRITICAL: Each compute pass ensures GPU work completes before next starts
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Layer {} Encoder", layer_idx)),
        });

        // Kernel 1: Attention normalization provider
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - AttnNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // <- GPU waits for AttnNorm to finish before QKV proceeds

        // Kernel 2: QKV generation + cache write
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - QKV", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        } // ← GPU waits for QKV to finish before proceeding

        // Kernel 2.5: QK Norm (Qwen3; no-op when qk_norm_enabled == 0)
        {
            let wg_qknorm = (q_len + kv_len).div_ceil(256);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - QKNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qk_norm);
            cpass.dispatch_workgroups(wg_qknorm, 1, 1);
        } // ← GPU waits for QKNorm to finish before Attn

        // Kernel 3: Attention (Q @ cached_K, softmax, weighted sum of cached_V)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - Attn", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_out);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for Attn to finish

        // Kernel 4: Attention output projection
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - AttnProj", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_proj);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for AttnProj to finish

        // Kernel 4.5: Post-attention norm correction (Gemma-2; no-op otherwise)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - PostAttnNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_post_attn_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for PostAttnNorm to finish

        // Kernel 5: FFN normalization broadcast (cooperative reduction)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFNNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_norm);
            // X=1: 256 threads share the token via local_invocation_id.
            cpass.dispatch_workgroups(1, 1, 1);
        } // ← GPU waits for FFNNorm to finish

        // Kernel 6: FFN gate/up + SiLU
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFN", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
            cpass.dispatch_workgroups(wg_ffn, 1, 1);
        } // ← GPU waits for FFN to finish

        // Kernel 7: FFN down + residual
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFNDown", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_down);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for FFNDown to finish

        // Kernel 6.5: Post-FFW norm correction (Gemma-2; no-op otherwise)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - PostFfwNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_post_ffw_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for PostFfwNorm to finish

        // 8. Readback result
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &staging_buffer, 0, dim * 4);
        queue.submit(Some(encoder.finish()));

        

        // NOTE: Cache increment happens ONCE per token (after all layers), not per layer
        // Caller must call kv_cache.increment() after processing all 22 layers

        self.readback_helper(device, &staging_buffer)
    }

    /// INT4 variant of run_layer_with_cache (TurboQuant — feat/turboquant-wgsl).
    ///
    /// Two-pass KV quantization:
    ///   1. main_qkv writes F32 K/V to staging buffers (bindings 7/8) — same as F32 path.
    ///   2. quantize_kv converts F32→INT4 at current_pos for all heads of this layer.
    ///   3. main_attn_out_int4 reads from INT4 packed+scale buffers (bindings 10-13).
    ///
    /// The kv_cache must have been created with KVCache::new_int4().
    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_with_cache_int4(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> Vec<f32> {
        let dim = params.dim as u64;

        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Activation Layer {} INT4", layer_idx)),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Temp State Layer {} INT4", layer_idx)),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets INT4"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params INT4"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let current_pos = kv_cache.get_seq_len();
        let cache_params = CacheParams {
            current_pos,
            seq_len: current_pos + 1,
            max_seq_len: kv_cache.max_len(),
            batch_size: 1,
            logical_pos_base: kv_cache.get_window_base(),
            pad1: 0, pad2: 0, pad3: 0,
        };
        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params INT4"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // INT4 layer bind group (14 bindings: 0-9 same as F32, 10-13 packed+scale read-only)
        // Used ONLY for main_attn_out_int4 — that pipeline was compiled against layer_layout_int4.
        let layer_bg_int4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} INT4 BindGroup", layer_idx)),
            layout: &self.layer_layout_int4,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: model.gpu_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1,  resource: activation_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2,  resource: temp_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3,  resource: offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4,  resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5,  resource: model.preflight.as_ref().unwrap().norm_bank_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6,  resource: model.preflight.as_ref().unwrap().rope_cache_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7,  resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8,  resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9,  resource: cache_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: kv_cache.get_k_packed_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: kv_cache.get_k_scale_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: kv_cache.get_v_packed_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: kv_cache.get_v_scale_buffer(layer_idx).as_entire_binding() },
            ],
        });

        // F32 bind group (10 bindings) for all other kernels (compiled against layer_layout).
        let layer_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} INT4 F32BG", layer_idx)),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: model.gpu_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: activation_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: model.preflight.as_ref().unwrap().norm_bank_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: model.preflight.as_ref().unwrap().rope_cache_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: cache_params_buffer.as_entire_binding() },
            ],
        });

        // quantize_kv bind group (7 bindings: f32_k, f32_v, packed_k, packed_v, scale_k, scale_v, params)
        let qkv_params = QuantizeKvParams {
            n_head_kv:  params.head_count_kv,
            head_dim:   params.head_dim,
            pos_offset: current_pos,
            _pad:       0,
        };
        let qkv_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QuantizeKvParams"),
            contents: bytemuck::bytes_of(&qkv_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let quant_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} QuantizeKV BindGroup", layer_idx)),
            layout: &self.quantize_kv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kv_cache.get_k_packed_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: kv_cache.get_v_packed_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: kv_cache.get_k_scale_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: kv_cache.get_v_scale_buffer(layer_idx).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: qkv_params_buf.as_entire_binding() },
            ],
        });

        let wg_dim  = params.dim.div_ceil(256);
        let q_len   = params.head_count * params.head_dim;
        let kv_len  = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv  = total_qkv.div_ceil(256);
        let wg_ffn  = (params.ffn_dim * 2).div_ceil(256);
        let wg_qknorm = (q_len + kv_len).div_ceil(256);

        // --- Fused single encoder: all INT4 kernels in order, one submit ---
        // wgpu guarantees sequential dispatch execution within a command encoder;
        // GPU memory barriers between passes replace all CPU round-trips.
        // Single CPU sync point: the staging readback below.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer INT4"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Layer {} INT4", layer_idx)),
            });
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("AttnNorm"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_attn_norm); cp.dispatch_workgroups(wg_dim, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("QKV"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_qkv); cp.dispatch_workgroups(wg_qkv, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("QKNorm"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_qk_norm); cp.dispatch_workgroups(wg_qknorm, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("QuantizeKV"), timestamp_writes: None }); cp.set_bind_group(0, &quant_bg, &[]); cp.set_pipeline(&self.quantize_kv_pipeline); cp.dispatch_workgroups(params.head_count_kv, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("AttnOutINT4"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg_int4, &[]); cp.set_pipeline(&self.layer_pipeline_attn_out_int4); cp.dispatch_workgroups(wg_dim, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("AttnProj"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_attn_proj); cp.dispatch_workgroups(wg_dim, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PostAttnNorm"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_post_attn_norm); cp.dispatch_workgroups(wg_dim, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FFNProj"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_ffn_proj); cp.dispatch_workgroups(wg_ffn, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FFNDown"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_ffn_down); cp.dispatch_workgroups(wg_dim, 1, 1); }
            { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PostFfwNorm"), timestamp_writes: None }); cp.set_bind_group(0, &layer_bg, &[]); cp.set_pipeline(&self.layer_pipeline_post_ffw_norm); cp.dispatch_workgroups(wg_dim, 1, 1); }
            enc.copy_buffer_to_buffer(&activation_buffer, 0, &staging_buffer, 0, dim * 4);
            queue.submit(Some(enc.finish()));
        }

        self.readback_helper(device, &staging_buffer)
    }

    /// Requantize all prefilled positions to INT4 across every layer.
    ///
    /// Call once after `run_full_model_prefill_chunked_with_cache_state` completes when
    /// `kv_cache.is_int4()` is true.  The F32 staging buffers (bindings 7/8) already contain
    /// all prefill positions written by the F32 prefill path; this converts them to INT4 so
    /// the decode loop (`run_layer_with_cache_int4`) can read them correctly.
    ///
    /// Dispatch: (n_head_kv, seq_len, 1) — each invocation handles one (head, position) pair.
    #[allow(clippy::too_many_arguments)]
    pub fn requantize_all_kv_int4(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kv_cache: &KVCache,
        n_head_kv: u32,
        head_dim: u32,
        seq_len: u32,
        n_layers: usize,
    ) {
        if seq_len == 0 { return; }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RequantizeAllKV"),
        });

        for layer_idx in 0..n_layers {
            let qkv_params = QuantizeKvParams {
                n_head_kv,
                head_dim,
                pos_offset: 0,
                _pad: 0,
            };
            let qkv_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("RequantKVParams L{}", layer_idx)),
                contents: bytemuck::bytes_of(&qkv_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let quant_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("RequantKV L{}", layer_idx)),
                layout: &self.quantize_kv_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: kv_cache.get_k_packed_buffer(layer_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: kv_cache.get_v_packed_buffer(layer_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: kv_cache.get_k_scale_buffer(layer_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: kv_cache.get_v_scale_buffer(layer_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: qkv_params_buf.as_entire_binding() },
                ],
            });

            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("RequantizeKV L{}", layer_idx)),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &quant_bg, &[]);
            cp.set_pipeline(&self.quantize_kv_pipeline);
            // x=head, y=position
            cp.dispatch_workgroups(n_head_kv, seq_len, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Debug version that extracts Q/K/V tensors for verification
    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_with_cache_debug(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> LayerDebugOutput {
        let dim = params.dim as u64;

        // 1. Create buffers (same as run_layer_with_cache)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Activation Layer {}", layer_idx)),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Temp State Layer {}", layer_idx)),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC for debug
            mapped_at_creation: false,
        });

        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let cache_params = CacheParams {
            current_pos: kv_cache.get_seq_len(),
            seq_len: kv_cache.get_seq_len() + 1,
            max_seq_len: kv_cache.max_len(),
            batch_size: 1,
            logical_pos_base: kv_cache.get_window_base(),
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} BindGroup", layer_idx)),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
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
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
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
                    resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding(),
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
        });

        let wg_dim = params.dim.div_ceil(256);
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = total_qkv.div_ceil(256);
        let ffn_total = params.ffn_dim * 2;
        let wg_ffn = ffn_total.div_ceil(256);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Layer {} Encoder (Debug)", layer_idx)),
        });

        // Kernel 1: Attention normalization provider
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - AttnNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // Kernel 2: QKV generation + cache write
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - QKV", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        }

        // CAPTURE Q from temp_state[dim..dim+q_len]
        let q_size = (params.head_count as u64) * (params.head_dim as u64) * 4;
        let q_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Q Staging"),
            size: q_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&temp_buffer, params.dim as u64 * 4, &q_staging, 0, q_size);

        // CAPTURE K from kv_cache_k for CURRENT position only
        let kv_slice_size = (params.head_count_kv as u64) * (params.head_dim as u64) * 4;
        let k_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("K Staging"),
            size: kv_slice_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_offset = (cache_params.current_pos as u64)
            * (params.head_count_kv as u64)
            * (params.head_dim as u64)
            * 4;
        encoder.copy_buffer_to_buffer(
            kv_cache.get_k_buffer(layer_idx),
            k_offset,
            &k_staging,
            0,
            kv_slice_size,
        );

        // CAPTURE V from kv_cache_v for CURRENT position only
        let v_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Staging"),
            size: kv_slice_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_offset = (cache_params.current_pos as u64)
            * (params.head_count_kv as u64)
            * (params.head_dim as u64)
            * 4;
        encoder.copy_buffer_to_buffer(
            kv_cache.get_v_buffer(layer_idx),
            v_offset,
            &v_staging,
            0,
            kv_slice_size,
        );

        // Continue with remaining kernels (2-5) - same as run_layer_with_cache
        // Kernel 2.5: QK Norm (Qwen3; no-op when qk_norm_enabled == 0)
        {
            let wg_qknorm = (q_len + kv_len).div_ceil(256);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - QKNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qk_norm);
            cpass.dispatch_workgroups(wg_qknorm, 1, 1);
        }

        // Kernel 3: Attention
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - Attn", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_out);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - AttnProj", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_proj);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // Kernel 4.5: Post-attention norm correction (Gemma-2; no-op otherwise)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - PostAttnNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_post_attn_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // CAPTURE post-attention state after AttnProj (before FFN)
        let post_attn_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Post-Attn Staging"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &post_attn_staging, 0, dim * 4);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFNNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_norm);
            // X=1: 256 threads share the token via local_invocation_id.
            cpass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFN", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
            cpass.dispatch_workgroups(wg_ffn, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFNDown", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_down);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // Kernel 6.5: Post-FFW norm correction (Gemma-2; no-op otherwise)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - PostFfwNorm", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_post_ffw_norm);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // CAPTURE FFN down output (stored at temp_state[ffn_dim*2..ffn_dim*2+dim])
        let ffn_debug_offset = (params.ffn_dim as u64) * 2 * 4;
        let ffn_down_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFN Down Staging"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &temp_buffer,
            ffn_debug_offset,
            &ffn_down_staging,
            0,
            dim * 4,
        );

        // Readback final output
        let output_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &output_staging, 0, dim * 4);

        queue.submit(Some(encoder.finish()));

        // Read all buffers
        let q_vals = self.readback_helper(device, &q_staging);
        let k_vals = self.readback_helper(device, &k_staging);
        let v_vals = self.readback_helper(device, &v_staging);
        let post_attn_vals = self.readback_helper(device, &post_attn_staging);
        let ffn_down_vals = self.readback_helper(device, &ffn_down_staging);
        let output = self.readback_helper(device, &output_staging);

        (
            output,
            post_attn_vals,
            ffn_down_vals,
            q_vals,
            k_vals,
            v_vals,
        )
    }

}

#[cfg(test)]
mod tests {
    // GPU-dispatch methods need a live device; end-to-end coverage lives in the
    // integration test suite under tests/.  Pure invariants are exercised in
    // pipeline/mod.rs and preflight.rs.
    use super::*;

    #[test]
    fn layer_params_size_is_gpu_aligned() {
        // LayerParams is copied verbatim into WGSL uniforms; size must be
        // a multiple of 4 bytes (WebGPU minimum alignment).
        assert_eq!(std::mem::size_of::<LayerParams>() % 4, 0);
    }
}
