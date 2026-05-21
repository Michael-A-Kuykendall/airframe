# Lambda Remote Session Briefing
**Date:** 2026-05-21  
**Instance:** 150.136.92.11 (Lambda Labs A100, 40 GB VRAM)  
**Working dir on Lambda:** `~/airframe`  
**Models dir:** `~/models/`

---

## What we're doing

Running the Airframe GPU inference server (`shimmy_server_gpu`) on an A100 via Vulkan/wgpu. The server loads a GGUF model onto the GPU and serves OpenAI-compatible `/v1/chat/completions` on port 8080.

**Current status:** Server starts and loads the model but **panics on first inference** with:
```
thread 'main' panicked: Error in Buffer::get_mapped_range: Validation Error
Buffer with 'Dequant Params' label is invalid
```

---

## Root cause (confirmed theory)

`wgpu::util::DeviceExt::create_buffer_init` works by:
1. Creating a staging buffer via `StagingBuffer::new` (uses `MAP_WRITE | COPY_SRC | TRANSIENT`)
2. Writing data into it via `get_mapped_range_mut`

If `StagingBuffer::new` fails for **any** reason (OOM, allocation error, etc.), it calls `handle_hal_error` which **permanently marks the wgpu Device as lost**. After that every subsequent `device.create_buffer` call returns `Fallible::Invalid("label")`, and `get_mapped_range_mut` on an invalid buffer calls `handle_error_fatal` → **panic**.

The 637 MB GGUF buffer upload in `loader.rs` is the prime suspect for triggering this. On Vulkan/Linux with wgpu-27, something about this large staging allocation corrupts device state. Even though the server prints "Model loaded to VRAM" (because the damage is silent), the device is already lost by the time inference begins.

---

## Files that need to be fixed on Lambda

All three files are in `~/airframe/src/backend/bindless/`.

### 1. `loader.rs` — THE ROOT CAUSE

**Line ~70–76** — the 637 MB GGUF buffer:

**Current (broken):**
```rust
let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("GGUF Raw"),
    contents: &raw_bytes,
    usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
});
```

**Replace with:**
```rust
let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("GGUF Raw"),
    size: raw_bytes.len() as u64,
    usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});
queue.write_buffer(&gpu_buffer, 0, &raw_bytes);
```

This requires `BindlessModel::load_from_disk` to accept `queue: &wgpu::Queue`. See call site changes below.

**Signature change:**
```rust
// Before:
pub fn load_from_disk(device: &wgpu::Device, path: &Path, spec: Option<&ModelSpec>) -> Self {

// After:
pub fn load_from_disk(device: &wgpu::Device, queue: &wgpu::Queue, path: &Path, spec: Option<&ModelSpec>) -> Self {
```

Also thread `queue` into `BindlessLoader::new(device, queue, ...)` and down into wherever the buffer is created.

---

### 2. `preflight.rs` — `upload_rope_table` and `build_norm_bank_from_ram`

Both functions use `create_buffer_init` for STORAGE buffers. Same fix: replace with `create_buffer(mapped_at_creation: false)` + `queue.write_buffer`.

**You'll need to check the exact signatures** — these functions likely already take `device` and `queue` as parameters (they're called during model init). Verify with:
```bash
grep -n 'fn upload_rope_table\|fn build_norm_bank' ~/airframe/src/backend/bindless/preflight.rs
grep -n 'create_buffer_init' ~/airframe/src/backend/bindless/preflight.rs
```

---

### 3. `pipeline.rs` — 29 inference-time calls

Every `create_buffer_init` in the inference pipeline (dequant params, attention params, etc.) needs the same treatment. There are ~29 of them. They all follow this pattern:

**Before:**
```rust
let foo_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Foo Params"),
    contents: bytemuck::bytes_of(&foo_data),
    usage: wgpu::BufferUsages::UNIFORM,
});
```

**After:**
```rust
let foo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Foo Params"),
    size: std::mem::size_of_val(&foo_data) as u64,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
queue.write_buffer(&foo_buffer, 0, bytemuck::bytes_of(&foo_data));
```

For `cast_slice` variants (arrays):
```rust
// contents: bytemuck::cast_slice(&arr),
// size: (arr.len() * std::mem::size_of::<T>()) as u64,
queue.write_buffer(&buf, 0, bytemuck::cast_slice(&arr));
```

Functions in pipeline.rs that take `device` need `queue` threaded in too. Check the existing function signatures — some may already have `queue`.

---

## Call site update in `shimmy_server_gpu.rs`

After the `load_from_disk` signature change, the call site (line ~329) needs to become:
```rust
let gpu_model = BindlessModel::load_from_disk(&device, &queue, &PathBuf::from(&model_path), Some(&spec));
```

Also check any test files that call `load_from_disk` — but those don't need to compile for the server binary, so they can be stubbed or ignored for now.

---

## How to confirm root cause before patching everything

Add this to `shimmy_server_gpu.rs` right after model loading completes (before the `[HTTP] Async listener spawned` line):

```rust
// Diagnostic: create a tiny test buffer to check if device is still valid
let test_buf = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("device_sanity_check"),
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
eprintln!("[Diag] Device sanity buffer created OK");
drop(test_buf);
```

If the server prints `[Diag] Device sanity buffer created OK`, device is still valid after load → problem is elsewhere.  
If it silently skips the `eprintln!`, device is lost during model loading → `loader.rs` is the confirmed root cause.

---

## Build command

```bash
source ~/.cargo/env
cd ~/airframe
cargo build --release --bin shimmy_server_gpu 2>&1
```

---

## Run command (after successful build)

```bash
killall shimmy_server_gpu 2>/dev/null; sleep 1
LIBSHIMMY_MODEL_PATH=/home/ubuntu/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
SHIMMY_PORT=8080 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
RUST_BACKTRACE=1 \
RUST_LOG=wgpu_core=warn \
nohup ~/airframe/target/release/shimmy_server_gpu > /tmp/server.log 2>&1 &
```

**CRITICAL:** `VK_ICD_FILENAMES` must always be set or wgpu falls back to lavapipe (software renderer, 128 MB limit).

Wait 15–20 seconds for model load, then:
```bash
tail -20 /tmp/server.log
```

Look for `[HTTP] Async listener spawned on 0.0.0.0:8080`.

---

## Test inference

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"temperature":0.0,"stream":false}'
```

Response will be async (queued). Get `job_id` from response, then poll:
```bash
curl -s "http://127.0.0.1:8080/api/repro/job-status?job_id=<JOB_ID>"
```

---

## After TinyLlama works — Gemma-2-2b

Check if model downloaded:
```bash
ls -lh ~/models/
```

If `gemma-2-2b-it-Q4_K_M.gguf` is present (~1.6 GB), restart server with:
```bash
LIBSHIMMY_MODEL_PATH=/home/ubuntu/models/gemma-2-2b-it-Q4_K_M.gguf ...
```

**Note:** Gemma-2-2b has a 2.19 GB output head which exceeds the WebGPU 2 GB buffer limit on Windows/DX12 — that's why we're validating on A100 (Vulkan, no such limit).

---

## Key facts about this codebase version

- Lambda has commit `639895c` ("chore: checkpoint console branch and release plan")
- Local Windows repo is on `openclaw-local-provider` branch — **do NOT overwrite Lambda files with local versions** — the APIs differ (different wgpu PollType signatures, etc.)
- `wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }` — this is the correct API on Lambda's wgpu-27
- `queue.submit([])` returns a `SubmissionIndex` — save it and pass to `PollType::Wait`

---

## Already applied patches (don't re-apply)

1. `load_output_head_f32` in `shimmy_server_gpu.rs` — already uses `create_buffer + queue.write_buffer` (not `create_buffer_init`)
2. `device.poll(PollType::Wait { submission_index: Some(flush_idx), timeout: None })` — already added after GGUF load
3. `load_output_head_f32` call site — already passes `&queue` as 4th argument

---

## Success criteria

- [ ] TinyLlama returns a coherent response to "What is 2+2?"
- [ ] Gemma-2-2b returns a coherent response
- [ ] No panics in `/tmp/server.log`
- [ ] `usage.prompt_tokens` and `usage.completion_tokens` present in response JSON
