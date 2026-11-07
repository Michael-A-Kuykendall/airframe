# Inference profiling — Airframe 0.4.5

The default `isf` feature includes opt-in scalar timing. With an installed
Shimmy 2.6.6 and an admitted model:

```sh
AIRFRAME_TRACE_TIMING=1 \
AIRFRAME_TRACE_TIMING_PATH=/tmp/airframe-profile.jsonl \
shimmy --kv-quant f32 serve --model-path /path/to/model.gguf
```

No path sends JSONL to stderr; an unusable sink falls back to stderr.
Without profiling enabled, Airframe does not request timestamp features or
allocate profiling report buffers. A build without `isf` reports profiling
unsupported. This does not eliminate unrelated diagnostics or their overhead.

`AIRFRAME_TRACE_TIMING_GPU=0` selects host-only timing. GPU timing requires
optional wgpu TIMESTAMP_QUERY support. Unsupported adapters, invalid samples,
capacity limits and readback timeouts are explicit; missing samples are not
measured zero-cost kernels.

## Accounting contract

The airframe.profile.v1 schema separates runtime loading from generation.
Generation IDs are process-local; starts and terminal summaries identify each
request. Prefill and decode are separate, including one-token prefill.

- compute_pass_count, dispatch_count and queue_submission_count count different
  operations; several dispatches can share a pass or submission.
- host_encode_ns, queue_submit_ns_sum and poll_wait_wall_ns_sum are wall-clock
  measurements. Poll/wait includes GPU execution and is not CPU overhead alone.
- gpu_ns_sum and gpu_measured_calls are per-stage GPU samples; partial coverage
  cannot establish total stage cost.
- profiling_readback_wall_ns_sum, dropped_measurements, counter_overflow and
  contaminated record diagnostic limitations.

Summed stage durations are not GPU wall time. Subtracting them from HTTP latency
does not measure CPU overhead. Timestamp encoding/readback adds work. Compare
capture off, GPU timing and host-only runs with identical source, model hash,
IDs, chunk, precision, backend and machine, separately recording cold and warm
requests. Do not overlap tensor captures or other heavy jobs.

## Release limits

Reference kernels and F32 KV define the validated path. Cooperative QKV,
attention projection, staged/vectorized FFN and cooperative FFN-down flags remain
experimental. Validated admission rejects them; this guide does not instruct
users to enable them or claim a performance fallback is certified.
INT4 and Metal/M4 performance/correctness are **NOT_RUN** for this release.
NVIDIA WSL measurements cannot certify another platform or resolve WSL driver
stability. No kernel, scheduling, prefix-reuse or throughput improvement follows
merely from enabling telemetry.

Timing reports contain scalar counts/timings, not prompt text, token IDs, model
paths, weights or generated text. Other capture/debug facilities have different
data exposure; inspect them separately before sharing logs.
