# Tiled GEMM Implementation Plan

## Problem
All matrix multiply kernels in `sh_layer_v1.wgsl` (and `sh_vit_layer.wgsl`) use scalar
row-per-thread GEMM — one thread computes one output element by serially looping over the
entire input dimension, with non-coalesced memory access. Correct results, ~5% GPU bandwidth
efficiency. Becomes a hard crash (Windows TDR watchdog) when vision batch sizes push a single
kernel past 2 seconds.

## Two branches of work

### Branch 1 — `fix/tiled-gemm-core` (public, ships first)
Rewrite the four hot entry points in `sh_layer_v1.wgsl`:
- `main_ffn_proj`
- `main_ffn_down`
- `main_attn_proj`
- `main_qkv`

**Gate condition:** Inside each entry point, branch on `params.batch_size`:
- `batch_size <= 32` → existing scalar path (all existing SHA baselines unaffected)
- `batch_size > 32` → new tiled path

This means the existing model test suite never touches new code. Zero SHA baseline churn.

### Branch 2 — vision feature branch (private, ships behind flag)
Same tiled rewrite for `sh_vit_layer.wgsl`:
- `main_vit_qkv`
- `main_vit_ffn_up`
- `main_vit_ffn_down`
- `layernorm_one_token` → cooperative workgroup reduction (one workgroup per token, not one thread)

---

## Tiled GEMM design — FSE-aligned Q4_K

### Why TILE_K = 256
Each Q4_K superblock is exactly 256 elements with shared `d` (scale) and `dmin` (min) across
all 256. Under FSE: `d` and `dmin` are **broadcast values** — extracted once into workgroup
shared memory, cost ≈ 0 per element. TILE_K = 256 aligns the tile boundary to the superblock
boundary exactly. Any other tile size splits a superblock across tiles and loses the broadcast.

### Workgroup shared memory layout
```wgsl
const TILE_K: u32 = 256u;
const TILE_N: u32 = 8u;   // output rows per workgroup

var<workgroup> tile_act:    array<f32, TILE_K>;    // activation slice — shared selector
var<workgroup> block_scale: array<f32, 8u>;        // 8 sub-block scales — broadcast
var<workgroup> block_min:   array<f32, 8u>;        // 8 sub-block mins — broadcast
var<workgroup> partial:     array<f32, TILE_N>;    // one accumulator per output row
```

### Execution per workgroup
1. Thread 0 reads superblock `d`, `dmin`, all 8 `get_scale_min_k4` values → shared memory
2. `workgroupBarrier()`
3. All 256 threads cooperatively load activation tile (thread i loads activation[k_base + i])
4. `workgroupBarrier()`
5. Each thread i (i < TILE_N) loops over 256 activation elements, reads nibble from weight,
   reconstructs value using shared scale/min, accumulates into `partial[i]`
6. `workgroupBarrier()`
7. Thread i writes `partial[i]` to output

### Dispatch shape change
Old: `dispatch(ceil(n_rows/256), batch, 1)` — one thread per output row
New: `dispatch(ceil(n_rows/TILE_N), ceil(n_tiles_k/1), batch)` — one workgroup per output tile

No changes to Rust, no changes to bind groups, no changes to call sites in `inference.rs`.

---

## Fast-fail math test (no GGUF, no GPU)

File: `tests/tiled_gemm_math.rs`

### Test 1 — `q4k_dequant_scalar_analytic`
Construct a minimal Q4_K superblock by hand (144 bytes):
- Set `d = 1.0` (f16), `dmin = 0.0` (f16)
- Set sub-block scales: all 8 pairs = (sc=1, m=0) via `get_scale_min_k4` encoding
- Set nibbles: element 0 = 3, element 1 = 7, element 255 = 15
- Expected dequant: element 0 = 3.0, element 1 = 7.0, element 255 = 15.0

Run through the Rust port of `dequant_q4k_elem`, assert exact values.

### Test 2 — `tiled_matmul_matches_scalar`
- 8 output rows × 256 inputs (one superblock per row)
- Known activation vector: `x[i] = f32(i) / 256.0`
- Random but deterministic nibble pattern (fixed seed)
- Scalar path: compute expected dot products
- Tiled path: same inputs, tiled accumulation
- Assert all 8 outputs match within `1e-4`

---

## Q4_K block layout reference (from `sh_layer_v1.wgsl`)
```
Offset  Size    Field
0       2       d        (fp16 super-scale)
2       2       dmin     (fp16 super-min)
4       12      scales   (6-bit packed, 8 sub-block sc + 8 sub-block m)
16      128     qs       (4-bit nibbles, 256 elements)
Total:  144 bytes
```

Sub-block j (j = 0..7), each covers 32 elements:
- Elements 0..31 of group g: sub-block `2g`, nibble = low nibble of qs byte
- Elements 32..63 of group g: sub-block `2g+1`, nibble = high nibble of qs byte

---

## Effort and risk
- Fibonacci: 13
- Confidence: 7/10
- Regression risk on existing SHA baselines: 0/10 (gated, never reached at batch≤32)
- Vision SHA baselines: generated fresh when vision ships
