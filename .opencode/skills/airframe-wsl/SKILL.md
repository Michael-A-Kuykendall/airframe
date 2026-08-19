---
name: airframe-wsl
description: "Use when dealing with WSL2 GPU/rendering setup, wgpu adapter selection, Vulkan/Dozen drivers, llvmpipe fallbacks, or any 'why is inference slow / why is it using CPU' question on this machine. Covers the verified WSL2 GPU dependency chain, the InstanceFlags fix, and the exact env vars + driver install that make shimmy/airframe run on the real GPU inside WSL2. Triggers: 'wsl', 'llvmpipe', 'gpu', 'adapter', 'dozen', 'dzn', 'vulkan', 'dxg', 'noncompliant', 'slow inference', 'cpu fallback'."
---

# Airframe/Shimmy on WSL2 — GPU Setup & Diagnosis

> **Canonical copy for this sandbox:**
> `airframe-workspace/.opencode/skills/airframe-wsl/SKILL.md`
> This is the top-level coordination space for both `airframe/` and `shimmy/`.
> A nested copy may exist under `airframe/.opencode/skills/airframe-wsl/`; if they
> disagree, the workspace copy wins.

## Why this skill exists

2026-08-17/18: a multi-session saga where every 4B+ model "ran slowly," the 8B
generate stalled at prefill layer 11, and the stack silently reported
`adapter="llvmpipe (LLVM …, 256 bits)" Cpu`. Multiple wrong hypotheses were
tried (missing drivers, Mesa dzn build from source, version gaps) before the
true root cause was found: **wgpu hides non-compliant Vulkan adapters unless a
flag is set in code.** This skill encodes the VERIFIED solution so the saga is
never repeated.

## The one-sentence answer

**On WSL2, wgpu needs BOTH (a) the Mesa Dozen (dzn) Vulkan driver installed AND
(b) `InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` set in our code (via
`InstanceFlags::default().with_env()`) — plus three env vars — or wgpu silently
falls back to llvmpipe (CPU).**

## The verified dependency chain (WSL2 → GPU)

```
wgpu (Rust, portable)  →  Vulkan (Linux GPU API)
                              ↓
Mesa Dozen (dzn) — Vulkan→D3D12 translator ICD   ← THE MISSING PIECE historically
                              ↓
/dev/dxg (WSL paravirtual GPU) → NVIDIA WSL driver → RTX 3060
```

- NVIDIA's WSL driver exposes the GPU to Linux as **CUDA + D3D12 (DirectML)
  only** — there is NO Linux Vulkan ICD from NVIDIA. Do NOT install a Linux
  NVIDIA driver inside WSL (NVIDIA's own CUDA-on-WSL guide forbids it).
- So to use Vulkan (which wgpu-on-Linux needs), you need **Dozen (dzn)** to
  translate Vulkan → D3D12 → `/dev/dxg`.
- `ubuntu`'s stock `mesa-vulkan-drivers` does **NOT** ship dzn. You need the
  **kisak PPA** (`ppa:kisak/kisak-mesa`) version.

## Verified root cause (the code bug that made it "impossible")

Even with dzn installed, wgpu **hides** non-conformant adapters (dzn is
"not a conformant Vulkan implementation, testing use only"). The gate is in
`wgpu-hal/src/vulkan/adapter.rs`:

```rust
if driver.conformance_version.major == 0 {
    if driver.driver_id == vk::DriverId::MOLTENVK { /* continue */ }
    else if self.shared.flags.contains(wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER) {
        /* continue */   // ← our fix makes this branch run
    } else {
        return None;      // ← dzn was hidden here → llvmpipe wins
    }
}
```

The env var `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` is ONLY honored when
our code calls `InstanceFlags::with_env()`. `InstanceDescriptor::default()`
reads nothing. So the fix is a **code change**, not a driver change.

## The fix (already applied — do not revert)

`InstanceFlags::default().with_env()` on every `wgpu::Instance` we create:

```rust
let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
    flags: wgpu::InstanceFlags::default().with_env(),
    ..Default::default()
});
```

Applied in: `airframe/src/runtime/gpu.rs` (engine) + all `airframe/src/bin/*.rs`
probes (layer_dump_gpu, stack_dump_gpu, quant_verify, decode_gate,
kv_chain_probe, kv_dump_probe, kv_head_probe, shimmy_server_gpu).
**Do NOT regress this to `InstanceDescriptor::default()`.**

## Setup to run on a fresh WSL2 box

Admin session (one time):

```bash
# 1. WSL2 must have the NVIDIA WSL driver (Windows side) — verify:
#    /usr/lib/wsl/lib/nvidia-smi  → should show RTX 3060
# 2. Install Mesa with Dozen via kisak PPA (NOT ubuntu's mesa):
sudo add-apt-repository -y ppa:kisak/kisak-mesa
sudo apt update
sudo apt install -y mesa-vulkan-drivers vulkan-tools
# 3. VERIFY dzn present:
dpkg -L mesa-vulkan-drivers | grep dzn
#    expect: /usr/lib/x86_64-linux-gnu/libvulkan_dzn.so
#            /usr/share/vulkan/icd.d/dzn_icd.json
```

Then for EVERY inference/probe run:

```bash
export WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```

## Verification (does our stack see the GPU?)

```bash
# Should print "Microsoft Direct3D12 (NVIDIA GeForce RTX 3060)" DiscreteGpu
env WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  airframe/target/debug/stack_dump_gpu <tinyllama.gguf> "The capital of France is" /tmp/x.json --top-k 2 \
  2>&1 | grep -i adapter
# PASS = NOT llvmpipe / Cpu. tinyllama output is bit-identical to CPU run
# (top1=3681 "▁Paris", FIRST_NAN_STAGE=none).
```

`vulkaninfo --summary` (vulkan-tools) should show the RTX 3060 under Dozen.

## Known pitfalls (each cost us hours — don't repeat)

1. **`VK_ICD_FILENAMES` pointing ONLY at dzn makes the loader say
   "Found no drivers!"** — dzn's `vkCreateInstance` returns
   `VK_ERROR_INCOMPATIBLE_DRIVER` when forced alone. Let it coexist with
   llvmpipe; wgpu + the flag does the right thing.
2. **`WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` without the CODE fix does
   nothing.** The env var is only read if code calls `with_env()`.
3. **Ubuntu's mesa won't ship dzn** — must use the kisak PPA. Verify with
   `dpkg -L | grep dzn`.
4. **The PPA add can silently fail** (e.g. `udo` typo) leaving Ubuntu's mesa.
   Always re-verify `dpkg -L mesa-vulkan-drivers | grep dzn` AFTER install.
5. **9P filesystem slowness:** repos/models on the Windows side (`/mnt/c/...`)
   accessed from WSL are brutally slow. Keep repos/models on the Linux ext4
   side (e.g. `/home/michael/...`). This is unrelated to the GPU fix.
6. **Hand-building Mesa dzn from source** (meson
   `-Dvulkan-drivers=microsoft-experimental`) produced a broken dzn that failed
   `vkCreateInstance`. Don't hand-build; use the PPA.
7. Do NOT install an NVIDIA Linux driver inside WSL — breaks the WSL stub
   (`/usr/lib/wsl/lib`). The Windows-side driver is the only driver needed.

## OS-agnostic testing matrix

| Venue | wgpu backend | GPU reachable? |
|---|---|---|
| Windows-native | DX12 | ✅ Real GPU |
| Native Linux (real/cloud/DGX) | Vulkan | ✅ Real GPU |
| CI container (GPU runner, nvidia-container-toolkit) | Vulkan | ✅ Real GPU |
| WSL2 (with this fix + kisak dzn) | Vulkan→Dozen→D3D12 | ✅ Real GPU |
| WSL2 without the fix / without dzn | llvmpipe | ❌ CPU only (fine for MATH box) |

## Anti-thrash rules

- **NaN/slowness is NOT "the environment."** First check the adapter line:
  `stack_dump_gpu ... 2>&1 | grep -i adapter`. If `Cpu`/`llvmpipe`, apply THIS
  skill before touching shader or inference code.
- **Do not hand-roll probes, read shader source, or blame the driver** before
  confirming the adapter. The tools print the answer.
- MATH box (quant_verify, layer_dump, PLAN/PEEL) works fine on llvmpipe —
  only the big-model CHAT/inference battery needs the GPU.
