# Contributing to Airframe

Thank you for your interest in contributing. Airframe is the GPU inference engine behind [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy) and is designed to be a clean, minimal, dependency-free Rust LLM runtime.

## Scope

Contributions are welcome in the following areas:

- **New model architectures** — additional transformer families (e.g., Gemma2, LLaMA 3.3, DeepSeek-V2)
- **Quantization types** — additional GGUF quant formats (Q5_K_M, IQ4_NL, etc.)
- **WGSL shader optimizations** — improved GPU throughput, reduced memory bandwidth
- **Evaluation / benchmark scripts** — additional accuracy benchmarks (HumanEval, MMLU, etc.)
- **Bug fixes and correctness improvements**
- **Documentation improvements**

**Not in scope for external contributions**: the `crates/libfse` policy engine is patent-pending. Please do not submit PRs that modify the FSE internals.

## Getting Started

```bash
git clone https://github.com/Michael-A-Kuykendall/airframe
cd airframe
cargo build
cargo test
```

For GPU-dependent tests and examples, set:

```bash
export LIBSHIMMY_MODEL_PATH=/path/to/any.gguf
cargo run --example simple_flight
```

## Code Style

- `cargo fmt` before committing
- `cargo clippy -- -D warnings` must pass
- Keep `unsafe` blocks out of non-GPU paths
- No new runtime dependencies without discussion

## Pull Request Process

1. Open an issue first for significant changes
2. Branch from `master`
3. Include a test or benchmark that validates your change
4. Reference any related issues in the PR description

## Patent Notice

The Fused Semantic Execution (FSE) subsystem in `crates/libfse` is covered by a pending US patent. Any contribution to that crate requires a contributor license agreement (CLA) to be executed separately. Contact michaelallenkuykendall@gmail.com before contributing to libfse.

All other code in this repository is MIT-licensed.
