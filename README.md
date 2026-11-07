<div align="center">
  <img src="https://raw.githubusercontent.com/Michael-A-Kuykendall/airframe/main/assets/airframe-logo.png" alt="Airframe" width="480" height="auto" />

  ### Airframe 0.4.5 — Rust/WGSL dense inference engine

  [![Crates.io](https://img.shields.io/crates/v/airframe.svg)](https://crates.io/crates/airframe)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

Airframe is the engine behind [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy).
**Default dependency licensing:** Shimmy → Airframe → libfse. Airframe depends
on libfse 0.1.4 under the FSE Noncommercial Evaluation License. Commercial use
or embedding of the default distribution requires separate permission for
libfse, as do derivative works under its terms. Airframe's MIT license does
not waive those restrictions; the complete default distribution is not
unrestricted open source. See [libfse licensing](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/LICENSE).

The runtime uses Rust, wgpu and WGSL. A backend API being portable does not mean
a model, operating system, driver or GPU has passed numerical certification.

**0.4.5:** shared checked GGUF parsing, validated dense descriptors/execution
plans, typed model admission, RoPE frequency factors, request state isolation,
Q/K normalization ownership and complete tensor-span window planning.
See [CHANGELOG](CHANGELOG.md) and the [workspace roadmap](ROADMAP.md).

## Release contract

Airframe admits validated dense **Llama and Qwen3** plans. Missing required
metadata, unsupported operations, graphs and kernel configurations reject
explicitly. Parsing a quantization or tokenizer does not prove a whole model
can execute correctly. Gemma2, MoE, MLA, recurrent/hybrid, state-space and
multimodal executors are not enabled.

The tested path uses **F32 KV and reference kernels**, with independent
dequantized-FP32 reference logits; maximum absolute error 0.1 and RMS 0.03.
These bounds and matching greedy decisions apply to exact tested fixtures,
not every model carrying the same family name.

| Fixture | Executed scope |
|---|---|
| Llama 3.2 3B Instruct Q4_K_M | context 512 |
| Qwen3-0.6B Q4_K_M | selected context 512, 2048, 4096 and 8192 modes |
| Qwen3-1.7B Q4_K_M | original 2032-token request, context 2048 |

Exact hashes, prompt IDs and mode limitations are in
[Shimmy's fixture matrix](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/SUPPORTED_MODELS.md).
Executed GPU evidence used NVIDIA through **WSL Dozen/Vulkan**. The
device-destruction and active-generation device-loss investigations remain open;
numerical parity is not WSL stability certification.

**NOT_RUN:** Metal/M4, native Windows/macOS/Linux GPU correctness, extended-context
Llama, larger unlisted Qwen fixtures, distinct alternate prefill chunks, INT4 KV
and experimental kernels. Do not infer those results from CI, parsing or
successful compilation.

## Library use

```toml
[dependencies]
airframe = "0.4.5"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use std::path::Path;
use airframe::runtime::gpu::{GpuRuntime, SamplingParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let runtime = GpuRuntime::load(Path::new("path/to/model.gguf")).await?;
    let text = runtime.generate(
        "The capital of France is", &SamplingParams::default(),
        None, None, None, None,
    )?;
    println!("{text}");
    Ok(())
}
```

Use an admitted model with its required metadata and sufficient device limits.
The path above is a placeholder, not a downloaded fixture. Loading does not
certify arbitrary GGUF files. For a running server, use
[Shimmy's quickstart](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/quickstart.md).

## Components and precision

- `core` and the shared ShimmyTok manifest parser validate tensor types, shapes,
  spans and metadata; descriptors and per-layer plans restrict execution.
- `backend/bindless` provides quantized weight decoding, projections, norms,
  RoPE, attention, FFN and sampling support through wgpu/WGSL.
- CPU dequantizers include F32/F16 and Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K.
  Decoder presence is not certification of every tensor layout/model.
- `runtime` manages generation and request-owned state. Repeated results are
  checked on declared configurations; there is no cross-machine/backend
  bit-identical or universal deterministic-output guarantee.
- libfse scans policy bytes/actions; callers must honor violations. Work includes
  matched actions, so it is not unconditionally constant in rule count.
- airframe-observe coordinates inference facts and callbacks using Dzero.
  Observation alone does not establish GPU arithmetic correctness.

## Profiling and experimental code

Opt-in scalar timing is described in [PROFILING](docs/PROFILING.md).
`AIRFRAME_TRACE_TIMING=1` enables reporting; absent GPU timestamps remain
unmeasured rather than zero cost. Timings are specific to the actual adapter.

INT4/TurboShimmy and cooperative/vectorized kernels remain experimental,
uncertified and excluded from the validated 0.4.5 release contract. Validated
admission rejects these configurations. Historical compression, throughput and
quality figures are not current support guarantees. Leave those options disabled.

No startup-latency, context-scaling, benchmark-score, memory-budget or universal
performance claim is made. Old references to unavailable TurboShimmy reports
and artifact directories have been removed rather than presented as evidence.

## Development

```bash
git clone https://github.com/Michael-A-Kuykendall/airframe
cd airframe
cargo build --locked
cargo fmt --check
```

GPU/fixture tests need their exact models and backend. Compilation and generic
unit tests do not certify a new model or platform. See
[CONTRIBUTING](CONTRIBUTING.md), [examples](examples/) and the
[release history](CHANGELOG.md). No hosted release runs or publication occur
without explicit maintainer approval.

## Ecosystem and licensing

Runtime: MIT, see [LICENSE](LICENSE).
The **separately licensed libfse** uses the
[FSE Noncommercial Evaluation License](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/LICENSE); commercial use,
embedding in products and derivative works require separate permission.
Its patent and third-party notices remain intact. airframe-observe is MIT with
its separate pending-patent notice. This README does not waive either notice.

- [libfse policy scanner](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/README.md)
- [airframe-observe facts and observers](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/airframe_observe/README.md)
- [ShimmyTok tokenizer and checked manifest](https://github.com/Michael-A-Kuykendall/shimmytok)
- [ShimmyJinja templates](https://github.com/Michael-A-Kuykendall/shimmyjinja)

## Support

This project is independently maintained. Licensing boundaries above apply to its components. Sponsorship helps fund ongoing development, certification, compatibility work, and releases.

[Sponsor on GitHub](https://github.com/sponsors/Michael-A-Kuykendall) · [Buy Me a Coffee](https://www.buymeacoffee.com/michaelakuykendall)

Any amount is appreciated. Suggested tiers:

- **$5/month** — Thanks
- **$15/month** — Supporter
- **$50/month** — Backer
- **Custom** — [Email for invoicing](mailto:michaelallenkuykendall@gmail.com)

---

## License

Runtime MIT — see [LICENSE](LICENSE).

**Inference runtime** (attention kernels, GGUF loader, quantization, WebGPU backend): MIT; see LICENSE and the separate component terms above.

**FSE subsystem** (`crates/libfse`): separately licensed under [its own terms](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/LICENSE), not covered by the runtime's MIT grant. Commercial use, embedding in products, or creation of derivative works requires a separate commercial license — contact michaelallenkuykendall@gmail.com. The existing pending-patent notice remains in that crate.

---

<sup>Trans rights are human rights. Airframe is built by and for everyone — discrimination has no place in our community or our code.</sup>
