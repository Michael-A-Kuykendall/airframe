# Changelog

## [0.1.4] — 2026-11-06

- Documentation errata: describe Dzero 0.1.0 fact coordination and observer callbacks accurately; distinguish observation from GPU numerical correctness and performance. Correct crate-root cost/signature prose without changing APIs or executable code.

- Include the packaged MIT license document matching the existing declaration.


- Package the observation contracts used by the certified dense foundation under a distinct version; retain Dzero 0.1.0.
- No timing/performance improvement or Metal certification is claimed.


## [0.1.3] — 2026-10-18

- Preserve EOS, extra-stop and token-limit terminal reasons in ISF state.
- Enforce the output budget before another GPU forward/sample operation.
- Count terminal sampled token IDs consistently without emitting stop text.
- Surface an empty decode-forward result as a generation failure.
- Add CPU-only regression coverage of the production saturation fabric.
