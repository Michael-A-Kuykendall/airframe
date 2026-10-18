# Changelog

## [0.1.3] — 2026-10-18

- Preserve EOS, extra-stop and token-limit terminal reasons in ISF state.
- Enforce the output budget before another GPU forward/sample operation.
- Count terminal sampled token IDs consistently without emitting stop text.
- Surface an empty decode-forward result as a generation failure.
- Add CPU-only regression coverage of the production saturation fabric.
