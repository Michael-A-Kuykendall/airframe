# Contributing to Airframe

Airframe is open source, not open contribution. Unsolicited pull requests are
not accepted. To seek co-maintainer approval, email
michaelallenkuykendall@gmail.com first and follow the maintainer's instructions.

## Approved development

Agree scope, mathematical contracts and reference fixtures before changing
operators, parsing or architecture admission. Preserve existing working paths;
metadata recognition must never silently enable an unsupported executor.

```bash
git clone https://github.com/Michael-A-Kuykendall/airframe
cd airframe
cargo build --locked
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
```

GPU tests require explicitly selected fixtures, token IDs, backend and precision.
Use test-specific requirements; there is no universal environment variable
selecting any GGUF as a sufficient correctness test. Unit-test or build success
does not substitute for full-vector reference comparisons and state checks.

Approved changes use focused feature branches and regression tests. Maintainer
approval is required for public integration, tags and publication. Do not use
release workflows to debug locally reproducible failures.

## License boundaries

The inference runtime is MIT; see [LICENSE](LICENSE). libfse has its
[FSE Noncommercial Evaluation License](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/LICENSE), including separate
permission for commercial use, product embedding and derivatives. Contact the
owner before contributing to it. Existing patent and attribution notices must
be preserved. Observe is MIT with its own pending-patent notice.
