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

The inference runtime and Observe are MIT. libfse is Apache-2.0, including the [additional grant for the existing 0.1.4 archive](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md). Commercial use, embedding, modification and redistribution are permitted subject to the applicable terms, without separate commercial permission. Preserve third-party attribution and applicable copyright/patent notices; see [stack licensing](LICENSING.md). Contributor admission and maintainer approval requirements above remain unchanged.
