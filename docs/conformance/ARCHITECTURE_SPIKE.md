# Conformance Architecture Spike

Status: proposed corrective architecture, 2026-08-08.

This document is the required pre-implementation decision record for
`airframe-8r6`. It replaces an unverified heuristic in the first CONF-2 draft:
that a raw GGUF reader can derive every tensor's exact byte length from the file.

## Evidence

The GGUF v3 specification defines a tensor directory entry as name, dimension
count, dimensions, `ggml_type`, and a data offset relative to `tensor_data`.
There is no tensor-size field. It requires offsets to be aligned to
`general.alignment`; that metadata value defaults to 32 and must be a multiple
of 8, not necessarily a power of two.

The physical bytes per tensor require `ggml_blck_size(type)` and
`ggml_type_size(type)`. Those values live in the GGML type-traits registry, not
in a GGUF file. Therefore a raw reader can prove directory syntax, offsets,
alignment, and bounded storage ranges, but cannot prove an exact quantized
payload length or physical overlap without a separate type-layout authority.

Pinned external sources for this decision:

- GGUF v3 structural format: `ggml` commit
  `30bf8685ed4eb0a47f2b06229543327749904150`,
  `docs/gguf.md`.
- GGML physical layout source: the same commit, `src/ggml.c`,
  `type_traits`, `ggml_blck_size`, and `ggml_type_size`.
- Local canonical supported-layout authority:
  `airframe_observe::quant_formula::QUANT_FORMULAS`. It currently covers type
  IDs `0`, `1`, `2`, `6`, `8`, `12`, `13`, and `14` only.

## Decision

### CONF-2 owns raw evidence only

The independent reader will parse the GGUF header, all metadata values, exact
tensor-directory bytes, and raw directory entries without importing Airframe's
loader, metadata projection, offset calculation, dispatch, cache, or inference
modules.

Its descriptor must retain:

- the original directory order;
- name, raw type ID, shape, and file-relative data offset;
- `tensor_data_start` and file length;
- a `storage_upper_bound`, calculated from the next higher declared offset or
  the file end; and
- byte-exact hashes for the full file and the raw tensor-directory region.

It must not expose `payload_size`, `row_size`, or `element_size` as though those
facts came from GGUF. An unknown type ID remains an opaque raw directory value,
not a CONF-2 parse failure.

### CONF-5 owns quant layout and exact spans

CONF-5 will use the canonical `quant_formula` registry for each supported type
to calculate exact payload bytes, validate the physical span against the
CONF-2 storage upper bound, and report unsupported types explicitly as red or
unsupported evidence. That is where exact size, row/block boundaries, and
physical-overlap validation belong.

This preserves independence from the production loader while obeying the
workspace rule that `quant_formula` is the math authority. It does not create
a duplicate hand-maintained layout table. CONF-0 must still decide whether the
current `airframe-observe` crate is narrow enough to be an allowed spec
dependency, or whether its registry needs extraction into a smaller neutral
spec crate.

### Capture protocol boundary is unresolved and blocks CONF-1

The committed CONF-1 put capture protocol types inside the test-only evaluator
crate while documenting that production Airframe emits them. That direction is
not executable: production code cannot depend on a test-only evaluator without
reversing the intended boundary. CONF-0 must decide the neutral protocol/schema
location before CONF-1 is reimplemented.

## Invalidated Work

The uncommitted `raw_gguf` draft is not acceptance evidence and is not a
baseline to salvage mechanically:

- it hand-codes GGML block/type sizes;
- it treats a calculated value as a raw GGUF tensor size;
- it rejects alignment as non-power-of-two despite the spec allowing any
  multiple of 8;
- it hashes reconstructed, name-sorted descriptors rather than exact directory
  bytes;
- it has no reader implementation and does not compile; and
- its views use unchecked offset/length arithmetic and unbounded allocations.

The committed `f3d9a51` CONF-1 implementation is also quarantined pending
re-audit: its dependency fixture comments out the forbidden imports, its test
only searches text, its claimed allowed production capture modules do not
exist, and the test-only crate is publishable by default.

## Replanned Bead Boundaries

1. **CONF-0, new spike:** establish the neutral protocol location, permitted
   specification dependency, raw-vs-derived ownership, and executable policy
   gates. It blocks CONF-1 and CONF-2.
2. **CONF-1, reopened:** create the approved neutral protocol/schema boundary,
   make the evaluator private (`publish = false`), and enforce imports using
   Cargo metadata plus source-policy tests that reject an actual forbidden
   fixture.
3. **CONF-2, reopened:** parse only raw GGUF structure and produce stable,
   byte-exact identity evidence. Structural malformed-input tests cover magic,
   v2/v3 layout, metadata/value bounds, dimensions, alignment, duplicate or
   out-of-range offsets, and integer overflow.
4. **CONF-5, amended:** introduce exact supported quant spans and dequant
   coverage through the approved spec registry. Unsupported IDs must remain
   visible evidence, not parser omissions.
5. **CONF-3 onward:** continue only after the revised CONF-1/CONF-2 gates are
   green. The existing dependency graph then remains meaningful.

## Required Gates Before Code Resumes

1. The architecture checker verifies the source pins and all sections in this
   document.
2. The bead graph shows CONF-0 blocking both reopened implementation beads.
3. A dependency-policy test rejects an uncommented fixture importing a
   forbidden production implementation.
4. The raw-reader API has no quant-layout or payload-size calculation.
5. The implementation branch is based on the correct conformance base, not a
   `2z6` regression branch.
