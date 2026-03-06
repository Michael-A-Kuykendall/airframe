# LibFSE: Strategic Plan (Proprietary Engine)

**Status:** CONFIDENTIAL
**License:** PROPRIETARY / EMBEDDED ENGINE
**Role:** Kernel-level enforcement infrastructure.

## 1. Core Mandate
`libfse` is **NOT** a utility library. It is a **patented execution kernel**. 
It exists to make `libshimmy` and `AI State Pilot` fail-closed and audit-proof.
It will **never** be published to crates.io.

## 2. Usage Model
-   **Internal Use Only:** We link it statically into our binaries (`shimmy`, `fse-cli`, `asp-agent`).
-   **No Public API:** The crate documentation is for *our* team only.
-   **Interface:** Exposes a `Policy Compiler` (takes JSON rules) and an `Execution Engine` (takes byte stream).

## 3. Immediate Action Plan

### A. Testing & Verification (Now)
1.  **Unit Tests:** Verify standard matching behavior works (Aho compatibility).
2.  **Safety Tests:** Verify `Reject` opcode stops immediately (Fail-Closed).
3.  **Overlap Tests:** Verify multiple rules on one token trigger correctly (Policy Fusion).
4.  **Zero-Alloc Verification:** Ensure `scan()` loop uses stable memory.

### B. Integration (Next)
1.  **FSE-CLI:** Build a tiny binary to benchmark `libfse` against `grep`.
2.  **Shimmy Hooks:** Wire `FseScanner` into `InferenceControl`.

---

## 4. Development Log

### Current State
- [x] `lib.rs` (Types)
- [x] `store.rs` (Compiler/Ops)
- [x] `scanner.rs` (Runtime)

### Next Steps
- [ ] Create `tests/integration_tests.rs`
- [ ] Implement `FseScanner::scan` unit tests in `src/scanner.rs`
