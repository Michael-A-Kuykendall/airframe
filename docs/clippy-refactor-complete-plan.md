# Clippy Refactor - Complete Execution Plan
**Date**: June 7, 2026  
**Current State**: 78 warnings (18 source + 60 test/bin warnings)  
**Target**: 0 warnings  
**Confidence Goal**: >90%  

---

## Executive Summary

The June 6 clippy fixes created temporary low-warning state, but were then **undone** by the Q4_K merge (`consolidate-fixes`) and subsequent `feat/starcoder-triage` work. This branch introduced:
- 68 new warnings
- Removal of `LayerDebugOutput` struct (reverted to tuple returns)
- New high-argument-count functions from Q4_K mixed-quantization support

**Root Cause**: The June 6 refactoring established good patterns (parameter structs), but the Q4_K feature merge reverted them and added more parameters. The current 78 warnings are a mix of:
- 28 "new code" warnings from Q4_K and starcoder-triage work
- 50 "mechanical/annoyance" warnings that can be auto-fixed

**Solution Strategy**: Follow the pattern established in June 6, but do it **completely** and **permanently** with a plan that:
1. Creates a robust parameter struct architecture
2. Applies it to ALL affected files systematically
3. Uses type aliases for complex types
4. Fixes remaining mechanical issues
5. Adds tests to verify the new patterns

---

## Phase 0: Pre-Work (15 minutes) - Establish Baseline

### 0.1 Git State Check
```bash
# Create backup branch
git checkout -b backup/clippy-refactor-2026-06-07

# Verify current state
cargo clippy --all-targets 2>&1 | tee /tmp/clippy_before.txt
grep -c "warning:" /tmp/clippy_before.txt  # Should show 78

# Document current warning distribution
cargo clippy --all-targets 2>&1 | grep "warning:" | grep -v "^warning:" > /tmp/warnings_raw.txt
```

### 0.2 Identify All Files with Warnings
```bash
# Get unique files with warnings
grep "^[^w]" /tmp/warnings_raw.txt | sed -E 's#^([^:]+):.*#\1#' | sort | uniq
```

Expected files (based on analysis):
1. `src/backend/bindless/pipeline/layer.rs` (6 warnings)
2. `src/backend/bindless/pipeline/inference.rs` (6 warnings)
3. `src/backend/bindless/pipeline/matmul.rs` (1 warning)
4. `src/backend/bindless/pipeline_shift.rs` (1 warning)
5. `src/bin/shimmy_server_gpu/server_inference.rs` (2 warnings)
6. `src/bin/shimmy_server_gpu.rs` (5 warnings)
7. `src/core/f16.rs` (2 warnings)
8. `src/core/ggml_types.rs` (1 warning)
9. `src/ops/dispatch.rs` (1 warning)
10. `src/math_bypass_control.rs` (1 warning)
11. `src/ops/reference/rope.rs` (1 warning)
12. `src/runtime/gpu.rs` (1 warning)
13. `src/backend/bindless/preflight.rs` (1 warning)
14. `tests/` files (remaining warnings)

---

## Phase 1: Create Infrastructure (30 minutes)

### 1.1 Create Parameter Struct Module
**File**: `src/backend/bindless/context.rs`

```rust
//! WebGPU execution context and parameter structures
//! Follows FSE principles: reduces O(N×M) to O(M) by batching parameter access

use super::loader::BindlessModel;
use crate::core::routing::ModelSpec;
use crate::backend::kv_cache::KVCache;
use crate::backend::metadata::{LayerOffsets, LayerParams};
use wgpu::Device;
use wgpu::Queue;

// ==================== Context Structs ====================

/// WebGPU execution context - eliminates device/queue/model parameter repetition
pub struct GpuExecutionContext<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub model: &'a BindlessModel,
}

/// Inference input parameters
pub struct InferenceInput<'a> {
    pub embeddings: &'a [f32],
    pub head_weights_override: Option<&'a wgpu::Buffer>,
}

/// Cache state parameters
pub struct CacheState<'a> {
    pub current_pos: u32,
    pub seq_len: u32,
    pub kv_state: Option<(&'a [wgpu::Buffer], &'a [wgpu::Buffer])>,
}

/// Layer execution parameters
pub struct LayerExecutionContext<'a> {
    pub kv_cache: &'a mut KVCache,
    pub layer_idx: usize,
    pub input: &'a [f32],
    pub offsets: LayerOffsets,
    pub params: LayerParams,
}

// ==================== Type Aliases ====================

/// Result type for layer processing
pub type LayerResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Complex iterator type for layer processing
pub type LayerIterator<'a, T> = Box<dyn Iterator<Item = Result<T, String>> + 'a>;

/// GPU buffer slice
pub type GpuBufferSlice<'a> = &'a [wgpu::Buffer];

/// Inference closure type
pub type InferenceClosure<'a> = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync + 'a>;
```

### 1.2 Update Module Exports
**File**: `src/backend/bindless/mod.rs`

Add:
```rust
pub mod context;
pub use context::*;
```

---

## Phase 2: Refactor Inference Pipeline (45 minutes)

### 2.1 Update inference.rs
**File**: `src/backend/bindless/pipeline/inference.rs`

#### 2.1.1 Current State (6 warnings)
- `run_full_model_with_cache_state`: 10 arguments (line 136)
- `run_full_model_prefill_chunked_with_cache_state`: 9 arguments (line 56)
- Complex types at lines 67, 147
- Loop counter at line 98

#### 2.1.2 Target State (0 warnings)
```rust
// Change signature from:
pub fn run_full_model_with_cache_state(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    input_embd: &[f32],
    head_weights_override: Option<&wgpu::Buffer>,
    current_pos: u32,
    kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
    spec: &ModelSpec,
    chunk_tokens: u32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>

// To:
pub fn run_full_model_with_cache_state(
    &self,
    ctx: &GpuExecutionContext,
    input: &InferenceInput,
    cache: &CacheState,
    spec: &ModelSpec,
    chunk_tokens: u32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>
```

### 2.2 Update layer.rs
**File**: `src/backend/bindless/pipeline/layer.rs`

#### 2.2.1 Current State (6 warnings)
- `run_layer_stepwise_test`: 8 args (line 211)
- `run_layer_with_cache`: 9 args (line 519)
- `run_layer_with_cache_int4`: 9 args (line 801)
- `requantize_all_kv_int4`: 8 args (line 970)
- `run_layer_with_cache_debug`: 9 args (line 1027)

#### 2.2.2 Target State (0 warnings)
```rust
// Change from:
pub fn run_layer_with_cache(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    input: &[f32],
    offsets: LayerOffsets,
    params: LayerParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>)

// To:
pub fn run_layer_with_cache(
    &self,
    ctx: &GpuExecutionContext,
    layer_ctx: &LayerExecutionContext,
) -> (Vec<f32>, Vec<f32>, Vec<f32>)
```

### 2.3 Update matmul.rs
**File**: `src/backend/bindless/pipeline/matmul.rs`

#### 2.3.1 Current State (1 warning)
- `run_lm_head_blob`: 10 args (line 220)

#### 2.3.2 Target State (0 warnings)
```rust
// Change from:
pub fn run_lm_head_blob(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    input: &[f32],
    head_weights: &wgpu::Buffer,
    head_weights_override: Option<&wgpu::Buffer>,
    spec: &ModelSpec,
) -> Vec<f32>

// To:
pub fn run_lm_head_blob(
    &self,
    ctx: &GpuExecutionContext,
    input: &InferenceInput,
    head_weights: &wgpu::Buffer,
    spec: &ModelSpec,
) -> Vec<f32>
```

### 2.4 Update pipeline_shift.rs
**File**: `src/backend/bindless/pipeline_shift.rs`

#### 2.4.1 Current State (1 warning)
- `execute`: 13 args (line 162)

#### 2.4.2 Target State (0 warnings)
```rust
// Change from:
pub fn execute(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    input: &[f32],
    offsets: LayerOffsets,
    params: LayerParams,
    spec: &ModelSpec,
    current_pos: u32,
    chunk_tokens: u32,
    kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
    layer_debug: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>

// To:
pub fn execute(
    &self,
    ctx: &GpuExecutionContext,
    layer_ctx: &LayerExecutionContext,
    spec: &ModelSpec,
    cache: &CacheState,
    debug: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>
```

---

## Phase 3: Update Server Inference (30 minutes)

### 3.1 Update server_inference.rs
**File**: `src/bin/shimmy_server_gpu/server_inference.rs`

#### 3.1.1 Current State (2 warnings)
- `build_layer_trace`: 11 args (line 219)
- `process_inference_job`: 13 args (line 1345)

#### 3.1.2 Target State (0 warnings)
```rust
// Create parameter structs in server_inference.rs or shared context
pub struct LayerTraceParams<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub model: &'a BindlessModel,
    pub layer_idx: usize,
    pub input: &'a [f32],
    pub offsets: LayerOffsets,
    pub params: LayerParams,
    pub cache: &'a mut KVCache,
    pub spec: &'a ModelSpec,
    pub debug: bool,
}

// Then change signature
pub fn build_layer_trace(
    params: &LayerTraceParams,
) -> LayerTraceOutput

// Similarly for process_inference_job
```

### 3.2 Update shimmy_server_gpu.rs
**File**: `src/bin/shimmy_server_gpu.rs`

#### 3.2.1 Current State (5 warnings)
- Various functions with too many args
- Identical if blocks

#### 3.2.2 Target State (0 warnings)
Apply same pattern - create parameter structs.

---

## Phase 4: Update Supporting Files (20 minutes)

### 4.1 Update rope.rs
**File**: `src/ops/reference/rope.rs`

#### 4.1.1 Current State (1 warning)
- Function with 8 args

#### 4.1.2 Target State
Create `RopeParams` struct following June 6 pattern.

### 4.2 Update gpu.rs
**File**: `src/runtime/gpu.rs`

#### 4.2.1 Current State (1 warning)
- Complex type at line 264

#### 4.2.2 Target State
Add type alias in context module.

### 4.3 Update preflight.rs
**File**: `src/backend/bindless/preflight.rs`

#### 4.3.1 Current State (1 warning)
- Loop variable issue

#### 4.3.2 Target State
Use `.enumerate()` pattern.

### 4.4 Update core files
**Files**: `src/core/f16.rs`, `src/core/ggml_types.rs`, `src/ops/dispatch.rs`, `src/math_bypass_control.rs`

#### 4.4.1 Current State (6 warnings)
- Excessive float precision (2)
- `len() == 0` instead of `is_empty()` (1)
- `default()` for unit struct (1)
- Complex type (1)
- Unused variable (1)

#### 4.4.2 Target State
Fix each individually with mechanical fixes.

---

## Phase 5: Test Code Cleanup (30 minutes)

### 5.1 Create Automated Fix Script
**File**: `scripts/fix_clippy_mechanical.ps1`

```powershell
# Replace vec![] with Vec::new() where appropriate
Get-ChildItem tests\*.rs -Recurse | ForEach-Object {
    (Get-Content $_) -replace 'vec!\[\]', 'Vec::new()' | Set-Content $_
}

# Fix is_empty() pattern
# Fix loop counter patterns
```

### 5.2 Run Mechanical Fixes
```bash
# Use clippy's built-in fix
cargo clippy --all-targets --fix --allow-dirty
```

### 5.3 Manual Test Fixes
- Remove dead code (`DeterminismProof`, `AirframeParityArtifact`, `get_tinyllama_spec`)
- Fix doc formatting
- Fix redundant operations

---

## Phase 6: Verification & Testing (30 minutes)

### 6.1 Verify Warning Count
```bash
cargo clippy --all-targets 2>&1 | tee /tmp/clippy_after.txt
grep -c "warning:" /tmp/clippy_after.txt  # Should show 0
```

### 6.2 Run Tests
```bash
cargo test --all-targets 2>&1 | tee /tmp/test_results.txt
```

### 6.3 Run Smoke Tests
```powershell
.\scripts\model_smoke_test.ps1 -IncludeLarge
```

### 6.4 Performance Check
Run inference on a small model to verify no regression:
```bash
cargo run --bin shimmy_eval -- --model path/to/model --prompt "test"
```

---

## Phase 7: Documentation & Guardrails (15 minutes)

### 7.1 Update Code Comments
Add to `src/backend/bindless/context.rs`:
```rust
//! FSE Pattern Implementation
//! 
//! This module establishes parameter structs to avoid O(N×M) parameter
//! explosion where N=functions, M=shared parameters.
//! 
//! Before: 10+ functions × 7 common params = 70+ parameter slots
//! After: 10 functions × 1 context struct = 10 parameter slots
```

### 7.2 Add CI Guard
Add to CI workflow:
```yaml
- name: Check Clippy
  run: cargo clippy --all-targets -- --deny warnings
```

### 7.3 Update CONTRIBUTING.md
Document the parameter struct pattern for future development.

---

## Expected Results

### Before
- 78 warnings (18 source + 60 test)
- High-argument-count functions throughout
- Complex type signatures
- Duplicate parameter patterns

### After
- 0 warnings
- All functions ≤7 arguments
- Type aliases for complex types
- Consistent context structure

### Time Budget
- Phase 0: 15 min
- Phase 1: 30 min
- Phase 2: 45 min
- Phase 3: 30 min
- Phase 4: 20 min
- Phase 5: 30 min
- Phase 6: 30 min
- Phase 7: 15 min
- **Total**: 215 minutes (3.6 hours)

### Confidence Triggers
1. **>90% confidence**: When all functions show ≤7 arguments in clippy output
2. **>95% confidence**: When all tests pass
3. **100% confidence**: When smoke tests show no regression

---

## Risk Mitigation

### Safety Measures
1. **Backup branch**: Created in Phase 0
2. **Incremental commits**: Each phase commits changes
3. **Test after each file**: Verify no regressions
4. **Revert plan**: `git reset --hard backup/clippy-refactor-2026-06-07`

### Rollback Strategy
If any phase fails:
1. `git log` to identify last good commit
2. `git reset --hard <last_good_commit>`
3. Re-evaluate approach

---

## Success Criteria

### Primary
- [ ] `cargo clippy --all-targets` returns 0 warnings
- [ ] No new warnings introduced

### Secondary
- [ ] All existing tests pass
- [ ] Smoke tests show no regression
- [ ] Inference latency unchanged

### Tertiary
- [ ] Code is more maintainable (parameter structs)
- [ ] Documentation updated
- [ ] CI guard added

---

## Next Steps

1. **Review this plan** - does it cover all 78 warnings?
2. **Execute Phase 0** - establish baseline
3. **Begin Phase 1** - create infrastructure

**Current Confidence**: 75% (down from 80% after learning June 6 work was undone)

**Confidence After Full Analysis**: 85% (plan is comprehensive and follows proven patterns)

**Confidence After Phase 1 Complete**: 90%+ (infrastructure established, patterns validated)

---

## Final Confidence Assessment

### Current Confidence: 88%

**Why 88%?**

**Factors Increasing Confidence**:
1. ✅ Pattern proven in June 6 work (`99469f8` - reduced 41 warnings to 10)
2. ✅ June 6 used exact same approach (parameter structs, type aliases)
3. ✅ All warnings are well-understood patterns from June 6
4. ✅ Plan follows FSE principles (one pass over shared patterns)
5. ✅ 80% of warnings are mechanical fixes or auto-fixable

**Factors Reducing Confidence**:
1. ⚠️ June 6 work was partially undone (LayerDebugOutput removed)
2. ⚠️ 68 new warnings added by Q4_K merge - need to verify patterns match
3. ⚠️ Some tests may have intentional patterns that clippy doesn't like
4. ⚠️ 13-argument function on `process_inference_job` may need creative solution

**Risk Areas**:
1. `process_inference_job` (13 args) - may need to split into multiple functions
2. `execute` (13 args) in pipeline_shift - same concern
3. Some test patterns may require more creative fixes

**What Would Get Me to 95%+**:
1. ✅ Running clippy with `--fix` to auto-fix mechanical issues first
2. ✅ Testing that parameter struct changes don't break existing callers
3. ✅ Verifying no regression in smoke tests

**What Would Get Me to 100%**:
1. ✅ Full test suite passes
2. ✅ Smoke tests show no regression
3. ✅ CI passes without warnings

### Confidence Triggers (To Reach 95%)

| Trigger | Action | Expected Outcome |
|---------|--------|------------------|
| Auto-fix completes | Run `cargo clippy --fix --all-targets` | 30-40 warnings removed |
| Phase 1 complete | Create context.rs with structs | 5-10 warnings reduced |
| Phase 2 complete | Refactor inference, layer, matmul | 15-20 warnings reduced |
| Phase 3 complete | Refactor server files | 8-10 warnings reduced |
| Phase 4 complete | Fix remaining core files | 5-7 warnings reduced |
| Phase 5 complete | Test code cleanup | 10-15 warnings reduced |
| Final verification | `cargo clippy --all-targets` | 0 warnings |

### Plan Validation

**This plan will succeed if**:
1. ✅ June 6 pattern is still valid (yes - it was just undone by Q4_K merge)
2. ✅ New code uses same parameter patterns (yes - all high-arg functions show same patterns)
3. ✅ Clippy `--fix` can handle mechanical issues (yes - 30+ warnings are auto-fixable)
4. ✅ No logic changes required (yes - we're just refactoring signatures)

**This plan may fail if**:
1. ❌ Some files have unique patterns not in June 6 (low risk - patterns are standard)
2. ❌ Parameter structs cause build failures (low risk - structural change only)
3. ❌ Test expectations break (medium risk - need to verify test expectations)

**Mitigation**: Run `cargo build --all-targets` after each phase to catch breaking changes early.

---

## Appendix: Complete Warning Inventory

### Source Files with Warnings

| File | Warning Count | Type |
|------|---------------|------|
| `src/ops/reference/rope.rs` | 1 | too_many_arguments (8/7) |
| `src/runtime/gpu.rs` | 1 | complex_type |
| `src/backend/bindless/pipeline/inference.rs` | 6 | too_many_args (9,10,10), complex_type (3) |
| `src/backend/bindless/pipeline/layer.rs` | 6 | too_many_args (8,9,9,8,9) |
| `src/backend/bindless/pipeline/matmul.rs` | 1 | too_many_args (10/7) |
| `src/backend/bindless/pipeline_shift.rs` | 1 | too_many_args (13/7) |
| `src/backend/bindless/preflight.rs` | 1 | loop_counter |
| `src/math_bypass_control.rs` | 1 | complex_type |
| `src/bin/shimmy_server_gpu.rs` | 5 | too_many_args (8), vec!, if identical |
| `src/bin/shimmy_server_gpu/server_inference.rs` | 2 | too_many_args (11,13) |
| `src/bin/frontier_compare.rs` | 2 | loop_counter, too_many_args (8) |
| `src/core/f16.rs` | 2 | excessive_precision |
| `src/core/ggml_types.rs` | 1 | len_zero |
| `src/ops/dispatch.rs` | 1 | default_unit |
| `src/backend/bindless/tests.rs` | 1 | unused_function |

### Test Files with Warnings

| File | Warning Count | Types |
|------|---------------|------|
| `tests/gpu_22layer_verify.rs` | 7 | vec!, unused Result (3), expect calls (3), if identical |
| `tests/math_pack_detection.rs` | 2 | doc_indent |
| `tests/parity.rs` | 2 | unused_struct (2) |
| `tests/spike_two_position_cache.rs` | 1 | unused_const |
| `tests/attention_f6_f7_verify.rs` | 2 | vec!, unused layer_output |
| `tests/template_pipeline.rs` | 1 | unnecessary_op |
| `tests/layer1_attention_forensics.rs` | 1 | unused Result |
| `tests/ffn_f8_verify.rs` | 2 | vec!, unused layer_output |
| `tests/generate_22layer_oracle.rs` | 2 | default_unit, unused enumerate |
| `tests/debug_layer1_offsets.rs` | 3 | operation_no_effect (3) |
| `tests/verify_norm_bank_extraction.rs` | 1 | operation_no_effect |
| `tests/tiled_gemm_math.rs` | 3 | loop_counter (3) |
| `tests/debug_weight_loading_layer1.rs` | 1 | unused var |

### Summary Lines (Not Actual Warnings)

| Crate | Count | Note |
|-------|-------|------|
| `airframe (lib)` | 1 | Aggregate summary |
| `airframe (lib test)` | 1 | Aggregate summary |
| `airframe (test *)` | 11 | Test summaries |
| `airframe (bin *)` | 6 | Bin summaries |

**Total Actual Warnings**: 59 (78 - 19 summary lines)

---

## Final Confidence: 90%+

### Why 90%+?

**Confirmed Pattern Match**:
- June 6 work (`99469f8`) reduced warnings from 41 to 10 using exactly the same approach
- Current high-argument-count functions use the same parameter patterns as June 6
- All 17 `too_many_arguments` warnings follow the same `(device, queue, model, ...)` pattern
- All 5 `complex_type` warnings are WebGPU buffer/iterator patterns June 6 already solved

**Automated Fix Potential**:
- 30+ warnings are auto-fixable with `cargo clippy --fix`
- 20+ warnings are mechanical fixes (vec!, is_empty, etc.)
- 9 warnings are unused variables/constants (easy to fix or remove)

**Pattern Validity**:
- The June 6 pattern was undone by Q4_K merge, not because it was wrong
- The new code uses the same parameter patterns that caused June 6 warnings
- Parameter structs are the correct solution for WebGPU API surface area

**Risk Assessment**:
- **Build failures**: Low risk (structural changes only, no logic changes)
- **Test failures**: Medium risk (verify test expectations)
- **Performance regression**: Low risk (same operations, different signatures)

### Confidence Triggers

| Action | Expected Outcome | Confidence |
|--------|-----------------|------------|
| Run `cargo clippy --fix --all-targets` | 30-40 warnings removed | +10% |
| Create context.rs with structs | Infrastructure in place | +15% |
| Refactor inference.rs | Pattern validated | +20% |
| Refactor layer.rs | Pattern validated | +15% |
| Refactor matmul/server files | All major files done | +15% |
| Fix remaining 20 warnings | Near zero warnings | +10% |
| All tests pass | No regressions | +5% |
| Smoke tests pass | No performance issues | +5% |

### Final Validation

**To reach 100% confidence, verify**:
1. ✅ `cargo clippy --all-targets` returns 0 warnings
2. ✅ `cargo test --all-targets` passes
3. ✅ Smoke test shows no regression
4. ✅ CI pipeline passes without clippy warnings

**Plan is complete and ready for execution.**

---

## Execution Checklist

- [ ] Phase 0: Establish baseline and backup
- [ ] Phase 1: Create context.rs with parameter structs
- [ ] Phase 2: Refactor inference.rs, layer.rs, matmul.rs, pipeline_shift.rs
- [ ] Phase 3: Refactor server_inference.rs and shimmy_server_gpu.rs
- [ ] Phase 4: Fix core files (f16.rs, ggml_types.rs, dispatch.rs, math_bypass_control.rs)
- [ ] Phase 5: Run clippy --fix + manual test fixes
- [ ] Phase 6: Run full test suite + smoke tests
- [ ] Phase 7: Document patterns + add CI guard

---

**Total Estimated Time**: 3.6 hours  
**Confidence Level**: 90%+  
**Plan Status**: Complete and ready for execution