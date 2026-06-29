# Test Regimes — Queued Testing for Human Execution

## Overview

When AI agent is dormant (you're using it), tests must be queued and run by human. This document defines test regimes that can be executed efficiently after development cycles complete.

---

## Test Execution Strategy

### Option A: Beads-Driven Testing (Recommended)
**Workflow:**
1. Agent completes dev work → files beads issues for testing needs
2. Human reviews `bd ready --json` to see pending tests
3. Human runs tests via `cargo test --test {name}` or `cargo run --bin {binary}`
4. Results logged in `.beads/test-results/`

**Advantages:**
- Tests linked to specific issues (traceability)
- Human sees what needs testing before running
- Can prioritize based on issue priority

### Option B: Scripted Test Regimes
**Workflow:**
1. Agent creates test regime scripts in `scripts/test-regimes/`
2. Human runs single command per regime: `./scripts/run-test-regime-{id}.ps1`
3. Results captured and reported back to agent

**Advantages:**
- One-command execution
- Batch testing capability
- Clear separation between dev and test phases

---

## Test Regimes — Current State (2026-06-19)

### Regime 1: TDR Model Stabilization Tests
**Linked Issues:** airframe-dna, pz9, guf, b41, o9e

**Test Commands:**
```powershell
# Qwen3-0.6B smoke test (QK-norm path)
$env:SHIMMY_MODEL_PATHS = "D:\shimmy-test-models\gguf_collection"
cargo run --release -- generate --name "Qwen3-0.6B" --prompt "Hello" --max-tokens 20

# Llama-3.2-3B smoke test (template fix validation)
cargo run --release -- generate --name "Llama-3.2-3B-Instruct" --prompt "test" --max-tokens 20

# Gemma-2-2B smoke test
cargo run --release -- generate --name "Gemma-2-2B" --prompt "test" --max-tokens 20
```

**Expected Results:**
- No NaN in output tensors
- Correct template wrapping (llama3 vs chatml)
- Frontier_compare layers < 0.001 MAE

---

### Regime 2: TDR Transport Layer Tests
**Linked Issues:** airframe-mbt, eri, dar, 68s

**Prerequisites:**
- TDR transport layer must be implemented first (airframe-mbt)
- Calibration data available (airframe-68s)

**Test Commands:**
```powershell
# Timestamp query smoke test
cargo run --release --bin tdr_calibrate -- help

# Encoder pool stress test
cargo run --release -- generate --name "TinyLlama" --prompt "test" --max-tokens 100

# ISF fact emission verification
# Check logs for [ISF] DispatchCompleted facts
```

---

### Regime 3: Frontiers & Vault Verification
**Linked Issues:** airframe-6ex (stderr cleanup), vault verification

**Test Commands:**
```powershell
# Run frontier_compare with all models
cargo run --release --bin frontier_compare -- \
  --model "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --prompt "Hello world" \
  --max-tokens 64

# Verify against vault oracles
python docs/vault_verify.py --models "Llama-3.2-3B-Instruct,Qwen3-1.7B,Qwen2-1_5b,Gemma-2-2B,TinyLlama"

# Run vault smoke test (TinyLlama Q6_K - known working)
cargo run --release -- generate --name "TinyLlama-1.1B-Q6_K" --prompt "test" --max-tokens 20
```

---

### Regime 4: CPU Parity Tests
**Test Files:** tests/*.rs (already exist)

**Run Commands:**
```powershell
# Run all CPU tests
cargo test --release

# Run specific test
cargo test --release --test cpu_layer_dump

# Run GPU-dependent tests (requires GPU + model)
cargo test --release -- --ignored
```

---

## Test Queue Management

### Beads Issues for Testing
Create beads issues when dev work completes:

```bash
# Example: after implementing TDR transport layer
bd issue create "airframe-test-transport-layer" \
  --title "Test TDR Transport Layer Implementation" \
  --priority 2 \
  --notes "Run frontier_compare on all models to verify transport layer stability. Also run calibration sweep (airframe-68s)."
```

### Test Results Tracking
Log results in `.beads/test-results/`:
```bash
mkdir -p .beads/test-results/$(date +%Y-%m-%d)
# After running tests:
echo "Model: Llama-3.2-3B, Status: PASS, MAE: 0.001" >> .beads/test-results/$(date +%Y-%m-%d).log
```

---

## Concurrent AI Testing — Can We Do It?

### Current Limitation
Running two AIs concurrently on same machine is problematic due to:
- Shared GPU (NVIDIA RTX 3060, 4GB VRAM)
- Memory pressure from both models + inference
- Context switching overhead

### Potential Workarounds

**Option 1: Small Model Test Bed**
- Run agent with tiny model (<500MB) for lightweight tasks
- Reserve larger models for human execution only
- Test if concurrent inference is viable (benchmark first)

**Option 2: Time-Sliced Execution**
- Agent queues tests in `scripts/test-regimes/`
- Human runs all queued tests when agent is dormant
- Agent resumes after test results available

**Option 3: Beads as Queue System**
```bash
# Agent creates test issues before going dormant
bd issue create "airframe-test-{id}" --priority 2

# Human sees pending tests in bd ready
bd ready --json

# Human runs tests, logs results to .beads/test-results/

# Agent resumes and reviews results
```

---

## Recommended Workflow

### When Agent Goes Dormant:
1. **Queue all pending tests** → beads issues or test regime scripts
2. **Update AGENTS.md** with current state and next steps
3. **Run `bd prime --export`** for context injection on resume

### Human Execution Phase:
1. Review `bd ready --json` to see what needs testing
2. Run tests according to regime priorities
3. Log results in `.beads/test-results/`
4. Optionally create new beads issues for discovered problems

### When Agent Resumes:
1. Review test results from previous session
2. Continue from last bead issue or start fresh with `bd ready --json`
3. Use `bd prime --export` to get full context dump

---

## Quick Start Commands

### Human: Run All Smoke Tests (5 min)
```powershell
$env:SHIMMY_MODEL_PATHS = "D:\shimmy-test-models\gguf_collection"
cargo run --release -- generate --name "Phi-3.5-mini-instruct" --prompt "Hello" --max-tokens 20
cargo run --release -- generate --name "Llama-3.2-1B-Instruct" --prompt "test" --max-tokens 20
cargo run --release -- generate --name "Qwen3-0.6B" --prompt "test" --max-tokens 20
```

### Human: Run Full Frontier Compare (15 min)
```powershell
cargo run --release --bin frontier_compare -- \
  --model "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --prompt "Hello world" \
  --max-tokens 64 \
  --validate-head-tile
```

### Human: Run Vault Verification (10 min)
```powershell
python docs/vault_verify.py --models "Llama-3.2-3B-Instruct,Qwen3-1.7B,Qwen2-1_5b,Gemma-2-2B,TinyLlama"
```

### Human: Run CPU Tests Only (no GPU)
```powershell
cargo test --release
```

---

## Summary

**Best Approach:** Beads-driven test queue + scripted regimes for efficiency.

**Why not concurrent AI testing?**
- GPU memory is bottleneck (4GB shared)
- Context switching overhead
- Better to let human run tests when agent is dormant

### Current Test Queue

| Issue ID | Title | Priority | Status |
|----------|-------|----------|--------|
| airframe-01o | Test Queue Management System | P2 | open |

**See `bd ready --json` for all pending tests.**

**Next Steps:**
1. Agent queues tests as beads issues before going dormant
2. Human runs tests, logs results to `.beads/test-results/`
3. Agent resumes and reviews test results from previous session
