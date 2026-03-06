# Airframe CLI & AI Dev Loop Integration Plan

## Goal
Integrate the fully resurrected Shimmy CLI (from the `shimmy-console` recovery branch) with the newly standalone `airframe` inference core. The objective is to create a totally local, bit-perfect, deterministic AI-driven development environment that replaces remote assistants.

## Pre-Flight Status
- `libfse` exists locally at `airframe/crates/libfse`.
- `shimmy-console` is confirmed to hold the recovered CLI components at branch `recovery/2025-12-20-cli-tools-recovered`.
- **CRITICAL AUDIT DISCOVERY**: The `Cargo.toml` inside the recovered `shimmy-console/console/Cargo.toml` is completely 100% null bytes due to the data erasure event. Attempting to copy it blindly will break the workspace. We must manually reconstruct it.

---

## Step-by-Step Implementation Plan

### Phase 1: Transplant the Resurrected CLI into the Sandbox
*Goal: Extract the recovered Rust source code, bypass the corrupted/wiped `Cargo.toml`, and graft it cleanly into the Airframe workspace.*

1. **Extract Source Files (Strict Bash Command):**
   Run the following terminal command from the `airframe` root to extract the `console` directory from the remote recovery branch safely into `crates/console`:
   ```bash
   cd C:/Users/micha/repos/shimmy-console
   git archive origin/recovery/2025-12-20-cli-tools-recovered console/ | tar -x -C ../airframe/crates/
   cd ../airframe
   ```

2. **Reconstruct Cargo.toml (Fix Null Bytes):**
   The recovered `console/Cargo.toml` is fundamentally corrupted. Delete it and create `crates/console/Cargo.toml` with the following explicitly audited dependencies (derived by running `rg "^use"` across the source tree):
   ```toml
   [package]
   name = "shimmy-console"   # Keep this name so existing "use shimmy_console::" imports don't break
   version = "0.1.0"
   edition = "2021"

   [lib]
   path = "src/lib.rs"

   [[bin]]
   name = "airframe-cli"
   path = "src/bin/system_test.rs" # The entrypoint logic in the recovered files

   [dependencies]
   airframe = { path = "../../" }
   async-trait = "0.1.74"
   clap = { version = "4.4", features = ["derive"] }
   tokio = { version = "1.0", features = ["full", "rt-multi-thread", "macros"] }
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   anyhow = "1.0"
   thiserror = "1.0"
   rusqlite = { version = "0.31", features = ["bundled"] }
   parking_lot = "0.12"
   schoolmarm = "0.1.1" # For formatting grammar validation hook
   ```

3. **Register Workspace Member:**
   Open `Cargo.toml` in the `airframe` root folder. Append the reconstructed CLI directory to the workspace definition so Cargo maps it:
   ```toml
   [workspace]
   members = [
       "crates/libfse",
       "crates/console"
   ]
   ```

4. **Verify Compilation Base (Fix Dangling Imports):**
   Run `cargo check -p shimmy-console`.
   *Audit Note:* Because this code was torn out of a crash, expect missing `plugin` or networking dependencies. Comment out any missing phantom files in `src/lib.rs` (e.g. `pub mod plugins;` if it throws errors) until the compiler gives a green light.

---

### Phase 2: Wire the "Bit-Perfect" Airframe Inference Core
*Goal: Remove all mock HTTP/WebSocket networking from the CLI and hardcode it directly to Airframe's determinist memory allocation (`generate_with_control`).*

1. **Reroute the Engine Dispatcher:**
   - Open `crates/console/src/adapters/ws_adapter.rs` (or `mock_adapter.rs`).
   - Remove the `tokio_tungstenite` network boundaries.
   - Inject the core memory engine: `airframe::runtime::Engine::new()`.
   - Call `engine.generate_with_control(prompt_ids, max_new_tokens, weights, &control, None)` directly in-process. This ensures we inherit the 100% bit-perfect guarantees.

2. **Implement the `schoolmarm` Hook (Syntax/Grammar Enforcer):**
   - Create a module `crates/console/src/hooks.rs`.
   - Implement the `airframe::control::InferenceControl` trait for a generic struct (e.g. `GrammarHook`).
   - In the `step()` hook logic, pass the `InferenceEvent.candidate_token` sequence text buffer to `schoolmarm::validate()`. If it fails the developer grammar constraints, halt appending to the KV cache immediately.

3. **Bind Context Management:**
   - Modify `crates/console/src/history.rs` (which uses `rusqlite`). 
   - Instead of injecting a bloated sequence of text backwards, pipe Airframe's internal sliding KV cache snapshot (`KvSnapshot`) metadata directly.

4. **Validate Determinism Execution:**
   - Write a unit test `tests/determinism.rs` in `crates/console/`.
   - Run a `generate_with_control` task using same weights/prompts 3 consecutive times. Assert the hash output bytes match with absolutely no drift.

---

### Phase 3: The Local AI Development Loop Setup (Rust Chain)
*Goal: Boot the loop into a terminal-attached agent that has system tools and zero network lag.*

1. **Activate Tooling Structs:**
   - Go to `crates/console/src/tools/`.
   - Expose the exact structs `file_ops::FileTool`, `git::GitTool`, and `command::CommandTool`.
   - Bind these inside the CLI initialization so the AI loop can call them.

2. **Unbuffered Streaming to Stdout:**
   - Check `crates/console/src/commands/chat.rs`.
   - Ensure the standard output `stdout` is unbuffered (`flush()` on every token) so `rustchain` automation pipelines can pipeline responses safely without packet tearing.

3. **End-to-End Compile Agent Check:**
   - Run the new binary: `cargo run -p shimmy-console --bin airframe-cli -- chat --model local`
   - Ask it to check `cargo check`, generate a file, and invoke git commands to prove the toolchain is self-sufficient.
