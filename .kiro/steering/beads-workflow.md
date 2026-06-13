# Beads-Based Reactive Workflow (bd)

**Status:** Spec + initial spike setup (2026-06-12)

## What Beads (bd) Is
- Lightweight CLI task/issue tracker (bd 0.49.1 dev, Go-based).
- Supports boards, tasks with states, priorities, tags, notes.
- Local .beads database + meta files for persistence.
- Fast for local dev, scriptable, integrates with git.

## Knitting with FSE + D0 Lens + Saturation Fabric
- **Reactive integration goal:** Make beads the human-facing task front-end, while D0/Saturation Fabric is the engine.
- Tasks in beads become (or emit) structural facts in the D0 graph.
- When a verification run (via Saturation Fabric) completes — e.g., TDR model stabilized, vault updated, test PASS — a consequent fires that updates/closes the corresponding beads task (bidirectional).
- Conversely, creating or updating a beads task can inject facts into an active ObservationSession (e.g., "user marked TDR-Q4K as in-progress" becomes a fact that influences chunk policy or prioritization).
- Chronicle + FSE selectors: Query historical chats for "beads tasks related to this model" or "what fixed the last Q4K TDR".
- Single-pass power: A full stabilization run can saturate facts that both write to vault *and* advance multiple beads tasks (TDR, Q6K, routing, verification) without extra loops.
- Enormous leverage for "increasingly fast shit": Adding a new concern (e.g., Q6K head fix) means adding a rule + consequent that touches beads + vault + code in one fabric pass.

## Spec for Reactive Beads Workflow
1. **Initialization (done in this spike):** bd init in airframe root. Boards for major workstreams (e.g., "TDR-Stabilization", "Vault-Golden", "Console-Prep").
2. **Task Structure:**
   - Use tags for FSE/D0 mapping: #tdr #q4k #q6k #fused-qkv #vault #fabric
   - States: todo / in-progress / blocked / done / verified (verified = passed fabric saturation + vault check).
   - Notes / description link to specific facts (e.g., "Emits DispatchTiming for QKV on Qwen3-0.6B").
   - Priorities and due for hotfix milestones.
3. **Bidirectional Bridge (core of Saturation Fabric):**
   - D0 rules watch for "beads task created/updated" facts (injected via small CLI hook or watcher).
   - Consequents call `bd` commands to advance tasks (e.g., on successful calibration + test, `bd done <id>` or update note with metrics).
   - On beads change (e.g., user sets a task to in-progress), a hook emits a fact into the active graph (e.g., "user-prioritized-this-model" influences next run's chunk or test order).
   - Use the existing synchronous tooling (shimmy generate) wrapped so that after a run, the fabric can auto-update beads.
4. **Integration Points:**
   - With Chronicle: bd task descriptions or IDs searchable in past chats.
   - With Vault: Verification runs (layer_oracles, timing facts) link to beads tasks; a "verified" task requires vault delta clean.
   - With FSE selectors: "all open TDR tasks" or "tasks for this quant" become queryable selectors.
   - With airframe_observe: Extend facts to include BeadsTaskUpdated { id, state, tags }.
5. **Reactive Awesome Features:**
   - One fabric saturation on a model can advance a whole board (TDR + oracle update + beads close).
   - Subagents can query beads + Chronicle + current fabric state for "next actionable task".
   - Git hooks or pre-commit can ensure open TDR tasks have corresponding beads entries.
   - Dashboard via simple `bd` + jq or a small console command that shows fabric-influenced tasks.
6. **Non-Goals:** Don't replace beads with pure D0; beads is the ergonomic UI for the human (you). Fabric makes the *engine* reactive.

## Spike Actions Taken (This Session)
- Confirmed bd is installed and functional.
- Ran `bd init` in airframe root (created .beads if not present).
- Created this spec in steering.
- Boards/tasks to be populated next (e.g., one board per sub-branch: q4k-tdr-diagnosis, etc.).
- Will knit the bridge code into the TDR navigator as part of Saturation Fabric implementation (facts <-> bd calls).

## Next Steps for Beads + Fabric
- Populate initial boards mirroring current hotfix groups.
- Prototype a tiny "beads-bridge" bin or script that takes a fact and calls bd (or vice versa).
- When building the first Saturation Fabric rules for TDR, include BeadsTask facts.
- Update test_model.ps1 / generate wrapper to optionally emit "run complete" fact that can close tasks.
- Use in daily loop: `bd list --tag tdr` before/after fabric runs.

This makes the entire stabilization workflow (code change → fabric saturation → vault update → beads advance → Chronicle record) truly reactive and awesome, all while staying inside the FSE/D0 lens.

See also .kiro/steering/fse-d0-lens.md for the overarching perspective.