# AI State Pilot: Component Naming Convention (Military)

## Architecture Overview
The system is hierarchical. The higher levels (AI State Pilot) set the policy, but the lower levels (Crew Chief) execute control on the ground.

## 1. AI State Pilot (The Control Plane)
**Role**: The overarching system, patents, and policy engine.
- Represents the entire product/suite.
- **Analogy**: The Pilot in Command + The Flight Computer.
- **Responsibility**: "I want to fly a safe mission." Defnes rules ("No hallucination"), loads flight plans (Prompts), and monitors instruments.

## 2. Crew Chief (The Self-Healing Layer)
**Role**: The active runtime guardian that fixes problems *before* they crash the plane.
- **Current Name**: "Redo Switch" / "Lane Assist" / "Fail-Fast Server".
- **New Name**: **Crew Chief**.
- **Analogy**: The Master Sergeant on the ground who refuses to sign off the jet if the pressure is low. He fixes the strut, swaps the tire, and *then* lets it fly.
- **Responsibility**: "This token (part) is defective. I am swapping it for a spare (Greedy Token) before I let it go."
- **Invariants**: 
  - "Nobody outranks the Crew Chief on safety." (Even if the Pilot wants to generate garbage, the Crew Chief says NO).
  - Two-Phase Commit is the Crew Chief's signature: Inspect -> Fix -> Sign Off.

## 3. Sortie (The Inference Run)
**Role**: A single generation request.
- **Current Name**: "Inference Request" / "Experiment".
- **New Name**: **Sortie**.
- **Analogy**: A mission. It starts (context load), flies (generation), and lands (EOS).
- **Responsibility**: Completion of the task.

## 4. Flight Recorder (The Telemetry)
**Role**: The metrics logging and audit trail.
- **Current Name**: "Metric Set" / "Polaroid" / "Fail Fast Report".
- **New Name**: **Flight Recorder**.
- **Analogy**: Black Box.
- **Responsibility**: Recording PPL, Entropy, Variance per token for post-sortie analysis.

## Required Refactors
We need to align the codebase to this terminology to enforce clarity.

| Old Term | New Term | Location |
| :--- | :--- | :--- |
| `shimmy_server` | `crew_chief_server` | Binary Name |
| `Redo Switch` | `crew_chief_intervention` | Function/Logic |
| `Fail Fast` | `abort_sortie` | Error Type |
| `Self-Healing` | `maintenance_action` | Log Event |
| `Metrics` | `telemetry` | Variables |

## Action Plan
1. Rename `shimmy_server.rs` to `crew_chief.rs` (or keep shimmy as the "Base Base" and Crew Chief as the module).
2. Update logs to use `[CREW CHIEF]` prefix instead of `[REDO]`.
3. Update docs to reflect the "Crew Chief" authority model.
