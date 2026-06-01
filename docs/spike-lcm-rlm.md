# Spike: LCM (Lossless Context Management) + RLM (Recursive Language Models)

**Date:** 2026-05-30  
**Status:** Research  
**Primary paper:** arXiv:2605.04050 — Clint Ehrlich, Theodore Blackman (Feb 2026)

---

## What Is It

LCMs are the problem that context windows are effectively lossy. When a conversation or task gets longer than the context window, something has to give. The current industry approach is:

- Let the model summarize its own history (fragile — the model decides what to keep)
- Truncate (loses the oldest content)
- Use RAG (loses continuity)

LCM proposes a different architecture: **the engine manages context, not the model.** Two mechanisms:

**1. Recursive Context Compression**  
A hierarchical summary DAG (directed acyclic graph) that automatically compacts older messages as context grows, while retaining lossless pointers back to every original message. You can always retrieve any prior state — nothing is actually deleted, just summarized and indexed. The model sees compressed context but can follow a pointer to retrieve the full original when it needs it.

**2. Recursive Task Partitioning**  
Engine-managed parallel primitives replace model-written loops. The canonical example is `LLM-Map` — instead of the model writing "for each file, do X" (which is fragile and can loop incorrectly), the engine handles the fan-out and aggregation. The model describes what to do to one item; the engine does the iteration.

The authors explicitly frame this as the move from GOTO statements to structured control flow — you give up maximum flexibility in exchange for termination guarantees and predictable behavior.

**The agent:** LCM-augmented coding agent called **Volt**. Benchmarked against Claude Code using Opus 4.6 on the OOLONG long-context eval. Volt outperforms Claude Code at every context length between 32K and 1M tokens.

---

## Relationship to RLM

RLM (Recursive Language Models) is the prior research paradigm LCM extends. RLM showed that you can build agents that recursively manipulate their own context — spawning sub-calls, summarizing results, building up an answer hierarchically. It proved the concept.

LCM's critique of RLM: when the model writes the recursion, you get all the flakiness of model-generated code. The recursion can fail, loop incorrectly, or lose state. LCM moves the recursion into the engine layer, giving you the same capability with deterministic guarantees.

---

## What I Found

**The paper is real and the results are striking.** Beating Claude Code on OOLONG at 32K–1M context is a meaningful benchmark. The architecture is conceptually sound. The "GOTO to structured control flow" analogy is a good one — this is the same move that made programming languages reliable.

**Weaknesses / things to verify:**
- The benchmark is OOLONG, which the authors appear to have constructed. Independent replication on other long-context evals not yet available.
- "Lossless" is a strong claim. The pointers are lossless, but the summaries that the model reads are not — they are LLM-generated summaries, which are inherently lossy. What's lossless is retrievability, not the working representation.
- The system requires tight coupling between the LLM client and the engine. Not a drop-in for existing inference servers.

---

## Relevance to Airframe / Shimmy

Airframe is an inference engine. Shimmy is the CLI/console layer. This is directly relevant.

| Question | Answer |
|---|---|
| Can we use LCM directly? | Not as a library — it's a conceptual architecture, not a package. The Volt agent is their implementation. |
| Is the concept applicable to our stack? | Yes — the inference loop in airframe (`server_inference.rs`) is exactly the right place to implement engine-managed context compression. |
| What would "LCM in airframe" look like? | The inference loop tracks a summary DAG. When context length approaches the model's limit, it triggers a summarization call, compresses older messages, and stores pointers. The model always sees a fresh context window. |
| Does this conflict with anything we're doing? | No. It's additive. It would be a new feature on top of the existing inference pipeline. |
| Is this harder than the math CAS work? | Yes, significantly. The math CAS is a detection + evaluation problem. LCM requires building a persistent state store, a summarization policy, and a retrieval mechanism. |

**The key insight to steal:** engine-managed context beats model-managed context. Don't ask the model to track what it remembers. Track it for the model and inject what it needs.

**Immediate applicable pattern:** the `ScanCursor` in libfse already embodies this idea in a narrow way — the engine manages scan state across tokens instead of asking the model to track it. LCM extends this philosophy to memory.

---

## Recommended Action

1. **Adopt the philosophy now, implement incrementally.** The design principle — engine manages state, model describes intent — should inform how we build shimmy's session layer.
2. **Short-term hook:** add a `context_budget` field to the inference config. When the token count exceeds it, the server triggers a background summarization pass before the next turn. Simple v1 of context compression.
3. **Longer term:** build the summary DAG. Store summaries + original message pointers in a sidecar file next to the GGUF. Persistent across sessions.
4. **Don't use LLM-Map yet.** That's a much bigger architectural change (parallel fan-out, aggregation). Compress first.

---

## References

- LCM paper: https://arxiv.org/abs/2605.04050
- OOLONG benchmark: referenced in the paper but not separately published yet (as of May 2026)
