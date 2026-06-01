# Spike: PageIndex — Vectorless RAG via Document Tree Structure

**Date:** 2026-05-30  
**Status:** Research  
**Primary reference:** Referenced in arXiv:2604.14222 (April 2026, Adaptive Query Routing)

---

## What Is It

PageIndex is a retrieval architecture that throws out the standard RAG stack entirely and replaces it with reasoning over document structure.

Standard RAG:
1. Chunk the document into fixed-size pieces
2. Embed each chunk into a vector
3. Store vectors in a vector database
4. At query time: embed the query, find nearest vectors, return matching chunks

PageIndex:
1. Parse the document into a tree (headings, sections, subsections, pages — the inherent structure the document already has)
2. At query time: give the LLM the tree structure (like a table of contents) and ask it to reason about where the answer would be
3. LLM navigates to the right node(s), retrieves the full text from those sections
4. Answer is grounded in the original document with no information lost to chunking

**Claimed result: 98.7% accuracy on FinanceBench** — a benchmark of 150 expert-annotated questions over real SEC 10-K and 10-Q financial filings.

---

## What I Found

PageIndex itself does not have a standalone arxiv paper by that name under my search. It is referenced as "[2]" in Lumer et al., which is referenced in arXiv:2604.14222. The core idea is captured in that secondary paper.

**The broader validation from arXiv:2604.14222 (Adaptive Query Routing, April 2026):**

This paper independently implemented three retrieval architectures and benchmarked them on financial, legal, and medical documents:

| Method | FinanceBench Score | Cross-reference Recall |
|---|---|---|
| Vector RAG | 0.821 | 91.7% |
| Tree Reasoning (PageIndex-style) | **0.938** | **100%** |
| Hybrid Adaptive | 0.901 | 100% |

Tree reasoning wins by **11.7 percentage points** on real FinanceBench data. Cross-reference recall (finding answers that span multiple sections of a document) is 100% for tree reasoning vs 91.7% for vector search.

The paper also shows hybrid beats vector on complex queries, and vector wins only on multi-document synthesis (Tier 4) — where you need to aggregate across many separate documents.

**Why tree reasoning beats vectors at cross-reference recall:** chunking destroys the hierarchical relationships in structured documents. A vector of a chunk about "revenue recognition policy" doesn't know it's under "Note 2: Accounting Policies" which is related to the revenue figures in the income statement. The tree preserves those relationships and the LLM can reason across them.

**The failure mode of PageIndex-style approaches:** unstructured documents with no inherent hierarchy. A long prose essay has no useful tree to build. Also: very large documents where the tree itself becomes too large to fit in context.

---

## Relevance to Airframe / Shimmy

Shimmy and airframe don't currently have a RAG or memory system. This is relevant to the future memory layer — specifically the question of how to let small models reason over external knowledge.

| Question | Answer |
|---|---|
| Is this directly usable today? | No — we have no document retrieval system to replace. |
| Is this relevant to the roadmap? | Yes. When we build session memory / document retrieval, this is the architecture to study. |
| Does it conflict with the vector DB paradigm? | Intentionally, yes. But vector search wins on multi-document synthesis. The lesson is: use tree reasoning for structured docs, vector for cross-document queries. |
| What's the small-model angle? | This is directly relevant to your thesis. Vector RAG requires embedding models. PageIndex only requires the inference model itself — the same model doing the generation. One model, no second embedding system. |
| What types of documents would benefit? | Code docs (natural tree structure: modules, functions, types), API specs, structured data schemas, legal docs, configuration files. All of these are already hierarchical. |

**The small-model advantage here is significant.** A 1B model with PageIndex over a well-structured codebase can answer questions about that codebase without needing a separate embedding model, a vector DB, or chunking infrastructure. The model navigates the document tree by reasoning, not by cosine similarity.

**What you'd build in airframe/shimmy:**
1. A document indexer that parses a codebase or doc set into a tree (section headers, file structure, class/function hierarchy)
2. At query time: give the model the tree as context, ask "which section has the answer?"
3. Retrieve the full text of that section
4. Answer grounded in the original source

This is buildable with no new infrastructure. The model is already running. The tree is just a JSON file.

---

## Recommended Action

1. **Prototype it.** Build a simple tree indexer for airframe's own docs (the `docs/` folder has a natural structure). Test it with shimmy — ask questions about the codebase, see if tree-guided retrieval beats naive "give the model all the docs."
2. **Target structured docs first.** Code, specs, READMEs — these are already hierarchical. Don't try it on unstructured prose.
3. **No vector DB needed.** This is a zero-dependency addition. The "database" is a JSON tree file. The "retriever" is a prompt that asks the model to navigate.
4. **Long-term:** combine with LCM. The document tree becomes part of the LCM context DAG — permanent external knowledge that the model can reference without burning context tokens on it.

---

## Summary Comparison Across All Three Spikes

| Spike | What it solves | Applicability now | Applicability v2.x |
|---|---|---|---|
| TurboQuant | Memory + speed at long context | Low (WGSL port needed, context limit deferred) | High for v2.1 |
| LCM | Context management across sessions | Medium (design principle now, implement incrementally) | High |
| PageIndex | Document retrieval without vectors | High (prototype-able today with no new deps) | Core of memory system |

PageIndex is the one you can touch this week. LCM is the philosophy to build toward. TurboQuant is the hardware layer that becomes relevant when we push to 8K+ context.

---

## References

- Primary reference: arXiv:2604.14222 (Adaptive Query Routing) — https://arxiv.org/abs/2604.14222
- FinanceBench benchmark: standard eval for financial document QA over SEC filings
- Lumer et al. (the original PageIndex comparative evaluation) — referenced as [1] in arXiv:2604.14222
