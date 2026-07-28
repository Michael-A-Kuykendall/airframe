# Stack compare

- A: `airframe/docs/fixtures/stack_minimal.json` engine=fixture
- B: `airframe/docs/fixtures/stack_diverge_b.json` engine=fixture_b
- prompt A: 'The capital of France is'
- prompt B: 'The capital of France is'

| Level | A | B | Status |
|-------|---|---|--------|
| L1 tokens | [785, 6722, 315, 9625, 374] | [785, 6722, 315, 9625, 374] | GREEN |
| L2 embd rms | 0.0169 | 0.0169 | GREEN |
| L3 residual L0 | rms=0.22 | rms=99.0 | RED |
| L6 argmax | 93367 ' Trom' | 93367 ' Trom' | GREEN |

**First diverge:** `L3` — residual diverge
