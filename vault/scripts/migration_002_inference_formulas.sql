-- ╔══════════════════════════════════════════════════════════════════════════╗
-- ║          VAULT MIGRATION 002 — Inference Formula Columns                  ║
-- ║                                                                             ║
-- ║  Adds `inference_formulas` table to store algebraic formula fingerprints    ║
-- ║  per (model, source, layer).                                                ║
-- ║                                                                             ║
-- ║  Formula columns are NOT raw tensors — they are dimensionless ratios that   ║
-- ║  represent HOW the inference path behaves algebraically. This lets you      ║
-- ║  compare airframe_gpu vs llama_cpp vs candle by asking:                     ║
-- ║  "Does our residual gain match within 2x log2-fold of the golden?"          ║
-- ║                                                                             ║
-- ║  Root cause that motivated this migration:                                  ║
-- ║  TinyLlama layer_dump_gpu produced all-NaN from layer 1 onward.             ║
-- ║  Diagnosed via layer_dump_gpu + observer platform. Root cause was           ║
-- ║  temp_buffer_size underallocated: formula spec gives 13312 but GPU needs    ║
-- ║  15872 (= n_embd + q_len + kv_len*2 + ff_dim*2). Fix in spec.rs            ║
-- ║  compute_derived() bumps to 16384.                                          ║
-- ║                                                                             ║
-- ║  Apply:  duckdb vault/vault.duckdb < vault/scripts/migration_002_inference_formulas.sql
-- ╚══════════════════════════════════════════════════════════════════════════╝

-- Update schema version
INSERT INTO schema_version (version, description, migrated_from)
VALUES (2, 'Add inference_formulas table for algebraic path comparison', 1);

-- ═══════════════════════════════════════════════════════════════════════════
-- INFERENCE FORMULAS
-- One row per (model, source, layer_idx, position).
-- Source is which inference engine produced this trace.
--
-- Formula columns (dimensionless ratios from trace_formula_diff.py):
--   output_energy     = std_dev of layer output hidden state
--   post_attn_energy  = std_dev after attention + residual add
--   ffn_energy        = std_dev of FFN output
--   residual_gain     = output_energy / post_attn_energy
--   ffn_gain          = ffn_energy / post_attn_energy
--   qk_balance        = std_dev(Q) / std_dev(K)
--   kv_mean_gap       = |mean(K) - mean(V)|
--
-- Golden source = 'airframe_cpu' (same CPU reference path as layer_oracles).
-- GPU source    = 'airframe_gpu'.
-- External refs = 'llama_cpp', 'candle'.
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE inference_formulas (
    id                  INTEGER PRIMARY KEY,
    model_id            INTEGER NOT NULL REFERENCES models(id),

    -- Which inference path produced this row
    source              VARCHAR NOT NULL,           -- airframe_cpu, airframe_gpu, llama_cpp, candle

    layer_idx           INTEGER NOT NULL,           -- 0 to n_layers-1; -1 = embedding
    position            INTEGER NOT NULL DEFAULT 0, -- token position in sequence
    input_token_id      INTEGER,

    -- Algebraic formula fingerprint (all dimensionless ratios)
    output_energy       REAL,                       -- std_dev of output hidden state
    post_attn_energy    REAL,                       -- std_dev after attention residual
    ffn_energy          REAL,                       -- std_dev of FFN output
    residual_gain       REAL,                       -- output_energy / post_attn_energy
    ffn_gain            REAL,                       -- ffn_energy / post_attn_energy
    qk_balance          REAL,                       -- std_dev(Q) / std_dev(K)
    kv_mean_gap         REAL,                       -- |mean(K) - mean(V)|

    -- Health flags (derived at insert time)
    has_nan             BOOLEAN DEFAULT FALSE,      -- any NaN in this layer?
    has_inf             BOOLEAN DEFAULT FALSE,      -- any Inf in this layer?

    -- Provenance
    git_commit          VARCHAR,
    created_at          TIMESTAMP DEFAULT NOW(),
    notes               TEXT,

    UNIQUE(model_id, source, layer_idx, position)
);

CREATE INDEX idx_formulas_model_source ON inference_formulas(model_id, source);
CREATE INDEX idx_formulas_nan         ON inference_formulas(has_nan);

-- ═══════════════════════════════════════════════════════════════════════════
-- FORMULA COMPARISONS  
-- One row per comparison run (gpu vs cpu golden, or airframe vs llama_cpp).
-- Stores aggregate divergence scores from trace_formula_diff.py.
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE formula_comparisons (
    id                  INTEGER PRIMARY KEY,
    model_id            INTEGER NOT NULL REFERENCES models(id),
    golden_source       VARCHAR NOT NULL,           -- e.g., airframe_cpu
    candidate_source    VARCHAR NOT NULL,           -- e.g., airframe_gpu

    -- Aggregate divergence (log2-fold scores from trace_formula_diff.py)
    mean_layer_score    REAL,                       -- mean log2-fold across all layers
    median_layer_score  REAL,
    max_layer_score     REAL,
    n_layer_points      INTEGER,

    -- Token-level divergence
    mean_top1_logit_fold REAL,                      -- log2-fold of top-1 logit
    shared_steps        INTEGER,

    -- First failure point (if any NaN/Inf found in candidate)
    first_nan_layer     INTEGER,                    -- NULL if no NaN
    first_nan_source    VARCHAR,

    -- Verdict
    passed              BOOLEAN NOT NULL,           -- TRUE if mean_layer_score < threshold
    threshold           REAL DEFAULT 2.0,           -- log2-fold tolerance

    git_commit          VARCHAR,
    created_at          TIMESTAMP DEFAULT NOW(),
    notes               TEXT
);

CREATE INDEX idx_comparisons_model   ON formula_comparisons(model_id, passed);
CREATE INDEX idx_comparisons_sources ON formula_comparisons(golden_source, candidate_source);

-- ═══════════════════════════════════════════════════════════════════════════
-- TEMP_BUFFER_CORRECTNESS — Enshrined fix for the TinyLlama NaN bug
-- 
-- This view documents the formula invariant that was violated:
--   temp_buffer_size must be >= n_embd + q_len + kv_len*2 + ff_dim*2
-- 
-- Query this view to audit any model for temp buffer undersizing.
-- ═══════════════════════════════════════════════════════════════════════════
CREATE VIEW temp_buffer_audit AS
SELECT
    name,
    arch,
    quant,
    n_embd,
    n_heads,
    n_heads_kv,
    head_dim,
    ff_dim,
    -- Required temp buffer size per new formula
    (n_embd + (n_heads * head_dim) + (n_heads_kv * head_dim * 2) + (ff_dim * 2)) AS required_temp_floats,
    -- Old (broken) formula: ff_dim*2 + n_embd only
    (ff_dim * 2 + n_embd)                                                          AS old_formula_floats,
    -- Safe allocation (next 1024 boundary)
    ((( n_embd + (n_heads * head_dim) + (n_heads_kv * head_dim * 2) + (ff_dim * 2) ) + 1023) / 1024 * 1024) AS correct_temp_buffer_size,
    -- Was the old formula sufficient?
    CASE WHEN (ff_dim * 2 + n_embd) >= (n_embd + (n_heads * head_dim) + (n_heads_kv * head_dim * 2) + (ff_dim * 2))
         THEN 'OK' ELSE 'UNDERALLOCATED' END AS old_formula_status
FROM models
ORDER BY name;

SELECT 'Migration 002 applied successfully. Tables: inference_formulas, formula_comparisons. View: temp_buffer_audit.' AS status;
