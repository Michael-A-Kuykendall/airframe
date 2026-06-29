-- ╔══════════════════════════════════════════════════════════════════════════╗
-- ║          VAULT MIGRATION 003 — Layer Diagnostics Table                    ║
-- ║                                                                           ║
-- ║  Adds `layer_diags` table to capture per-layer GPU diagnostic metadata    ║
-- ║  from frontier_compare traces: quant types, weight offsets, routing        ║
-- ║  policy codes.                                                             ║
-- ║                                                                           ║
-- ║  Motivation: V projection produced garbage at layer 2+ for Llama-3.2      ║
-- ║  while Q and K were fine. This table makes per-layer quant type and       ║
-- ║  offset data queryable alongside formula signatures and oracle RMS values. ║
-- ║                                                                           ║
-- ║  Apply: duckdb vault/vault.duckdb < vault/scripts/migration_003_layer_diags.sql
-- ╚══════════════════════════════════════════════════════════════════════════╝

-- Update schema version
INSERT INTO schema_version (version, description, migrated_from)
VALUES (3, 'Add layer_diags table for per-layer GPU diagnostic metadata', 2);

-- ═══════════════════════════════════════════════════════════════════════════
-- LAYER DIAGS
-- One row per (model_id, layer_idx, source_trace).
-- Populated by vault_verify.py when processing frontier_compare traces.
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE layer_diags (
    id              INTEGER PRIMARY KEY,
    model_id        INTEGER NOT NULL REFERENCES models(id),
    layer_idx       INTEGER NOT NULL,

    -- Per-tensor quantization types (GGML type codes)
    q_quant         INTEGER NOT NULL DEFAULT 0,
    k_quant         INTEGER NOT NULL DEFAULT 0,
    v_quant         INTEGER NOT NULL DEFAULT 0,
    ffn_gate_quant  INTEGER NOT NULL DEFAULT 0,
    ffn_down_quant  INTEGER NOT NULL DEFAULT 0,
    ffn_up_quant    INTEGER NOT NULL DEFAULT 0,
    attn_out_quant  INTEGER NOT NULL DEFAULT 0,

    -- Weight byte offsets in the blob
    v_offset        BIGINT NOT NULL DEFAULT 0,
    q_offset        BIGINT NOT NULL DEFAULT 0,
    k_offset        BIGINT NOT NULL DEFAULT 0,
    ffn_gate_offset BIGINT NOT NULL DEFAULT 0,
    ffn_down_offset BIGINT NOT NULL DEFAULT 0,
    ffn_up_offset   BIGINT NOT NULL DEFAULT 0,

    -- Routing policy codes from LayerParams
    ffn_kind        INTEGER NOT NULL DEFAULT 0,
    qkv_layout      INTEGER NOT NULL DEFAULT 0,
    qk_norm         INTEGER NOT NULL DEFAULT 0,
    post_norm       INTEGER NOT NULL DEFAULT 0,
    layer_norm      INTEGER NOT NULL DEFAULT 0,
    batch_count     INTEGER NOT NULL DEFAULT 1,

    -- Provenance
    source_trace    VARCHAR,           -- filename of the frontier_compare trace
    created_at      TIMESTAMP DEFAULT NOW(),

    UNIQUE(model_id, layer_idx, source_trace)
);

CREATE INDEX idx_layer_diags_model ON layer_diags(model_id);
CREATE INDEX idx_layer_diags_quant ON layer_diags(model_id, v_quant);

SELECT 'Migration 003 applied successfully. Table: layer_diags.' AS status;
