-- ╔══════════════════════════════════════════════════════════════════════════╗
-- ║                        GOLDEN REFERENCE VAULT SCHEMA                      ║
-- ║                     Supports Local ↔ Cloud Bounce Workflow                ║
-- ║                              Version 1.0                                   ║
-- ╚══════════════════════════════════════════════════════════════════════════╝

-- ═══════════════════════════════════════════════════════════════════════════
-- SCHEMA VERSION - Track schema changes across local/cloud sync
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE schema_version (
    version         INTEGER PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT NOW(),
    description     VARCHAR,
    migrated_from   INTEGER
);

INSERT INTO schema_version (version, description) VALUES (1, 'Initial vault schema');

-- ═══════════════════════════════════════════════════════════════════════════
-- MODELS - One row per unique (model_name, quant) combination
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE models (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR NOT NULL,           -- e.g., "TinyLlama-1.1B-Chat-v1.0"
    gguf_filename       VARCHAR NOT NULL,           -- e.g., "TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
    arch                VARCHAR NOT NULL,           -- llama, mistral, phi, gemma, qwen2, qwen3
    quant               VARCHAR NOT NULL,           -- q4_0, q4_k, q6_k, f16, f32
    
    -- Core dimensions (from GGUF metadata)
    n_layers            INTEGER NOT NULL,
    n_heads             INTEGER NOT NULL,
    n_heads_kv          INTEGER NOT NULL,
    head_dim            INTEGER NOT NULL,
    n_embd              INTEGER NOT NULL,           -- embedding dimension
    ff_dim              INTEGER NOT NULL,           -- feed-forward dimension
    n_vocab             INTEGER NOT NULL,
    n_ctx               INTEGER NOT NULL,
    
    -- RoPE config
    rope_base           REAL,
    rope_scale          REAL,
    rope_dim            INTEGER,
    rope_scaling_type   VARCHAR,                    -- linear, yarn, none
    
    -- Special features (for vault filtering)
    rms_eps             REAL,
    has_qk_norm         BOOLEAN DEFAULT FALSE,      -- Qwen3 only
    uses_layer_norm     BOOLEAN DEFAULT FALSE,      -- Phi only
    attn_logit_softcap  REAL DEFAULT 0.0,           -- Gemma2: 50.0
    final_logit_softcap REAL DEFAULT 0.0,           -- Gemma2: 30.0
    expert_count        INTEGER,                    -- Mixtral: 8
    expert_used_count   INTEGER,                    -- Mixtral: 2
    
    -- File info
    gguf_path           VARCHAR NOT NULL,           -- absolute path to GGUF file
    file_size           BIGINT,                     -- bytes
    file_sha256         VARCHAR,                    -- for integrity
    
    -- Metadata
    created_at          TIMESTAMP DEFAULT NOW(),
    oracle_git_commit   VARCHAR,                    -- commit that generated oracle
    notes               TEXT,
    
    UNIQUE(name, quant)
);

-- ═══════════════════════════════════════════════════════════════════════════
-- TENSOR METADATA - All tensors in the GGUF file (for debugging weight loading)
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE tensor_metadata (
    id                  INTEGER PRIMARY KEY,
    model_id            INTEGER NOT NULL REFERENCES models(id),
    tensor_name         VARCHAR NOT NULL,           -- e.g., "blk.0.attn_q.weight"
    tensor_type         VARCHAR NOT NULL,           -- e.g., "Q4_K", "F32"
    n_dims              INTEGER NOT NULL,           -- number of dimensions
    dim_0               BIGINT,                     -- first dimension
    dim_1               BIGINT,                     -- second dimension
    dim_2               BIGINT,                     -- third dimension (if exists)
    tensor_offset       BIGINT NOT NULL,            -- byte offset in GGUF file
    size_bytes          BIGINT NOT NULL,            -- size in bytes
    n_quant_blocks      BIGINT,                     -- for quants: number of blocks
    
    UNIQUE(model_id, tensor_name)
);

CREATE INDEX idx_tensor_model ON tensor_metadata(model_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- LAYER ORACLES - Golden traces for each (model, layer, operation, position)
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE layer_oracles (
    id                  INTEGER PRIMARY KEY,
    model_id            INTEGER NOT NULL REFERENCES models(id),
    layer_idx           INTEGER NOT NULL,           -- 0 to n_layers-1, -1 for embedding
    operation           VARCHAR NOT NULL,           -- embedding, rms_norm, attention_q, etc.
    position            INTEGER NOT NULL,           -- token position in sequence (0=bos)
    input_token_id      INTEGER,                    -- token ID at this position
    
    -- Expected metrics (computed from CPU reference)
    expected_rms        REAL,                       -- RMS diff from CPU (0.0 = golden)
    expected_max        REAL,                       -- max abs diff
    expected_nan        INTEGER DEFAULT 0,          -- expected NaN count
    expected_inf        INTEGER DEFAULT 0,          -- expected Inf count
    
    -- Data storage (Parquet files with actual F32 values)
    cpu_blob_path       VARCHAR,                    -- path to parquet with F32 values
    cpu_blob_hash       VARCHAR,                    -- SHA256 of blob file
    checksum            BIGINT,                     -- row-wise F32 checksum
    
    -- Metadata
    created_at          TIMESTAMP DEFAULT NOW(),
    oracle_git_commit   VARCHAR,
    notes               TEXT,
    
    UNIQUE(model_id, layer_idx, operation, position)
);

-- Common query indexes
CREATE INDEX idx_oracles_model_op ON layer_oracles(model_id, operation);
CREATE INDEX idx_oracles_model_layer ON layer_oracles(model_id, layer_idx);
CREATE INDEX idx_oracles_position ON layer_oracles(position);

-- ═══════════════════════════════════════════════════════════════════════════
-- VERIFICATION RUNS - CI/CD integration point
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE verification_runs (
    id                  INTEGER PRIMARY KEY,
    model_id            INTEGER NOT NULL REFERENCES models(id),
    run_type            VARCHAR NOT NULL,           -- ci, manual, regression, debug
    git_commit          VARCHAR NOT NULL,
    timestamp           TIMESTAMP DEFAULT NOW(),
    
    -- Runtime config
    backend             VARCHAR NOT NULL,           -- airframe_gpu, airframe_cpu, llama_cpp
    kv_quant            VARCHAR,                    -- none, int4 (for KV cache quant)
    test_name           VARCHAR,                    -- which test was run
    
    -- Results
    passed              BOOLEAN NOT NULL,
    rms_diff_avg        REAL,
    rms_diff_max        REAL,
    nan_count           INTEGER DEFAULT 0,
    inf_count           INTEGER DEFAULT 0,
    duration_ms         REAL,
    
    -- Error tracking
    error_message       TEXT,
    first_fail_layer    INTEGER,                    -- which layer failed first
    first_fail_operation VARCHAR,                   -- which operation failed
    
    -- Metadata
    notes               TEXT,
    
    UNIQUE(git_commit, model_id, run_type, timestamp)
);

-- Query indexes
CREATE INDEX idx_runs_passed ON verification_runs(passed, timestamp DESC);
CREATE INDEX idx_runs_commit ON verification_runs(git_commit);
CREATE INDEX idx_runs_model ON verification_runs(model_id, passed);

-- ═══════════════════════════════════════════════════════════════════════════
-- SYNC LOG - Track local ↔ cloud bounces
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE sync_log (
    id                  INTEGER PRIMARY KEY,
    sync_direction      VARCHAR NOT NULL,           -- push, pull, merge
    timestamp           TIMESTAMP DEFAULT NOW(),
    git_commit          VARCHAR,
    records_added       INTEGER,
    records_updated     INTEGER,
    conflicts           INTEGER DEFAULT 0,
    resolution          VARCHAR,                    -- local_wins, cloud_wins, manual
    notes               TEXT
);

-- ═══════════════════════════════════════════════════════════════════════════
-- VAULT CONFIG - Per-installation settings
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE vault_config (
    key                 VARCHAR PRIMARY KEY,
    value               VARCHAR,
    updated_at          TIMESTAMP DEFAULT NOW()
);

-- Default config
INSERT INTO vault_config (key, value) VALUES 
    ('sync_mode', 'git_lfs'),
    ('conflict_resolution', 'local_wins_for_oracles'),
    ('rms_threshold', '1e-5'),
    ('max_abs_threshold', '1e-4'),
    ('ci_backend', 'airframe_gpu');

-- Print confirmation
SELECT 'Vault schema initialized successfully!' AS status;