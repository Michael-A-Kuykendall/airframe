use rusqlite::{params, Connection, Result as DbResult};

/// Open (or create) the audit database and run migrations.
pub fn open_or_create(path: &str) -> DbResult<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    migrate(&conn)?;
    Ok(conn)
}

/// Run schema migrations idempotently.
fn migrate(conn: &Connection) -> DbResult<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            arch TEXT NOT NULL,
            gguf_path TEXT NOT NULL,
            n_layers INTEGER NOT NULL,
            n_embd INTEGER NOT NULL,
            n_head INTEGER NOT NULL,
            n_kv_head INTEGER NOT NULL,
            head_dim INTEGER NOT NULL,
            n_ffn INTEGER NOT NULL,
            rms_norm_eps REAL NOT NULL,
            rope_freq_base REAL,
            rope_dim_count INTEGER,
            file_type INTEGER,
            git_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tensor_quants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL REFERENCES models(id),
            tensor_name TEXT NOT NULL,
            quant_type INTEGER NOT NULL,
            shape TEXT NOT NULL DEFAULT '[]',
            offset_bytes INTEGER,
            UNIQUE(model_id, tensor_name)
        );

        CREATE TABLE IF NOT EXISTS audit_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL REFERENCES models(id),
            status TEXT NOT NULL DEFAULT 'running',
            seed INTEGER,
            input_text TEXT,
            num_tokens INTEGER,
            total_layers INTEGER,
            failed_checks INTEGER DEFAULT 0,
            first_fail_layer INTEGER,
            first_fail_check TEXT,
            duration_ms INTEGER,
            git_hash TEXT,
            error_msg TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS captured_tensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL REFERENCES models(id),
            run_id INTEGER NOT NULL REFERENCES audit_runs(id),
            layer_idx INTEGER NOT NULL,
            tensor_name TEXT NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            rms REAL NOT NULL,
            checksum TEXT NOT NULL,
            max_abs REAL NOT NULL,
            min REAL NOT NULL,
            mean REAL NOT NULL,
            has_nan INTEGER NOT NULL DEFAULT 0,
            has_inf INTEGER NOT NULL DEFAULT 0,
            n_elements INTEGER NOT NULL,
            first_20 TEXT,
            seed INTEGER,
            git_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS invariant_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES audit_runs(id),
            model_id INTEGER NOT NULL REFERENCES models(id),
            layer_idx INTEGER,
            tensor_name TEXT,
            check_name TEXT NOT NULL,
            passed INTEGER NOT NULL,
            metric_value REAL,
            threshold REAL,
            detail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_captured_run ON captured_tensors(run_id);
        CREATE INDEX IF NOT EXISTS idx_captured_layer ON captured_tensors(model_id, layer_idx);
        CREATE INDEX IF NOT EXISTS idx_checks_run ON invariant_checks(run_id);
        CREATE INDEX IF NOT EXISTS idx_checks_fail ON invariant_checks(run_id, passed);
        ",
    )?;
    Ok(())
}

/// Insert a model record. Returns model_id.
#[allow(clippy::too_many_arguments)]
pub fn insert_model(
    conn: &Connection,
    name: &str,
    arch: &str,
    gguf_path: &str,
    n_layers: i32,
    n_embd: i32,
    n_head: i32,
    n_kv_head: i32,
    head_dim: i32,
    n_ffn: i32,
    rms_norm_eps: f64,
    rope_freq_base: Option<f64>,
    rope_dim_count: Option<i32>,
    file_type: Option<i32>,
    git_hash: &str,
) -> DbResult<i64> {
    conn.execute(
        "INSERT INTO models (name, arch, gguf_path, n_layers, n_embd, n_head, n_kv_head, head_dim, n_ffn, rms_norm_eps, rope_freq_base, rope_dim_count, file_type, git_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
        params![name, arch, gguf_path, n_layers, n_embd, n_head, n_kv_head, head_dim, n_ffn, rms_norm_eps, rope_freq_base, rope_dim_count, file_type, git_hash],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Insert a tensor quant mapping.
pub fn insert_tensor_quant(
    conn: &Connection,
    model_id: i64,
    tensor_name: &str,
    quant_type: i32,
    shape: &str,
    offset_bytes: Option<i64>,
) -> DbResult<()> {
    conn.execute(
        "INSERT OR IGNORE INTO tensor_quants (model_id, tensor_name, quant_type, shape, offset_bytes) VALUES (?1,?2,?3,?4,?5)",
        params![model_id, tensor_name, quant_type, shape, offset_bytes],
    )?;
    Ok(())
}

/// Start a new audit run. Returns run_id.
pub fn start_run(
    conn: &Connection,
    model_id: i64,
    seed: Option<i64>,
    input_text: &str,
    num_tokens: i32,
    total_layers: i32,
    git_hash: &str,
) -> DbResult<i64> {
    conn.execute(
        "INSERT INTO audit_runs (model_id, status, seed, input_text, num_tokens, total_layers, git_hash) VALUES (?1,'running',?2,?3,?4,?5,?6)",
        params![model_id, seed, input_text, num_tokens, total_layers, git_hash],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Mark a run as passed.
pub fn finish_run_ok(conn: &Connection, run_id: i64, duration_ms: i64) -> DbResult<()> {
    conn.execute(
        "UPDATE audit_runs SET status='passed', duration_ms=?1 WHERE id=?2",
        params![duration_ms, run_id],
    )?;
    Ok(())
}

/// Mark a run as failed with details.
pub fn finish_run_fail(
    conn: &Connection,
    run_id: i64,
    duration_ms: i64,
    first_fail_layer: Option<i32>,
    first_fail_check: Option<&str>,
    failed_checks: i32,
) -> DbResult<()> {
    conn.execute(
        "UPDATE audit_runs SET status='failed', duration_ms=?1, first_fail_layer=?2, first_fail_check=?3, failed_checks=?4 WHERE id=?5",
        params![duration_ms, first_fail_layer, first_fail_check, failed_checks, run_id],
    )?;
    Ok(())
}

/// Mark a run as errored.
pub fn finish_run_error(
    conn: &Connection,
    run_id: i64,
    duration_ms: i64,
    error_msg: &str,
) -> DbResult<()> {
    conn.execute(
        "UPDATE audit_runs SET status='error', duration_ms=?1, error_msg=?2 WHERE id=?3",
        params![duration_ms, error_msg, run_id],
    )?;
    Ok(())
}

/// Insert a captured tensor row with computed stats.
#[allow(clippy::too_many_arguments)]
pub fn insert_captured_tensor(
    conn: &Connection,
    model_id: i64,
    run_id: i64,
    layer_idx: i32,
    tensor_name: &str,
    position: i32,
    data: &[f32],
    seed: Option<i64>,
    git_hash: &str,
) -> DbResult<()> {
    let rms = compute_rms(data);
    let checksum = compute_checksum(data);
    let (max_abs, min, mean) = compute_stats(data);
    let has_nan = data.iter().any(|x| x.is_nan()) as i32;
    let has_inf = data.iter().any(|x| x.is_infinite()) as i32;
    let n_elements = data.len() as i32;
    let first_20 = if data.len() <= 20 {
        serde_json::to_string(data).unwrap_or_default()
    } else {
        let s: Vec<f32> = data.iter().take(20).copied().collect();
        serde_json::to_string(&s).unwrap_or_default()
    };

    conn.execute(
        "INSERT INTO captured_tensors (model_id, run_id, layer_idx, tensor_name, position, rms, checksum, max_abs, min, mean, has_nan, has_inf, n_elements, first_20, seed, git_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
        params![model_id, run_id, layer_idx, tensor_name, position, rms, checksum, max_abs, min, mean, has_nan, has_inf, n_elements, first_20, seed, git_hash],
    )?;
    Ok(())
}

/// Insert an invariant check result.
#[allow(clippy::too_many_arguments)]
pub fn insert_invariant_check(
    conn: &Connection,
    run_id: i64,
    model_id: i64,
    layer_idx: Option<i32>,
    tensor_name: Option<&str>,
    check_name: &str,
    passed: bool,
    metric_value: Option<f64>,
    threshold: Option<f64>,
    detail: &str,
) -> DbResult<()> {
    conn.execute(
        "INSERT INTO invariant_checks (run_id, model_id, layer_idx, tensor_name, check_name, passed, metric_value, threshold, detail) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
        params![run_id, model_id, layer_idx, tensor_name, check_name, passed as i32, metric_value, threshold, detail],
    )?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn compute_rms(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let sq: f32 = v.iter().map(|x| x * x).sum();
    (sq / v.len() as f32).sqrt() as f64
}

fn compute_checksum(v: &[f32]) -> String {
    let c: i64 = v
        .iter()
        .map(|x| x.to_bits() as i64)
        .fold(0i64, |a, b| a.wrapping_add(b));
    format!("{:016x}", c)
}

fn compute_stats(v: &[f32]) -> (f64, f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut max_abs = 0.0f32;
    let mut min = f32::MAX;
    let mut sum = 0.0f32;
    for &x in v {
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
        if x < min {
            min = x;
        }
        sum += x;
    }
    (max_abs as f64, min as f64, (sum / v.len() as f32) as f64)
}
