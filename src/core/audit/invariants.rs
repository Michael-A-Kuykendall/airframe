use rusqlite::{params, Connection};

use super::db;

/// Default RMS ratio threshold per tensor type.
fn smoothness_threshold(tensor_name: &str) -> f64 {
    match tensor_name {
        "layer_output" => 3.0,  // tight — should always be smooth
        "attn_out" => 3.0,      // same as layer_output
        "ffn_down" => 20.0,     // loose — naturally oscillates 10-13x
        "q" | "k" | "v" => 0.0, // skip — all near-zero, ratio meaningless
        _ => 5.0,
    }
}

/// Minimum RMS magnitude before a smoothness ratio is meaningful.
///
/// Below this floor the tensor is still in the "wakeup" regime (attention has
/// not yet built up at layer 0), where small absolute values produce inflated
/// ratios that are NOT real divergences. Mirrors the q/k/v "near-zero, ratio
/// meaningless" policy.
fn smoothness_floor(tensor_name: &str) -> f64 {
    match tensor_name {
        "attn_out" | "layer_output" => 0.1,
        _ => 0.01,
    }
}

/// Run the full invariant check suite for a given run.
///
/// Returns (passed_count, failed_count, first_fail_check_name).
pub fn run_all(
    conn: &Connection,
    run_id: i64,
    model_id: i64,
    total_layers: i32,
) -> (i32, i32, Option<String>) {
    let mut passed = 0i32;
    let mut failed = 0i32;
    let mut first_fail: Option<String> = None;

    for layer in 0..total_layers {
        let tensors = match get_layer_tensors(conn, run_id, layer) {
            Ok(t) => t,
            Err(e) => {
                let _ = db::insert_invariant_check(
                    conn,
                    run_id,
                    model_id,
                    Some(layer),
                    None,
                    "layer_exists",
                    false,
                    None,
                    None,
                    &format!("no tensors captured: {e}"),
                );
                failed += 1;
                if first_fail.is_none() {
                    first_fail = Some(format!("L{layer}:layer_exists"));
                }
                continue;
            }
        };

        for (tname, t) in &tensors {
            for (check, ok, metric, threshold, detail) in vec![
                ("no_nan", t.has_nan == 0, None, None, String::new()),
                ("no_inf", t.has_inf == 0, None, None, String::new()),
            ] {
                let _ = db::insert_invariant_check(
                    conn,
                    run_id,
                    model_id,
                    Some(layer),
                    Some(tname),
                    check,
                    ok,
                    metric,
                    threshold,
                    &detail,
                );
                if ok {
                    passed += 1;
                } else {
                    failed += 1;
                    if first_fail.is_none() {
                        first_fail = Some(format!("L{layer}:{tname}:{check}"));
                    }
                }
            }

            // RMS sanity: should be finite and non-negative
            if t.rms.is_nan() || t.rms.is_infinite() {
                let detail = format!("rms={:.6}", t.rms);
                let _ = db::insert_invariant_check(
                    conn,
                    run_id,
                    model_id,
                    Some(layer),
                    Some(tname),
                    "rms_finite",
                    false,
                    Some(t.rms),
                    None,
                    &detail,
                );
                failed += 1;
                if first_fail.is_none() {
                    first_fail = Some(format!("L{layer}:{tname}:rms_finite"));
                }
            } else {
                passed += 1;
            }
        }
    }

    // Cross-layer RMS smoothness — per-tensor-type thresholds
    let tensor_names = ["q", "k", "v", "attn_out", "ffn_down", "layer_output"];
    for tname in &tensor_names {
        let threshold = smoothness_threshold(tname);
        if threshold == 0.0 {
            continue;
        } // skip

        let rms_vals: Vec<(i32, f64)> = (0..total_layers)
            .filter_map(|l| {
                get_tensor_rms(conn, run_id, l, tname)
                    .ok()
                    .flatten()
                    .map(|rms| (l, rms))
            })
            .collect();

        for pair in rms_vals.windows(2) {
            let (prev_l, prev_rms) = pair[0];
            let (cur_l, cur_rms) = pair[1];
            if prev_rms == 0.0 {
                continue;
            }
            let floor = smoothness_floor(tname);
            if prev_rms < floor || cur_rms < floor {
                continue;
            }

            let ratio = (cur_rms / prev_rms).max(prev_rms / cur_rms);
            let ok = ratio < threshold;
            let detail =
                format!("L{prev_l} rms={prev_rms:.6} L{cur_l} rms={cur_rms:.6} ratio={ratio:.4}");
            let _ = db::insert_invariant_check(
                conn,
                run_id,
                model_id,
                Some(cur_l),
                Some(tname),
                "rms_smoothness",
                ok,
                Some(ratio),
                Some(threshold),
                &detail,
            );
            if ok {
                passed += 1;
            } else {
                failed += 1;
                if first_fail.is_none() {
                    first_fail = Some(format!("L{cur_l}:{tname}:rms_smoothness"));
                }
            }
        }
    }

    // Energy monotonicity: layer_output RMS should generally increase
    let lo_rms: Vec<(i32, f64)> = (0..total_layers)
        .filter_map(|l| {
            get_tensor_rms(conn, run_id, l, "layer_output")
                .ok()
                .flatten()
                .map(|rms| (l, rms))
        })
        .collect();
    for pair in lo_rms.windows(2) {
        let (prev_l, prev_rms) = pair[0];
        let (cur_l, cur_rms) = pair[1];
        if prev_rms == 0.0 {
            continue;
        }
        let drop_pct = (prev_rms - cur_rms) / prev_rms;
        let ok = drop_pct < 0.20;
        if !ok {
            let detail = format!(
                "L{prev_l} rms={prev_rms:.6} -> L{cur_l} rms={cur_rms:.6} drop={drop_pct:.4}"
            );
            let _ = db::insert_invariant_check(
                conn,
                run_id,
                model_id,
                Some(cur_l),
                Some("layer_output"),
                "energy_monotonic",
                false,
                Some(drop_pct),
                Some(0.20),
                &detail,
            );
            failed += 1;
            if first_fail.is_none() {
                first_fail = Some(format!("L{cur_l}:layer_output:energy_monotonic"));
            }
        } else {
            passed += 1;
        }
    }

    (passed, failed, first_fail)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

struct CapturedTensorSummary {
    rms: f64,
    has_nan: i32,
    has_inf: i32,
}

fn get_layer_tensors(
    conn: &Connection,
    run_id: i64,
    layer_idx: i32,
) -> Result<Vec<(String, CapturedTensorSummary)>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT tensor_name, rms, has_nan, has_inf FROM captured_tensors WHERE run_id=?1 AND layer_idx=?2 ORDER BY tensor_name"
    )?;
    let rows = stmt.query_map(params![run_id, layer_idx], |row| {
        Ok((
            row.get::<_, String>(0)?,
            CapturedTensorSummary {
                rms: row.get(1)?,
                has_nan: row.get(2)?,
                has_inf: row.get(3)?,
            },
        ))
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

fn get_tensor_rms(
    conn: &Connection,
    run_id: i64,
    layer_idx: i32,
    tensor_name: &str,
) -> Result<Option<f64>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT rms FROM captured_tensors WHERE run_id=?1 AND layer_idx=?2 AND tensor_name=?3 LIMIT 1"
    )?;
    let mut rows = stmt.query_map(params![run_id, layer_idx, tensor_name], |row| {
        row.get::<_, f64>(0)
    })?;
    match rows.next() {
        Some(Ok(v)) => Ok(Some(v)),
        _ => Ok(None),
    }
}
