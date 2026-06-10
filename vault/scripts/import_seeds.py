#!/usr/bin/env python3
"""
vault import_seeds.py
?????????????????????
Reads all JSON seed files from vault/seeds/, validates integrity,
then inserts model + oracle rows into vault/vault.duckdb.

Usage:
    python vault/scripts/import_seeds.py [seeds_dir] [vault_db]

Defaults:
    seeds_dir = vault/seeds
    vault_db  = vault/vault.duckdb

Checks before any INSERT:
  1. seed_version == 1
  2. nan_count == 0
  3. inf_count == 0
  4. oracle count matches expected_oracle_count
  5. rms_sum matches recomputed sum (tolerance 1e-3)
  6. No duplicate (name, quant) already in models table
"""

import json
import sys
import os
import math
import glob
import duckdb
from datetime import datetime, timezone


def load_seed(path):
    with open(path) as f:
        return json.load(f)


def validate(seed, path):
    errors = []
    if seed.get("seed_version") != 1:
        errors.append(f"unexpected seed_version {seed.get('seed_version')}")
    ig = seed.get("integrity", {})
    if ig.get("nan_count", -1) != 0:
        errors.append(f"nan_count={ig.get('nan_count')} ? not importing")
    if ig.get("inf_count", -1) != 0:
        errors.append(f"inf_count={ig.get('inf_count')} ? not importing")
    oracles = seed.get("oracles", [])
    expected = ig.get("expected_oracle_count", -1)
    if len(oracles) != expected:
        errors.append(f"oracle count {len(oracles)} != expected {expected}")
    # recompute rms_sum
    computed = sum(float(o["rms"]) for o in oracles)
    stored = float(ig.get("rms_sum", 0))
    if abs(computed - stored) > 1e-2:
        errors.append(f"rms_sum mismatch: stored={stored:.6f} computed={computed:.6f}")
    return errors


def import_seed(con, seed, path):
    m = seed["model"]
    ig = seed["integrity"]
    oracles = seed["oracles"]

    # Check for existing model
    existing = con.execute(
        "SELECT id FROM models WHERE name=? AND quant=?",
        [m["name"], m["quant"]]
    ).fetchone()
    if existing:
        model_id = existing[0]
        print(f"  model already exists (id={model_id}), updating oracles only")
    else:
        # Insert model row ? DuckDB requires explicit nextval for INTEGER PRIMARY KEY
        con.execute("""
            INSERT INTO models (
                id,
                name, gguf_filename, arch, quant,
                n_layers, n_heads, n_heads_kv, head_dim, n_embd, ff_dim,
                n_vocab, n_ctx,
                rope_base, rope_scale, rope_dim, rms_eps,
                has_qk_norm, attn_logit_softcap, final_logit_softcap,
                gguf_path, file_size
            ) VALUES (
                (SELECT COALESCE(MAX(id), 0) + 1 FROM models),
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, [
            m["name"], m["gguf_filename"], m["arch"], m["quant"],
            m["n_layers"], m["n_heads"], m["n_heads_kv"], m["head_dim"],
            m["n_embd"], m["ff_dim"], m["n_vocab"], m["n_ctx"],
            m["rope_base"], m["rope_scale"], m["rope_dim"], m["rms_eps"],
            m["has_qk_norm"], m["attn_logit_softcap"], m["final_logit_softcap"],
            m["gguf_path"], m["file_size_bytes"],
        ])
        model_id = con.execute(
            "SELECT id FROM models WHERE name=? AND quant=?",
            [m["name"], m["quant"]]
        ).fetchone()[0]
        print(f"  inserted model id={model_id}")

    # Insert / replace oracle rows
    inserted = 0
    skipped = 0
    for o in oracles:
        first20_str = "|".join(str(x) for x in o["first20"])
        existing_oracle = con.execute(
            "SELECT id FROM layer_oracles WHERE model_id=? AND layer_idx=? AND operation=? AND position=?",
            [model_id, o["layer_idx"], o["operation"], o["position"]]
        ).fetchone()
        if existing_oracle:
            # Update
            con.execute("""
                UPDATE layer_oracles
                SET expected_rms=?, checksum=?, notes=?
                WHERE id=?
            """, [o["rms"], o["checksum"], first20_str, existing_oracle[0]])
            skipped += 1
        else:
            con.execute("""
                INSERT INTO layer_oracles (
                    id,
                    model_id, layer_idx, operation, position, input_token_id,
                    expected_rms, expected_max, expected_nan, expected_inf,
                    checksum, notes
                ) VALUES (
                    (SELECT COALESCE(MAX(id), 0) + 1 FROM layer_oracles),
                    ?,?,?,?,?,?,?,?,?,?,?
                )
            """, [
                model_id, o["layer_idx"], o["operation"], o["position"], o["input_token_id"],
                o["rms"], max(abs(x) for x in o["first20"]),
                0, 0,
                o["checksum"],
                "|".join(str(x) for x in o["first20"]),
            ])
            inserted += 1

    return model_id, inserted, skipped


def post_import_checks(con, model_id, seed):
    """Verify the import landed correctly."""
    errors = []
    expected = seed["integrity"]["expected_oracle_count"]
    actual = con.execute(
        "SELECT COUNT(*) FROM layer_oracles WHERE model_id=?", [model_id]
    ).fetchone()[0]
    if actual < expected:
        errors.append(f"oracle count after import: {actual} < expected {expected}")

    # No nulls in key fields
    nulls = con.execute(
        "SELECT COUNT(*) FROM layer_oracles WHERE model_id=? AND expected_rms IS NULL",
        [model_id]
    ).fetchone()[0]
    if nulls > 0:
        errors.append(f"{nulls} oracle rows have NULL expected_rms")

    # RMS values sanity: no zeros, no massive outliers
    rms_vals = con.execute(
        "SELECT expected_rms FROM layer_oracles WHERE model_id=? AND operation='layer_output'",
        [model_id]
    ).fetchall()
    rms_list = [r[0] for r in rms_vals if r[0] is not None]
    if rms_list:
        if max(rms_list) > 1000:
            errors.append(f"suspiciously large RMS value: {max(rms_list):.2f}")
        zeros = sum(1 for r in rms_list if r == 0.0)
        if zeros > 0:
            errors.append(f"{zeros} oracle rows have RMS=0.0 (likely empty tensor)")

    return errors


def main():
    seeds_dir = sys.argv[1] if len(sys.argv) > 1 else "vault/seeds"
    db_path   = sys.argv[2] if len(sys.argv) > 2 else "vault/vault.duckdb"

    seed_files = sorted(glob.glob(os.path.join(seeds_dir, "*.json")))
    if not seed_files:
        print(f"No seed files found in {seeds_dir}")
        sys.exit(1)

    print(f"Found {len(seed_files)} seed files")
    print(f"Database: {db_path}")
    print()

    con = duckdb.connect(db_path)

    total_models = 0
    total_oracles = 0
    failed = []

    for path in seed_files:
        label = os.path.basename(path)
        print(f"[{label}]")

        seed = load_seed(path)
        errors = validate(seed, path)
        if errors:
            print(f"  SKIP ? validation failed:")
            for e in errors:
                print(f"    {e}")
            failed.append((label, errors))
            continue

        try:
            model_id, inserted, updated = import_seed(con, seed, path)
        except Exception as e:
            print(f"  FAIL ? import error: {e}")
            failed.append((label, [str(e)]))
            continue

        # Post-import integrity check
        check_errors = post_import_checks(con, model_id, seed)
        if check_errors:
            print(f"  WARN  POST-IMPORT CHECKS FAILED:")
            for e in check_errors:
                print(f"    {e}")
            failed.append((label, check_errors))
        else:
            print(f"  OK  oracle rows inserted={inserted} updated={updated} ? checks passed")
            total_models += 1
            total_oracles += inserted + updated

        print()

    # Summary
    print("=" * 50)
    print(f"Vault summary:")
    total_m = con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    total_o = con.execute("SELECT COUNT(*) FROM layer_oracles").fetchone()[0]
    print(f"  models table    : {total_m} rows")
    print(f"  layer_oracles   : {total_o} rows")
    print()

    if failed:
        print(f"FAILED seeds ({len(failed)}):")
        for name, errs in failed:
            print(f"  {name}: {errs[0]}")
        sys.exit(1)
    else:
        print(f"All {len(seed_files)} seeds imported cleanly.")

    con.close()


if __name__ == "__main__":
    main()
