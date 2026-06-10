#!/usr/bin/env python3
"""
vault vault_certify.py
----------------------
Compare Airframe oracle rows against Candle probe output.
Marks each oracle row as: certified, disputed, or provisional.

Usage:
    python vault/scripts/vault_certify.py [vault_db] [candle_seeds_dir]

Defaults:
    vault_db        = vault/vault.duckdb
    candle_seeds_dir = vault/seeds/candle

Certification tiers:
    certified   - Airframe and Candle agree within tolerance
    disputed    - Airframe and Candle disagree beyond tolerance (needs investigation)
    provisional - No Candle probe data available for this model

Tolerances:
    layer_output:  delta_rms < 0.01  (1% - hidden states may differ slightly by quant path)
    final_logits:  delta_rms < 1.0   (logit scale differences are expected due to dequant)
    Note: if delta > tolerance but both are NaN-free and checksums are nonzero, row is
    'disputed' not 'invalid' - it needs investigation, not rejection.
"""

import json
import sys
import os
import glob
import duckdb
from datetime import datetime, timezone

LAYER_OUTPUT_TOLERANCE = 0.01
FINAL_LOGITS_TOLERANCE = 1.0   # wide tolerance for logit comparison - quant path differences expected

def load_candle_seed(path):
    with open(path) as f:
        return json.load(f)

def certify_model(con, model_id, model_name, candle_seed):
    """Compare vault oracles against candle probe data for one model."""
    candle_layers = {l['layer_idx']: l for l in candle_seed.get('layers', [])}
    candle_version = candle_seed.get('tool', 'unknown')

    vault_oracles = con.execute(
        "SELECT id, layer_idx, operation, expected_rms FROM layer_oracles WHERE model_id=?",
        [model_id]
    ).fetchall()

    certified = 0
    disputed = 0
    skipped = 0

    for oracle_id, layer_idx, operation, airframe_rms in vault_oracles:
        candle_entry = candle_layers.get(layer_idx)

        # Also try matching final_logits by operation name
        if candle_entry is None and operation == 'final_logits':
            candle_entry = candle_layers.get(-1)

        if candle_entry is None:
            # No candle data for this layer - stay provisional
            skipped += 1
            continue

        candle_rms = float(candle_entry.get('rms', 0))
        delta = abs(airframe_rms - candle_rms) if (airframe_rms and candle_rms) else None

        tolerance = FINAL_LOGITS_TOLERANCE if operation == 'final_logits' else LAYER_OUTPUT_TOLERANCE

        if delta is None:
            status = 'provisional'
        elif delta <= tolerance:
            status = 'certified'
            certified += 1
        else:
            status = 'disputed'
            disputed += 1

        # Record in cross_validations table
        con.execute("""
            INSERT OR REPLACE INTO cross_validations (
                model_id, layer_idx, operation,
                airframe_rms, candle_rms, delta_rms,
                candle_version, validated_at, pass, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, [
            model_id, layer_idx, operation,
            airframe_rms, candle_rms, delta,
            candle_version,
            datetime.now(timezone.utc).isoformat(),
            status == 'certified',
            f"delta={delta:.6f} tolerance={tolerance}" if delta is not None else "no candle data"
        ])

    return certified, disputed, skipped


def ensure_cross_validations_table(con):
    """Create cross_validations table if it doesn't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS cross_validations (
            id              INTEGER,
            model_id        INTEGER NOT NULL REFERENCES models(id),
            layer_idx       INTEGER NOT NULL,
            operation       VARCHAR NOT NULL,
            airframe_rms    REAL,
            candle_rms      REAL,
            delta_rms       REAL,
            candle_version  VARCHAR NOT NULL,
            validated_at    TIMESTAMP,
            pass            BOOLEAN NOT NULL,
            notes           TEXT,
            UNIQUE(model_id, layer_idx, operation, candle_version)
        )
    """)


def main():
    vault_db       = sys.argv[1] if len(sys.argv) > 1 else "vault/vault.duckdb"
    candle_dir     = sys.argv[2] if len(sys.argv) > 2 else "vault/seeds/candle"

    candle_files = glob.glob(os.path.join(candle_dir, "*.json"))
    if not candle_files:
        print(f"No candle seed files found in {candle_dir}")
        print("Run candle_probe on your models first.")
        sys.exit(1)

    con = duckdb.connect(vault_db)
    ensure_cross_validations_table(con)

    print(f"Found {len(candle_files)} candle probe files")
    print(f"Database: {vault_db}")
    print()

    total_certified = 0
    total_disputed = 0
    total_skipped = 0

    for candle_path in candle_files:
        label = os.path.basename(candle_path)
        seed = load_candle_seed(candle_path)
        source = seed.get('source_gguf', '')
        gguf_name = os.path.basename(source) if source else label

        # Find matching model in vault
        rows = con.execute(
            "SELECT id, name, quant FROM models WHERE gguf_filename=? OR gguf_path LIKE ?",
            [gguf_name, f"%{gguf_name}%"]
        ).fetchall()

        if not rows:
            print(f"[{label}] SKIP - no matching model in vault for {gguf_name}")
            continue

        model_id, model_name, quant = rows[0]
        print(f"[{model_name} / {quant}]")

        certified, disputed, skipped = certify_model(con, model_id, model_name, seed)
        total_certified += certified
        total_disputed += disputed
        total_skipped += skipped

        if disputed > 0:
            print(f"  WARN  certified={certified} DISPUTED={disputed} provisional={skipped}")
        else:
            print(f"  OK  certified={certified} provisional={skipped}")
        print()

    # Summary
    print("-" * 50)
    print(f"Certification summary:")
    total = total_certified + total_disputed + total_skipped
    print(f"  certified   : {total_certified}/{total}")
    print(f"  disputed    : {total_disputed}/{total}")
    print(f"  provisional : {total_skipped}/{total}")
    print()

    if total_disputed > 0:
        print("WARN  Disputed rows need investigation.")
        print("   Check delta_rms values: high delta on final_logits is expected (quant path)")
        print("   High delta on layer_output rows is a bug signal.")
    elif total_certified > 0:
        print(f"OK  {total_certified} oracle rows certified against Candle reference.")

    # Show detailed disputed rows
    if total_disputed > 0:
        print()
        print("Disputed rows:")
        rows = con.execute("""
            SELECT m.name, m.quant, cv.layer_idx, cv.operation,
                   cv.airframe_rms, cv.candle_rms, cv.delta_rms
            FROM cross_validations cv
            JOIN models m ON cv.model_id = m.id
            WHERE cv.pass = false
            ORDER BY cv.delta_rms DESC
            LIMIT 20
        """).fetchall()
        for r in rows:
            print(f"  {r[0]}/{r[1]} layer={r[2]} op={r[3]} "
                  f"airframe={r[4]:.4f} candle={r[5]:.4f} delta={r[6]:.4f}")

    con.close()


if __name__ == "__main__":
    main()
