#!/usr/bin/env python3
"""
vault import_seeds.py
=====================
Reads JSON seed files, auto-heals integrity metadata, then inserts
model + oracle rows into vault/vault.duckdb idempotently.

Usage:
    python vault/scripts/import_seeds.py [seeds_dir] [vault_db]

Defaults:
    seeds_dir = vault/seeds
    vault_db  = vault/vault.duckdb

Auto-healing:
    - Fixes "unknown" quant by extracting from gguf_filename
    - Recalculates rms_sum if stale
    - Fixes expected_oracle_count if wrong

Idempotency:
    - Upserts models by (normalized_name, quant) — no duplicates
    - Upserts oracles by (model_id, layer_idx, operation, position)
"""

import json
import sys
import os
import re
import glob
import duckdb

QUANT_PATTERN = re.compile(r'(Q[0-9]_[A-Z0-9_]+|F[0-9]{2})$', re.IGNORECASE)

# Known non-standard GGUF filenames -> correct quant
KNOWN_QUANTS = {
    'phi3-mini-4k-instruct-q4.gguf': 'q4_0',
}


def extract_quant(gguf_filename):
    """Derive quant from GGUF filename."""
    if gguf_filename in KNOWN_QUANTS:
        return KNOWN_QUANTS[gguf_filename]
    base = gguf_filename.replace('.gguf', '').replace('.GGUF', '')
    m = QUANT_PATTERN.search(base)
    if m:
        return m.group(1).lower()
    return 'unknown'


def heal_seed(seed):
    """Auto-heal integrity metadata in seed dict. Returns list of changes."""
    changes = []
    m = seed.get('model', {})
    ig = seed.get('integrity', {})

    # Fix quant from filename (only if currently unknown AND can extract)
    gguf = m.get('gguf_filename', '') or os.path.basename(seed.get('source_gguf', ''))
    if gguf:
        correct = extract_quant(gguf)
        old_quant = m.get('quant', '')
        if correct and old_quant == 'unknown' and correct != 'unknown' and old_quant != correct:
            m['quant'] = correct
            changes.append(f'quant: {old_quant} -> {correct}')
        elif correct and correct == 'unknown' and old_quant != 'unknown':
            # Keep the non-standard quant (like 'phi3-mini-4k-instruct-q4') — don't overwrite
            pass

    # Fix rms_sum
    oracles = seed.get('oracles', [])
    actual_sum = sum(float(o['rms']) for o in oracles)
    stored = float(ig.get('rms_sum', 0))
    if abs(actual_sum - stored) > 1e-2:
        ig['rms_sum'] = actual_sum
        changes.append(f'rms_sum: {stored:.6f} -> {actual_sum:.6f}')

    # Fix expected_oracle_count
    expected = ig.get('expected_oracle_count', -1)
    if expected != len(oracles):
        ig['expected_oracle_count'] = len(oracles)
        changes.append(f'count: {expected} -> {len(oracles)}')

    return changes


def find_existing_model(con, name, quant):
    """Case-insensitive model lookup by (name, quant)."""
    return con.execute(
        "SELECT id, name FROM models WHERE LOWER(name) = LOWER(?) AND quant = ?",
        [name, quant]
    ).fetchone()


def import_seed(con, seed):
    m = seed['model']
    ig = seed['integrity']
    oracles = seed['oracles']

    # Check for existing model (case-insensitive)
    existing = find_existing_model(con, m['name'], m['quant'])
    if existing:
        model_id, existing_name = existing
        # Update name to match seed if different case
        if existing_name != m['name']:
            con.execute("UPDATE models SET name = ? WHERE id = ?", [m['name'], model_id])
        print(f"  model exists (id={model_id}), updating oracles")
    else:
        con.execute("""
            INSERT INTO models (
                id, name, gguf_filename, arch, quant,
                n_layers, n_heads, n_heads_kv, head_dim, n_embd, ff_dim,
                n_vocab, n_ctx, rope_base, rope_scale, rope_dim, rms_eps,
                has_qk_norm, attn_logit_softcap, final_logit_softcap,
                gguf_path, file_size
            ) VALUES (
                (SELECT COALESCE(MAX(id), 0) + 1 FROM models),
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, [
            m['name'], m['gguf_filename'], m['arch'], m['quant'],
            m['n_layers'], m['n_heads'], m['n_heads_kv'], m['head_dim'],
            m['n_embd'], m['ff_dim'], m['n_vocab'], m['n_ctx'],
            m['rope_base'], m['rope_scale'], m['rope_dim'], m['rms_eps'],
            m['has_qk_norm'], m['attn_logit_softcap'], m['final_logit_softcap'],
            m['gguf_path'], m['file_size_bytes'],
        ])
        model_id = find_existing_model(con, m['name'], m['quant'])[0]
        print(f"  insert model id={model_id}")

    # Upsert oracles
    inserted = 0
    updated = 0
    for o in oracles:
        first20_str = '|'.join(str(x) for x in o['first20'])
        existing_o = con.execute(
            "SELECT id FROM layer_oracles WHERE model_id=? AND layer_idx=? AND operation=? AND position=?",
            [model_id, o['layer_idx'], o['operation'], o['position']]
        ).fetchone()
        if existing_o:
            con.execute("""
                UPDATE layer_oracles SET expected_rms=?, checksum=?, notes=?
                WHERE id=?
            """, [o['rms'], o['checksum'], first20_str, existing_o[0]])
            updated += 1
        else:
            con.execute("""
                INSERT INTO layer_oracles (
                    id, model_id, layer_idx, operation, position, input_token_id,
                    expected_rms, expected_max, expected_nan, expected_inf,
                    checksum, notes
                ) VALUES (
                    (SELECT COALESCE(MAX(id), 0) + 1 FROM layer_oracles),
                    ?,?,?,?,?,?,?,?,?,?,?
                )
            """, [
                model_id, o['layer_idx'], o['operation'], o['position'], o['input_token_id'],
                o['rms'], max(abs(x) for x in o['first20']),
                0, 0,
                o['checksum'],
                first20_str,
            ])
            inserted += 1

    return model_id, inserted, updated


def main():
    seeds_dir = sys.argv[1] if len(sys.argv) > 1 else 'vault/seeds'
    db_path   = sys.argv[2] if len(sys.argv) > 2 else 'vault/vault.duckdb'

    seed_files = sorted(glob.glob(os.path.join(seeds_dir, '*.json')))
    candle_dir = os.path.join(seeds_dir, 'candle')
    seed_files = [f for f in seed_files if not f.startswith(candle_dir)]

    print(f"Found {len(seed_files)} seed files")
    print(f"Database: {db_path}")
    print()

    con = duckdb.connect(db_path)
    imported = 0
    total_o = 0

    for path in seed_files:
        label = os.path.basename(path)
        print(f'[{label}]')

        with open(path, encoding='utf-8') as f:
            seed = json.load(f)

        # Auto-heal
        if seed.get('seed_version') != 1:
            print(f'  SKIP: unsupported seed_version {seed.get("seed_version")}')
            continue
        ig = seed.get('integrity', {})
        if ig.get('nan_count', -1) != 0 or ig.get('inf_count', -1) != 0:
            print(f'  SKIP: nan_count={ig.get("nan_count")} inf_count={ig.get("inf_count")}')
            continue

        changes = heal_seed(seed)
        if changes:
            for c in changes:
                print(f'  heal: {c}')
            # Persist heal to disk
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(seed, f, indent=2)

        try:
            model_id, ins, upd = import_seed(con, seed)
            total_o += ins + upd
            imported += 1
            print(f'  OK  inserted={ins} updated={upd} total={(ins + upd)}')
        except Exception as e:
            print(f'  FAIL: {e}')
            con.rollback()

        print()

    # Summary
    con.commit()
    total_m = con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    total_o_db = con.execute("SELECT COUNT(*) FROM layer_oracles").fetchone()[0]
    print('=' * 50)
    print(f'Imported {imported}/{len(seed_files)} seeds')
    print(f'models: {total_m}, layer_oracles: {total_o_db}')
    con.close()


if __name__ == '__main__':
    main()
