#!/usr/bin/env python3
"""Cert ledger: record family runs and MATH/CHAT checkoffs.

Uses DuckDB if importable, else SQLite at cert/ledger.sqlite.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _connect(db_path: Path):
    try:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path.with_suffix(".duckdb")))
        backend = "duckdb"
    except Exception:
        import sqlite3

        con = sqlite3.connect(str(db_path.with_suffix(".sqlite")))
        backend = "sqlite"
    return con, backend


def init_schema(con, backend: str) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS family_runs (
            id INTEGER PRIMARY KEY,
            family_id VARCHAR,
            ts VARCHAR,
            git_sha VARCHAR,
            math_ok BOOLEAN,
            chat_ok BOOLEAN,
            n_reds INTEGER,
            report_path VARCHAR,
            reds_path VARCHAR
        )
        """
        if backend == "sqlite"
        else """
        CREATE TABLE IF NOT EXISTS family_runs (
            id INTEGER,
            family_id VARCHAR,
            ts VARCHAR,
            git_sha VARCHAR,
            math_ok BOOLEAN,
            chat_ok BOOLEAN,
            n_reds INTEGER,
            report_path VARCHAR,
            reds_path VARCHAR
        )
        """
    )
    if backend == "duckdb":
        # duckdb has no autoincrement PK the same way; use sequence-less max+1
        pass
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS reds (
            run_id INTEGER,
            code VARCHAR,
            detail VARCHAR
        )
        """
    )
    # The advertised-models table (what generate_models_table.py reads). One row
    # per certified model/quant combo. This is the schema the model-matrix reader
    # depends on; keep in sync with query in generate_models_table.py.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS math_runs (
            model_id VARCHAR,
            family VARCHAR,
            quant VARCHAR,
            certified INTEGER,
            ts VARCHAR
        )
        """
    )
    try:
        con.commit()
    except Exception:
        pass


def next_id(con, backend: str) -> int:
    row = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM family_runs").fetchone()
    return int(row[0])


def record_run(
    db_path: Path,
    *,
    family_id: str,
    math_ok: bool,
    chat_ok: bool | None,
    n_reds: int,
    report_path: str,
    reds_path: str,
    git_sha: str = "",
    red_codes: list[tuple[str, str]] | None = None,
) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con, backend = _connect(db_path)
    init_schema(con, backend)
    rid = next_id(con, backend)
    ts = datetime.now(timezone.utc).isoformat()
    chat = False if chat_ok is None else chat_ok
    con.execute(
        """
        INSERT INTO family_runs
        (id, family_id, ts, git_sha, math_ok, chat_ok, n_reds, report_path, reds_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [rid, family_id, ts, git_sha, math_ok, chat, n_reds, report_path, reds_path],
    )
    for code, detail in red_codes or []:
        con.execute(
            "INSERT INTO reds (run_id, code, detail) VALUES (?, ?, ?)",
            [rid, code, detail],
        )
    # Advertised-model row: parse family_id "qwen3-0.6b-q4-k-m" into
    # (family=qwen3, model=qwen3-0.6b, quant=q4-k-m). Certifiable = math green.
    # naive split: drop the trailing quant segment if it looks like qXY/k.
    parts = family_id.split("-")
    quant = ""
    if len(parts) >= 2 and parts[-1][0:1] in ("q", "Q") and "." not in parts[-1]:
        quant = parts[-1]
    model_id = "-".join(parts[:-1]) if quant else family_id
    family = parts[0] if parts else family_id
    con.execute(
        "INSERT INTO math_runs (model_id, family, quant, certified, ts) VALUES (?, ?, ?, ?, ?)",
        [model_id, family, quant, 1 if math_ok else 0, ts],
    )
    try:
        con.commit()
    except Exception:
        pass
    con.close()
    return rid


def list_status(db_path: Path) -> None:
    if (
        not db_path.with_suffix(".duckdb").exists()
        and not db_path.with_suffix(".sqlite").exists()
    ):
        # try both via connect
        pass
    con, backend = _connect(db_path)
    init_schema(con, backend)
    rows = con.execute(
        """
        SELECT family_id, ts, math_ok, chat_ok, n_reds
        FROM family_runs
        ORDER BY ts DESC
        """
    ).fetchall()
    print(f"backend={backend} path_base={db_path}")
    if not rows:
        print("(no runs)")
        return
    print(f"{'family_id':32} {'math':5} {'chat':5} {'n_reds':6} ts")
    for r in rows[:50]:
        print(f"{r[0]:32} {str(r[2]):5} {str(r[3]):5} {r[4]:6} {r[1]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cert" / "ledger",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--family-id", required=True)
    rec.add_argument("--reds-json", type=Path, required=True)
    rec.add_argument("--report", type=Path, default=None)
    rec.add_argument(
        "--chat-ok", choices=["true", "false", "unknown"], default="unknown"
    )
    rec.add_argument("--git-sha", default="")

    sub.add_parser("status")

    args = ap.parse_args(argv)
    if args.cmd == "status":
        list_status(args.db)
        return 0

    reds_doc = json.loads(args.reds_json.read_text(encoding="utf-8"))
    math_ok = bool(reds_doc.get("math_ok"))
    n_reds = int(reds_doc.get("n_reds") or len(reds_doc.get("reds") or []))
    chat_ok = None if args.chat_ok == "unknown" else args.chat_ok == "true"
    red_codes = [
        (r.get("code", "?"), str(r.get("detail", "")))
        for r in (reds_doc.get("reds") or [])
    ]
    report = str(args.report or args.reds_json.parent / "REPORT.md")
    rid = record_run(
        args.db,
        family_id=args.family_id,
        math_ok=math_ok,
        chat_ok=chat_ok,
        n_reds=n_reds,
        report_path=report,
        reds_path=str(args.reds_json),
        git_sha=args.git_sha,
        red_codes=red_codes,
    )
    print(f"recorded run_id={rid} math_ok={math_ok} n_reds={n_reds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
