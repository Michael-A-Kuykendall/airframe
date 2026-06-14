#!/usr/bin/env bash
# export_code_to_md.sh
# Exports all relevant Rust source + WGSL files from airframe and shimmy
# into a single markdown document for cloud review.
#
# Usage:
#   bash scripts/export_code_to_md.sh > docs/internal/code-export-$(date +%Y-%m-%d).md
#
# What it captures:
#   - All .rs files in airframe/src/ and airframe/crates/
#   - All .wgsl shader files
#   - shimmy/src/ engine and API files
#   - Cargo.toml for both repos
#   - Current git log (last 20 commits per repo)
#   - Current diff (uncommitted changes)

AIRFRAME=/c/Users/micha/repos/airframe
SHIMMY=/c/Users/micha/repos/shimmy
DATE=$(date +%Y-%m-%d)

echo "# Airframe + Shimmy Full Code Export"
echo "**Generated**: $DATE"
echo "**Branch (airframe)**: $(cd $AIRFRAME && git branch --show-current)"
echo "**Branch (shimmy)**: $(cd $SHIMMY && git branch --show-current)"
echo ""

# ─── GIT LOG ───────────────────────────────────────────────────────────────

echo "## Recent Git History"
echo ""
echo "### airframe (last 20 commits)"
echo '```'
cd $AIRFRAME && git log --oneline -20
echo '```'
echo ""
echo "### shimmy (last 20 commits)"
echo '```'
cd $SHIMMY && git log --oneline -20
echo '```'
echo ""

# ─── GIT DIFF ──────────────────────────────────────────────────────────────

echo "## Uncommitted Changes"
echo ""
echo "### airframe diff"
echo '```diff'
cd $AIRFRAME && git diff HEAD
echo '```'
echo ""
echo "### shimmy diff"
echo '```diff'
cd $SHIMMY && git diff HEAD
echo '```'
echo ""

# ─── CARGO TOML ────────────────────────────────────────────────────────────

echo "## Cargo Manifests"
echo ""
echo "### airframe/Cargo.toml"
echo '```toml'
cat $AIRFRAME/Cargo.toml
echo '```'
echo ""
echo "### shimmy/Cargo.toml"
echo '```toml'
cat $SHIMMY/Cargo.toml
echo '```'
echo ""

# ─── AIRFRAME SOURCE FILES ──────────────────────────────────────────────────

echo "## Airframe Source Files"
echo ""

# Core inference path — most important files first
PRIORITY_FILES=(
  "src/runtime/gpu.rs"
  "src/backend/bindless/pipeline/inference.rs"
  "src/backend/bindless/pipeline/dequant.rs"
  "src/backend/bindless/loader.rs"
  "src/backend/bindless/metadata.rs"
  "src/backend/bindless/preflight.rs"
  "src/backend/bindless/kv_cache.rs"
  "src/core/spec.rs"
  "src/core/routing.rs"
)

for f in "${PRIORITY_FILES[@]}"; do
  FULL="$AIRFRAME/$f"
  if [ -f "$FULL" ]; then
    echo "### $f"
    echo '```rust'
    cat "$FULL"
    echo '```'
    echo ""
  fi
done

# WGSL shaders
echo "## WGSL Shaders"
echo ""
for wgsl in $AIRFRAME/src/backend/bindless/*.wgsl; do
  name=$(basename "$wgsl")
  echo "### $name"
  echo '```wgsl'
  cat "$wgsl"
  echo '```'
  echo ""
done

# Remaining Rust files (not already listed)
echo "## Remaining Airframe Source"
echo ""
find $AIRFRAME/src -name "*.rs" | sort | while read f; do
  rel="${f#$AIRFRAME/}"
  # Skip already-covered priority files
  skip=0
  for p in "${PRIORITY_FILES[@]}"; do
    if [ "$rel" = "$p" ]; then skip=1; break; fi
  done
  if [ $skip -eq 0 ]; then
    echo "### $rel"
    echo '```rust'
    cat "$f"
    echo '```'
    echo ""
  fi
done

# airframe_observe crate
echo "## airframe_observe Crate"
echo ""
find $AIRFRAME/crates -name "*.rs" | sort | while read f; do
  rel="${f#$AIRFRAME/}"
  echo "### $rel"
  echo '```rust'
  cat "$f"
  echo '```'
  echo ""
done

# ─── SHIMMY SOURCE FILES ────────────────────────────────────────────────────

echo "## Shimmy Source Files"
echo ""
find $SHIMMY/src -name "*.rs" | sort | while read f; do
  rel="${f#$SHIMMY/}"
  echo "### $rel"
  echo '```rust'
  cat "$f"
  echo '```'
  echo ""
done
