#!/usr/bin/env bash
# Copy previously downloaded external algorithm checkouts into ./external
# (git-ignored; weights exceed GitHub's 100 MB file limit and are never pushed).
#
# Usage:
#   scripts/setup_external_tools.sh [SOURCE_DIR]
#
# SOURCE_DIR resolution order (no hardcoded absolute paths):
#   1. first argument
#   2. $TROP2_EXTERNAL_SRC environment variable
#   3. ./external itself (already-populated checkouts are kept in place)
# RFdiffusion: https://github.com/RosettaCommons/RFdiffusion  (weights included)
# ProteinMPNN: https://github.com/dauparas/ProteinMPNN
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/external"

SRC="${1:-${TROP2_EXTERNAL_SRC:-$DST}}"

mkdir -p "$DST"

if [ "$SRC" = "$DST" ] && [ ! -d "$DST/RFdiffusion" ] && [ ! -d "$DST/ProteinMPNN" ]; then
    echo "[setup] no source given and $DST is empty." >&2
    echo "[setup] clone the repos above into any directory, then:" >&2
    echo "[setup]   scripts/setup_external_tools.sh /path/to/that/directory" >&2
    exit 1
fi

for tool in ProteinMPNN RFdiffusion; do
    if [ -d "$SRC/$tool" ]; then
        if [ -d "$DST/$tool" ]; then
            echo "[setup] $tool already present at $DST/$tool (skipped)"
        else
            echo "[setup] copying $tool -> $DST/$tool"
            cp -r "$SRC/$tool" "$DST/$tool"
        fi
    else
        echo "[setup] WARNING: $tool not found in $SRC" >&2
    fi
done

echo "[setup] done. Edit configs/tools.yaml paths if you keep tools elsewhere."
