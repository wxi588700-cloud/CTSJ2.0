#!/usr/bin/env bash
# Copy previously downloaded external algorithm checkouts into ./external
# (git-ignored; weights exceed GitHub's 100 MB file limit and are never pushed).
#
# Usage: scripts/setup_external_tools.sh [SOURCE_DIR]
#   SOURCE_DIR defaults to /home/protein_design2026/external
set -euo pipefail

SRC="${1:-/home/protein_design2026/external}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/external"

mkdir -p "$DST"

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
