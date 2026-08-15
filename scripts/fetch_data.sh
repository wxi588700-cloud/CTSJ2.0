#!/usr/bin/env bash
# Fetch the primary input structures/sequences from public sources (PRD 7.1).
# Files are small; committed to the repo for reproducibility.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDB="$ROOT/data/raw/pdb"
FASTA="$ROOT/data/raw/fasta"
mkdir -p "$PDB" "$FASTA"

for id in 7E5N 7E5M 7PEE 4MZV; do
    if [ ! -s "$PDB/$id.cif" ]; then
        echo "[fetch] $id.cif"
        curl -fsSL "https://files.rcsb.org/download/$id.cif" -o "$PDB/$id.cif"
    fi
done

curl -fsSL "https://rest.uniprot.org/uniprotkb/P09758.fasta" -o "$FASTA/TROP2_human.fasta"
curl -fsSL "https://rest.uniprot.org/uniprotkb/P16422.fasta" -o "$FASTA/EpCAM_human.fasta"

echo "[fetch] done: 7E5N(cis) 7E5M(trans) 7PEE(ECD) 4MZV(EpCAM) + FASTA"
