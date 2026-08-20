"""Shared I/O helpers: structures (gemmi), FASTA, CSV, hashes.

Structures are handled as mmCIF internally (PRD 5.2); PDB is only used at
tool boundaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import gemmi
import numpy as np

# ---------------------------------------------------------------- hashing --

def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(payload: str | dict | list) -> str:
    """Stable hash of a JSON-able object (sorted keys)."""
    if isinstance(payload, str):
        blob = payload.encode()
    else:
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ------------------------------------------------------------------ json/yaml --

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # encoding fix: never locale-dependent (Windows cp936 would mangle/crash
    # on non-ASCII paths or payloads written with ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stable_hash(text: str) -> int:
    """Process-independent stable hash (SHA-256 derived).

    MUST be used instead of builtin ``hash()`` for anything feeding a random
    seed or a file name: builtin ``hash()`` on str is salted per process
    (PYTHONHASHSEED), which silently broke the reproducibility claims.
    """
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------- fasta --

def read_fasta(path: Path | str) -> dict[str, str]:
    """Return {record_id: sequence}. Record id is the header up to first whitespace."""
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_protein_sequence(seq: str, min_len: int = 20, max_len: int = 400) -> list[str]:
    """Return list of validation errors (empty list == valid)."""
    errors = []
    seq = seq.upper().strip()
    if not seq:
        errors.append("empty sequence")
        return errors
    bad = sorted(set(seq) - VALID_AA - {"X"})
    if bad:
        errors.append(f"invalid characters: {bad}")
    if len(seq) < min_len:
        errors.append(f"length {len(seq)} < minimum {min_len}")
    if len(seq) > max_len:
        errors.append(f"length {len(seq)} > maximum {max_len}")
    if seq.count("X") > max(1, len(seq) // 20):
        errors.append("too many unknown residues (X)")
    return errors


# -------------------------------------------------------------- structures --

def read_structure(path: Path | str) -> gemmi.Structure:
    """Read mmCIF or PDB (by suffix) into a gemmi Structure with entities set up."""
    path = Path(path)
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def write_cif(st: gemmi.Structure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = st.make_mmcif_document()
    doc.write_file(str(path), gemmi.cif.Style.Aligned)
    return path


def polymer_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    """Protein residues of a chain (skip waters/ligands/glycans)."""
    out = []
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        if info is None:
            continue
        if info.is_amino_acid():
            out.append(res)
    return out


def chain_sequence(chain: gemmi.Chain) -> str:
    return gemmi.one_letter_code([r.name for r in polymer_residues(chain)])


def residue_one_letter(res: gemmi.Residue) -> str:
    code = gemmi.find_tabulated_residue(res.name)
    if code is None:
        return "X"
    one = code.one_letter_code.upper()
    return one if one in VALID_AA else "X"


def atom_coords(residues: Iterable[gemmi.Residue], names: Iterable[str] | None = None) -> np.ndarray:
    """Stack coordinates of all atoms (optionally filtered by name) in residue order."""
    wanted = set(names) if names else None
    pts = []
    for res in residues:
        for atom in res:
            if wanted is None or atom.name in wanted:
                pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not pts:
        return np.zeros((0, 3))
    return np.asarray(pts, dtype=np.float64)


def ca_coords(chain: gemmi.Chain) -> np.ndarray:
    pts = []
    for res in polymer_residues(chain):
        ca = res.find_atom("CA", "*")
        if ca is not None:
            pts.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3))


def find_residue(chain: gemmi.Chain, number: int, altloc_first: bool = True) -> gemmi.Residue | None:
    """Find first protein residue with the given author sequence number."""
    for res in polymer_residues(chain):
        if res.seqid.num == number:
            return res
    return None


def first_protein_chain(st: gemmi.Structure, chain_name: str | None = None) -> gemmi.Chain:
    model = st[0]
    if chain_name is not None:
        ch = model.find_chain(chain_name)
        if ch is None:
            raise KeyError(f"chain {chain_name!r} not found in {st.name}; have {[c.name for c in model]}")
        return ch
    for ch in model:
        if len(polymer_residues(ch)) >= 10:
            return ch
    raise ValueError(f"no protein chain with >=10 residues found in {st.name}")


def extract_chain_structure(st: gemmi.Structure, chain_name: str, out_path: Path,
                            new_name: str | None = None) -> Path:
    """Write a single-chain (protein-only) structure to mmCIF."""
    src = first_protein_chain(st, chain_name)
    new = gemmi.Structure()
    new.name = f"{st.name}_{src.name}" if new_name is None else new_name
    new.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    ch = gemmi.Chain(new_name or src.name)
    for res in polymer_residues(src):
        ch.add_residue(res.clone())
    model.add_chain(ch)
    new.add_model(model)
    new.setup_entities()
    return write_cif(new, out_path)


def iter_protein_chains(st: gemmi.Structure) -> Iterator[gemmi.Chain]:
    for ch in st[0]:
        if len(polymer_residues(ch)) >= 10:
            yield ch


AA3_TO_1 = {  # minimal fallback map for sequence ops
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
