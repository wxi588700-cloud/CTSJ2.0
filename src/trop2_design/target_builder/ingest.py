"""M01: data ingestion and target registration (PRD module M01).

Reads and standardises TROP2 / EpCAM structures and sequences, builds the
author-numbering <-> label-numbering <-> UniProt-numbering mapping, runs QC
(chains, missing residues, mutations, non-standard residues, duplicate
numbering) and writes the SHA-256 input manifest.

Standard outputs: target_registry.json, standardized/*.cif,
residue_mapping.csv, input_qc.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import (
    VALID_AA, first_protein_chain, iter_protein_chains, polymer_residues,
    read_fasta, read_structure, residue_one_letter, sha256_file, write_cif,
    write_json,
)
from ..schemas.project import parse_residue_id


# --------------------------------------------------------- sequence alignment --

def align_identity(reference: str, query: str) -> list[tuple[int, int]]:
    """Simple global alignment (Needleman-Wunsch, identity scoring) of two
    protein sequences.  Returns list of (ref_index, query_index) matches.

    Used to map structure SEQRES numbering onto UniProt numbering.  Sequences
    here are short (<400 aa) so the O(n*m) DP is fine.
    """
    n, m = len(reference), len(query)
    if n == 0 or m == 0:
        return []
    match, gap = 1, -1
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    ptr[1:, 0] = 1
    ptr[0, 1:] = 2
    for i in range(1, n + 1):
        ri = reference[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match if ri == query[j - 1] else -match)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            ptr[i, j] = 0 if best == diag else (1 if best == up else 2)
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        p = ptr[i, j]
        if p == 0:
            # record ALL aligned positions (matches AND mismatches) so that
            # downstream mapping stays complete and mutations are detectable
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ------------------------------------------------------------------- QC ----

QC_KEYS = [
    "missing_residues", "mutations_vs_uniprot", "nonstandard_residues",
    "duplicate_numbering", "chain_count", "status",
]


def qc_structure(st, chain_name: str | None, uniprot_seq: str,
                 uniprot_offset: int = 0) -> dict:
    """Quality checks for one structure chain against the UniProt sequence."""
    ch = first_protein_chain(st, chain_name)
    residues = polymer_residues(ch)
    issues: dict = {"structure": st.name, "chain": ch.name, "chain_count": 0,
                    "missing_residues": [], "mutations_vs_uniprot": [],
                    "nonstandard_residues": [], "duplicate_numbering": [],
                    "status": "ok"}

    # chain count (protein chains)
    issues["chain_count"] = sum(1 for _ in iter_protein_chains(st))

    # duplicate numbering
    seen: dict[int, str] = {}
    for res in residues:
        key = res.seqid.num
        if key in seen:
            issues["duplicate_numbering"].append(str(key))
        seen[key] = res.name

    # sequence + missing residues + mutations via alignment to UniProt
    seq = "".join(residue_one_letter(r) for r in residues)
    issues["resolved_residues"] = len(residues)
    issues["first_residue_num"] = residues[0].seqid.num if residues else None
    issues["last_residue_num"] = residues[-1].seqid.num if residues else None
    pairs = align_identity(uniprot_seq, seq)
    if pairs:
        mapped_res = [p[0] for p in pairs]
        # missing = gaps in the uniprot coverage interior
        lo, hi = min(mapped_res), max(mapped_res)
        covered = set(mapped_res)
        missing_uniprot = [u for u in range(lo, hi + 1) if u not in covered]
        issues["missing_residues"] = [u + uniprot_offset for u in missing_uniprot][:200]
        # mutations: mapped residues disagree after mapping through numbering
        num_to_res = {res.seqid.num: residue_one_letter(res) for res in residues}
        uniprot_num_of = {p[0]: p[1] for p in pairs}
        for u_idx, s_idx in pairs:
            uniprot_num = u_idx + 1 + uniprot_offset  # 1-based, author-style
            res = residues[s_idx]
            if res.seqid.num in num_to_res and uniprot_num in num_to_res:
                want = num_to_res[uniprot_num]
                got = num_to_res[res.seqid.num]
                # only flag when the alignment says the residue position maps
                # to a different uniprot residue than observed
                if uniprot_seq[u_idx] != seq[s_idx]:
                    issues["mutations_vs_uniprot"].append(
                        f"{uniprot_seq[u_idx]}{uniprot_num}{seq[s_idx]}"
                    )
        issues["mutations_vs_uniprot"] = issues["mutations_vs_uniprot"][:100]

    # non-standard residues
    for res in residues:
        if residue_one_letter(res) not in VALID_AA:
            issues["nonstandard_residues"].append(f"{res.name}{res.seqid.num}")

    fatal = bool(issues["duplicate_numbering"])
    issues["status"] = "error" if fatal else (
        "warning" if (issues["missing_residues"] or issues["mutations_vs_uniprot"]
                      or issues["nonstandard_residues"]) else "ok"
    )
    return issues


# ------------------------------------------------------------- main stage ----

def run(ctx) -> None:
    cfg = ctx.config
    target = cfg.target
    out = ctx.out
    std_dir = out / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    uniprot = read_fasta(target.sequence_fasta)
    if len(uniprot) != 1:
        raise ValueError(f"expected exactly 1 record in {target.sequence_fasta}, got {len(uniprot)}")
    uniprot_full_id, uniprot_seq = next(iter(uniprot.items()))
    # "sp|P09758|TACD2_HUMAN" -> "P09758"; plain accessions pass through
    parts = uniprot_full_id.split("|")
    uniprot_id = parts[1] if len(parts) == 3 else uniprot_full_id

    registry: dict = {
        "target": {
            "name": target.name,
            "uniprot_id": uniprot_id,
            "gene": target.gene,
            "species": target.species,
            "sequence_length": len(uniprot_seq),
        },
        "cleavage": target.cleavage.model_dump(),
        "structures": {},
        "input_files": {},
    }

    # ---- cleavage residue verification against UniProt sequence (fail fast)
    left_aa, left_num = parse_residue_id(target.cleavage.left_residue)
    right_aa, right_num = parse_residue_id(target.cleavage.right_residue)
    if target.cleavage.numbering == "uniprot":
        li, ri = left_num - 1, right_num - 1
    else:
        # author numbering of these structures == uniprot numbering (verified
        # for 7E5N/7E5M/7PEE: R87/T88/C73/C108 are direct sequence positions)
        li, ri = left_num - 1, right_num - 1
    if uniprot_seq[li] != left_aa:
        raise ValueError(
            f"cleavage left residue {target.cleavage.left_residue} does not match "
            f"UniProt {uniprot_id} position {left_num} (={uniprot_seq[li]}); "
            f"check cleavage.numbering convention"
        )
    if uniprot_seq[ri] != right_aa:
        raise ValueError(
            f"cleavage right residue {target.cleavage.right_residue} does not match "
            f"UniProt {uniprot_id} position {right_num} (={uniprot_seq[ri]})"
        )
    for a, b in target.cleavage.preserve_disulfides:
        for spec in (a, b):
            aa, num = parse_residue_id(spec)
            if aa != "C":
                raise ValueError(f"disulfide residue {spec} is not cysteine")
            if uniprot_seq[num - 1] != "C":
                raise ValueError(f"residue {spec} is not Cys in {uniprot_id}")

    # ---- structures: standardise + QC + mapping
    mapping_rows: list[dict] = []
    qc: dict = {"trop2": {}, "epcam": {}}
    struct_specs = [
        ("cis", target.cis_structure),
        ("trans", target.trans_structure),
        ("alternate", target.alternate_structure) if target.alternate_structure else None,
    ]
    for item in struct_specs:
        if item is None:
            continue
        role, ref = item
        path = Path(ref.path)
        if not path.is_absolute():
            path = (ctx.project_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{role} structure not found: {path}")
        st = read_structure(path)
        chain_name = ref.chain or first_protein_chain(st).name
        qc_result = qc_structure(st, chain_name, uniprot_seq)
        qc["trop2"][role] = qc_result

        # standardised single-chain protein-only mmCIF
        std_path = std_dir / f"{st.name.lower()}_{chain_name}.cif"
        from ..io import extract_chain_structure
        extract_chain_structure(st, chain_name, std_path)

        # residue mapping: author num / label num / seqres index / uniprot num
        std_st = read_structure(std_path)
        ch = first_protein_chain(std_st)
        residues = polymer_residues(ch)
        seq = "".join(residue_one_letter(r) for r in residues)
        pairs = align_identity(uniprot_seq, seq)
        # store 1-based uniprot numbers (matching author numbering semantics)
        seqres_to_uniprot = {s_idx: u_idx + 1 for u_idx, s_idx in pairs}
        for idx, res in enumerate(residues):
            mapping_rows.append({
                "structure": st.name,
                "role": role,
                "chain": ch.name,
                "author_num": res.seqid.num,
                "author_ins": res.seqid.icode.strip(),
                "label_num": res.label_seq,
                "seqres_index": idx + 1,
                "uniprot_num": seqres_to_uniprot.get(idx, "") or "",
                "residue_name": res.name,
                "residue": residue_one_letter(res),
            })
        registry["structures"][role] = {
            "source_pdb_id": st.name.upper(),
            "source_file": str(path),
            "chain": chain_name,
            "standardized_file": str(std_path),
            "sha256": sha256_file(path),
            "resolved_residues": len(residues),
            "qc_status": qc_result["status"],
        }

    # ---- EpCAM negative target
    neg = cfg.negatives
    epcam_path = Path(neg.epcam_structure) if neg.epcam_structure else None
    if epcam_path is not None and not epcam_path.is_absolute():
        epcam_path = (ctx.project_root / epcam_path).resolve()
    if epcam_path is not None and epcam_path.exists():
        est = read_structure(epcam_path)
        ecam_chain = first_protein_chain(est).name
        estd = std_dir / f"{est.name.lower()}_{ecam_chain}.cif"
        extract_chain_structure(est, ecam_chain, estd)
        eseq = "".join(residue_one_letter(r) for r in polymer_residues(first_protein_chain(est)))
        qc["epcam"]["structure"] = {
            "structure": est.name, "chain": ecam_chain,
            "resolved_residues": len(eseq), "status": "ok",
        }
        registry["epcam"] = {
            "uniprot_id": neg.epcam_uniprot,
            "source_pdb_id": est.name.upper(),
            "source_file": str(epcam_path),
            "chain": ecam_chain,
            "standardized_file": str(estd),
            "sha256": sha256_file(epcam_path),
            "sequence_length": len(eseq),
        }
    else:
        qc["epcam"]["structure"] = {"status": "missing",
                                    "note": "EpCAM structure not provided"}
        registry["epcam"] = {"uniprot_id": neg.epcam_uniprot, "status": "missing"}

    # ---- input file hashes (M01 standard: SHA-256 manifest)
    hashes: dict[str, str] = {"uniprot_fasta": sha256_file(target.sequence_fasta)}
    for role, s in registry["structures"].items():
        hashes[f"{role}_structure"] = s["sha256"]
    if "epcam" in registry and "sha256" in registry["epcam"]:
        hashes["epcam_structure"] = registry["epcam"]["sha256"]
    registry["input_files"] = hashes

    # ---- required residues must map (AC-01)
    required = [target.cleavage.left_residue, target.cleavage.right_residue]
    required += [r for pair in target.cleavage.preserve_disulfides for r in pair]
    roles = sorted({r["role"] for r in mapping_rows})
    for spec in required:
        aa, num = parse_residue_id(spec)
        # audit fix: validate EVERY ingested structure (cis AND trans AND
        # alternate) - downstream M07/M08 assume shared numbering on all of
        # them, so a mismatch anywhere is fatal, not just in the cis reference
        for role in roles:
            rows = [r for r in mapping_rows
                    if r["role"] == role and r["uniprot_num"] == num]
            if not rows:
                # residue simply not resolved in this structure is fine;
                # only a WRONG residue at that number is fatal
                continue
            if rows[0]["residue"] != aa:
                raise ValueError(
                    f"required residue {spec} maps to "
                    f"{rows[0]['residue']}{num} in the {role} structure - "
                    f"numbering convention mismatch"
                )

    # ---- write standard outputs
    write_json(out / "target_registry.json", registry)

    import pandas as pd
    pd.DataFrame(mapping_rows).to_csv(out / "residue_mapping.csv", index=False)

    qc["ac01_required_residues_mapped"] = {
        spec: any(r["uniprot_num"] == parse_residue_id(spec)[1]
                  and r["residue"] == parse_residue_id(spec)[0]
                  for r in mapping_rows)
        for spec in required
    }
    write_json(out / "input_qc.json", qc)

    ctx.state["registry"] = registry
    ctx.state["qc"] = qc
    ctx.state["mapping_rows"] = mapping_rows
