"""M04 stage: candidate generation + import with unified validation.

Two entry points per PRD: RFdiffusion generation (adapter; deterministic
scaffold fallback on CPU-only machines) and external FASTA / PDB import
(AC-04: legal candidates get stable IDs, illegal ones are rejected with
recorded reasons).

Standard outputs: candidates/raw/*.pdb, candidates.fasta,
candidate_manifest.csv, generation_log.json.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import gemmi
import numpy as np

from ..io import (
    first_protein_chain, polymer_residues, read_fasta, read_json,
    read_structure, validate_protein_sequence, write_cif, write_json,
)
from .adapters import RfdiffusionAdapter
from .scaffolds import generate_scaffolds


def stable_candidate_id(*parts) -> str:
    """Stable full-pipeline candidate id (PRD 7.3: never renumbered by rank)."""
    blob = ":".join(str(p) for p in parts).encode()
    return "CAND-" + hashlib.sha256(blob).hexdigest()[:8].upper()


def backbone_from_ca(ca: np.ndarray) -> gemmi.Structure:
    """Materialise a CA-only backbone structure (one GLY residue per CA;
    ProteinMPNN/heuristic design works from CA + ideal geometry)."""
    st = gemmi.Structure()
    st.name = "scaffold"
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("B")
    for i, p in enumerate(ca, start=1):
        res = gemmi.Residue()
        res.name = "GLY"
        res.seqid = gemmi.SeqId(i, " ")
        res.het_flag = "A"
        res.subchain = f"B{i}"
        a = gemmi.Atom()
        a.name = "CA"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(*p)
        a.occ = 1.0
        a.b_iso = 30.0
        res.add_atom(a)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


def check_geometry(ca: np.ndarray) -> list[str]:
    """Basic backbone sanity: bond-length outliers and self-clashes."""
    errors = []
    if len(ca) < 10:
        errors.append(f"too few CA atoms ({len(ca)})")
        return errors
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    bad = int(((d < 2.5) | (d > 5.0)).sum())
    if bad > len(d) * 0.05:
        errors.append(f"{bad} CA-CA bond outliers")
    dm = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    np.fill_diagonal(dm, 99.0)
    if len(dm) > 3:
        close = dm[np.triu_indices(len(ca), k=3)]
        clashes = int((close < 3.2).sum())
        if clashes > 0:
            errors.append(f"{clashes} long-range CA clashes < 3.2 A")
    return errors


def import_fasta_candidates(path: Path, len_range: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    """Import external binder sequences; returns (accepted, rejected)."""
    accepted, rejected = [], []
    records = read_fasta(path)
    lo, hi = len_range
    for header, seq in records.items():
        errors = validate_protein_sequence(seq, min_len=lo, max_len=hi)
        # repeat-content check
        runs = max((len(m.group(0)) for m in __import__("re").finditer(r"(.)\1*", seq)), default=0)
        if runs > 8:
            errors.append(f"low-complexity repeat run of {runs}")
        seq = seq.upper()
        if errors:
            rejected.append({"header": header, "sequence": seq, "errors": errors})
        else:
            accepted.append({
                "candidate_id": stable_candidate_id("import", header, hashlib.sha256(seq.encode()).hexdigest()[:12]),
                "header": header, "sequence": seq, "source": "import",
            })
    return accepted, rejected


def import_pdb_candidates(directory: Path, len_range: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    lo, hi = len_range
    for pdb in sorted(directory.glob("*.pdb")) + sorted(directory.glob("*.cif")):
        try:
            st = read_structure(pdb)
            model = st[0]
            protein_chains = [c for c in model if len(polymer_residues(c)) >= lo]
            if len(protein_chains) != 1:
                rejected.append({"file": pdb.name,
                                 "errors": [f"expected 1 protein chain, found {len(protein_chains)}"]})
                continue
            ch = protein_chains[0]
            seq = "".join(gemmi.one_letter_code([r.name for r in polymer_residues(ch)]))
            n = len(polymer_residues(ch))
            if not (lo <= n <= hi):
                rejected.append({"file": pdb.name,
                                 "errors": [f"length {n} outside {lo}-{hi}"]})
                continue
            ca = np.array([[a.find_atom("CA", "*").pos.x,
                            a.find_atom("CA", "*").pos.y,
                            a.find_atom("CA", "*").pos.z]
                           for a in [polymer_residues(ch)[0]]])  # placeholder replaced below
            cas = []
            for r in polymer_residues(ch):
                ca = r.find_atom("CA", "*")
                if ca is not None:
                    cas.append([ca.pos.x, ca.pos.y, ca.pos.z])
            geo_errors = check_geometry(np.asarray(cas))
            if geo_errors:
                rejected.append({"file": pdb.name, "errors": geo_errors})
                continue
            accepted.append({
                "candidate_id": stable_candidate_id("import", pdb.name, sha_of_file := __import__("hashlib").sha256(pdb.read_bytes()).hexdigest()[:12]),
                "file": str(pdb), "sequence": seq, "source": "import",
            })
        except Exception as exc:  # corrupt files must fail explicitly
            rejected.append({"file": pdb.name, "errors": [f"parse failure: {exc}"]})
    return accepted, rejected


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    raw_dir = out / "candidates" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log: dict = {"rfdiffusion": {}, "fallback": {}, "import": {}}
    manifest: list[dict] = []
    fasta_records: list[tuple[str, str]] = []

    # ---- T88 patch centroid from M03
    epitope = read_json(out / "epitope_patch.json")
    patch_centroid = np.mean([r["centroid"] for r in epitope["residues"][:8]], axis=0)

    # ---- hotspot list
    hotspots = []
    hot_file = out / "hotspots.txt"
    for line in hot_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            hotspots.append(line.split()[0].lstrip("#"))

    n_gen = cfg.design.max_candidates  # imports are additive on top

    # ---- 1) RFdiffusion adapter (smoke-capable)
    adapter = RfdiffusionAdapter(ctx.tools.rfdiffusion if ctx.tools else None,
                                 workdir=out / "rfdiffusion_work")
    # target for diffusion: first cleaved state
    import pandas as pd
    state_df = pd.read_csv(out / "state_manifest.csv")
    cleaved = state_df[(state_df.kind == "cleaved") & state_df.audit_passed]
    target_cif = Path(cleaved.iloc[0].file)
    result = adapter.design_binder(
        target_cif, hotspots, n=n_gen, seed=ctx.seed,
        binder_len=cfg.design.binder_length,
    )
    if result.ok:
        log["rfdiffusion"] = {"status": "ok", "n_pdbs": len(result.pdbs)}
        for pdb in result.pdbs:
            st = read_structure(pdb)
            ch = first_protein_chain(st, None)
            seq = gemmi.one_letter_code([r.name for r in polymer_residues(ch)])
            cid = stable_candidate_id("rfdiffusion", pdb.name)
            dest = raw_dir / f"{cid}.cif"
            write_cif(st, dest)
            manifest.append({"candidate_id": cid, "source": "rfdiffusion",
                             "sequence": seq, "file": str(dest),
                             "length": len(seq), "backbone_family": "rfdiffusion"})
            fasta_records.append((f"{cid}|rfdiffusion", seq))
    else:
        log["rfdiffusion"] = {"status": "unavailable", "reason": result.reason}

    # ---- 2) deterministic scaffold fallback (keeps pipeline runnable)
    used_fallback = "rfdiffusion" not in {m["source"] for m in manifest}
    if used_fallback:
        # ALL target heavy atoms of the first cleaved state, so placement
        # keeps a safe distance from side chains, not just the CA trace
        target_pts = None
        try:
            from ..io import polymer_residues as _pr
            tgt_st = read_structure(target_cif)
            pts = [[a.pos.x, a.pos.y, a.pos.z] for ch in tgt_st[0]
                   for r in _pr(ch) for a in r
                   if a.element.name not in ("H", "D")]
            target_pts = np.asarray(pts) if pts else None
        except Exception:
            target_pts = None
        # anchor = the T88 free alpha-amino N atom (the hard-gate contact)
        anchor = None
        try:
            from ..schemas.project import parse_residue_id
            _, right_num = parse_residue_id(cfg.target.cleavage.right_residue)
            for ch in tgt_st[0]:
                for r in polymer_residues(ch):
                    if r.seqid.num == right_num:
                        n_atom = r.find_atom("N", "*")
                        if n_atom is not None:
                            anchor = np.array([n_atom.pos.x, n_atom.pos.y, n_atom.pos.z])
                        break
                if anchor is not None:
                    break
        except Exception:
            anchor = None
        scaffolds = generate_scaffolds(n_gen, cfg.design.binder_length,
                                       patch_centroid, seed=ctx.seed,
                                       target_pts=target_pts,
                                       anchor_point=anchor)
        for key, ca, contacts, topo in scaffolds:
            geo = check_geometry(ca)
            if geo:
                continue
            cid = stable_candidate_id("fallback", key, ctx.seed)
            st = backbone_from_ca(ca)
            dest = raw_dir / f"{cid}.cif"
            write_cif(st, dest)
            np.save(raw_dir / f"{cid}_contacts.npy", contacts)
            manifest.append({"candidate_id": cid, "source": "scaffold_fallback",
                             "sequence": "", "file": str(dest),
                             "length": len(ca), "backbone_family": topo,
                             "contacts_file": str(raw_dir / f"{cid}_contacts.npy")})
        log["fallback"] = {"status": "ok", "n_generated": len(manifest)}

    # ---- 3) imports
    if cfg.design.import_fasta and Path(cfg.design.import_fasta).exists():
        acc, rej = import_fasta_candidates(Path(cfg.design.import_fasta),
                                           cfg.design.binder_length)
        for c in acc:
            dest = raw_dir / f"{c['candidate_id']}.fasta.txt"
            dest.write_text(c["sequence"])
            manifest.append({"candidate_id": c["candidate_id"], "source": "import",
                             "sequence": c["sequence"], "file": str(dest),
                             "length": len(c["sequence"]), "backbone_family": "imported"})
            fasta_records.append((f"{c['candidate_id']}|import", c["sequence"]))
        log["import"]["fasta"] = {"accepted": len(acc), "rejected": rej}
    if cfg.design.import_pdb_dir and Path(cfg.design.import_pdb_dir).exists():
        acc, rej = import_pdb_candidates(Path(cfg.design.import_pdb_dir),
                                         cfg.design.binder_length)
        for c in acc:
            st = read_structure(c["file"])
            dest = raw_dir / f"{c['candidate_id']}.cif"
            write_cif(st, dest)
            manifest.append({"candidate_id": c["candidate_id"], "source": "import",
                             "sequence": c["sequence"], "file": str(dest),
                             "length": c["length"], "backbone_family": "imported"})
            fasta_records.append((f"{c['candidate_id']}|import", c["sequence"]))
        log["import"]["pdb"] = {"accepted": len(acc), "rejected": rej}

    if not manifest:
        raise RuntimeError("M04 produced zero candidates (generation + import both empty)")

    # limit to max_candidates, deterministic order
    manifest = manifest[: cfg.design.max_candidates]

    import pandas as pd
    pd.DataFrame(manifest).to_csv(out / "candidate_manifest.csv", index=False)
    from ..io import write_fasta
    write_fasta(out / "candidates.fasta", fasta_records)
    log["final_candidate_count"] = len(manifest)
    write_json(out / "generation_log.json", log)
    ctx.state["candidates"] = manifest
