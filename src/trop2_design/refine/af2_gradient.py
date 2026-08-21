"""M04b: AF2 gradient binder refinement (BindCraft-style, ported recipe).

Ported from trop2-binder's production pipeline where the identical
ColabDesign binder protocol reached Boltz-verified ipTM 0.716.  For each
cleaved state this stage runs gradient trajectories against the T88
neo-epitope (top patch residues + T88), producing sequence-optimised
binder candidates that enter the regular M05/M06 evaluation chain via the
native-sequence passthrough.

Standard outputs: gradient_design/ (inner script artifacts),
candidate_manifest.csv (+ appended rows), candidates.fasta.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from ..io import polymer_residues, read_json, read_structure, write_cif, write_json
from ..io.common import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[3]
INNER_SCRIPT = REPO_ROOT / "scripts" / "af2_gradient_inner.py"


def prepare_target_pdb(cif_path: Path, workdir: Path) -> tuple[Path, dict[int, str]]:
    """Convert a cleaved-state mmCIF to fixed-column PDB with single-letter
    chain IDs (A, B, ...); returns (pdb_path, {resnum: chain_letter})."""
    workdir.mkdir(parents=True, exist_ok=True)
    st = gemmi.read_structure(str(cif_path))
    import string
    chain_of: dict[int, str] = {}
    for i, ch in enumerate(st[0]):
        letter = string.ascii_uppercase[i % 26]
        for r in polymer_residues(ch):
            chain_of.setdefault(r.seqid.num, letter)
        ch.name = letter
    st.setup_entities()
    out = (workdir / "target_input.pdb").resolve()
    st.write_pdb(str(out))
    return out, chain_of


def select_gradient_hotspots(out: Path, top_n: int,
                             chain_map: dict[int, str] | None = None,
                             radius: float = 10.0) -> list[str]:
    """chain_map: {resnum: chain_letter} from prepare_target_pdb (dynamic,
    no hardcoded BODY/NFR assumptions)."""
    """Top patch residues by epitope score (the 0.15 hotspot cut ignored -
    it degenerates to T88-only) + T88 always included; returns author-number
    residues as '{resname}{num}'."""
    patch = read_json(out / "epitope_patch.json")
    residues = patch.get("residues") or []
    t88 = next((r for r in residues
                if r.get("resnum") == 88 and r.get("chain") == "BODY"), None)
    t88_c = np.array(t88["centroid"]) if t88 else np.zeros(3)
    scored = []
    for rec in residues:
        c = np.array(rec.get("centroid", [0, 0, 0]))
        d = float(np.linalg.norm(c - t88_c))
        sasa = rec.get("mean_sasa", 0.0) or 0.0
        std = rec.get("sasa_std", 0.0) or 0.0
        score = (1.0 - min(d / radius, 1.0))
        score *= (0.4 + 0.6 * min(sasa / 120.0, 1.0))
        score *= (1.0 - min(std / 60.0, 0.8))
        scored.append((score, rec))
    scored.sort(key=lambda t: -t[0])
    picked = [rec for _s, rec in scored[: top_n]]
    if t88 is not None and t88 not in picked:
        picked.insert(0, t88)
    chain_map = chain_map or {}
    out_hs = []
    for rec in picked[:top_n]:
        letter = chain_map.get(rec["resnum"])
        if letter is None:
            # dynamic fallback: first converted chain (A) - with a warning
            print(f"[M04b][warn] residue {rec['resnum']} not in chain map; "
                  f"defaulting to chain A")
            letter = "A"
        out_hs.append(f"{letter}{rec['resnum']}")
    return out_hs


class Af2GradientAdapter:
    """Runs scripts/af2_gradient_inner.py in the design env (GPU via ssh)."""

    def __init__(self, spec, workdir: Path):
        self.spec = spec
        self.workdir = Path(workdir)

    def available(self) -> tuple[bool, str]:
        if self.spec is None:
            return False, "tools.yaml has no af2design entry"
        if not INNER_SCRIPT.exists():
            return False, f"inner script missing: {INNER_SCRIPT}"
        py = self._interpreter()
        if not Path(py).exists() and not shlex.which(py):
            return False, f"interpreter not found: {py}"
        params = getattr(self.spec, "params", None)
        if not params or not Path(params).expanduser().exists():
            return False, f"AF2 params dir missing: {params}"
        return True, "ok"

    def _interpreter(self) -> str:
        py = str(self.spec.python).strip() if self.spec.python else ""
        if py.startswith("~"):
            return str(Path(py).expanduser())
        return py or "python"

    def design(self, target_pdb: Path, chain_str: str, hotspots: list[str],
               binder_len: int, n_traj: int, seed: int,
               timeout_s: int | None = None) -> list[dict]:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"af2design unavailable: {why}")
        target_pdb = Path(target_pdb).resolve()
        out_dir = (self.workdir / "gradient_design").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self._interpreter(), str(INNER_SCRIPT),
            "--target", str(target_pdb),
            "--chain", chain_str,
            "--hotspot", ",".join(hotspots),
            "--binder-len", str(binder_len),
            "--n-trajectories", str(n_traj),
            "--seed", str(seed),
            "--out", str(out_dir),
            "--data-dir", str(Path(self.spec.params).expanduser()),
        ]
        timeout_s = timeout_s or n_traj * 1500 + 900
        ssh_host = getattr(self.spec, "ssh_host", None)
        if ssh_host:
            dev = os.environ.get("TROP2_AF2_DEVICE",
                                 os.environ.get("TROP2_BOLTZ_DEVICE", "6"))
            # shared-card guard: AF2 gradient needs ~30GB (0.60 x 48GB);
            # wait rather than OOM on a violated-allocation card
            from ..io.gpu_wait import wait_gpu_free
            if not wait_gpu_free(ssh_host, dev, min_free_mb=32_000,
                                 max_wait_min=180, label="af2-gradient"):
                raise RuntimeError(
                    f"GPU{dev} stayed busy (<32GB free) for 180min - AF2 "
                    f"gradient cannot run; refusing silent degradation")
            remote = (
                f"cd {shlex.quote(str(out_dir))} && "
                f"CUDA_VISIBLE_DEVICES={dev} "
                f"XLA_FLAGS=--xla_gpu_enable_triton_gemm=false "
                f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 "
                f"PYTHONHASHSEED=0 "
                + " ".join(shlex.quote(a) for a in argv))
            argv = ["ssh", ssh_host, remote]
        else:
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_s, cwd=str(self.workdir))
        log_path = self.workdir / "gradient_run.log"
        log_path.write_text((proc.stdout or "")[-8000:] + "\n--stderr--\n"
                            + (proc.stderr or "")[-4000:], encoding="utf-8")
        results_json = out_dir / "design_results.json"
        if proc.returncode != 0 or not results_json.exists():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or ["<no stderr>"]
            raise RuntimeError(f"af2design exit {proc.returncode}: {tail[0][:200]}")
        return json.loads(results_json.read_text(encoding="utf-8"))


def run(ctx) -> None:
    from ..generation.generate import stable_candidate_id

    cfg = ctx.config
    out = ctx.out
    grad_cfg = getattr(cfg.design, "gradient", None)
    if grad_cfg is None or not grad_cfg.enabled:
        log = {"status": "disabled"}
        write_json(out / "gradient_log.json", log)
        ctx.state["gradient"] = log
        return

    spec = getattr(ctx.tools, "af2design", None) if ctx.tools else None
    adapter = Af2GradientAdapter(spec, out / "gradient_work")
    ok, why = adapter.available()
    if not ok:
        # same fail-fast policy as the other generation paths
        cfg.resources.forbid_proxy_degradation(f"AF2 gradient design ({why})")
        log = {"status": "unavailable", "reason": why}
        write_json(out / "gradient_log.json", log)
        ctx.state["gradient"] = log
        return

    state_df = pd.read_csv(out / "state_manifest.csv")
    cleaved = state_df[(state_df.kind == "cleaved") & state_df.audit_passed]
    target_cif = Path(cleaved.iloc[0].file)
    target_pdb, chain_of = prepare_target_pdb(target_cif, out / "gradient_work")

    hot_pdb = select_gradient_hotspots(out, grad_cfg.hotspot_top_n, chain_of)

    results = adapter.design(
        target_pdb, "A,B", hot_pdb,
        binder_len=grad_cfg.binder_len, n_traj=grad_cfg.n_traj,
        seed=ctx.seed + 7919,  # disjoint seed space from other stages
    )
    designed = [r for r in results if r.get("status") == "designed"]

    raw_dir = out / "candidates" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(out / "candidate_manifest.csv").to_dict("records")
    fasta_path = out / "candidates.fasta"
    fasta_lines = [ln for ln in fasta_path.read_text(encoding="utf-8").splitlines()
                   if ln] if fasta_path.exists() else []

    # max_candidates semantics v2 (production lesson): gradient candidates
    # are the HIGHEST-VALUE designs (0.716-recipe) and must not be squeezed
    # out by M04 filling the whole budget - reserve a quota for them:
    #   reserved = min(n_traj, ceil(max_candidates / 2))
    #   M04 keeps max_candidates - reserved slots (enforced in generate.py)
    # gradient then fills up to the reserve even if M04 under-delivers.
    import math
    reserved = min(grad_cfg.n_traj,
                   math.ceil(cfg.design.max_candidates / 2))
    budget = max(0, cfg.design.max_candidates - len(man) + reserved)
    budget = min(budget, len(designed))  # cannot add more than designed
    n_added = 0
    for r in designed:
        if n_added >= budget:
            break
        pdb = Path(r["pdb"])
        if not pdb.exists():
            continue
        st = read_structure(pdb)
        binder = None
        # colabdesign MERGES the two target chains into its chain A; the
        # binder is the chain whose polymer length equals the design length
        for ch in st[0]:
            if len(polymer_residues(ch)) == r["length"]:
                binder = ch
                break
        if binder is None:
            continue
        binder_only = gemmi.Structure()
        binder_only.name = pdb.stem
        bmodel = gemmi.Model("1")
        bmodel.add_chain(binder.clone())
        binder_only.add_model(bmodel)
        binder_only.setup_entities()
        cid = stable_candidate_id("af2_gradient", f"{pdb.stem}_{stable_hash(r['sequence'])}")
        dest = raw_dir / f"{cid}.cif"
        write_cif(binder_only, dest)
        man.append({"candidate_id": cid, "source": "af2_gradient",
                    "sequence": r["sequence"], "file": str(dest),
                    "length": r["length"], "backbone_family": "af2_gradient",
                    # AF2 (ColabDesign) SELF-REPORTED metrics - formally
                    # integrated for provenance; NOT a substitute for the
                    # Boltz-2 cross-validation in M06 (binder-project
                    # evidence: AF2 self-ipTM ~0.1 while Boltz measured 0.7)
                    "af2_plddt": r.get("binder_plddt"),
                    "af2_iptm": r.get("i_ptm"),
                    "af2_ipae": r.get("i_pae")})
        fasta_lines += [f">{cid}|af2_gradient|plddt={r['binder_plddt']}|"
                        f"af2_iptm={r['i_ptm']}", r["sequence"]]
        n_added += 1

    pd.DataFrame(man).to_csv(out / "candidate_manifest.csv", index=False)
    if fasta_lines:
        fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    log = {"status": "ok", "n_designed": len(designed), "n_added": n_added,
           "max_candidates_budget": budget,
           "budget_note": ("budget exhausted by M04 - no gradient candidates "
                           "added" if budget == 0 and designed else ""),
           "hotspots": hot_pdb, "results": results}
    write_json(out / "gradient_log.json", log)
    ctx.state["gradient"] = log
