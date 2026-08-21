"""M09: developability and risk (PRD M09).

Pure-python implementations of ProtParam-style properties (MW, pI, net
charge, GRAVY), a CamSol-style solubility score, an exposure-weighted
aggregation estimate (Aggrescan3D-style), sequence liability rules
(deamidation, oxidation, isomerisation, protease motifs, unpaired Cys,
N-glycosylation motifs, N-terminal cyclisation) and an MHC-II presentation
risk screen (NetMHCIIpan adapter when configured; deterministic propensity
fallback flagged 'proxy'; tool-missing -> 'review' per AC-11).

Standard outputs: developability_metrics.csv, liability_flags.csv,
immunogenicity_hits.csv.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_fasta, write_json

# ------------------------------------------------------------ ProtParam data --

RESIDUE_MW = {  # residue (minus H2O) masses Da
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}
WATER = 18.010565

PKA = {  # EMBOSS-style pKa set
    "C": 8.5, "D": 3.9, "E": 4.1, "H": 6.5, "K": 10.8, "R": 12.5, "Y": 10.1,
    "Nterm": 8.6, "Cterm": 3.6,
}
HYDROPHOBICITY_KD = {  # Kyte-Doolittle
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
AGGREGATION_PROPENSITY = {  # TANGO/AGGRESCAN-style qualitative weights
    "A": 1.1, "C": 0.8, "D": -1.5, "E": -1.5, "F": 1.9, "G": 0.9,
    "H": -0.5, "I": 1.9, "K": -1.8, "L": 1.9, "M": 1.4, "N": -0.8,
    "P": -2.0, "Q": -0.8, "R": -1.8, "S": -0.6, "T": -0.4, "V": 1.6,
    "W": 1.5, "Y": 1.1,
}


def molecular_weight(seq: str) -> float:
    return sum(RESIDUE_MW.get(a, 110.0) for a in seq) + WATER


def net_charge(seq: str, ph: float) -> float:
    charge = 0.0
    for a in seq:
        if a in ("K", "R"):
            charge += 1.0 / (1.0 + 10 ** (ph - PKA[a]))
        elif a in ("H",):
            charge += 1.0 / (1.0 + 10 ** (ph - PKA[a]))
        elif a in ("D", "E", "C", "Y"):
            charge -= 1.0 / (1.0 + 10 ** (PKA[a] - ph))
    charge += 1.0 / (1.0 + 10 ** (ph - PKA["Nterm"]))
    charge -= 1.0 / (1.0 + 10 ** (PKA["Cterm"] - ph))
    return charge


def isoelectric_point(seq: str) -> float:
    lo, hi = 2.0, 13.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def gravy(seq: str) -> float:
    return sum(HYDROPHOBICITY_KD.get(a, 0.0) for a in seq) / max(len(seq), 1)


def camsol_like_score(seq: str) -> float:
    """CamSol-style solubility (0-1 normalised): hydrophobicity windows
    penalised, net charge and proline/glycine rewarded."""
    n = len(seq)
    if n == 0:
        return 0.0
    win = 7
    worst = 1.0
    scores = []
    for i in range(n - win + 1):
        w = seq[i:i + win]
        scores.append(sum(HYDROPHOBICITY_KD.get(a, 0) for a in w) / win)
    worst_window = max(scores) if scores else 0.0
    charge = abs(net_charge(seq, 7.4))
    pg = (seq.count("P") + seq.count("G")) / n
    raw = 1.2 - 0.28 * worst_window - 0.12 * abs(charge - 4.0) + 0.8 * min(pg, 0.18)
    return float(np.clip(raw, 0.0, 1.0))


def aggregation_hotspots(seq: str, contacts_per_res=None) -> tuple[float, list[int]]:
    """Exposure-weighted aggregation risk in [0,1] plus hotspot positions.

    With a contact map (fallback scaffolds) buried hydrophobics are
    down-weighted, mimicking Aggrescan3D's structural correction.
    """
    n = len(seq)
    if n == 0:
        return 0.0, []
    exposed = np.ones(n)
    if contacts_per_res is not None and len(contacts_per_res) == n:
        exposed = np.clip(1.0 - contacts_per_res / 10.0, 0.15, 1.0)
    prop = np.array([AGGREGATION_PROPENSITY.get(a, 0.0) for a in seq])
    weighted = prop * exposed
    risk = float(np.clip(weighted.mean() / 1.2 * 0.5 + 0.5, 0.0, 1.0))
    hot = [i + 1 for i in range(n) if weighted[i] > 1.5]
    return round(risk, 3), hot[:20]


# ------------------------------------------------------------ liability rules --

def sequence_liabilities(seq: str) -> list[dict]:
    flags = []
    n = len(seq)
    # unpaired cysteines
    cys = [i for i, a in enumerate(seq, 1) if a == "C"]
    if cys and len(cys) % 2 == 1:
        flags.append({"liability": "unpaired_cys", "positions": cys,
                      "severity": "high", "note": "odd number of Cys -> free thiol risk"})
    # deamidation NG/NS/NT/NA
    for motif in ("NG", "NS", "NT", "NA"):
        pos = [i for i in range(n - 1) if seq[i:i + 2] == motif]
        if pos:
            flags.append({"liability": f"deamidation_{motif}", "positions":
                          [p + 1 for p in pos], "severity": "medium"})
    # isomerisation DG/DS/DT
    for motif in ("DG", "DS", "DT"):
        pos = [i for i in range(n - 1) if seq[i:i + 2] == motif]
        if pos:
            flags.append({"liability": f"isomerisation_{motif}",
                          "positions": [p + 1 for p in pos], "severity": "medium"})
    # oxidation: exposed M/W counted generously
    pos = [i for i, a in enumerate(seq, 1) if a in "MW"]
    if len(pos) > 4:
        flags.append({"liability": "oxidation_MW", "positions": pos, "severity": "low"})
    # protease motifs
    for motif in ("KR", "RR", "KK", "XR", "KX"):
        if motif in ("XR", "KX"):
            continue
        pos = [i for i in range(n - 1) if seq[i:i + 2] == motif]
        if pos:
            flags.append({"liability": f"protease_{motif}",
                          "positions": [p + 1 for p in pos], "severity": "low"})
    # unintended N-glycosylation N-X-S/T (X != P)
    for i in range(n - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            flags.append({"liability": "nglycosylation_motif",
                          "positions": [i + 1], "severity": "medium"})
    # N-terminal cyclisation
    if seq and seq[0] in "QE":
        flags.append({"liability": "nterm_cyclisation", "positions": [1], "severity": "low"})
    # renal clearance note for small bare proteins (PRD M09)
    if 60 <= n <= 120:
        flags.append({"liability": "rapid_renal_clearance_risk",
                      "positions": [], "severity": "info",
                      "note": "60-120 aa bare miniprotein; expect fast kidney clearance"})
    return flags


# ---------------------------------------------------------------- MHC-II risk --

class NetMHCIIpanAdapter:
    def __init__(self, spec):
        self.spec = spec

    def available(self) -> bool:
        if self.spec is None:
            return False
        return bool(shutil.which("netMHCIIpan") or
                    (self.spec.command and shutil.which(self.spec.command[0])))


MHC2_PROPENSITY = {  # allele-agnostic 9-core enrichment heuristic (proxy)
    "F": 1.4, "W": 1.5, "Y": 1.2, "I": 1.3, "L": 1.3, "V": 1.1, "M": 1.2,
    "A": 0.9, "C": 0.8, "G": 0.7, "P": 0.5, "S": 0.7, "T": 0.7,
    "N": 0.6, "Q": 0.7, "D": 0.5, "E": 0.5, "K": 0.5, "R": 0.5, "H": 0.6,
}


def mhc2_risk_peptides(seq: str, top_k: int = 10) -> list[dict]:
    """15-mer MHC-II presentation propensity screen (deterministic proxy).

    Flagged 'proxy'; replace with NetMHCIIpan-4.3 output when the tool is
    installed (adapter above) - per PRD this is risk ranking only, never a
    claim of non-immunogenicity.
    """
    n = len(seq)
    if n < 15:
        return []
    scored = []
    for i in range(n - 14):
        pep = seq[i:i + 15]
        core = pep[3:12]
        score = sum(MHC2_PROPENSITY.get(a, 0.8) for a in core) / 9.0
        scored.append((round(score, 3), i + 1, pep))
    scored.sort(reverse=True)
    return [{"peptide": p, "start": s, "propensity": sc}
            for sc, s, p in scored[:top_k]]


# ------------------------------------------------------------- main stage ----

def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    pos = pd.read_csv(out / "positive_state_metrics.csv")
    agg = pos[pos.state_id == "AGGREGATE"]
    shortlist = agg[agg.positive_state_pass_rate > 0]
    if shortlist.empty:
        pd.DataFrame(columns=["candidate_id", "design_name", "mw_da", "pI",
                              "solubility_score", "aggregation_risk"]).to_csv(
            out / "developability_metrics.csv", index=False)
        pd.DataFrame(columns=["candidate_id", "liability", "severity"]).to_csv(
            out / "liability_flags.csv", index=False)
        pd.DataFrame(columns=["candidate_id", "peptide", "propensity"]).to_csv(
            out / "immunogenicity_hits.csv", index=False)
        ctx.state["developability"] = []
        return

    mono = pd.read_csv(out / "monomer_metrics.csv")
    cand_manifest = pd.read_csv(out / "candidate_manifest.csv").set_index("candidate_id")

    adapter = NetMHCIIpanAdapter(ctx.tools.netmhc2pan if ctx.tools else None)
    netmhc_available = adapter.available()
    if not netmhc_available:
        # audit fix: MHC-II risk silently degrades to a deterministic proxy
        ctx.config.resources.forbid_proxy_degradation(
            "NetMHCIIpan immunogenicity screening")
    elif netmhc_available:
        print("[M09][warn] netMHCIIpan binary found but no execution adapter "
              "is implemented - MHC-II risk stays heuristic (labelled proxy)")

    rows: list[dict] = []
    flags_rows: list[dict] = []
    immuno_rows: list[dict] = []

    for _, arow in shortlist.iterrows():
        cid = arow.candidate_id
        sub = mono[(mono.candidate_id == cid) & (mono.status == "pass")]
        if sub.empty:
            continue
        for _, mrow in sub.iterrows():
            seq = str(mrow.sequence)
            contacts = None
            cand = cand_manifest.loc[cid]
            cfile = cand.get("contacts_file") if hasattr(cand, "get") else None
            if isinstance(cfile, str) and Path(cfile).exists():
                contacts = np.load(cfile).sum(axis=1)
            agg_risk, hotspots = aggregation_hotspots(seq, contacts)
            sol = camsol_like_score(seq)
            mw = molecular_weight(seq)
            pi = isoelectric_point(seq)
            charge = net_charge(seq, 7.4)
            grav = gravy(seq)
            liabilities = sequence_liabilities(seq)

            mhc_peptides = mhc2_risk_peptides(seq)
            if mhc_peptides:
                mhc2_risk = round(float(np.mean(
                    [p["propensity"] for p in mhc_peptides[:5]])), 3)
            else:
                mhc2_risk = 0.0
            # audit fix (external review P0): the adapter has NO execution
            # path (available() only) - the value is ALWAYS the heuristic;
            # labelling it netmhc2pan when the binary exists was a
            # provenance lie.
            mhc_source = "proxy"

            rows.append({
                "candidate_id": cid, "design_name": mrow.design_name,
                "length": len(seq),
                "mw_da": round(mw, 1),
                "pI": pi,
                "net_charge_pH7.4": round(charge, 2),
                "gravy": round(grav, 3),
                "solubility_score": round(sol, 3),
                "aggregation_risk": agg_risk,
                "aggregation_hotspots": ";".join(map(str, hotspots)),
                "mhc2_risk": mhc2_risk,
                "mhc2_source": mhc_source,
                "liability_count": len([f for f in liabilities if f["severity"] != "info"]),
                "developability_flags": ";".join(
                    f["liability"] for f in liabilities if f["severity"] != "info"),
            })
            for f in liabilities:
                flags_rows.append({
                    "candidate_id": cid, "design_name": mrow.design_name,
                    "liability": f["liability"], "severity": f["severity"],
                    "positions": ";".join(map(str, f.get("positions", []))),
                    "note": f.get("note", ""),
                })
            for p in mhc_peptides[:5]:
                immuno_rows.append({
                    "candidate_id": cid, "design_name": mrow.design_name,
                    "peptide": p["peptide"], "start": p["start"],
                    "propensity": p["propensity"], "source": mhc_source,
                })

    pd.DataFrame(rows).to_csv(out / "developability_metrics.csv", index=False)
    pd.DataFrame(flags_rows).to_csv(out / "liability_flags.csv", index=False)
    pd.DataFrame(immuno_rows).to_csv(out / "immunogenicity_hits.csv", index=False)
    write_json(out / "developability_log.json", {
        "netmhc2pan_available": netmhc_available,
        "note": "MHC-II numbers are ranking proxies unless netmhc2pan ran",
    })
    ctx.state["developability"] = rows
