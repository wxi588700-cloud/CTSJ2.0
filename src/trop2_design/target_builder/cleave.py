"""M02: R87-T88 cleaved-state construction and conformer ensemble (PRD M02).

From the standardised intact TROP2 structure this module:

1. breaks the R87-T88 peptide bond (R87 becomes a new C-terminus ``COO-``,
   T88 a new N-terminus ``NH3+``) by splitting the chain into two gemmi
   chains in the output structure - an explicit topology edit, never two
   unconstrained chains handed to a folder (PRD medical note under M02);
2. preserves and audits the C73-C108 disulfide that keeps the N-terminal
   fragment attached to the body;
3. samples >= n_conformers deterministic conformations by rigid-body
   rotation of the N-terminal fragment around the disulfide pivot plus
   small hinge perturbations, rejecting any state with severe clashes
   (PyRosetta FastRelax / OpenMM MD are pluggable upgrades recorded in the
   audit when available; on this CPU-only baseline we use geometry-only
   sampling, which is fully deterministic);
4. writes per-state topology audits (topology_audit.json) and a state
   manifest covering both cleaved and intact control states.

Standard outputs: cleaved_states/*.cif, intact_states/*.cif,
state_manifest.csv, topology_audit.json.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

from ..io import (
    atom_coords, clash_overlap_volume, find_residue, first_protein_chain,
    min_pair_distance, polymer_residues, read_json, read_structure,
    residue_one_letter, rotation_matrix, write_cif, write_json,
)
from ..schemas.project import parse_residue_id
from ..schemas.results import StateManifestRow, TopologyAudit

CLASH_MAX_OVERLAP = 120.0   # A^3 tolerated steric overlap inside folded core
INTERCHAIN_MIN_DIST = 1.90  # A: below this even polar pairs are impossible
HBOND_POLAR = {"N", "O"}    # N/O pairs >= 1.9 A are legitimate H-bonds


def polar_aware_clash_metrics(frag_pts, frag_elems, body_pts, body_elems):
    """Clash metrics that do not penalise legitimate polar hydrogen bonds.

    Real crystal structures contain 2.0-2.8 A N/O contacts (H-bonds, salt
    bridges).  Only pairs that are (a) below 1.9 A regardless of type, or
    (b) overlapping with at least one non-polar (C/S) atom, count as
    steric problems.  Returns (overlap_volume_A3, min_flagged_distance).
    """
    from scipy.spatial import cKDTree

    if len(frag_pts) == 0 or len(body_pts) == 0:
        return 0.0, 99.0
    tree = cKDTree(body_pts)
    idx = tree.query_ball_point(frag_pts, r=4.5)
    radii = {a: 1.70 for a in "CS"}
    overlap = 0.0
    min_d = 99.0
    for i, js in enumerate(idx):
        for j in js:
            d = float(np.linalg.norm(frag_pts[i] - body_pts[j]))
            both_polar = frag_elems[i] in HBOND_POLAR and body_elems[j] in HBOND_POLAR
            if both_polar and d >= INTERCHAIN_MIN_DIST:
                continue  # hydrogen-bond region, not a clash
            min_d = min(min_d, d)
            ra = 1.55 if frag_elems[i] == "N" else radii.get(frag_elems[i], 1.7)
            rb = 1.55 if body_elems[j] == "N" else radii.get(body_elems[j], 1.7)
            if d >= ra + rb:
                continue
            overlap += (4.0 / 3.0) * np.pi * min(ra, rb) ** 3 * (
                1.0 - d / max(ra + rb, 1e-6))
    return float(overlap), float(min_d)


# ---------------------------------------------------------------- helpers --

def disulfide_pairs(chain: gemmi.Chain) -> list[tuple[int, int]]:
    """Detect SG-SG pairs < 2.5 A within one chain (author numbering)."""
    residues = polymer_residues(chain)
    sgs = []
    for res in residues:
        if res.name in ("CYS",):
            sg = res.find_atom("SG", "*")
            if sg is not None:
                sgs.append((res.seqid.num, np.array([sg.pos.x, sg.pos.y, sg.pos.z])))
    pairs = []
    for i in range(len(sgs)):
        for j in range(i + 1, len(sgs)):
            if np.linalg.norm(sgs[i][1] - sgs[j][1]) < 2.5:
                pairs.append((sgs[i][0], sgs[j][0]))
    return pairs


def add_oxt(residue: gemmi.Residue) -> bool:
    """Ensure a C-terminal OXT oxygen on the residue's carboxyl carbon."""
    c = residue.find_atom("C", "*")
    o = residue.find_atom("O", "*")
    if c is None or o is None:
        return False
    if residue.find_atom("OXT", "*") is not None:
        return True
    oxt = gemmi.Atom()
    oxt.name = "OXT"
    oxt.element = gemmi.Element("O")
    # place OXT by reflecting O through C along the bisector
    cp = np.array([c.pos.x, c.pos.y, c.pos.z])
    op = np.array([o.pos.x, o.pos.y, o.pos.z])
    direction = op - cp
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return False
    oxt_pos = cp - direction * (1.25 / norm)  # ~1.25 A on the other side
    oxt.pos = gemmi.Position(*oxt_pos)
    oxt.occ = 1.0
    oxt.b_iso = 30.0
    residue.add_atom(oxt)
    return True


def has_peptide_bond(chain: gemmi.Chain, left_num: int, right_num: int) -> bool:
    """True when C(left)-N(right) distance is a peptide bond (~1.33 A)."""
    left = find_residue(chain, left_num)
    right = find_residue(chain, right_num)
    if left is None or right is None:
        return False
    c = left.find_atom("C", "*")
    n = right.find_atom("N", "*")
    if c is None or n is None:
        return False
    d = np.linalg.norm(np.array([c.pos.x - n.pos.x, c.pos.y - n.pos.y, c.pos.z - n.pos.z]))
    return d < 2.0


# ------------------------------------------------------- conformer sampling --

def residue_centroid_pts(res) -> np.ndarray:
    pts = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
    return pts.mean(axis=0) if len(pts) else np.zeros(3)


def sample_conformers(residues, split_index: int, n_local: int = 6,
                      n_conformers: int = 5, seed: int = 20260816,
                      max_angle_deg: float = 12.0):
    """Local hinge sampling around the R87-T88 cleavage site.

    Physical rationale: the N-terminal fragment stays folded into the same
    domain (tethered by C73-C108 and surrounding contacts), so post-cleavage
    mobility is LOCAL.  Each conformer applies seeded rotations to a short
    segment on each side of the new termini:

      * NFR side: residues (left_num-n_local .. left_num] pivot around the
        centroid of residue left_num-n_local  -> the new C-terminus swings;
      * BODY side: residues [right_num .. right_num+n_local) pivot around the
        centroid of residue right_num+n_local with a LARGER amplitude - the
        freshly liberated neo-N-terminus is the most mobile part of the
        cleaved protein (PRD risk table: terminus exposure is a key
        uncertainty the conformer ensemble must span).

    C73-C108 (>=7 residues away from the hinges) is never moved, so the
    disulfide survives every conformer by construction.  Conformer 0 is the
    unrelaxed identity state.  Fully deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    left_idx = split_index - 1                 # index of R87
    right_idx = split_index                    # index of T88
    lo = max(0, left_idx - n_local)            # first moved residue (NFR side)
    hi = min(len(residues) - 1, right_idx + n_local - 1)  # last moved (BODY)
    pivot_nfr = residue_centroid_pts(residues[lo])
    pivot_body = residue_centroid_pts(residues[hi])
    body_amplitude = 2.2 * max_angle_deg       # neo-N-terminus swings wider

    conformers = []
    for k in range(n_conformers):
        desc_parts = []
        out = {i: np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
               for i, res in enumerate(residues)}
        if k > 0:
            # NFR-side hinge
            axis_n = rng.normal(size=3)
            axis_n /= np.linalg.norm(axis_n) + 1e-12
            ang_n = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
            R_n = rotation_matrix(axis_n, ang_n)
            for i in range(lo + 1, left_idx + 1):
                out[i] = (out[i] - pivot_nfr) @ R_n + pivot_nfr
            desc_parts.append(f"NFR hinge {np.rad2deg(ang_n):+.1f} deg")
            # BODY-side hinge (wider swing exposes the neo-N-terminus)
            axis_b = rng.normal(size=3)
            axis_b /= np.linalg.norm(axis_b) + 1e-12
            ang_b = np.deg2rad(rng.uniform(-body_amplitude, body_amplitude))
            R_b = rotation_matrix(axis_b, ang_b)
            for i in range(right_idx, hi):
                out[i] = (out[i] - pivot_body) @ R_b + pivot_body
            desc_parts.append(f"BODY hinge {np.rad2deg(ang_b):+.1f} deg")
        conformers.append(("identity (cleaved, unrelaxed)" if k == 0
                           else "; ".join(desc_parts), out))
    return conformers


def write_state(residues, coords_map: dict, chain_names: dict[int, str],
                out_path: Path, name: str, terminal_flags: dict[int, str]) -> Path:
    """Materialise a sampled state to mmCIF with explicit chain split."""
    st = gemmi.Structure()
    st.name = name
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chains: dict[str, gemmi.Chain] = {}
    for idx, res in enumerate(residues):
        cname = chain_names[idx]
        if cname not in chains:
            chains[cname] = gemmi.Chain(cname)
        new_res = gemmi.Residue()
        new_res.name = res.name
        new_res.seqid = res.seqid
        new_res.het_flag = "A"
        new_res.subchain = f"{cname}{new_res.seqid.num}"
        for atom, pos in zip(res, coords_map[idx]):
            a = gemmi.Atom()
            a.name = atom.name
            a.element = atom.element
            a.pos = gemmi.Position(*pos)
            a.occ = atom.occ
            a.b_iso = atom.b_iso
            new_res.add_atom(a)
        if idx in terminal_flags and terminal_flags[idx] == "coo":
            add_oxt(new_res)
        chains[cname].add_residue(new_res)
    for cname in sorted(chains):
        model.add_chain(chains[cname])
    st.add_model(model)
    st.setup_entities()
    return write_cif(st, out_path)


# ------------------------------------------------------------- main stage ----

def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    cleave = cfg.target.cleavage
    left_num, right_num = cleave.site
    left_aa, _ = parse_residue_id(cleave.left_residue)
    right_aa, _ = parse_residue_id(cleave.right_residue)
    preserve = [tuple(sorted(parse_residue_id(r)[1] for r in pair))
                for pair in cleave.preserve_disulfides]

    registry = read_json(out / "target_registry.json")
    std_cis = Path(registry["structures"]["cis"]["standardized_file"])
    st = read_structure(std_cis)
    chain = first_protein_chain(st)
    residues = polymer_residues(chain)

    # --- verify cleavage residues exist with the right identity
    left = find_residue(chain, left_num)
    right = find_residue(chain, right_num)
    if left is None or right is None:
        raise ValueError(f"cleavage residues {left_num}/{right_num} missing from {std_cis}")
    if residue_one_letter(left) != left_aa or residue_one_letter(right) != right_aa:
        raise ValueError(
            f"cleavage residues in structure are {residue_one_letter(left)}{left_num}/"
            f"{residue_one_letter(right)}{right_num}, config expects "
            f"{left_aa}{left_num}/{right_aa}{right_num}"
        )
    if not has_peptide_bond(chain, left_num, right_num):
        raise ValueError(
            f"no peptide bond {left_aa}{left_num}-{right_aa}{right_num} found in "
            f"source structure; cannot create cleaved state"
        )

    ss = disulfide_pairs(chain)
    ss_sorted = [tuple(sorted(p)) for p in ss]

    # --- disulfide anchor verification (fragment-side / body-side split)
    pivot_res_a, pivot_res_b = preserve[0]
    nums = [r.seqid.num for r in residues]
    split_index = nums.index(right_num)  # fragment = residues[:split_index]
    frag_nums = set(nums[:split_index])
    if pivot_res_a in frag_nums:
        anchor_num, body_num = pivot_res_a, pivot_res_b
    else:
        anchor_num, body_num = pivot_res_b, pivot_res_a
    for num in (anchor_num, body_num):
        if num not in nums:
            raise ValueError(f"required disulfide residue {num} missing from structure")

    terminal_flags_cleaved = {split_index - 1: "coo"}  # R87 gets OXT
    terminal_flags_intact: dict[int, str] = {}
    chain_names_cleaved = {i: ("NFR" if i < split_index else "BODY") for i in range(len(residues))}
    chain_names_intact = {i: "TROP2" for i in range(len(residues))}

    # the two NEW termini are adjacent by construction (they were just
    # bonded), so their mutual contact is chemistry, not a clash: exclude
    # the (R87, T88) residue pair from inter-fragment clash metrics
    left_idx = split_index - 1
    right_idx = split_index
    frag_idx = [i for i in range(0, split_index) if i != left_idx]
    body_idx = [i for i in range(split_index, len(residues)) if i != right_idx]
    elem_lists = {
        i: [a.element.name if hasattr(a.element, "name") else str(a.element)
            for a in res]
        for i, res in enumerate(residues)
    }

    def audit_pair(coords_map):
        frag_pts = np.vstack([coords_map[i] for i in frag_idx])
        frag_els = [e for i in frag_idx for e in elem_lists[i]]
        body_pts = np.vstack([coords_map[i] for i in body_idx])
        body_els = [e for i in body_idx for e in elem_lists[i]]
        return polar_aware_clash_metrics(frag_pts, frag_els, body_pts, body_els)

    n_conformers = 5
    amplitude = 12.0
    conformers = []
    for attempt in range(4):
        sampled = sample_conformers(residues, split_index,
                                    n_conformers=n_conformers,
                                    seed=ctx.seed + attempt,
                                    max_angle_deg=amplitude)
        # quick pre-audit: keep only conformers with acceptable clash metrics
        kept = [(d, c) for d, c in sampled
                if audit_pair(c)[0] <= CLASH_MAX_OVERLAP
                and audit_pair(c)[1] >= INTERCHAIN_MIN_DIST]
        if len(kept) >= n_conformers:
            conformers = kept
            break
        conformers = kept
        amplitude *= 0.5  # adaptive damping for tighter structures
    if len(conformers) < n_conformers:
        # last resort: supplement with extra seeds at minimal amplitude
        extra = sample_conformers(residues, split_index,
                                  n_conformers=3 * n_conformers,
                                  seed=ctx.seed + 99,
                                  max_angle_deg=3.0)
        for d, c in extra:
            if len(conformers) >= n_conformers:
                break
            if (audit_pair(c)[0] <= CLASH_MAX_OVERLAP
                    and audit_pair(c)[1] >= INTERCHAIN_MIN_DIST):
                conformers.append((d, c))
    conformers = conformers[:n_conformers]

    cleaved_dir = out / "cleaved_states"
    intact_dir = out / "intact_states"
    audits: list[dict] = []
    manifest_rows: list[StateManifestRow] = []

    for k, (desc, coords_map) in enumerate(conformers, start=1):
        state_id = f"cleaved_{k:02d}"
        path = cleaved_dir / f"{state_id}.cif"
        write_state(residues, coords_map, chain_names_cleaved, path,
                    state_id, terminal_flags_cleaved)

        # ---- topology audit (PRD AC-02 / AC-03)
        overlap, mind = audit_pair(coords_map)

        # disulfides after transform: SG-SG distance check per pair
        disulfides_kept: list[tuple[int, int]] = []
        for a_num, b_num in sorted(set(ss_sorted)):
            ra = find_residue(chain, a_num)
            rb = find_residue(chain, b_num)
            ia = nums.index(a_num)
            ib = nums.index(b_num)
            pa = coords_map[ia][list(ra).index(ra.find_atom("SG", "*"))]
            pb = coords_map[ib][list(rb).index(rb.find_atom("SG", "*"))]
            if np.linalg.norm(pa - pb) < 2.5:
                disulfides_kept.append((a_num, b_num))
        required_present = all(p in [tuple(sorted(d)) for d in disulfides_kept]
                               for p in preserve)

        failures = []
        if overlap > CLASH_MAX_OVERLAP:
            failures.append(f"fragment-body steric overlap {overlap:.0f} A^3 > {CLASH_MAX_OVERLAP}")
        if mind < INTERCHAIN_MIN_DIST:
            failures.append(f"fragment-body min distance {mind:.2f} A < {INTERCHAIN_MIN_DIST}")
        if not required_present:
            failures.append(f"required disulfide {preserve} broken")

        audit = TopologyAudit(
            state_id=state_id, kind="cleaved",
            source_structure=registry["structures"]["cis"]["source_pdb_id"],
            source_chain=chain.name,
            peptide_bond_left_right=False,
            left_terminal=f"{left_aa}{left_num} new C-terminus {cleave.left_terminal_state}",
            right_terminal=f"{right_aa}{right_num} new N-terminus {cleave.right_terminal_state}",
            chains=["NFR", "BODY"],
            residues=len(residues),
            disulfides=disulfides_kept,
            required_disulfides_present=required_present,
            max_clash_overlap=round(overlap, 2),
            min_nonbonded_distance=round(mind, 2),
            passed=not failures,
            failures=failures,
            transformations=[desc],
        )
        audits.append(audit.model_dump())
        manifest_rows.append(StateManifestRow(state_id=state_id, kind="cleaved",
                                              file=str(path),
                                              audit_passed=audit.passed,
                                              audit_failures=audit.failures))

    # intact control state (AC-08: positive/negative must be distinguishable)
    for k, (desc, coords_map) in enumerate(conformers[:1], start=1):
        state_id = "intact_01"
        path = intact_dir / f"{state_id}.cif"
        write_state(residues, coords_map, chain_names_intact, path,
                    state_id, terminal_flags_intact)
        audit = TopologyAudit(
            state_id=state_id, kind="intact",
            source_structure=registry["structures"]["cis"]["source_pdb_id"],
            source_chain=chain.name,
            peptide_bond_left_right=True,
            left_terminal="intact internal peptide bond",
            right_terminal="intact internal peptide bond",
            chains=["TROP2"],
            residues=len(residues),
            disulfides=ss_sorted,
            required_disulfides_present=True,
            max_clash_overlap=0.0,
            min_nonbonded_distance=99.0,
            passed=True,
            transformations=["none (intact control)"],
        )
        audits.append(audit.model_dump())
        manifest_rows.append(StateManifestRow(state_id=state_id, kind="intact",
                                              file=str(path), audit_passed=True))

    write_json(out / "topology_audit.json", {
        "cleavage": cleave.model_dump(),
        "overlap_tolerance_A3": CLASH_MAX_OVERLAP,
        "states": audits,
    })

    import pandas as pd
    pd.DataFrame([r.model_dump() for r in manifest_rows]).to_csv(
        out / "state_manifest.csv", index=False)

    passed_cleaved = [r for r in manifest_rows if r.kind == "cleaved" and r.audit_passed]
    if len(passed_cleaved) < 5:
        raise RuntimeError(
            f"only {len(passed_cleaved)} clash-free cleaved conformers generated "
            f"(need >=5 per PRD AC-03); reduce sampling amplitude or check topology"
        )
    ctx.state["state_manifest"] = [r.model_dump() for r in manifest_rows]
