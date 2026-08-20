"""Geometry toolkit: SASA (Shrake-Rupley), clashes, superposition, contacts.

Pure numpy implementations so every number is deterministic and dependency
free (FreeSASA/PyRosetta remain optional adapters at tool boundaries).
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ radii --

VDW_RADII: dict[str, float] = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "H": 1.10, "SE": 1.90,
}
PROBE = 1.4  # water probe radius (A)

# ------------------------------------------------------------ sphere points --

def sphere_points(n: int = 960) -> np.ndarray:
    """Deterministic quasi-uniform sphere sampling (golden spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    pts = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )
    return pts


# ------------------------------------------------------------------- SASA --

def sasa(coords: np.ndarray, elements: np.ndarray, n_points: int = 960,
         probe: float = PROBE) -> np.ndarray:
    """Per-atom solvent-accessible surface area (Shrake-Rupley, A^2).

    coords: (N,3); elements: (N,) strings like 'C','N','O','S'.
    Vectorised neighbour test; deterministic.
    """
    n = len(coords)
    radii = np.array([VDW_RADII.get(e, 1.7) for e in elements]) + probe
    pts = sphere_points(n_points)
    areas = np.zeros(n)
    if n == 0:
        return areas
    # neighbour grid via KD-tree
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    max_r = radii.max()
    pairs = tree.query_pairs(r=2 * max_r, output_type="ndarray")
    # occupied[p, j] counts blocked points; use dense bool only for efficiency
    occ = np.zeros((n, n_points), dtype=bool)
    for i, j in pairs:
        d = np.linalg.norm(coords[i] - coords[j])
        ri, rj = radii[i], radii[j]
        if d >= ri + rj:
            continue
        # points on i's sphere blocked by j
        vec = coords[j] - coords[i]
        if d < 1e-8:
            continue
        cos_limit = (d * d + ri * ri - rj * rj) / (2.0 * d * ri)
        cos_limit = np.clip(cos_limit, -1.0, 1.0)
        dots = pts @ (vec / d)
        blocked = dots > cos_limit
        occ[i] |= blocked
        # symmetric: points on j's sphere blocked by i
        vec2 = -vec
        cos_limit2 = (d * d + rj * rj - ri * ri) / (2.0 * d * rj)
        cos_limit2 = np.clip(cos_limit2, -1.0, 1.0)
        dots2 = pts @ (vec2 / d)
        occ[j] |= dots2 > cos_limit2
    free_frac = 1.0 - occ.sum(axis=1) / n_points
    areas = 4.0 * np.pi * radii**2 * free_frac
    return areas


POLAR_ELEMENTS = {"N", "O"}


def residue_sasa(residues, probe: float = PROBE) -> dict[int, float]:
    """Total per-residue SASA keyed by residue seqid.num (gemmi residues in)."""
    coords, elems, owner = [], [], []
    for idx, res in enumerate(residues):
        for atom in res:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            elems.append(atom.element.name if hasattr(atom.element, "name") else str(atom.element))
            owner.append(idx)
    if not coords:
        return {}
    coords = np.asarray(coords)
    per_atom = sasa(coords, np.array(elems), probe=probe)
    out: dict[int, float] = {}
    for idx, res in enumerate(residues):
        mask = np.array(owner) == idx
        out[res.seqid.num] = float(per_atom[mask].sum())
    return out


# ---------------------------------------------------------------- clashes --

def min_pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 99.0
    from scipy.spatial import cKDTree

    d, _ = cKDTree(b).query(a, k=1)
    return float(d.min())


def clash_count(a: np.ndarray, b: np.ndarray, cutoff: float = 3.0) -> int:
    """Number of heavy-atom pairs closer than cutoff (inter-chain clashes)."""
    if len(a) == 0 or len(b) == 0:
        return 0
    from scipy.spatial import cKDTree

    tree = cKDTree(b)
    idx = tree.query_ball_point(a, r=cutoff)
    return int(sum(len(x) for x in idx))


def clash_overlap_volume(a: np.ndarray, b: np.ndarray, vdwr_a=None, vdwr_b=None) -> float:
    """Crude steric overlap volume (A^3) between two atom sets."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    from scipy.spatial import cKDTree

    tree = cKDTree(b)
    idx = tree.query_ball_point(a, r=4.5)
    vol = 0.0
    for i, js in enumerate(idx):
        for j in js:
            d = float(np.linalg.norm(a[i] - b[j]))
            ra = vdwr_a[i] if vdwr_a is not None else 1.7
            rb = vdwr_b[j] if vdwr_b is not None else 1.7
            if d >= ra + rb:
                continue
            vol += (4.0 / 3.0) * np.pi * min(ra, rb) ** 3 * (1.0 - d / max(ra + rb, 1e-6))
    return float(vol)


# ------------------------------------------------------- superposition (Kabsch) --

def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotation R, translation t) so that ``mobile @ R + t ~= target``.

    Uses the standard SVD solution with determinant correction.
    """
    mu_m = mobile.mean(axis=0)
    mu_t = target.mean(axis=0)
    P = mobile - mu_m
    Q = target - mu_t
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).T  # row-vector convention: mobile @ R
    t = mu_t - mu_m @ R
    return R, t


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(a) != len(b):
        return float("nan")
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def superpose_by_number(mobile_res, target_res, max_pairs: int = 60,
                        min_pairs: int = 10):
    """Kabsch-superpose two residue lists paired by residue ``seqid.num``.

    Returns ``(R, t, fit_rmsd, n_pairs)`` so that ``ca(mobile) @ R + t``
    aligns onto the matched target CAs.

    Pairing by residue NUMBER (never by list order) is mandatory here: the
    cleaved BODY chain starts at T88 while assembly/intact chains start at
    residue 32 - sequential pairing silently mis-registers the two traces by
    56 residues and superposes the wrong residues onto each other (this is
    exactly the M08 bug that produced all-zero cis_block/trans_occlusion in
    every run before the fix).
    """
    def ca_by_num(res_list):
        out = {}
        for r in res_list:
            at = r.find_atom("CA", "*")
            if at is not None and r.seqid.num not in out:
                out[r.seqid.num] = (at.pos.x, at.pos.y, at.pos.z)
        return out

    m = ca_by_num(mobile_res)
    t = ca_by_num(target_res)
    common = sorted(set(m) & set(t))
    if len(common) < min_pairs:
        raise ValueError(
            f"superpose_by_number: only {len(common)} residues matched by "
            f"number (need >= {min_pairs}); the two traces do not share a "
            f"numbering convention - refusing to guess an alignment")
    sel = common[:max_pairs]
    A = np.array([m[k] for k in sel])
    B = np.array([t[k] for k in sel])
    R, tvec = kabsch(A, B)
    fit = A @ R + tvec
    return R, tvec, rmsd(fit, B), len(sel)


# ---------------------------------------------------------------- contacts --

def contacts_within(a: np.ndarray, b: np.ndarray, cutoff: float = 4.5):
    """Yield (i, j, distance) pairs across two coordinate arrays."""
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return []
    tree = cKDTree(b)
    pairs = tree.query_ball_point(a, r=cutoff)
    out = []
    for i, js in enumerate(pairs):
        for j in js:
            out.append((i, j, float(np.linalg.norm(a[i] - b[j]))))
    return out


# ------------------------------------------------------- rotations for sampling --

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation (row-vector convention)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )
    return R
