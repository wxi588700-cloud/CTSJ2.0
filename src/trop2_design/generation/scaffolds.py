"""Deterministic de-novo mini-protein backbone generator (M04 fallback).

Generates ideal-geometry 3-helix-bundle backbones (60-120 aa) positioned
against the T88-neo-terminus epitope patch.  This is the CPU-only fallback
used when the RFdiffusion checkout/GPU is unavailable; every candidate it
produces is tagged ``source=scaffold_fallback`` so downstream consumers can
distinguish it from real diffusion output (PRD: generation and validation
models must be separable and auditable).
"""
from __future__ import annotations

import numpy as np

HELIX_RISE = 1.50          # A per residue along the helix axis
HELIX_TWIST = np.deg2rad(100.0)
CA_RADIUS = 2.30           # A, CA distance from helix axis

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def ideal_helix(n_res: int, axis: np.ndarray, origin: np.ndarray,
                phase: float = 0.0) -> np.ndarray:
    """Ideal alpha-helix CA trace (n_res, 3) along ``axis``."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # orthonormal basis
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(axis, u)
    pts = []
    for i in range(n_res):
        t = phase + i * HELIX_TWIST
        radial = CA_RADIUS * (np.cos(t) * u + np.sin(t) * v)
        pts.append(origin + i * HELIX_RISE * axis + radial)
    return np.asarray(pts)


def relax_geometry(ca: np.ndarray, iters: int = 120) -> np.ndarray:
    """Deterministic geometry relaxation: bond springs (target 3.8 A) +
    non-bonded repulsion below 4.2 A for |i-j| >= 3.  Fixes loop spacing and
    inter-helix clashes while staying fully reproducible."""
    ca = ca.copy()
    n = len(ca)
    idx = np.arange(n)
    for _ in range(iters):
        diff = ca[:, None, :] - ca[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, 99.0)
        force = np.zeros_like(ca)
        if n > 1:
            b_vec = ca[1:] - ca[:-1]
            b_len = np.linalg.norm(b_vec, axis=1)
            # pull endpoints together when b_len > 3.8 (and push apart when short)
            spring = (b_len - 3.8)[:, None] * 0.20
            unit_b = b_vec / (b_len[:, None] + 1e-9)
            force[:-1] += spring * unit_b
            force[1:] -= spring * unit_b
        mask = (d < 4.2) & (np.abs(idx[:, None] - idx[None, :]) >= 3)
        if mask.any():
            unit = diff / (d[:, :, None] + 1e-9)
            push = (4.2 - d) * 0.15
            push[~mask] = 0.0
            force += (push[:, :, None] * unit).sum(axis=1)
        ca += force
        if np.abs(force).max() < 1e-4:
            break
    return ca


def helix_bundle(n_helices: int, lengths: list[int], loop_len: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (ca_coords (N,3), contact_map_binary (N,N)) for a helical bundle.

    Helices are anti-parallel, packed at ~10 A axis separation (typical
    helix-helix packing distance), connected by loops with adaptive spacing,
    then relaxed (``relax_geometry``) so every bond and contact passes
    downstream geometry checks deterministically.
    """
    separation = 10.0
    coords: list[np.ndarray] = []
    sign = 1.0
    for k, n in enumerate(lengths):
        origin = np.array([separation * np.cos(np.pi / 3.0 * k),
                           separation * np.sin(np.pi / 3.0 * k), 0.0])
        phase = rng.uniform(0, 2 * np.pi)
        helix = ideal_helix(n, np.array([0.0, 0.0, 1.0]) * sign,
                            origin - np.array([0.0, 0.0, n * HELIX_RISE / 2]) * sign,
                            phase=phase)
        if k > 0:
            prev_last = coords[-1]
            gap = float(np.linalg.norm(helix[0] - prev_last))
            n_loop = max(loop_len, int(np.ceil(gap / 3.5)) - 1)
            for j in range(1, n_loop + 1):
                frac = j / (n_loop + 1)
                coords.append(prev_last + (helix[0] - prev_last) * frac
                              + np.array([0.0, 0.6 * np.sin(np.pi * frac), 0.0]))
        coords.extend(list(helix))
        sign *= -1.0
    ca = np.asarray(coords)
    ca = relax_geometry(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    idx = np.arange(len(ca))
    contacts = (d < 10.0) & (np.abs(idx[:, None] - idx[None, :]) >= 3)
    return ca, contacts.astype(np.int8)


def place_against_patch(ca: np.ndarray, patch_centroid: np.ndarray,
                        rng: np.random.Generator,
                        target_pts: np.ndarray | None = None,
                        protein_centroid: np.ndarray | None = None,
                        anchor_point: np.ndarray | None = None) -> np.ndarray:
    """Rigidly place a scaffold against the epitope patch.

    Strategy: approach the patch along its OUTWARD NORMAL (from the protein
    centroid through the patch centroid), with the scaffold long axis kept
    tangential and a sampled lateral offset (±6 A) plus spin for pose
    diversity.  The placement distance is scanned from far outside inward
    until the CA+CB probe makes vdW contact (3.8-5.2 A) with the target
    heavy atoms - contact at the patch, never interpenetration.
    """
    from ..io.geometry import rotation_matrix

    centre = ca.mean(axis=0)
    centred = ca - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    long_axis = vt[0]

    if protein_centroid is None:
        protein_centroid = patch_centroid - np.array([0.0, 0.0, 20.0])
    normal = np.asarray(patch_centroid, dtype=float) - np.asarray(protein_centroid, dtype=float)
    normal /= np.linalg.norm(normal) + 1e-12
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)

    # lateral offset + spin sampling for pose diversity
    lateral = rng.uniform(-6.0, 6.0, size=2)
    approach_centre = (np.asarray(patch_centroid, dtype=float)
                       + u * lateral[0] + v * lateral[1])

    # orient long axis perpendicular to the approach normal (tangential)
    tangent = np.cross(normal, np.cross(long_axis, normal))
    tn = np.linalg.norm(tangent)
    if tn > 1e-9:
        tangent /= tn
        w = np.cross(long_axis, tangent)
        s = float(np.linalg.norm(w))
        if s > 1e-9:
            centred = centred @ rotation_matrix(w / s, float(np.arcsin(np.clip(s, -1, 1))))
    centred = centred @ rotation_matrix(long_axis, rng.uniform(0, 2 * np.pi))

    # interface probe = CA trace + outward CB pseudo-atoms (2.5 A)
    centroid = centred.mean(axis=0)
    radial = centred - centroid
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.where(norms > 1e-6, norms, 1.0)
    probe = np.vstack([centred, centred + 2.5 * radial])

    if target_pts is None or len(target_pts) == 0:
        return centred + approach_centre + normal * 12.0

    from scipy.spatial import cKDTree

    tree = cKDTree(target_pts)

    def contact_stats(pts):
        d, _ = tree.query(pts, k=1)
        return float(d.min()), int((d < 3.0).sum())

    best = None
    best_anchor_dist = np.inf
    patch_centroid = np.asarray(patch_centroid, dtype=float)
    for attempt in range(8):
        lat = lateral if attempt == 0 else rng.uniform(-9.0, 9.0, size=2)
        centre_i = patch_centroid + u * lat[0] + v * lat[1]
        # re-spin the scaffold for this attempt
        centred_i = centred @ rotation_matrix(long_axis, rng.uniform(0, 2 * np.pi))
        centroid_i = centred_i.mean(axis=0)
        radial_i = centred_i - centroid_i
        norms_i = np.linalg.norm(radial_i, axis=1, keepdims=True)
        radial_i = radial_i / np.where(norms_i > 1e-6, norms_i, 1.0)
        probe_i = np.vstack([centred_i, centred_i + 2.5 * radial_i])
        accepted = None
        accepted_dist = None
        for dist in np.arange(70.0, 6.0, -0.25):
            mind, overlaps = contact_stats(probe_i + centre_i + normal * dist)
            if overlaps == 0 and mind <= 5.5:
                accepted = centred_i + centre_i + normal * dist
                accepted_dist = dist
                break
            if overlaps > 0:
                break
        if accepted is not None:
            placed_probe = probe_i + centre_i + normal * accepted_dist
            anchor_d = float(np.linalg.norm(placed_probe - patch_centroid,
                                            axis=1).min())
            if anchor_d < best_anchor_dist:
                best_anchor_dist = anchor_d
                best = accepted
    if best is None:
        best = centred + approach_centre + normal * 14.0

    # terminus-directed greedy docking: slide the accepted pose toward the
    # anchor (the T88 free N atom) in 0.5 A steps while overlap-free, so the
    # binder's edge approaches the neo-N-terminus as closely as sterics allow
    if anchor_point is not None:
        anchor_point = np.asarray(anchor_point, dtype=float)
        shift = np.zeros(3)
        centroid_f = best.mean(axis=0)
        radial_f = best - centroid_f
        norms_f = np.linalg.norm(radial_f, axis=1, keepdims=True)
        radial_f = radial_f / np.where(norms_f > 1e-6, norms_f, 1.0)
        probe_f = np.vstack([best, best + 2.5 * radial_f])
        for _ in range(24):
            d = np.linalg.norm(probe_f + shift - anchor_point, axis=1)
            i = int(np.argmin(d))
            gap = d[i]
            if gap <= 4.2:
                break
            step_dir = (anchor_point - (probe_f[i] + shift))
            step_dir /= np.linalg.norm(step_dir) + 1e-12
            step = step_dir * 0.5
            _, overlaps = contact_stats(probe_f + shift + step)
            if overlaps > 0:
                break
            shift += step
        best = best + shift
    return best


def generate_scaffolds(n_candidates: int, binder_len_range: tuple[int, int],
                       patch_centroid, seed: int, target_pts=None,
                       anchor_point=None):
    """Yield (candidate_key, ca_coords, contacts, topology_name)."""
    rng = np.random.default_rng(seed)
    out = []
    length_pool = []
    for total in range(binder_len_range[0], binder_len_range[1] + 1, 4):
        length_pool.append(total)
    topologies = [
        ("3hlx", [0.45, 0.30, 0.25], 4),
        ("3hlx", [0.34, 0.33, 0.33], 3),
        ("2hlx", [0.55, 0.45], 5),
        ("4hlx", [0.32, 0.24, 0.24, 0.20], 3),
    ]
    for i in range(n_candidates):
        total = int(length_pool[(seed + i * 7) % len(length_pool)])
        topo = topologies[(seed + i) % len(topologies)]
        lengths = [max(12, int(total * f) - topo[2] // 2) for f in topo[1]]
        # renormalise to hit total approximately
        scale = (total - topo[2] * (len(lengths) - 1)) / sum(lengths)
        lengths = [max(12, int(round(l * scale))) for l in lengths]
        ca, contacts = helix_bundle(len(lengths), lengths, topo[2], rng)
        protein_centroid = (np.asarray(target_pts, dtype=float).mean(axis=0)
                            if target_pts is not None and len(target_pts) else None)
        placed = place_against_patch(ca, np.asarray(patch_centroid, dtype=float),
                                     rng, target_pts=target_pts,
                                     protein_centroid=protein_centroid,
                                     anchor_point=anchor_point)
        key = f"fallback_s{i+1:02d}_len{len(ca)}_{topo[0]}"
        out.append((key, placed, contacts, topo[0]))
    return out
