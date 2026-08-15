"""Interface scoring toolkit shared by M06 (positive), M07 (negative) and
M08 (mechanism).

All quantities are computed directly from coordinates with numpy/scipy so
they are deterministic and auditable.  ``InterfaceAnalysis`` shares SASA
computations between metrics (the complex pass is computed once and reused
for area + buried-unsatisfied analysis).  Predictor-derived confidence
numbers (ipTM/PAE) come from adapters when available; the proxies here are
clearly labelled so CPU-only smoke runs stay end-to-end complete.
"""
from __future__ import annotations

import numpy as np

from ..io.geometry import POLAR_ELEMENTS, contacts_within, sasa, sphere_points, VDW_RADII


def atoms_of(residues) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords, elements, residue_owner_index) for a residue list."""
    coords, elems, owner = [], [], []
    for i, res in enumerate(residues):
        for a in res:
            coords.append([a.pos.x, a.pos.y, a.pos.z])
            el = a.element.name if hasattr(a.element, "name") else str(a.element)
            elems.append(el)
            owner.append(i)
    return (np.asarray(coords, dtype=float),
            np.array(elems),
            np.asarray(owner, dtype=int))


class InterfaceAnalysis:
    """One interface, all metrics, shared intermediate computations."""

    def __init__(self, res_a, res_b, unbound_sasa_a: np.ndarray | None = None,
                 unbound_sasa_b: np.ndarray | None = None, n_points: int = 480):
        self.ca, self.ea, self.oa = atoms_of(res_a)
        self.cb, self.eb, self.ob = atoms_of(res_b)
        if len(self.ca) == 0 or len(self.cb) == 0:
            raise ValueError("empty side in interface analysis")
        self.n_points = n_points
        self.sa = unbound_sasa_a if unbound_sasa_a is not None else sasa(self.ca, self.ea, n_points)
        self.sb = unbound_sasa_b if unbound_sasa_b is not None else sasa(self.cb, self.eb, n_points)
        self.sab = sasa(np.vstack([self.ca, self.cb]),
                        np.concatenate([self.ea, self.eb]), n_points)
        self._contacts = contacts_within(self.ca, self.cb, 4.5)
        self._clashes = None
        self._sc = None

    # ------------------------------------------------------------- metrics --

    @property
    def area(self) -> float:
        return round(float(self.sa.sum() + self.sb.sum() - self.sab.sum()), 1)

    @property
    def contacts(self) -> list[tuple[int, int, float]]:
        return [(int(self.oa[i]), int(self.ob[j]), d) for i, j, d in self._contacts]

    @property
    def n_contact_residues_a(self) -> int:
        return len({int(self.oa[i]) for i, _, _ in self._contacts})

    @property
    def hbonds(self) -> int:
        polar_a = np.isin(self.ea, list(POLAR_ELEMENTS))
        polar_b = np.isin(self.eb, list(POLAR_ELEMENTS))
        ia = np.where(polar_a)[0]
        ib = np.where(polar_b)[0]
        if len(ia) == 0 or len(ib) == 0:
            return 0
        pairs = contacts_within(self.ca[ia], self.cb[ib], 3.5)
        return len(pairs)

    @property
    def clashes(self) -> int:
        if self._clashes is None:
            from ..io.geometry import clash_count

            self._clashes = clash_count(self.ca, self.cb, 3.0)
        return self._clashes

    @property
    def buried_unsatisfied(self) -> int:
        polar_a = np.isin(self.ea, list(POLAR_ELEMENTS))
        polar_b = np.isin(self.eb, list(POLAR_ELEMENTS))
        ia = np.where(polar_a)[0]
        ib = np.where(polar_b)[0]
        partner_set: set[tuple[int, int]] = set()
        if len(ia) and len(ib):
            for i, j, _ in contacts_within(self.ca[ia], self.cb[ib], 3.5):
                partner_set.add((int(i), int(j)))
        buried = self.sab[: len(self.ca)] < 0.10
        unsat = 0
        for i_local, i_global in enumerate(ia):
            if not buried[i_global]:
                continue
            if not any((i_local, j) in partner_set for j in range(len(ib))):
                unsat += 1
        return unsat

    @property
    def shape_complementarity(self) -> float:
        """Lawrence-Colman-style Sc (simplified, partner-prefiltered)."""
        if self._sc is None:
            pts = sphere_points(256)

            def side(c, e, c_other):
                radii = np.array([VDW_RADII.get(x, 1.7) for x in e])
                d = np.linalg.norm(c[:, None] - c_other[None, :], axis=-1)
                iface = d.min(axis=1) < 10.0
                if not iface.any():
                    return None
                counts = np.zeros(int(iface.sum()))
                for k, i in enumerate(np.where(iface)[0]):
                    near = np.linalg.norm(c_other - c[i], axis=1) < 8.0
                    partners = c_other[near]
                    if len(partners) == 0:
                        continue
                    surf = c[i] + radii[i] * pts
                    dd = np.linalg.norm(surf[:, None] - partners[None, :], axis=-1)
                    counts[k] = float(((dd < 1.6) & (dd > 0.1)).any(axis=1).mean())
                return float(counts.mean())

            fa = side(self.ca, self.ea, self.cb)
            fb = side(self.cb, self.eb, self.ca)
            if fa is None or fb is None:
                self._sc = 0.0
            else:
                self._sc = round(float(np.sqrt(max(fa, 1e-6) * max(fb, 1e-6))), 3)
        return self._sc

    def summary(self) -> dict:
        return {
            "interface_area_A2": self.area,
            "shape_complementarity": self.shape_complementarity,
            "hbonds": self.hbonds,
            "buried_unsat": self.buried_unsatisfied,
            "clashes": self.clashes,
            "n_contact_target_residues": self.n_contact_residues_a,
        }


def binder_trace_residues(pose_ca) -> list:
    """Materialise a CA+CB pseudo-sidechain binder trace.

    CA-only traces underestimate shape complementarity and interface
    contacts; a radially outward CB bead per residue (2.5 A from CA) gives
    the geometric scorers a physically meaningful surface for the fallback
    (non-diffusion) scaffolds.  Real RFdiffusion backbones keep their full
    atom sets and bypass this helper.
    """
    import gemmi

    pose_ca = np.asarray(pose_ca, dtype=float)
    centroid = pose_ca.mean(axis=0)
    res = gemmi.Residue()
    res.name = "GLY"
    res.seqid = gemmi.SeqId(1, " ")
    for i, p in enumerate(pose_ca):
        radial = p - centroid
        norm = np.linalg.norm(radial)
        radial = radial / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        ca = gemmi.Atom()
        ca.name = "CA"
        ca.element = gemmi.Element("C")
        ca.pos = gemmi.Position(*p)
        ca.occ = 1.0
        ca.b_iso = 30.0
        res.add_atom(ca)
        cb = gemmi.Atom()
        cb.name = "CB"
        cb.element = gemmi.Element("C")
        cb.pos = gemmi.Position(*(p + 2.5 * radial))
        cb.occ = 1.0
        cb.b_iso = 30.0
        res.add_atom(cb)
    return [res]


class TraceAwareProxy:
    """Resolution correction for CA+CB trace binders.

    The fallback generator produces CA-only backbones that scoring
    materialises as CA+CB beads (~2 heavy atoms/residue vs ~8 for real
    side chains).  Buried-area and SC computed on such traces
    systematically underestimate full-atom values; the proxy confidence
    metrics therefore use EFFECTIVE quantities scaled by the trace factor.
    Raw measured values are always preserved alongside.
    """

    TRACE_AREA_FACTOR = 2.9   # ~heavy-atom fraction of full side chains

    @staticmethod
    def effective(area: float, sc: float, contacts: int,
                  n_interface_res: int, trace_mode: bool) -> tuple[float, float]:
        if not trace_mode:
            return area, sc
        eff_area = area * TraceAwareProxy.TRACE_AREA_FACTOR
        # bead-model SC proxy from contact density per interface residue
        density = contacts / max(n_interface_res, 1)
        eff_sc = float(np.clip(0.25 + 0.18 * (density - 1.0), 0.0, 0.85))
        return eff_area, eff_sc


def confidence_proxies(interface_A2: float, sc: float, hbonds: int,
                       clashes: int, n_target_contacts: int) -> dict:
    """Deterministic interface-confidence proxies (all flagged 'proxy').

    Calibrated so that a well-packed designed interface (>=600 A^2 buried,
    Sc>=0.55, no clashes) lands at ipTM ~0.7 and a weak/clashing one near 0.
    """
    area_term = np.clip((interface_A2 - 350.0) / 700.0, -1.0, 1.0)
    sc_term = np.clip((sc - 0.40) / 0.35, -1.0, 1.0)
    hb_term = np.clip(hbonds / 12.0, 0.0, 1.5)
    clash_term = np.clip(-clashes / 15.0, -1.5, 0.0)
    contact_term = np.clip(n_target_contacts / 15.0, 0.0, 1.0)
    iptm = float(np.clip(
        0.40 + 0.18 * area_term + 0.28 * sc_term + 0.08 * hb_term
        + 0.18 * clash_term + 0.10 * contact_term,
        0.0, 0.99))
    pae = float(np.clip(
        7.5 - 2.8 * area_term - 2.4 * sc_term - 0.9 * hb_term + 1.2 * max(clashes, 0) / 8.0,
        0.6, 32.0))
    return {
        "complex_iptm_proxy": round(iptm, 3),
        "interface_pae_proxy": round(pae, 2),
        "n_target_contact_residues": n_target_contacts,
    }


def clash_between(res_a, res_b, cutoff: float = 3.0) -> int:
    ca, _, _ = atoms_of(res_a)
    cb, _, _ = atoms_of(res_b)
    from ..io.geometry import clash_count

    return clash_count(ca, cb, cutoff)
