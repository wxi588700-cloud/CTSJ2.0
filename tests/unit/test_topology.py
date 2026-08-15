"""Geometry + topology unit tests (AC-02 / AC-03 core logic)."""
from __future__ import annotations

import numpy as np
import pytest

from trop2_design.io import (
    find_residue, first_protein_chain, polymer_residues, read_structure,
)
from trop2_design.io.geometry import kabsch, rmsd, rotation_matrix, sasa
from trop2_design.target_builder.cleave import (
    add_oxt, disulfide_pairs, has_peptide_bond, sample_conformers,
)


class TestGeometry:
    def test_sasa_single_atom(self):
        # isolated carbon: SASA = 4*pi*(1.7+1.4)^2
        area = sasa(np.array([[0.0, 0.0, 0.0]]), np.array(["C"]))[0]
        expected = 4 * np.pi * (1.7 + 1.4) ** 2
        assert area == pytest.approx(expected, rel=0.01)

    def test_kabsch_identity(self):
        pts = np.random.default_rng(1).normal(size=(50, 3))
        R, t = kabsch(pts, pts)
        assert rmsd(pts @ R + t, pts) < 1e-8

    def test_kabsch_known_rotation(self):
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(100, 3))
        R_true = rotation_matrix(np.array([0.2, 0.5, 0.8]), 0.7)
        moved = pts @ R_true + np.array([1.0, -2.0, 0.5])
        R, t = kabsch(moved, pts)
        assert rmsd(moved @ R + t, pts) < 1e-6

    def test_rotation_preserves_distance(self):
        R = rotation_matrix(np.array([1.0, 2.0, 3.0]), 1.234)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


class TestCleavageTopology:
    """AC-02: R|T peptide bond must be absent after cleavage, terminals
    correct, disulfide preserved."""

    def test_fixture_has_peptide_bond_and_disulfide(self, mini_target):
        st = read_structure(mini_target)
        ch = first_protein_chain(st)
        assert has_peptide_bond(ch, 60, 61)
        pairs = disulfide_pairs(ch)
        assert (20, 75) in [tuple(sorted(p)) for p in pairs]

    def test_add_oxt(self, mini_target):
        st = read_structure(mini_target)
        ch = first_protein_chain(st)
        res = find_residue(ch, 60)
        assert res.find_atom("OXT", "*") is None
        assert add_oxt(res)
        assert res.find_atom("OXT", "*") is not None

    def test_conformers_preserve_disulfide_and_are_distinct(self, mini_target):
        st = read_structure(mini_target)
        ch = first_protein_chain(st)
        residues = polymer_residues(ch)
        nums = [r.seqid.num for r in residues]
        split = nums.index(61)
        n_local = 6
        confs = sample_conformers(residues, split, n_local=n_local,
                                  n_conformers=6, seed=20260816)
        assert len(confs) == 6
        i20, i75 = nums.index(20), nums.index(75)
        left_idx, right_idx = split - 1, split
        hinge = set(range(left_idx - n_local + 1, left_idx + 1)) | \
                set(range(right_idx, right_idx + n_local))
        orig = {i: np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
                for i, res in enumerate(residues)}
        for desc, cm in confs:
            # disulfide far outside the hinge segments -> never moved
            assert np.allclose(cm[i20], orig[i20]), desc
            assert np.allclose(cm[i75], orig[i75]), desc
            # residues far from the cleavage site unchanged
            for far in (0, 10, nums.index(100)):
                assert np.allclose(cm[far], orig[far]), desc
        # conformer 0 = identity; others must perturb the hinge
        assert np.allclose(confs[0][1][left_idx], orig[left_idx])
        moved = 0
        for _, cm in confs[1:]:
            if not np.allclose(cm[left_idx], orig[left_idx], atol=1e-6):
                moved += 1
        assert moved >= 4
        # different conformers differ from each other
        a = np.vstack([confs[1][1][i] for i in sorted(hinge)])
        b = np.vstack([confs[2][1][i] for i in sorted(hinge)])
        assert not np.allclose(a, b)
