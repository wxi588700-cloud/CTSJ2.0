"""Workflow engine tests: caching, resume and reproducibility (AC-14/AC-15)."""
from __future__ import annotations

from pathlib import Path

import pytest

from trop2_design.io import read_json, write_json
from trop2_design.schemas.results import RunManifest
from trop2_design.workflow.engine import RunContext, Stage, WorkflowRunner


def make_ctx(tmp_path: Path) -> RunContext:
    return RunContext(project_root=tmp_path, out_dir=tmp_path / "out",
                      config=None, tools=None, seed=42)


def make_manifest() -> RunManifest:
    return RunManifest(run_id="test", config_hash="x" * 16,
                       config_copy={}, seed=42)


class TestCachingAndResume:
    def test_cache_hit_on_second_run(self, tmp_path):
        ctx = make_ctx(tmp_path)
        ctx.out.mkdir(parents=True)
        counter = {"n": 0}

        def stage1(c):
            counter["n"] += 1
            write_json(c.out / "a.json", {"value": counter["n"]})

        st = Stage("S1", stage1, outputs=[ctx.out / "a.json"])
        runner = WorkflowRunner([st], ctx, make_manifest())
        runner.run()
        assert counter["n"] == 1
        assert (ctx.out / "a.json").exists()

        runner2 = WorkflowRunner([st], ctx, make_manifest())
        runner2.run()
        assert counter["n"] == 1  # cached, not re-executed
        statuses = [s.status for s in runner2.manifest.stages]
        assert "cached" in statuses

    def test_failure_does_not_clobber_previous_success(self, tmp_path):
        ctx = make_ctx(tmp_path)
        ctx.out.mkdir(parents=True)

        def stage_ok(c):
            write_json(c.out / "ok.json", {"a": 1})

        def stage_bad(c):
            raise RuntimeError("boom")

        s1 = Stage("S1", stage_ok, outputs=[ctx.out / "ok.json"])
        s2 = Stage("S2", stage_bad, depends_on=["S1"])
        runner = WorkflowRunner([s1, s2], ctx, make_manifest())
        with pytest.raises(Exception):
            runner.run()
        # prior success intact + recorded
        assert (ctx.out / "ok.json").exists()
        statuses = {s.stage: s.status for s in runner.manifest.stages}
        assert statuses["S1"] == "ok"
        assert statuses["S2"] == "failed"
        # resume skips S1 (cached) and retries S2
        def stage_bad_now_ok(c):
            write_json(c.out / "fixed.json", {"b": 2})

        s2b = Stage("S2", stage_bad_now_ok, depends_on=["S1"],
                    outputs=[ctx.out / "fixed.json"])
        runner3 = WorkflowRunner([s1, s2b], ctx, make_manifest())
        runner3.run()
        statuses3 = {s.stage: s.status for s in runner3.manifest.stages}
        assert statuses3["S1"] == "cached"
        assert statuses3["S2"] == "ok"

    def test_identical_seed_identical_output(self, tmp_path):
        """AC-14: same inputs+seed -> same discrete results."""
        def stage(c):
            import numpy as np

            rng = np.random.default_rng(c.seed)
            write_json(c.out / "r.json", {"v": int(rng.integers(0, 10**9))})

        outs = []
        for _ in range(2):
            d = tmp_path / f"run_{len(outs)}"
            ctx = RunContext(project_root=tmp_path, out_dir=d, config=None,
                             tools=None, seed=20260816)
            ctx.out.mkdir(parents=True)
            WorkflowRunner([Stage("S", stage, outputs=[d / "r.json"])],
                           ctx, make_manifest()).run()
            outs.append(read_json(d / "r.json"))
        assert outs[0] == outs[1]

    def test_cycle_detection(self, tmp_path):
        ctx = make_ctx(tmp_path)
        s1 = Stage("A", lambda c: None, depends_on=["B"])
        s2 = Stage("B", lambda c: None, depends_on=["A"])
        with pytest.raises(ValueError, match="cyclic|unknown"):
            WorkflowRunner([s1, s2], ctx, make_manifest())

    def test_unknown_dependency(self, tmp_path):
        ctx = make_ctx(tmp_path)
        s = Stage("A", lambda c: None, depends_on=["NOPE"])
        with pytest.raises(ValueError, match="unknown stage"):
            WorkflowRunner([s], ctx, make_manifest())
