"""M00: workflow orchestration, content-hash caching, resume and run audit.

A tiny internal DAG (PRD 5.2 allows "Snakemake 或内部 DAG").  Each stage
declares inputs/outputs; the runner skips a stage when its cache key (config
hash + input content hashes + code version + seed) matches a previous
successful run whose outputs still exist.  Failures never clobber prior
successes, enabling mid-pipeline resume (AC-15).
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..schemas.results import RunManifest, StageStatus, utcnow
from ..io import content_hash, sha256_file, write_json

try:
    from importlib.metadata import version as _pkg_ver
    CODE_VERSION = _pkg_ver("trop2-cis-dimer-inhibitor")
except Exception:
    CODE_VERSION = "3.3.0"


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def python_version() -> str:
    return sys.version.split()[0]


# ------------------------------------------------------------------ stages --

@dataclass
class Stage:
    name: str                                   # e.g. "M02_cleaved_states"
    fn: Callable[["RunContext"], None]          # executes the stage, writes outputs
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    description: str = ""


def _model_content_hash(model) -> str:
    """Stable content hash of a pydantic config model (None-safe)."""
    if model is None:
        return "none"
    try:
        return content_hash(model.model_dump(mode="json"))
    except Exception:
        return content_hash(str(model))


class RunContext:
    """Everything a stage may need: paths, config, tools, and shared state."""

    def __init__(self, project_root: Path, out_dir: Path, config, tools, seed: int):
        self.project_root = project_root
        self.out = out_dir
        self.config = config
        self.tools = tools
        self.seed = seed
        self.state: dict = {}          # in-memory handoff between stages
        self.artifacts: dict = {}      # stage -> list of produced paths


class WorkflowRunner:
    def __init__(self, stages: list[Stage], ctx: RunContext, manifest: RunManifest,
                 cache_dir: Path | None = None):
        self.stages = stages
        self.ctx = ctx
        self.manifest = manifest
        self.cache_dir = cache_dir or (ctx.out / ".cache")
        self.status_path = ctx.out / "task_status.csv"
        self._order_stages()
        self._executed: list[str] = []

    # ------------------------------------------------------------ ordering --

    def _order_stages(self) -> None:
        by_name = {s.name: s for s in self.stages}
        ordered: list[Stage] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(s: Stage) -> None:
            if s.name in done:
                return
            if s.name in visiting:
                raise ValueError(f"cyclic dependency at {s.name}")
            visiting.add(s.name)
            for dep in s.depends_on:
                if dep not in by_name:
                    raise ValueError(f"stage {s.name} depends on unknown stage {dep}")
                visit(by_name[dep])
            visiting.discard(s.name)
            done.add(s.name)
            ordered.append(s)

        for s in self.stages:
            visit(s)
        self.stages = ordered

    # ------------------------------------------------------------ cache key --

    def _cache_key(self, stage: Stage) -> str:
        payload = {
            "code": CODE_VERSION,
            "stage": stage.name,
            "seed": self.ctx.seed,
            "input_hashes": sorted(
                (str(p), sha256_file(p)) for p in stage.inputs if p.exists()
            ),
            "outputs": sorted(str(p) for p in stage.outputs),
            # audit fix (external review P1): config/tools were NOT part of
            # the key - changing thresholds or design parameters but reusing
            # the same run_id could wrongly hit the old cache
            "config": _model_content_hash(self.ctx.config),
            "tools": _model_content_hash(self.ctx.tools),
        }
        return content_hash(payload)

    def _cache_record(self, stage: Stage) -> Path:
        return self.cache_dir / f"{stage.name}.json"

    # ---------------------------------------------------------------- run --

    def run(self, only: set[str] | None = None, skip: set[str] | None = None) -> RunManifest:
        skip = skip or set()
        for stage in self.stages:
            if only is not None and stage.name not in only:
                continue
            if stage.name in skip:
                self._record(StageStatus(stage=stage.name, status="skipped"))
                continue
            key = self._cache_key(stage)
            rec = self._cache_record(stage)
            outputs_exist = all(p.exists() for p in stage.outputs) if stage.outputs else False
            if rec.exists() and outputs_exist:
                import json as _json
                try:
                    prior = _json.loads(rec.read_text(encoding="utf-8"))
                    if prior.get("cache_key") == key and prior.get("status") == "ok":
                        self._record(StageStatus(stage=stage.name, status="cached",
                                                 cache_key=key,
                                                 note="cache hit, outputs reused"))
                        self._executed.append(stage.name)
                        continue
                except Exception:
                    pass
            status = StageStatus(stage=stage.name, status="running", cache_key=key,
                                 started=utcnow())
            t0 = time.time()
            try:
                stage.fn(self.ctx)
                status.status = "ok"
                status.finished = utcnow()
                status.duration_s = round(time.time() - t0, 3)
                self._record(status)
                write_json(rec, {"cache_key": key, "status": "ok",
                                 "outputs": [str(p) for p in stage.outputs],
                                 "ts": utcnow()})
                self._executed.append(stage.name)
            except Exception as exc:  # noqa: BLE001 - explicit failure capture
                status.status = "failed"
                status.finished = utcnow()
                status.duration_s = round(time.time() - t0, 3)
                status.note = f"{type(exc).__name__}: {exc}"
                self._record(status)
                self.manifest.failures.append(f"{stage.name}: {status.note}")
                raise StageFailure(stage.name, status.note) from exc
        return self.manifest

    # -------------------------------------------------------------- records --

    def _record(self, status: StageStatus) -> None:
        self.manifest.stages.append(status)
        # append-only task_status.csv (M00 standard output)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.status_path.exists()
        with open(self.status_path, "a") as fh:
            if header:
                fh.write("stage,status,cache_key,started,finished,duration_s,note\n")
            note = status.note.replace('"', "'")
            fh.write(
                f'{status.stage},{status.status},"{status.cache_key}",'
                f'{status.started},{status.finished},{status.duration_s},"{note}"\n'
            )
        write_json(self.ctx.out / "run_manifest.json",
                   self.manifest.model_dump())


class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"stage {stage} failed: {message}")
        self.stage = stage
        self.message = message
