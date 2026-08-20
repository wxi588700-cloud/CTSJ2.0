"""RFdiffusion adapter (M04) - wraps the previously downloaded checkout.

The adapter is invoked through the tool's own python environment if
configured in ``tools.yaml``; availability is probed before the run and the
outcome (used / unavailable) is recorded in generation_log.json so smoke
tests on CPU-only machines stay deterministic.
"""
from __future__ import annotations

import shutil
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationResult:
    ok: bool
    pdbs: list[Path] = field(default_factory=list)
    log: str = ""
    reason: str = ""


class RfdiffusionAdapter:
    """Subprocess adapter for RFdiffusion hotspot binder design."""

    def __init__(self, spec, workdir: Path):
        self.spec = spec
        self.workdir = Path(workdir)

    def available(self) -> tuple[bool, str]:
        if self.spec is None:
            return False, "tools.yaml has no rfdiffusion entry"
        root = Path(self.spec.root)
        if not root.exists():
            return False, f"rfdiffusion root missing: {root}"
        script = root / "scripts" / "run_inference.py"
        if not script.exists():
            return False, f"run_inference.py missing under {root}"
        weights = self.spec.weights
        if weights is not None and not Path(weights).exists():
            return False, f"weights missing: {weights}"
        # audit fix: available() never validated the interpreter - with
        # python=null it probed bare 'python' (absent here) and launched died
        py = self._interpreter()
        if shutil.which(py) is None and not Path(py).exists():
            return False, f"interpreter not found: {py}"
        return True, "ok"

    def _interpreter(self) -> str:
        """Resolve the interpreter: spec.python, else the RUNNING python
        (not bare 'python' which need not exist on PATH)."""
        return str(self.spec.python) if self.spec.python else sys.executable

    def _argv(self, target_pdb: Path, hotspots: list[str], n: int, seed: int,
              binder_len: tuple[int, int]) -> list[str]:
        root = Path(self.spec.root)
        py = self._interpreter()
        argv = [
            py, str(root / "scripts" / "run_inference.py"),
            f"inference.output_prefix={self.workdir / 'rf'}",
            f"inference.input_pdb={target_pdb}",
            "inference.num_designs={n}".format(n=n),
            f"inference.ckpt_override_path={self.spec.weights}" if self.spec.weights else "",
            "ppi.hotspot_res={hs}".format(hs=",".join(hotspots)),
            f"inference.seed={seed}",
            "inference.design_minlength={}".format(binder_len[0]),
            "inference.design_maxlength={}".format(binder_len[1]),
        ]
        return [a for a in argv if a]

    def design_binder(self, target_pdb: Path, hotspots: list[str], n: int,
                      seed: int, binder_len: tuple[int, int],
                      timeout_s: int = 7200) -> GenerationResult:
        ok, why = self.available()
        if not ok:
            return GenerationResult(ok=False, reason=why)
        self.workdir.mkdir(parents=True, exist_ok=True)
        argv = self._argv(target_pdb, hotspots, n, seed, binder_len)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout_s, cwd=str(self.workdir))
        except FileNotFoundError as exc:
            return GenerationResult(ok=False, reason=f"failed to launch: {exc}")
        except subprocess.TimeoutExpired:
            return GenerationResult(ok=False, reason="timeout")
        pdbs = sorted(self.workdir.glob("rf*.pdb"))
        return GenerationResult(
            ok=(proc.returncode == 0 and bool(pdbs)),
            pdbs=pdbs,
            log=proc.stdout[-4000:] + proc.stderr[-4000:],
            reason="" if proc.returncode == 0 else f"exit {proc.returncode}",
        )
