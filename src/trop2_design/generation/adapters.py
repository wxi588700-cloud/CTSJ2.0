"""RFdiffusion adapter (M04) - wraps the previously downloaded checkout.

The adapter is invoked through the tool's own python environment if
configured in ``tools.yaml``; availability is probed before the run and the
outcome (used / unavailable) is recorded in generation_log.json so smoke
tests on CPU-only machines stay deterministic.
"""
from __future__ import annotations

import os
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
        if shutil.which(py) is None and not Path(py).expanduser().exists():
            return False, f"interpreter not found: {py}"
        return True, "ok"

    def _interpreter(self) -> str:
        """Resolve the interpreter: spec.python (~/ expanded), else the
        RUNNING python (not bare 'python' which need not exist on PATH)."""
        py = str(self.spec.python).strip() if self.spec.python else sys.executable
        return str(Path(py).expanduser()) if py.startswith("~") else py

    def _argv(self, target_pdb: Path, hotspots: list[str], n: int, seed: int,
              binder_len: tuple[int, int], contigs: str | None = None) -> list[str]:
        # absolute paths: the subprocess (and the ssh variant) cd's into the
        # workdir first, so a relative root would break resolution
        root = Path(self.spec.root).expanduser().resolve()
        py = self._interpreter()
        argv = [
            py, str(root / "scripts" / "run_inference.py"),
            f"inference.output_prefix={self.workdir / 'rf'}",
            f"inference.input_pdb={target_pdb}",
            "inference.num_designs={n}".format(n=n),
            (f"inference.ckpt_override_path="
             f"{Path(self.spec.weights).expanduser().resolve()}"
             if self.spec.weights else ""),
            "ppi.hotspot_res=[{hs}]".format(hs=",".join(hotspots)),
            # NOTE: no inference.seed key in this fork (seeds internally per
            # design index); binder design REQUIRES an explicit contig string
            # alongside ppi hotspots (contigmap.length alone is rejected)
            (f"contigmap.contigs=[{contigs} {binder_len[0]}-{binder_len[1]}]"
             if contigs else
             "contigmap.length={}-{}".format(binder_len[0], binder_len[1])),
        ]
        return [a for a in argv if a]

    def _prepare_input(self, target_pdb: Path, hotspots: list[str]):
        """Convert mmCIF to fixed-column PDB (RFdiffusion parses PDB only)
        with single-letter chain IDs, remapping '{aa}{num}' hotspots to
        '{chain}{num}' against the renamed chains."""
        import re as _re
        if target_pdb.suffix.lower() != ".cif":
            return target_pdb, hotspots, None
        import gemmi
        st = gemmi.read_structure(str(target_pdb))
        import string as _string
        for i, ch in enumerate(st[0]):
            ch.name = _string.ascii_uppercase[i % 26]
        st.setup_entities()
        out_pdb = self.workdir / "target_input.pdb"
        st.write_pdb(str(out_pdb))
        # explicit contig string (this fork REQUIRES it with ppi hotspots):
        # every input chain kept fixed, binder free-designed at the tail
        parts = []
        for ch in st[0]:
            nums = [r.seqid.num for r in ch]
            if len(nums) >= 2:
                parts.append(f"{ch.name}{nums[0]}-{nums[-1]}/0")
        contigs = " ".join(parts)
        mapped = []
        for hs in hotspots:
            num_s = _re.sub(r"[^0-9]", "", hs)
            if not num_s:
                continue
            num = int(num_s)
            for ch in st[0]:
                if any(r.seqid.num == num for r in ch):
                    mapped.append(f"{ch.name}{num}")
                    break
        return out_pdb, (mapped or hotspots), contigs

    def design_binder(self, target_pdb: Path, hotspots: list[str], n: int,
                      seed: int, binder_len: tuple[int, int],
                      timeout_s: int = 7200) -> GenerationResult:
        ok, why = self.available()
        if not ok:
            return GenerationResult(ok=False, reason=why)
        self.workdir.mkdir(parents=True, exist_ok=True)
        target_pdb, hotspots, contigs = self._prepare_input(target_pdb, hotspots)
        argv = self._argv(target_pdb, hotspots, n, seed, binder_len, contigs)
        ssh_host = getattr(self.spec, "ssh_host", None)
        if ssh_host:
            # GPU dispatch over NFS-shared home (mirrors the boltz adapter):
            # workdir/paths must live under the shared home (they do - the
            # run tree is under the repo), /tmp is NOT shared between nodes
            import shlex
            dev = os.environ.get("TROP2_BOLTZ_DEVICE", "6")
            # shared-card guard: wait for free memory instead of OOM (a
            # neighbour process once held 46/48GB and we silently degraded)
            from ..io.gpu_wait import wait_gpu_free
            if not wait_gpu_free(ssh_host, dev, min_free_mb=10_000,
                                 max_wait_min=60, label="rfdiffusion"):
                return GenerationResult(
                    ok=False, pdbs=[], log="",
                    reason=(f"GPU{dev} stayed busy (<10GB free) for 60min - "
                            f"refusing to degrade silently"))
            remote = (f"cd {shlex.quote(str(self.workdir))} && "
                      f"CUDA_VISIBLE_DEVICES={dev} HYDRA_FULL_ERROR=1 "
                      + " ".join(shlex.quote(a) for a in argv))
            argv = ["ssh", ssh_host, remote]
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
            reason="" if proc.returncode == 0 else (
                f"exit {proc.returncode}: " +
                ((proc.stderr or "").strip().splitlines() or [""])[-1][:200]),
        )
