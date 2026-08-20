import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.mini_target import MINI_FASTA, SEQ, build_mini_target  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def mini_target(tmp_path_factory) -> Path:
    return build_mini_target(tmp_path_factory.mktemp("mini") / "mini_target.cif")


@pytest.fixture(scope="session")
def mini_fasta(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("mini") / "mini.fasta"
    p.write_text(MINI_FASTA, encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def mini_seq() -> str:
    return SEQ
