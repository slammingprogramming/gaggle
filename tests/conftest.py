from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def reset_read_only_tmp_artifacts(request: pytest.FixtureRequest):
    tmp_path = request.getfixturevalue("tmp_path") if "tmp_path" in request.fixturenames else None
    yield
    if tmp_path is None:
        return
    for path in sorted(tmp_path.rglob("*"), reverse=True):
        try:
            os.chmod(path, 0o700)
        except OSError:
            continue
