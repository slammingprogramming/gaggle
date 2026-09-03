from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from uuid import UUID

import pytest

from gaggle.core.config import RuntimeConfig
from gaggle.core.triage import TriageService
from gaggle.ingest.service import IngestService
from gaggle.storage.repository import Repository

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_source_clip(source_root: Path, camera_id: str, name: str) -> Path:
    camera_dir = source_root / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    path = camera_dir / name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:rate=15:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _ingest_reference_clip(tmp_path: Path) -> tuple[Repository, UUID, Path]:
    source_root = tmp_path / "source"
    source_path = _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    config.storage.ingest_mode = "reference"
    config.storage.set_read_only = False  # keep the fixture file writable for the test

    manifest = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(manifest)
    clip_id = manifest.copied_files[0].clip_id
    return repository, clip_id, source_path


def test_convert_reference_to_copy_leaves_the_source_and_adds_a_workspace_copy(
    tmp_path: Path,
) -> None:
    repository, clip_id, source_path = _ingest_reference_clip(tmp_path)

    row = TriageService(repository).convert_ingest_mode(clip_id, "copy", actor="tester")

    assert row.ingest_mode == "copy"
    assert Path(row.stored_path).exists()
    assert Path(row.stored_path).is_relative_to(repository.workspace.originals)
    assert source_path.exists()  # untouched


def test_convert_reference_to_move_removes_the_source(tmp_path: Path) -> None:
    repository, clip_id, source_path = _ingest_reference_clip(tmp_path)

    row = TriageService(repository).convert_ingest_mode(clip_id, "move", actor="tester")

    assert row.ingest_mode == "move"
    assert Path(row.stored_path).exists()
    assert Path(row.stored_path).is_relative_to(repository.workspace.originals)
    assert not source_path.exists()


def test_converting_a_copy_mode_clip_to_reference_is_refused(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()  # default ingest_mode is "copy"

    manifest = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(manifest)
    clip_id = manifest.copied_files[0].clip_id

    with pytest.raises(ValueError, match="only converting from 'reference'"):
        TriageService(repository).convert_ingest_mode(clip_id, "move", actor="tester")


def test_conversion_refuses_when_the_external_source_is_gone(tmp_path: Path) -> None:
    repository, clip_id, source_path = _ingest_reference_clip(tmp_path)
    source_path.unlink()

    with pytest.raises(RuntimeError, match="no longer exists"):
        TriageService(repository).convert_ingest_mode(clip_id, "copy", actor="tester")


def test_conversion_refuses_when_the_external_source_has_changed(tmp_path: Path) -> None:
    repository, clip_id, source_path = _ingest_reference_clip(tmp_path)
    source_path.write_bytes(source_path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="does not match indexed hash"):
        TriageService(repository).convert_ingest_mode(clip_id, "copy", actor="tester")
