from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gaggle.core.config import RuntimeConfig
from gaggle.ingest.service import IngestService
from gaggle.storage.repository import Repository

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_track.gpx"


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


def test_ingest_copies_a_colocated_gpx_track_as_a_sidecar(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")
    shutil.copy(FIXTURE_GPX, source_root / "front" / "track.gpx")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()

    manifest = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)

    assert len(manifest.copied_files) == 1
    clip = manifest.copied_files[0]
    gps_artifacts = [a for a in clip.sidecar_artifacts if a.artifact_type == "gps_track"]
    assert len(gps_artifacts) == 1
    copied_path = Path(gps_artifacts[0].path)
    assert copied_path.exists()
    assert copied_path.read_bytes() == FIXTURE_GPX.read_bytes()
    # A workspace-local copy, not a pointer back at the source directory --
    # matches the `*.samples.json` sidecar's own copy-not-reference behavior.
    assert copied_path != (source_root / "front" / "track.gpx")


def test_ingest_without_a_gpx_file_produces_no_gps_track_sidecar(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()

    manifest = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)

    clip = manifest.copied_files[0]
    assert [a for a in clip.sidecar_artifacts if a.artifact_type == "gps_track"] == []


def test_ingest_with_multiple_gpx_files_deterministically_picks_one(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")
    shutil.copy(FIXTURE_GPX, source_root / "front" / "b_track.gpx")
    shutil.copy(FIXTURE_GPX, source_root / "front" / "a_track.gpx")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()

    manifest = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)

    clip = manifest.copied_files[0]
    gps_artifacts = [a for a in clip.sidecar_artifacts if a.artifact_type == "gps_track"]
    assert len(gps_artifacts) == 1  # never both, never a crash -- sorted-first wins
