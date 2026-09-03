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


def test_copy_mode_leaves_source_untouched(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_path = _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    config.storage.ingest_mode = "copy"

    manifest = IngestService(repository.workspace, config).ingest_directory(source_root)

    assert len(manifest.copied_files) == 1
    clip = manifest.copied_files[0]
    assert clip.ingest_mode == "copy"
    assert source_path.exists()  # untouched
    assert Path(clip.stored_path).exists()
    assert Path(clip.stored_path) != source_path
    assert Path(clip.stored_path).is_relative_to(repository.workspace.originals)


def test_move_mode_relocates_the_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_path = _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    config.storage.ingest_mode = "move"

    manifest = IngestService(repository.workspace, config).ingest_directory(source_root)

    clip = manifest.copied_files[0]
    assert clip.ingest_mode == "move"
    assert not source_path.exists()  # relocated, not copied
    assert Path(clip.stored_path).exists()
    assert Path(clip.stored_path).is_relative_to(repository.workspace.originals)


def test_reference_mode_never_touches_the_source_location(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_path = _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    config.storage.ingest_mode = "reference"
    config.storage.set_read_only = False  # keep the fixture file writable for cleanup

    manifest = IngestService(repository.workspace, config).ingest_directory(source_root)

    clip = manifest.copied_files[0]
    assert clip.ingest_mode == "reference"
    assert source_path.exists()
    assert Path(clip.stored_path) == source_path.resolve()
    # Nothing was written into workspace/originals/ for the media file itself.
    assert not any((repository.workspace.originals / "front").glob("*.mp4"))


def test_reference_mode_still_places_sidecars_in_the_workspace(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_path = _make_source_clip(source_root, "front", "20260101_120000.mp4")
    sidecar_path = source_path.with_suffix(f"{source_path.suffix}.samples.json")
    sidecar_path.write_text('{"motion_series": []}', encoding="utf-8")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    config.storage.ingest_mode = "reference"
    config.storage.set_read_only = False

    manifest = IngestService(repository.workspace, config).ingest_directory(source_root)

    clip = manifest.copied_files[0]
    assert len(clip.sidecar_artifacts) == 1
    sidecar_workspace_path = Path(clip.sidecar_artifacts[0].path)
    assert sidecar_workspace_path.is_relative_to(repository.workspace.originals)
    assert sidecar_workspace_path.exists()
