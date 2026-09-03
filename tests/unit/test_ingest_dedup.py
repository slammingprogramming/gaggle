from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gaggle.core.config import RuntimeConfig
from gaggle.ingest.service import IngestService, _walk_files
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


def test_reingesting_the_same_file_is_skipped_by_default(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()

    first = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(first)
    assert len(first.copied_files) == 1

    second = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(second)

    assert len(second.copied_files) == 0
    assert len(repository.database.list_media()) == 1


def test_reingest_proceeds_if_the_first_copy_no_longer_exists(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()

    first = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(first)
    stored_path = Path(first.copied_files[0].stored_path)
    stored_path.chmod(stored_path.stat().st_mode | 0o600)  # clear read-only for cleanup
    stored_path.unlink()

    second = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(second)

    assert len(second.copied_files) == 1


def test_dedupe_can_be_disabled(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    config.storage.dedupe_on_ingest = False

    first = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(first)
    second = IngestService(
        repository.workspace, config, database=repository.database
    ).ingest_directory(source_root)
    repository.index_ingest_manifest(second)

    assert len(second.copied_files) == 1
    assert len(repository.database.list_media()) == 2


def test_ingest_without_a_database_handle_still_works(tmp_path: Path) -> None:
    """Dedup against prior runs needs a database handle, but ingest itself
    must not require one -- it should just fall back to "dedup impossible,
    always ingest" (still deduping within the same run via in-memory hash
    tracking)."""

    source_root = tmp_path / "source"
    _make_source_clip(source_root, "front", "20260101_120000.mp4")

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()

    manifest = IngestService(repository.workspace, config).ingest_directory(source_root)
    assert len(manifest.copied_files) == 1


def test_walk_files_skips_a_symlink_loop(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "front").mkdir(parents=True)
    real_file = root / "front" / "clip.mp4"
    real_file.write_bytes(b"not a real video, just a fixture for the walk test")

    loop_link = root / "front" / "loop"
    try:
        os.symlink(root, loop_link, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks isn't permitted in this environment")

    found = sorted(_walk_files(root))
    assert found == [real_file]


def test_walk_files_follows_a_non_looping_symlinked_directory(tmp_path: Path) -> None:
    root = tmp_path / "source"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(parents=True)
    other_file = elsewhere / "clip.mp4"
    other_file.write_bytes(b"fixture")
    root.mkdir(parents=True)

    linked = root / "linked"
    try:
        os.symlink(elsewhere, linked, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks isn't permitted in this environment")

    found = sorted(_walk_files(root))
    assert found == [linked / "clip.mp4"]
