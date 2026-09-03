from __future__ import annotations

import hashlib
import mimetypes
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from gaggle.core.config import RuntimeConfig
from gaggle.ingest.probe import ProbeError, probe_media
from gaggle.schemas.camera import Camera
from gaggle.schemas.common import ArtifactReference, HashDigest
from gaggle.schemas.media import IngestManifest, MediaClip
from gaggle.storage.database import TimelineDatabase
from gaggle.storage.filesystem import WorkspacePaths
from gaggle.utils.filesystem import (
    copy_file_preserve_metadata,
    safe_relpath,
    set_read_only,
)
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)
SUPPORTED_EXTENSIONS: Final[set[str]] = {".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3", ".m4a"}
DEFAULT_UNPROBED_DURATION_SECONDS: Final[float] = 10.0
TIMESTAMP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?P<date>\d{8})[_-](?P<time>\d{6})"),
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[T_](?P<time>\d{2}-\d{2}-\d{2})"),
)


class IngestService:
    def __init__(
        self,
        workspace: WorkspacePaths,
        config: RuntimeConfig,
        database: TimelineDatabase | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        # Optional: enables dedup-on-ingest (checking a new file's hash
        # against already-indexed clips from prior ingest runs). Ingest
        # still works without it -- dedup is simply unavailable, not a
        # hard requirement -- since some callers (e.g. standalone scripts)
        # may not have a database handle at ingest time.
        self.database = database

    def ingest_directory(self, source_root: Path) -> IngestManifest:
        self.workspace.ensure_layout()
        # Every camera discovered in one ingest run shares this derived
        # site_id, so cameras from the same source root (e.g. a dashcam
        # rig's front/rear/interior subfolders) keep cross-syncing with
        # zero configuration -- while a security camera ingested in a
        # *separate* run naturally gets a different site_id and stays
        # isolated. See normalize/sync.py's site-scoped grouping.
        source_root_hash = hashlib.sha256(str(source_root.resolve()).encode()).hexdigest()
        default_site_id = f"site-{source_root_hash[:12]}"
        clips: list[MediaClip] = []
        hashes: list[HashDigest] = []
        seen_hashes: set[str] = set()
        registered_camera_ids: set[str] = set()
        for path in _walk_files(source_root):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            clip = self._ingest_file(
                source_root, path, seen_hashes, default_site_id, registered_camera_ids
            )
            if clip is None:
                continue
            clips.append(clip)
            hashes.append(HashDigest(value=clip.sha256))
        manifest = IngestManifest(
            run_id=new_uuid(),
            created_at=utc_now(),
            source_root=str(source_root.resolve()),
            copied_files=clips,
            config_snapshot=self.config.model_dump(mode="json"),
            hashes=hashes,
        )
        manifest_path = self.workspace.ingest / f"{manifest.run_id}.json"
        self.workspace.write_json(manifest_path, manifest.model_dump(mode="json"))
        LOGGER.info(
            "ingest_completed",
            source_root=str(source_root),
            clip_count=len(clips),
            default_site_id=default_site_id,
        )
        return manifest

    def _ingest_file(
        self,
        source_root: Path,
        path: Path,
        seen_hashes: set[str],
        default_site_id: str,
        registered_camera_ids: set[str],
    ) -> MediaClip | None:
        camera_id = path.parent.name or "unknown-camera"
        self._register_camera_if_needed(camera_id, default_site_id, registered_camera_ids)
        sha256 = hash_file(path, self.config.storage.hash_algorithm)

        if self.config.storage.dedupe_on_ingest and self._is_duplicate(sha256, seen_hashes):
            LOGGER.info("ingest_duplicate_skipped", path=str(path), sha256=sha256)
            return None
        seen_hashes.add(sha256)

        observed_start, timestamp_source, confidence = _infer_timestamp(path)

        probe_metadata: dict[str, object] = {"source_relpath": safe_relpath(path, source_root)}
        duration_seconds, fps = self._probe_duration_and_fps(path, probe_metadata)
        observed_end = observed_start + timedelta(seconds=duration_seconds)

        relative_destination = Path(camera_id) / f"{sha256[:12]}_{path.name}"
        ingest_mode = self.config.storage.ingest_mode
        stored_path = self._place_original(path, relative_destination, ingest_mode)
        sidecar_workspace_path = self.workspace.originals / relative_destination
        sidecar_artifacts = self._copy_sidecars(path, sidecar_workspace_path)
        clip = MediaClip(
            clip_id=new_uuid(),
            camera_id=camera_id,
            source_path=str(path.resolve()),
            stored_path=str(stored_path.resolve()),
            filename=path.name,
            media_type=_infer_media_type(path),
            byte_size=path.stat().st_size if path.exists() else stored_path.stat().st_size,
            sha256=sha256,
            observed_start=observed_start,
            observed_end=observed_end,
            original_timestamp_source=timestamp_source,
            timestamp_confidence=confidence,
            fps=fps,
            duration_seconds=duration_seconds,
            sidecar_artifacts=sidecar_artifacts,
            metadata=probe_metadata,
            ingest_mode=ingest_mode,
        )
        LOGGER.info(
            "clip_ingested",
            camera_id=camera_id,
            stored_path=str(stored_path),
            ingest_mode=ingest_mode,
            duration_seconds=duration_seconds,
            probe_status=probe_metadata.get("probe_status"),
        )
        return clip

    def _register_camera_if_needed(
        self, camera_id: str, default_site_id: str, registered_camera_ids: set[str]
    ) -> None:
        """Auto-register a minimal camera record on first-seen `camera_id`
        within this run -- never overwrites an already-registered camera's
        metadata (including a manually-edited `site_id`), so registration
        stays purely additive and the zero-setup workflow is unaffected.
        `registered_camera_ids` just avoids a redundant per-file database
        round-trip for a camera already handled earlier in this same run.
        """

        if self.database is None or camera_id in registered_camera_ids:
            return
        registered_camera_ids.add(camera_id)
        self.database.register_camera_if_absent(
            Camera(camera_id=camera_id, site_id=default_site_id, created_at=utc_now())
        )

    def _is_duplicate(self, sha256: str, seen_hashes: set[str]) -> bool:
        """True if ``sha256`` has already been ingested -- either earlier in
        this same run, or in a prior run whose copy still exists on disk.

        A hash match against a prior run's row whose file no longer exists
        (e.g. purged, or a ``reference``-mode source that's since vanished)
        is *not* treated as a duplicate -- that evidence is genuinely gone
        from the workspace and re-acquiring it should proceed normally,
        not be silently refused.
        """

        if sha256 in seen_hashes:
            return True
        if self.database is None:
            return False
        existing = self.database.get_media_by_sha256(sha256)
        if existing is None:
            return False
        return Path(existing.stored_path).exists()

    def _place_original(
        self,
        source: Path,
        relative_destination: Path,
        ingest_mode: Literal["copy", "move", "reference"],
    ) -> Path:
        """Put the original where ``ingest_mode`` says it should live.

        See ``core/config.py::StorageConfig.ingest_mode`` for the full
        tradeoffs of each mode. This is the one place that decision is
        acted on -- every downstream stage just reads ``MediaClip.stored_path``
        and doesn't need to know which mode produced it.
        """

        if ingest_mode == "reference":
            if self.config.storage.set_read_only:
                try:
                    set_read_only(source)
                except OSError as error:
                    LOGGER.warning(
                        "reference_read_only_failed",
                        path=str(source),
                        reason=str(error),
                    )
            return source.resolve()
        if ingest_mode == "move":
            return self.workspace.move_original(
                source, relative_destination, read_only=self.config.storage.set_read_only
            )
        return self.workspace.copy_original(
            source, relative_destination, read_only=self.config.storage.set_read_only
        )

    def _probe_duration_and_fps(
        self, path: Path, probe_metadata: dict[str, object]
    ) -> tuple[float, float | None]:
        """Extract real duration/fps via ffprobe, degrading explicitly (never silently).

        Any failure to probe is recorded in ``probe_metadata`` rather than
        swallowed, so a downstream reviewer can tell a clip's duration is a
        fallback estimate rather than a measurement.
        """

        try:
            result = probe_media(path)
        except ProbeError as error:
            LOGGER.warning("media_probe_failed", path=str(path), reason=str(error))
            probe_metadata["probe_status"] = "failed"
            probe_metadata["probe_error"] = str(error)
            return DEFAULT_UNPROBED_DURATION_SECONDS, None
        probe_metadata["probe_status"] = "ok"
        probe_metadata["probe_tool_version"] = result.probe_tool_version
        probe_metadata["video_codec"] = result.video_codec
        probe_metadata["audio_codec"] = result.audio_codec
        probe_metadata["has_audio"] = result.has_audio
        probe_metadata["width"] = result.width
        probe_metadata["height"] = result.height
        duration = (
            result.duration_seconds
            if result.duration_seconds > 0
            else DEFAULT_UNPROBED_DURATION_SECONDS
        )
        if result.duration_seconds <= 0:
            probe_metadata["probe_status"] = "ok_no_duration"
        return duration, result.fps

    def _copy_sidecars(self, source: Path, sidecar_base_path: Path) -> list[ArtifactReference]:
        """Copy a `*.samples.json` fixture sidecar and/or a `.gpx` GPS
        track (if present) into the workspace.

        Always lands under ``workspace/originals/...`` regardless of
        ``ingest_mode`` -- under "reference" mode the media file itself
        stays untouched at its source location, but a sidecar copy is still
        workspace-local so it doesn't write into (or depend on) that
        external location.
        """

        artifacts: list[ArtifactReference] = []
        sample_sidecar = source.with_suffix(f"{source.suffix}.samples.json")
        if sample_sidecar.exists():
            destination = sidecar_base_path.with_name(f"{sidecar_base_path.name}.samples.json")
            copy_file_preserve_metadata(sample_sidecar, destination)
            if self.config.storage.set_read_only:
                destination.chmod(destination.stat().st_mode & ~0o222)
            artifacts.append(
                ArtifactReference(
                    path=str(destination.resolve()),
                    artifact_type="sample_metrics",
                    created_at=utc_now(),
                    sha256=hash_file(destination),
                )
            )

        gpx_source = self._find_gpx_track(source)
        if gpx_source is not None:
            destination = sidecar_base_path.with_name(f"{sidecar_base_path.name}.gpx")
            copy_file_preserve_metadata(gpx_source, destination)
            if self.config.storage.set_read_only:
                destination.chmod(destination.stat().st_mode & ~0o222)
            artifacts.append(
                ArtifactReference(
                    path=str(destination.resolve()),
                    artifact_type="gps_track",
                    created_at=utc_now(),
                    sha256=hash_file(destination),
                )
            )
        return artifacts

    def _find_gpx_track(self, source: Path) -> Path | None:
        """A GPX track is session-level (one file can cover many clips),
        not per-file like the `*.samples.json` fixture sidecar -- look for
        any `.gpx` file in the same camera directory as `source`, matched
        by presence rather than filename correlation. If more than one
        exists, the first in sorted order wins (deterministic) and a
        warning is logged, since this project's telemetry support assumes
        one GPS track per ingest session -- see
        `detection/telemetry_analysis.py`'s module docstring and
        `docs/limitations.md` for that scope boundary.
        """

        candidates = sorted(source.parent.glob("*.gpx"))
        if not candidates:
            return None
        if len(candidates) > 1:
            LOGGER.warning(
                "multiple_gpx_tracks_found",
                directory=str(source.parent),
                using=str(candidates[0]),
                ignored=[str(c) for c in candidates[1:]],
            )
        return candidates[0]


def _walk_files(root: Path) -> Iterator[Path]:
    """Recursively yield every file under ``root``, in deterministic
    (sorted) order, refusing to descend into a symlinked directory that
    would revisit a directory already walked -- a plain ``rglob("*")``
    has no such protection and can hang or loop forever on a symlink
    cycle (e.g. a directory symlinked to one of its own ancestors).

    Every directory's resolved real path is tracked in ``visited`` as it's
    entered, regardless of whether it was reached via a symlink or not --
    a cycle can only ever close through a symlink, but the *other* end of
    the cycle is often an ordinary directory reached earlier in the walk,
    so only guarding symlinked directories against themselves would miss
    that case.
    """

    root_real = root.resolve()
    yield from _walk_dir(root, root_real, {root_real})


def _walk_dir(directory: Path, root_real: Path, visited: set[Path]) -> Iterator[Path]:
    try:
        entries = sorted(directory.iterdir())
    except OSError as error:
        LOGGER.warning("ingest_walk_dir_failed", path=str(directory), reason=str(error))
        return
    for entry in entries:
        if entry.is_file():
            yield entry
            continue
        if not entry.is_dir():
            continue  # broken symlink or other non-file, non-dir entry
        try:
            real = entry.resolve()
        except OSError as error:
            LOGGER.warning("ingest_symlink_unresolvable", path=str(entry), reason=str(error))
            continue
        if real == root_real or real in root_real.parents or real in visited:
            LOGGER.warning("ingest_symlink_loop_skipped", path=str(entry), target=str(real))
            continue
        visited.add(real)
        yield from _walk_dir(entry, root_real, visited)


def _infer_media_type(path: Path) -> Literal["video", "audio", "image", "unknown"]:
    guessed, _encoding = mimetypes.guess_type(path.name)
    if guessed is None:
        return "unknown"
    if guessed.startswith("video"):
        return "video"
    if guessed.startswith("audio"):
        return "audio"
    if guessed.startswith("image"):
        return "image"
    return "unknown"


def _infer_timestamp(
    path: Path,
) -> tuple[datetime, Literal["filename", "mtime", "sidecar", "manual"], float]:
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(path.stem)
        if match is None:
            continue
        raw_date = match.group("date")
        raw_time = match.group("time")
        normalized = f"{raw_date} {raw_time.replace('-', ':')}"
        format_string = "%Y%m%d %H%M%S" if len(raw_date) == 8 else "%Y-%m-%d %H:%M:%S"
        parsed = datetime.strptime(normalized, format_string).replace(tzinfo=UTC)
        return parsed, "filename", 0.7
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return mtime, "mtime", 0.3
