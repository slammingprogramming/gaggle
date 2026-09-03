from __future__ import annotations

from datetime import timedelta

from gaggle.core.config import RuntimeConfig
from gaggle.schemas.media import EventWindow, NormalizationManifest, WindowManifest
from gaggle.storage.filesystem import WorkspacePaths
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class WindowingService:
    """Generates deterministic sliding windows over normalized clip intervals.

    Windowing always operates on *corrected* (sync-adjusted) timestamps, not
    the raw per-camera observed timestamps, so that a window genuinely
    represents "this span of real-world time across all cameras" rather than
    each camera's own uncorrected clock.
    """

    def __init__(self, workspace: WorkspacePaths, config: RuntimeConfig) -> None:
        self.workspace = workspace
        self.config = config

    def generate(self, normalization_manifest: NormalizationManifest) -> WindowManifest:
        if not normalization_manifest.clips:
            manifest = WindowManifest(
                run_id=new_uuid(),
                created_at=utc_now(),
                windows=[],
                source_normalization_run_id=normalization_manifest.run_id,
            )
            self.workspace.write_json(
                self.workspace.windows / f"{manifest.run_id}.json",
                manifest.model_dump(mode="json"),
            )
            return manifest
        window_duration = timedelta(seconds=self.config.pipeline.window_duration_seconds)
        stride = timedelta(seconds=self.config.pipeline.window_stride_seconds)
        start = min(clip.corrected_start for clip in normalization_manifest.clips)
        end = max(clip.corrected_end for clip in normalization_manifest.clips)
        cursor = start
        windows: list[EventWindow] = []
        while cursor < end:
            window_end = min(cursor + window_duration, end)
            overlapping = [
                clip
                for clip in normalization_manifest.clips
                if clip.corrected_start < window_end and clip.corrected_end > cursor
            ]
            if overlapping:
                windows.append(
                    EventWindow(
                        window_id=new_uuid(),
                        start=cursor,
                        end=window_end,
                        involved_cameras=sorted({clip.camera_id for clip in overlapping}),
                        clip_ids=[clip.clip_id for clip in overlapping],
                        rationale=(
                            "fixed deterministic sliding window over sync-corrected clip intervals"
                        ),
                    )
                )
            cursor += stride
        manifest = WindowManifest(
            run_id=new_uuid(),
            created_at=utc_now(),
            windows=windows,
            source_normalization_run_id=normalization_manifest.run_id,
        )
        output_path = self.workspace.windows / f"{manifest.run_id}.json"
        self.workspace.write_json(output_path, manifest.model_dump(mode="json"))
        LOGGER.info("windows_generated", count=len(windows), run_id=str(manifest.run_id))
        return manifest
