from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from gaggle.core.config import RuntimeConfig
from gaggle.schemas.media import EventWindow, NormalizedClip
from gaggle.schemas.signal import Signal


@dataclass(frozen=True)
class DetectionInputs:
    workspace_root: Path
    windows: list[EventWindow]
    clips: list[NormalizedClip]
    config: RuntimeConfig


class Detector:
    """Base interface for all detectors, built-in or plugin-provided.

    Detectors emit ``Signal`` instances -- evidence, never conclusions. A
    detector implementation should be a pure function of its inputs: given
    the same media bytes, windows, and config, it must produce the same
    signals every time it runs (see the project's reproducibility
    requirements). ``name`` and ``version`` are recorded on every signal a
    detector produces so results can always be traced back to the exact
    detector logic that generated them.
    """

    name: str = "detector"
    version: str = "1.0.0"

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        raise NotImplementedError


def match_window_id(
    windows: Sequence[EventWindow],
    camera_id: str,
    start: datetime,
    end: datetime,
) -> UUID | None:
    """Find the window that fully contains ``[start, end)`` for ``camera_id``.

    Windows are checked in the order given (chronological, from
    ``WindowingService``), so the earliest-starting containing window wins
    when sliding windows overlap. Returns ``None`` when no window claims the
    interval, in which case the caller must drop the sample rather than
    inventing a window for it.
    """

    for window in windows:
        if camera_id not in window.involved_cameras:
            continue
        if window.start <= start and window.end >= end:
            return window.window_id
    return None
