"""A registered camera source -- optional metadata layered on top of the
free-form `camera_id` string tag already used throughout `schemas/media.py`,
`schemas/signal.py`, and `schemas/recognition.py`.

Registration is never required for the system to work: `ingest/service.py`
auto-registers a minimal record (`source_type="other"`) the first time a
new `camera_id` is seen, so the existing zero-setup dashcam workflow is
completely unaffected by this. Richer metadata (a real `source_type`,
`indoor`, a `site_id`) is opt-in via `camera register`/`camera update`.

`site_id` is what scopes cross-camera time synchronization
(`normalize/sync.py`): cameras sharing a `site_id` are candidates for
overlap-based alignment (one dashcam rig's simultaneous cameras); cameras
with no `site_id`, or different `site_id`s, are never cross-synced against
each other -- correct for independent security cameras with unrelated
clocks, which would otherwise risk being spuriously aligned just because
their recording times happened to overlap. `ingest/service.py` derives a
deterministic default `site_id` per ingest run so existing dashcam users
keep cross-syncing with zero configuration; see its docstring.
"""

from __future__ import annotations

from typing import Literal

from gaggle.schemas.common import StrictModel, UTCDateTime

CAMERA_SCHEMA_VERSION = "1.0.0"

CameraSourceType = Literal[
    "dashcam", "security_ip", "security_usb", "nvr_export", "doorbell", "other"
]


class Camera(StrictModel):
    schema_version: str = CAMERA_SCHEMA_VERSION
    camera_id: str
    label: str | None = None
    source_type: CameraSourceType = "other"
    indoor: bool | None = None
    site_id: str | None = None
    created_at: UTCDateTime
    notes: str = ""
