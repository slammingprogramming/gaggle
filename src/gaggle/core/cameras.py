"""Camera registry lookups -- thin wrapper over `storage/database.py`'s
Camera CRUD, mirroring `core/recognition.py::RecognitionService`'s shape.
Used by `ingest/service.py` (auto-registration), `normalize/sync.py` (site-
scoped grouping), and `cli/app.py`'s `camera` command group.

Registration is always optional: `camera_id` remains a free-form string
everywhere else in the schema. Nothing here is required for the pipeline to
work end to end with zero camera setup, exactly as before this entity
existed.
"""

from __future__ import annotations

from gaggle.schemas.camera import Camera
from gaggle.storage.database import CameraRow
from gaggle.storage.repository import Repository
from gaggle.utils.time import utc_now


class CameraRepository:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def register_if_absent(self, camera_id: str, site_id: str | None) -> None:
        camera = Camera(camera_id=camera_id, site_id=site_id, created_at=utc_now())
        self.repository.database.register_camera_if_absent(camera)

    def register(self, camera: Camera) -> None:
        self.repository.database.upsert_camera(camera)

    def get(self, camera_id: str) -> CameraRow | None:
        return self.repository.database.get_camera(camera_id)

    def list(self) -> list[CameraRow]:
        return self.repository.database.list_cameras()

    def site_id_by_camera(self) -> dict[str, str]:
        return self.repository.database.site_id_by_camera()

    def update(
        self,
        camera_id: str,
        label: str | None = None,
        source_type: str | None = None,
        indoor: bool | None = None,
        site_id: str | None = None,
        notes: str | None = None,
    ) -> CameraRow | None:
        return self.repository.database.update_camera(
            camera_id,
            label=label,
            source_type=source_type,
            indoor=indoor,
            site_id=site_id,
            notes=notes,
        )
