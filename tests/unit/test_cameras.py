from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gaggle.core.cameras import CameraRepository
from gaggle.schemas.camera import Camera
from gaggle.storage.repository import Repository

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def test_register_if_absent_creates_a_minimal_camera(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)

    cameras.register_if_absent("front", site_id="site-abc")

    row = cameras.get("front")
    assert row is not None
    assert row.camera_id == "front"
    assert row.site_id == "site-abc"
    assert row.source_type == "other"


def test_register_if_absent_never_overwrites_existing_metadata(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)
    cameras.register(
        Camera(
            camera_id="front",
            label="Front dashcam",
            source_type="dashcam",
            site_id="user-chosen-site",
            created_at=BASE,
        )
    )

    cameras.register_if_absent("front", site_id="auto-derived-site")

    row = cameras.get("front")
    assert row is not None
    assert row.label == "Front dashcam"
    assert row.source_type == "dashcam"
    assert row.site_id == "user-chosen-site"


def test_register_overwrites_unconditionally(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)
    cameras.register(Camera(camera_id="porch", source_type="other", created_at=BASE))

    cameras.register(
        Camera(
            camera_id="porch",
            label="Porch cam",
            source_type="security_ip",
            indoor=False,
            created_at=BASE,
        )
    )

    row = cameras.get("porch")
    assert row is not None
    assert row.label == "Porch cam"
    assert row.source_type == "security_ip"
    assert row.indoor is False


def test_list_returns_every_registered_camera_sorted_by_id(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)
    cameras.register(Camera(camera_id="rear", created_at=BASE))
    cameras.register(Camera(camera_id="front", created_at=BASE))

    listed = [c.camera_id for c in cameras.list()]

    assert listed == ["front", "rear"]


def test_site_id_by_camera_omits_cameras_with_no_site(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)
    cameras.register(Camera(camera_id="front", site_id="site-a", created_at=BASE))
    cameras.register(Camera(camera_id="unregistered-site", site_id=None, created_at=BASE))

    mapping = cameras.site_id_by_camera()

    assert mapping == {"front": "site-a"}


def test_update_only_changes_passed_fields(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)
    cameras.register(
        Camera(
            camera_id="front",
            label="Front",
            source_type="dashcam",
            site_id="site-a",
            created_at=BASE,
        )
    )

    updated = cameras.update("front", indoor=False)

    assert updated is not None
    assert updated.indoor is False
    assert updated.label == "Front"  # untouched
    assert updated.site_id == "site-a"  # untouched


def test_update_returns_none_for_unregistered_camera(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cameras = CameraRepository(repository)

    assert cameras.update("never-registered", label="x") is None
