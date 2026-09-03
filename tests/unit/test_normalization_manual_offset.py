from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from gaggle.core.config import RuntimeConfig, SyncConfig
from gaggle.normalize.service import NormalizationService
from gaggle.normalize.sync import SessionSyncResult
from gaggle.storage.filesystem import WorkspacePaths

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_service(
    tmp_path: Path, manual_offset_overrides: dict[str, float]
) -> NormalizationService:
    workspace = WorkspacePaths(tmp_path / "workspace")
    config = RuntimeConfig(sync=SyncConfig(manual_offset_overrides=manual_offset_overrides))
    return NormalizationService(workspace, config)


def _make_result(camera_id: str, is_reference: bool) -> SessionSyncResult:
    return SessionSyncResult(
        camera_id=camera_id,
        session_id=f"{camera_id}#000",
        clip_ids=["clip-1"],
        original_start=BASE,
        original_end=BASE + timedelta(seconds=60),
        corrected_start=BASE,
        corrected_end=BASE + timedelta(seconds=60),
        offset_seconds=5.0,
        drift_seconds_per_hour=0.0,
        confidence=0.8,
        is_reference=is_reference,
        reference_camera_id=None if is_reference else "front",
        rationale="test rationale",
    )


def test_manual_offset_shifts_a_non_reference_session(tmp_path: Path) -> None:
    service = _make_service(tmp_path, {"rear": 30.0})
    result = _make_result("rear", is_reference=False)

    adjusted = service._apply_manual_offset(result)

    assert adjusted.corrected_start == result.corrected_start + timedelta(seconds=30.0)
    assert adjusted.corrected_end == result.corrected_end + timedelta(seconds=30.0)
    assert adjusted.offset_seconds == result.offset_seconds + 30.0
    assert "manual_offset_overrides['rear']" in adjusted.rationale


def test_manual_offset_is_a_no_op_when_unconfigured(tmp_path: Path) -> None:
    service = _make_service(tmp_path, {})
    result = _make_result("rear", is_reference=False)

    adjusted = service._apply_manual_offset(result)

    assert adjusted is result  # untouched, not just equal


def test_manual_offset_is_a_no_op_when_configured_as_zero(tmp_path: Path) -> None:
    service = _make_service(tmp_path, {"rear": 0.0})
    result = _make_result("rear", is_reference=False)

    adjusted = service._apply_manual_offset(result)

    assert adjusted is result


def test_manual_offset_is_ignored_for_the_reference_session(tmp_path: Path) -> None:
    """A configured override targeting whichever camera turns out to be
    the sync reference for its group is a documented no-op (see
    NormalizationService._apply_manual_offset's docstring for why:
    shifting the reference alone would misalign it from the sessions
    already aligned to it) -- confirmed here rather than assumed."""

    service = _make_service(tmp_path, {"front": 45.0})
    result = _make_result("front", is_reference=True)

    adjusted = service._apply_manual_offset(result)

    assert adjusted is result
    assert adjusted.corrected_start == result.corrected_start
