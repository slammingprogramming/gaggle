from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from gaggle.inference.service import InferenceService
from gaggle.schemas.signal import Signal

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _signal(signal_type: str, confidence: float, window_id, camera_id: str = "front") -> Signal:
    return Signal(
        id=uuid4(),
        source="test",
        signal_type=signal_type,  # type: ignore[arg-type]
        timestamp_start=BASE,
        timestamp_end=BASE + timedelta(seconds=1),
        confidence=confidence,
        camera_id=camera_id,
        window_id=window_id,
    )


def test_isolated_gunshot_retention_caps_confidence_at_point_six() -> None:
    window_id = uuid4()
    signals = [_signal("gunshot", 0.95, window_id)]

    hypotheses = InferenceService(load_rule_plugins=False).infer(signals)

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.rule_name == "isolated_gunshot_retention"
    assert hypothesis.confidence == 0.60
    assert hypothesis.contributing_signal_ids == [signals[0].id]


def test_isolated_gunshot_retention_uses_mean_confidence_when_below_cap() -> None:
    window_id = uuid4()
    signals = [_signal("gunshot", 0.30, window_id)]

    hypotheses = InferenceService(load_rule_plugins=False).infer(signals)

    assert len(hypotheses) == 1
    assert hypotheses[0].confidence == 0.30


def test_gunshot_plus_motion_produces_a_corroborated_hypothesis() -> None:
    window_id = uuid4()
    gunshot = _signal("gunshot", 0.80, window_id)
    motion = _signal("motion", 0.60, window_id)

    hypotheses = InferenceService(load_rule_plugins=False).infer([gunshot, motion])

    rule_names = {h.rule_name for h in hypotheses}
    assert "gunshot_plus_motion" in rule_names
    assert "isolated_gunshot_retention" not in rule_names
    corroborated = next(h for h in hypotheses if h.rule_name == "gunshot_plus_motion")
    assert set(corroborated.contributing_signal_ids) == {gunshot.id, motion.id}
    assert corroborated.confidence == pytest.approx(0.80)


def test_no_gunshot_signal_produces_no_gunshot_hypothesis() -> None:
    window_id = uuid4()
    signals = [_signal("motion", 0.5, window_id)]

    hypotheses = InferenceService(load_rule_plugins=False).infer(signals)

    rule_names = {h.rule_name for h in hypotheses}
    assert "isolated_gunshot_retention" not in rule_names
    assert "gunshot_plus_motion" not in rule_names
