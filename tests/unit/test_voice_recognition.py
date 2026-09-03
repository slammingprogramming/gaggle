from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaggle.enrichment.voice import (
    IncrementalVoiceClusterer,
    VoiceAnalysisError,
    VoiceSegment,
    compute_voiceprint,
    detect_voice_segments,
)

SAMPLE_RATE = 16_000


def _synthesize_voice(
    duration_seconds: float,
    f0: float,
    formants: list[tuple[float, float]],
    noise_seed: int = 0,
    noise_amplitude: float = 0.02,
) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(SAMPLE_RATE * duration_seconds), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * f0 * t)
    for freq, amp in formants:
        signal = signal + amp * np.sin(2 * np.pi * freq * t)
    rng = np.random.default_rng(noise_seed)
    signal = signal + rng.normal(0, noise_amplitude, size=signal.shape)
    return signal


def _synthesize_pure_tone(duration_seconds: float, frequency: float) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(SAMPLE_RATE * duration_seconds), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)


def test_voice_activity_detection_finds_two_speech_segments() -> None:
    duration = 6.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = np.zeros_like(t)

    def add_voice(start: float, end: float, f0: float, formants: list[tuple[float, float]]):
        mask = (t >= start) & (t < end)
        seg_t = t[mask]
        signal = 0.3 * np.sin(2 * np.pi * f0 * seg_t)
        for freq, amp in formants:
            signal = signal + amp * np.sin(2 * np.pi * freq * seg_t)
        audio[mask] += signal

    add_voice(1.0, 2.5, 120, [(600, 0.3), (1200, 0.2)])
    add_voice(3.5, 5.0, 220, [(900, 0.3), (2400, 0.2)])
    rng = np.random.default_rng(3)
    audio = audio + rng.normal(0, 0.02, size=audio.shape)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.8

    segments = detect_voice_segments(audio, SAMPLE_RATE)

    assert len(segments) == 2
    assert 0.8 <= segments[0].start_offset_seconds <= 1.2
    assert 2.2 <= segments[0].end_offset_seconds <= 2.6
    assert 3.3 <= segments[1].start_offset_seconds <= 3.7


def test_voice_activity_detection_rejects_a_pure_tone() -> None:
    """A pure tone (standing in for a horn or steady engine hum) must not be
    mistaken for speech -- this is the whole reason spectral flatness is
    part of the VAD gate, not just energy."""

    tone = _synthesize_pure_tone(2.0, 440.0)
    segments = detect_voice_segments(tone, SAMPLE_RATE)
    assert segments == []


def test_voice_activity_detection_rejects_silence() -> None:
    silence = np.zeros(SAMPLE_RATE * 2)
    segments = detect_voice_segments(silence, SAMPLE_RATE)
    assert segments == []


def test_voice_activity_detection_rejects_empty_input() -> None:
    with pytest.raises(VoiceAnalysisError):
        detect_voice_segments(np.array([]), SAMPLE_RATE)


def test_compute_voiceprint_has_expected_shape() -> None:
    audio = _synthesize_voice(2.0, 120, [(600, 0.3), (1200, 0.2)])
    observation = compute_voiceprint(audio, SAMPLE_RATE)
    assert observation.voiceprint.shape == (26,)
    assert 0.0 <= observation.energy_confidence <= 1.0


def test_compute_voiceprint_rejects_too_short_segment() -> None:
    tiny = np.zeros(10)
    with pytest.raises(VoiceAnalysisError):
        compute_voiceprint(tiny, SAMPLE_RATE)


def test_same_voice_scores_much_closer_than_different_voices() -> None:
    """The core empirical claim this whole capability rests on. Verified
    across several noise seeds during development (not just one lucky
    sample) before the default cluster_distance_threshold was chosen --
    see enrichment/voice.py's module docstring."""

    from gaggle.enrichment.voice import _cosine_distance

    same_voice_distances = []
    diff_voice_distances = []
    for seed in range(4):
        a1 = _synthesize_voice(1.5, 120, [(600, 0.3), (1200, 0.2)], noise_seed=seed)
        a2 = _synthesize_voice(1.5, 121, [(605, 0.3), (1190, 0.2)], noise_seed=seed + 100)
        b = _synthesize_voice(1.5, 220, [(900, 0.3), (2400, 0.2)], noise_seed=seed)

        vp_a1 = compute_voiceprint(a1, SAMPLE_RATE).voiceprint
        vp_a2 = compute_voiceprint(a2, SAMPLE_RATE).voiceprint
        vp_b = compute_voiceprint(b, SAMPLE_RATE).voiceprint

        same_voice_distances.append(_cosine_distance(vp_a1, vp_a2))
        diff_voice_distances.append(_cosine_distance(vp_a1, vp_b))

    assert max(same_voice_distances) < 0.02
    assert min(diff_voice_distances) > 0.05
    assert max(same_voice_distances) < min(diff_voice_distances)


def test_clusterer_merges_the_same_voice_across_two_appearances(tmp_path: Path) -> None:
    a1 = _synthesize_voice(1.5, 120, [(600, 0.3), (1200, 0.2)], noise_seed=1)
    a2 = _synthesize_voice(1.5, 121, [(605, 0.3), (1195, 0.2)], noise_seed=2)
    vp_a1 = compute_voiceprint(a1, SAMPLE_RATE).voiceprint
    vp_a2 = compute_voiceprint(a2, SAMPLE_RATE).voiceprint

    clusterer = IncrementalVoiceClusterer(tmp_path / "model.json")
    id1, _dist1, is_new1 = clusterer.match_or_create_cluster(vp_a1)
    id2, _dist2, is_new2 = clusterer.match_or_create_cluster(vp_a2)

    assert is_new1 is True
    assert is_new2 is False
    assert id1 == id2


def test_clusterer_separates_two_different_voices(tmp_path: Path) -> None:
    a = _synthesize_voice(1.5, 120, [(600, 0.3), (1200, 0.2)], noise_seed=1)
    b = _synthesize_voice(1.5, 220, [(900, 0.3), (2400, 0.2)], noise_seed=1)
    vp_a = compute_voiceprint(a, SAMPLE_RATE).voiceprint
    vp_b = compute_voiceprint(b, SAMPLE_RATE).voiceprint

    clusterer = IncrementalVoiceClusterer(tmp_path / "model.json")
    id_a, _dist_a, is_new_a = clusterer.match_or_create_cluster(vp_a)
    id_b, _dist_b, is_new_b = clusterer.match_or_create_cluster(vp_b)

    assert is_new_a is True
    assert is_new_b is True
    assert id_a != id_b


def test_clusterer_persists_and_reloads(tmp_path: Path) -> None:
    a = _synthesize_voice(1.5, 120, [(600, 0.3), (1200, 0.2)], noise_seed=1)
    vp_a = compute_voiceprint(a, SAMPLE_RATE).voiceprint

    model_path = tmp_path / "model.json"
    clusterer = IncrementalVoiceClusterer(model_path)
    cluster_id, _dist, _is_new = clusterer.match_or_create_cluster(vp_a)
    clusterer.save()

    reloaded = IncrementalVoiceClusterer(model_path)
    centroid = reloaded.get_cluster_centroid(cluster_id)
    assert centroid is not None
    assert np.allclose(centroid, vp_a)


def test_predict_nearest_cluster_does_not_mutate_state(tmp_path: Path) -> None:
    a = _synthesize_voice(1.5, 120, [(600, 0.3), (1200, 0.2)], noise_seed=1)
    vp_a = compute_voiceprint(a, SAMPLE_RATE).voiceprint

    clusterer = IncrementalVoiceClusterer(tmp_path / "model.json")
    assert clusterer.predict_nearest_cluster(vp_a) == (None, 0.0)

    cluster_id, _dist, _is_new = clusterer.match_or_create_cluster(vp_a)
    centroid_before = clusterer.get_cluster_centroid(cluster_id)

    clusterer.predict_nearest_cluster(vp_a)
    clusterer.predict_nearest_cluster(vp_a)
    centroid_after = clusterer.get_cluster_centroid(cluster_id)

    assert np.array_equal(centroid_before, centroid_after)


def test_voice_segment_offsets_are_within_the_source_clip() -> None:
    segment = VoiceSegment(1.0, 2.5)
    assert segment.end_offset_seconds > segment.start_offset_seconds
