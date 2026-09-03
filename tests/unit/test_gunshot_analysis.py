from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import gaggle.detection.gunshot_analysis as gunshot_analysis
from gaggle.detection.gunshot_analysis import (
    GUNSHOT_LIKE_CLASS_NAMES,
    GunshotDetectionError,
    analyze_gunshot_events,
    ensure_gunshot_model,
    sherpa_onnx_available,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def test_gunshot_like_class_names_are_the_expected_curated_set() -> None:
    assert GUNSHOT_LIKE_CLASS_NAMES == {
        "Gunshot, gunfire",
        "Machine gun",
        "Artillery fire",
        "Cap gun",
    }
    # Deliberately excluded -- see the module docstring on why conflating
    # these with actual gunfire would be misleading, not just imprecise.
    assert "Fireworks" not in GUNSHOT_LIKE_CLASS_NAMES
    assert "Firecracker" not in GUNSHOT_LIKE_CLASS_NAMES


def test_sherpa_onnx_available_returns_false_instead_of_raising_on_a_non_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors `insightface_available()`'s own broadened except-Exception
    test -- see that test's docstring for the real Windows DLL-conflict
    failure mode this guards against for every optional ONNX-adjacent
    dependency in this project, sherpa-onnx included."""

    import builtins

    real_import = builtins.__import__

    def _broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sherpa_onnx":
            raise OSError("a simulated broken native-extension load")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    assert sherpa_onnx_available() is False


# -- ensure_gunshot_model: the test sandbox has no network access (see
# AGENTS.md) -- every test here mocks the download rather than hitting
# the real k2-fsa/sherpa-onnx GitHub release.


def _fake_archive_bytes(
    model_bytes: bytes = b"fake-model", labels_bytes: bytes = b"fake-labels"
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        for member_name, content in (
            (gunshot_analysis._MODEL_MEMBER, model_bytes),
            (gunshot_analysis._LABELS_MEMBER, labels_bytes),
        ):
            info = tarfile.TarInfo(name=member_name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _mock_download(monkeypatch: pytest.MonkeyPatch, archive_bytes: bytes) -> None:
    import hashlib

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return archive_bytes

    monkeypatch.setattr(gunshot_analysis.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(
        gunshot_analysis, "_ARCHIVE_SHA256", hashlib.sha256(archive_bytes).hexdigest()
    )


def test_ensure_gunshot_model_downloads_extracts_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _fake_archive_bytes(
        labels_bytes=b"index,mid,display_name\n427,/m/032s66,Gunshot"
    )
    _mock_download(monkeypatch, archive_bytes)

    model_path, labels_path = ensure_gunshot_model(cache_dir=tmp_path)
    assert model_path.read_bytes() == b"fake-model"
    assert "Gunshot" in labels_path.read_text(encoding="utf-8")

    # Re-calling must not re-download -- mtime stays unchanged, and it
    # would work anyway even if the mocked urlopen were removed.
    first_mtime = model_path.stat().st_mtime
    monkeypatch.undo()
    model_path_again, _ = ensure_gunshot_model(cache_dir=tmp_path)
    assert model_path_again == model_path
    assert model_path.stat().st_mtime == first_mtime


def test_ensure_gunshot_model_rejects_a_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _fake_archive_bytes()

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return archive_bytes

    monkeypatch.setattr(gunshot_analysis.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(gunshot_analysis, "_ARCHIVE_SHA256", "0" * 64)

    with pytest.raises(GunshotDetectionError, match="hash mismatch"):
        ensure_gunshot_model(cache_dir=tmp_path)


def test_ensure_gunshot_model_wraps_a_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("network is unreachable")

    monkeypatch.setattr(gunshot_analysis.urllib.request, "urlopen", _raise)

    with pytest.raises(GunshotDetectionError, match="failed to download"):
        ensure_gunshot_model(cache_dir=tmp_path)


def test_ensure_gunshot_model_rejects_a_missing_archive_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        info = tarfile.TarInfo(name="unrelated/file.txt")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"hi!"))
    _mock_download(monkeypatch, buffer.getvalue())

    with pytest.raises(GunshotDetectionError, match="missing expected member"):
        ensure_gunshot_model(cache_dir=tmp_path)


# -- analyze_gunshot_events: a fake tagger double exercises the real
# windowing/thresholding/event-construction logic deterministically,
# without needing the real (network-fetched) model at all.


class _FakeStream:
    def __init__(self, on_compute: list[tuple[str, float]]) -> None:
        self._results = on_compute

    def accept_waveform(self, sample_rate: int, waveform: object) -> None:
        del sample_rate, waveform


class _FakeTagger:
    """Always reports the same fixed set of (name, prob) entries for
    every window classified -- enough to test filtering/windowing
    without depending on real classifier output."""

    def __init__(self, entries: list[tuple[str, float]]) -> None:
        self._entries = entries
        self.windows_classified = 0

    def create_stream(self) -> _FakeStream:
        return _FakeStream(self._entries)

    def compute(self, stream: _FakeStream) -> list[SimpleNamespace]:
        self.windows_classified += 1
        return [SimpleNamespace(name=name, prob=prob) for name, prob in self._entries]


def _make_tone_clip(path: Path, duration_seconds: float = 5.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=16000:duration={duration_seconds}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available")
def test_analyze_gunshot_events_filters_by_class_name_and_confidence(tmp_path: Path) -> None:
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, duration_seconds=3.0)

    tagger = _FakeTagger([("Speech", 0.99), ("Gunshot, gunfire", 0.42)])
    result = analyze_gunshot_events(clip, tagger, confidence_threshold=0.5)

    assert result.has_audio is True
    assert result.events == []  # "Speech" isn't gunshot-like; "Gunshot..." is below threshold


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available")
def test_analyze_gunshot_events_reports_matches_above_threshold(tmp_path: Path) -> None:
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, duration_seconds=3.0)

    tagger = _FakeTagger([("Gunshot, gunfire", 0.91), ("Fireworks", 0.85)])
    result = analyze_gunshot_events(clip, tagger, confidence_threshold=0.5, window_seconds=1.0)

    assert result.has_audio is True
    assert result.events
    assert all(event.class_name == "Gunshot, gunfire" for event in result.events)
    assert all(event.confidence == pytest.approx(0.91) for event in result.events)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available")
def test_analyze_gunshot_events_windows_the_clip_at_the_configured_stride(tmp_path: Path) -> None:
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, duration_seconds=4.0)

    tagger = _FakeTagger([])
    analyze_gunshot_events(clip, tagger, window_seconds=1.0, hop_seconds=1.0)

    # ~4 one-second, non-overlapping windows over a 4s clip.
    assert 3 <= tagger.windows_classified <= 5


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available")
def test_analyze_gunshot_events_reports_no_audio_for_a_video_only_source(tmp_path: Path) -> None:
    video_only = tmp_path / "video_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x64:rate=5:duration=1",
            "-an",
            str(video_only),
        ],
        check=True,
    )

    result = analyze_gunshot_events(video_only, _FakeTagger([]))
    assert result.has_audio is False
    assert result.events == []


def test_analyze_gunshot_events_rejects_nonpositive_window_or_hop(tmp_path: Path) -> None:
    fake_path = tmp_path / "does-not-need-to-exist.wav"
    tagger = _FakeTagger([])
    with pytest.raises(ValueError):
        analyze_gunshot_events(fake_path, tagger, window_seconds=0)
    with pytest.raises(ValueError):
        analyze_gunshot_events(fake_path, tagger, hop_seconds=-1)
