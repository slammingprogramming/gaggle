from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gaggle.core.config import RuntimeConfig, load_config


def test_runtime_config_defaults() -> None:
    config = RuntimeConfig()
    assert config.pipeline.window_duration_seconds == 10
    assert config.storage.hash_algorithm == "sha256"
    assert config.sync.session_gap_seconds == 120.0
    assert config.detection.use_fixture_signals_when_available is True
    assert config.pipeline.max_event_duration_seconds == 120.0
    assert config.detection.audio_extraction_timeout_seconds == 300.0
    assert config.enrichment.voice.audio_extraction_timeout_seconds == 300.0


def test_max_event_duration_seconds_can_be_disabled() -> None:
    config = RuntimeConfig(pipeline={"max_event_duration_seconds": None})
    assert config.pipeline.max_event_duration_seconds is None


def test_runtime_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"not_a_real_field": True})


def test_load_config_from_profile_yaml(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
active_profile: strict
profiles:
  strict:
    detection:
      motion_threshold: 0.5
    sync:
      session_gap_seconds: 30
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.active_profile == "strict"
    assert config.detection.motion_threshold == 0.5
    assert config.sync.session_gap_seconds == 30.0
    # unspecified sections still get sensible defaults
    assert config.scoring.high_threshold == 0.80


def test_environment_overrides_apply_on_top_of_file_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("active_profile: default\n", encoding="utf-8")
    monkeypatch.setenv("DASHCAM_SENTINEL__DETECTION__MOTION_THRESHOLD", "0.42")
    config = load_config(config_path)
    assert config.detection.motion_threshold == 0.42
    monkeypatch.delenv("DASHCAM_SENTINEL__DETECTION__MOTION_THRESHOLD", raising=False)


def test_environment_overrides_do_not_leak_between_calls() -> None:
    assert "DASHCAM_SENTINEL__DETECTION__MOTION_THRESHOLD" not in os.environ
    config = load_config(None)
    assert config.detection.motion_threshold == 0.20


def test_security_outdoor_example_profile_loads_and_validates() -> None:
    config = load_config(Path("examples/config/security-outdoor.yaml"))
    assert config.active_profile == "security-outdoor"
    assert config.detection.motion_threshold > RuntimeConfig().detection.motion_threshold
    assert (
        config.detection.optical_flow.roi_divergence_delta_threshold
        > RuntimeConfig().detection.optical_flow.roi_divergence_delta_threshold
    )
    assert config.sync.session_gap_seconds > RuntimeConfig().sync.session_gap_seconds


def test_security_indoor_example_profile_loads_and_validates() -> None:
    config = load_config(Path("examples/config/security-indoor.yaml"))
    assert config.active_profile == "security-indoor"
    assert config.detection.motion_threshold < RuntimeConfig().detection.motion_threshold


def test_default_api_key_env_var_does_not_trigger_the_secret_warning(capsys) -> None:
    load_config(None)
    assert "api_key_env_var_looks_like_a_secret" not in capsys.readouterr().out


def test_loading_a_pasted_api_key_as_env_var_name_warns(tmp_path, capsys) -> None:
    """Regression test for a real incident: a user pasted their actual
    OpenRouter API key (`sk-or-v1-...`) directly into
    `enrichment.cloud.api_key_env_var`, which is supposed to hold an
    environment variable *name*, not a secret. The mistake silently
    failed (`os.environ.get()` found nothing) and printed the real key to
    the terminal on every enrich run via the `env_var` log field. This
    must be caught and flagged at config load time."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
active_profile: default
profiles:
  default:
    enrichment:
      cloud:
        enabled: true
        api_key_env_var: sk-or-v1-e2a9c1bd5ea5f3f8fc640b29265bef03aee277fa
""",
        encoding="utf-8",
    )

    load_config(config_path)

    output = capsys.readouterr().out
    assert "api_key_env_var_looks_like_a_secret_not_a_variable_name" in output


def _all_key_paths(payload: dict, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Every leaf key path in a nested dict, e.g. `{"a": {"b": 1}}` ->
    `{("a", "b")}`. Used to compare *structure* (which fields exist),
    never values.

    A dict-valued field that defaults to `{}` (e.g.
    `SyncConfig.manual_offset_overrides`) is treated as a leaf itself,
    not recursed into -- an empty dict has no sub-keys to recurse into,
    so without this the field itself would never appear in the returned
    path set at all, a real blind spot found while adding
    `manual_offset_overrides`: the drift check below would have silently
    never noticed *that field itself* going undocumented, only sub-keys
    of a non-empty default."""

    paths: set[tuple[str, ...]] = set()
    for key, value in payload.items():
        path = (*prefix, key)
        if isinstance(value, dict) and value:
            paths |= _all_key_paths(value, path)
        else:
            paths.add(path)
    return paths


def test_example_config_documents_every_runtime_config_field() -> None:
    """Regression test for real, observed drift: many config fields added
    to core/config.py over time (vehicle_appearance, signing, encounters,
    the face/plate detector+device fields, dedupe_on_ingest,
    max_event_duration_seconds, and others) were never added to
    examples/config.yaml, so a user reading it had no way to discover
    they existed short of reading the source -- they just silently took
    their defaults. This checks *structure*, not values (an example may
    deliberately show a non-default value for something) -- only that
    every field the code knows about is explicitly present in the
    example file, not silently absent from it."""

    default_paths = _all_key_paths(RuntimeConfig().model_dump())
    default_paths.discard(("active_profile",))  # lives outside profiles.default in the file

    raw = yaml.safe_load(Path("examples/config.yaml").read_text(encoding="utf-8"))
    documented_paths = _all_key_paths(raw["profiles"]["default"])

    missing = default_paths - documented_paths
    assert not missing, (
        f"fields present in RuntimeConfig but missing from examples/config.yaml: {sorted(missing)}"
    )


def test_an_unconventional_but_valid_looking_env_var_name_does_not_warn(tmp_path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
active_profile: default
profiles:
  default:
    enrichment:
      cloud:
        api_key_env_var: MY_CUSTOM_KEY_NAME
""",
        encoding="utf-8",
    )

    load_config(config_path)

    assert "api_key_env_var_looks_like_a_secret" not in capsys.readouterr().out
