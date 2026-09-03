from __future__ import annotations

from pathlib import Path

import pytest

from gaggle.core import cli_config


@pytest.fixture(autouse=True)
def _redirect_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_config, "config_path", lambda: tmp_path / "cli_config.json")


def test_get_default_actor_returns_none_when_unset() -> None:
    assert cli_config.get_default_actor() is None


def test_set_then_get_default_actor_roundtrips() -> None:
    cli_config.set_default_actor("ash")
    assert cli_config.get_default_actor() == "ash"


def test_set_default_actor_overwrites_previous_value() -> None:
    cli_config.set_default_actor("ash")
    cli_config.set_default_actor("morgan")
    assert cli_config.get_default_actor() == "morgan"


def test_get_default_actor_tolerates_a_corrupt_config_file(tmp_path: Path) -> None:
    path = tmp_path / "cli_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json{{{", encoding="utf-8")
    assert cli_config.get_default_actor() is None


def test_resolve_actor_prefers_the_explicit_argument() -> None:
    cli_config.set_default_actor("configured")
    assert cli_config.resolve_actor("explicit") == "explicit"


def test_resolve_actor_falls_back_to_the_configured_default() -> None:
    cli_config.set_default_actor("configured")
    assert cli_config.resolve_actor(None) == "configured"


def test_resolve_actor_raises_when_neither_is_available() -> None:
    with pytest.raises(ValueError, match="no --actor given"):
        cli_config.resolve_actor(None)
