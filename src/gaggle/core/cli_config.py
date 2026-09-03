"""Per-machine CLI convenience configuration.

Deliberately **not** part of the shared workspace `config.yaml` -- who is
running the CLI is a personal preference tied to this machine/user, not
workspace state other users or machines sharing the same workspace should
inherit. Stored at `platformdirs.user_config_dir("gaggle") /
"cli_config.json"`, mirroring `core/models.py`'s
`platformdirs.user_cache_dir("gaggle")` precedent for per-machine (not
per-workspace) state.

Added so a user doesn't have to retype `--actor <name>` on every single
attributed CLI command -- `gaggle config set-actor <name>` once, then every
command that takes `--actor` falls back to this when the flag is omitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import platformdirs


def config_path() -> Path:
    return Path(platformdirs.user_config_dir("gaggle")) / "cli_config.json"


def _load() -> dict[str, object]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _save(data: dict[str, object]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_default_actor() -> str | None:
    value = _load().get("default_actor")
    return value if isinstance(value, str) and value else None


def set_default_actor(name: str) -> None:
    data = _load()
    data["default_actor"] = name
    _save(data)


def resolve_actor(actor: str | None) -> str:
    """Every CLI command that writes an attributed record calls this with
    its raw `--actor` value instead of using that value directly -- falls
    back to the configured default actor if `--actor` wasn't passed, and
    raises a clear, actionable error (never silently attributes to an
    empty string) if neither is set."""

    if actor:
        return actor
    default = get_default_actor()
    if default:
        return default
    raise ValueError(
        "no --actor given and no default actor configured; "
        "run 'gaggle config set-actor <name>' once, or pass --actor"
    )
