from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Stable entry-point group names third-party packages register plugins
#: under (see ``docs/plugin-authoring.md``). Kept as constants so pipeline
#: code and plugin authors reference the same strings.
DETECTOR_PLUGIN_GROUP = "gaggle.plugins.detectors"
INFERENCE_RULE_PLUGIN_GROUP = "gaggle.plugins.inference_rules"
EXPORTER_PLUGIN_GROUP = "gaggle.plugins.exporters"
REVIEW_EXTENSION_PLUGIN_GROUP = "gaggle.plugins.review_extensions"


def load_plugins(group: str) -> list[Any]:
    """Load and instantiate every plugin registered under ``group``.

    A plugin that fails to load or instantiate is logged and skipped rather
    than aborting the whole pipeline (plugin isolation) -- one broken
    third-party plugin must never prevent the built-in, dependency-free
    pipeline from running.
    """

    plugins: list[Any] = []
    for entry in entry_points().select(group=group):
        try:
            loaded = entry.load()
            plugin = loaded() if isinstance(loaded, type) else loaded
        except Exception as error:
            LOGGER.error("plugin_load_failed", group=group, name=entry.name, reason=str(error))
            continue
        plugins.append(plugin)
    return plugins
