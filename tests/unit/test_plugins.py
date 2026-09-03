from __future__ import annotations

from gaggle.plugins.registry import load_plugins


def test_load_plugins_returns_empty_list_when_none_registered() -> None:
    assert load_plugins("gaggle.plugins.detectors_test_nonexistent_group") == []


def test_load_plugins_isolates_a_broken_plugin(monkeypatch) -> None:
    class WorkingPlugin:
        name = "working"
        version = "1.0.0"

    class _FakeEntry:
        def __init__(self, name: str, loader) -> None:
            self.name = name
            self._loader = loader

        def load(self):
            return self._loader()

    def broken_loader() -> None:
        raise RuntimeError("boom")

    working_entry = _FakeEntry("working", lambda: WorkingPlugin)
    broken_entry = _FakeEntry("broken", broken_loader)

    class FakeEntryPoints:
        def select(self, group: str) -> list[_FakeEntry]:
            return [broken_entry, working_entry]

    monkeypatch.setattr("gaggle.plugins.registry.entry_points", lambda: FakeEntryPoints())

    plugins = load_plugins("test.group")
    assert len(plugins) == 1
    assert plugins[0].name == "working"
