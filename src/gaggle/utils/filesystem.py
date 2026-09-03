from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def copy_file_preserve_metadata(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def set_read_only(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode & ~stat.S_IWRITE)


def delete_even_if_read_only(path: Path) -> None:
    """Clear a read-only attribute (if any) immediately before deleting.

    Windows' ``os.unlink()`` raises ``PermissionError: [WinError 5]
    Access is denied`` on a read-only file -- unlike POSIX, where only the
    containing directory's write permission governs whether a file can be
    unlinked, regardless of the file's own mode. Every file this project
    deletes under normal operation (a confirmed-deletion original, a
    dedup-superseded ingest source) may have been marked read-only by
    ``set_read_only`` above (the `storage.set_read_only: true` default) or
    by the external source itself -- a real ``PermissionError`` was hit
    purging a real workspace, not a hypothetical. Best-effort: if the
    chmod itself fails (e.g. no permission at all), the subsequent
    ``unlink()`` still raises its own clear error, no worse than before.
    """

    try:
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    except OSError:
        pass
    path.unlink()


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.write("\n")


def list_files_sorted(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file())


def safe_relpath(path: Path, root: Path) -> str:
    return os.fspath(path.resolve().relative_to(root.resolve()))
