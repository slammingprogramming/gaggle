from __future__ import annotations

from uuid import UUID, uuid4


def new_uuid() -> UUID:
    return uuid4()
