from __future__ import annotations

from gaggle.storage.database import TimelineDatabase, TimelineQuery


class TimelineService:
    """Read-only querying over the SQLite index.

    All filtering here is delegated to ``TimelineQuery`` /
    ``TimelineDatabase.query_events`` so there is exactly one place that
    knows how to translate a query into SQL; this service is just a typed,
    documented façade for callers (CLI, review UI).
    """

    def __init__(self, database: TimelineDatabase) -> None:
        self.database = database

    def list_events(self, query: TimelineQuery | None = None) -> list[dict[str, object]]:
        rows = self.database.query_events(query or TimelineQuery())
        return [
            {
                "event_id": row.event_id,
                "event_path": row.event_path,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "severity": row.severity,
                "confidence": row.confidence,
                "preservation_state": row.preservation_state,
                "review_decision": row.review_decision,
                "camera_count": row.camera_count,
                "cameras": row.cameras_csv.split(",") if row.cameras_csv else [],
                "revision": row.revision,
            }
            for row in rows
        ]
