"""Small explicit sqlite3 connection and transaction boundary."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

from app.domain.models import AttendanceEventDraft
from app.storage.schema import initialize_schema


class Database:
    """An initialized local SQLite database with explicit transactions."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            database_path = str(self.path)
        else:
            database_path = ":memory:"

        self.connection = sqlite3.connect(database_path, timeout=timeout, isolation_level=None)
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            initialize_schema(self.connection)
            self.connection.execute("PRAGMA journal_mode = WAL")
        except Exception:
            self.connection.close()
            raise

    @property
    def journal_mode(self) -> str:
        """Return SQLite's effective journal mode for this connection."""
        return str(self.connection.execute("PRAGMA journal_mode").fetchone()[0]).upper()

    @property
    def foreign_keys_enabled(self) -> bool:
        return bool(self.connection.execute("PRAGMA foreign_keys").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit successful work or roll it back while propagating errors."""
        if self.connection.in_transaction:
            raise RuntimeError("nested transactions are not supported")
        self.connection.execute("BEGIN")
        try:
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def insert_attendance_event(self, event_id: str, event: AttendanceEventDraft) -> None:
        """Persist a validated domain event using its timezone-aware timestamp."""
        if not isinstance(event_id, str):
            raise TypeError("event_id must be a string")
        if not isinstance(event, AttendanceEventDraft):
            raise TypeError("event must be an AttendanceEventDraft")
        self.connection.execute(
            "INSERT INTO attendance_events "
            "(event_id, worker_id, event_type, camera_id, event_timestamp, recognition_confidence, track_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                event.employee_id,
                event.event_type.value,
                event.camera_id,
                event.timestamp.isoformat(),
                event.recognition_confidence,
                event.track_id,
            ),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
