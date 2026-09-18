"""Initial SQLite schema for identity metadata and attendance outbox state."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1


class UnsupportedSchemaVersionError(RuntimeError):
    """Raised when a database was created by a newer schema."""


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS identity_metadata (
        worker_id NOT NULL PRIMARY KEY,
        is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        CHECK (typeof(worker_id) = 'integer' AND worker_id > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attendance_events (
        event_id NOT NULL PRIMARY KEY
            CHECK (
                typeof(event_id) = 'text'
                AND length(replace(replace(replace(trim(event_id), char(9), ''), char(10), ''), char(13), '')) > 0
            ),
        worker_id NOT NULL,
        event_type TEXT NOT NULL CHECK (event_type IN ('ENTRY', 'EXIT')),
        camera_id TEXT NOT NULL CHECK (length(trim(camera_id)) > 0),
        event_timestamp TEXT NOT NULL CHECK (
            typeof(event_timestamp) = 'text'
            AND length(trim(event_timestamp)) > 0
            AND event_timestamp GLOB '????-??-??T??:??:??*'
            AND (
                event_timestamp LIKE '%+__:__'
                OR event_timestamp LIKE '%-__:__'
                OR event_timestamp LIKE '%Z'
            )
        ),
        recognition_confidence REAL NOT NULL
            CHECK (recognition_confidence >= 0.0 AND recognition_confidence <= 1.0),
        track_id TEXT NOT NULL CHECK (length(trim(track_id)) > 0),
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY (worker_id) REFERENCES identity_metadata(worker_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outbox (
        event_id NOT NULL PRIMARY KEY
            CHECK (
                typeof(event_id) = 'text'
                AND length(replace(replace(replace(trim(event_id), char(9), ''), char(10), ''), char(13), '')) > 0
            ),
        payload TEXT NOT NULL,
        delivery_state TEXT NOT NULL DEFAULT 'PENDING'
            CHECK (delivery_state IN ('PENDING', 'DELIVERED', 'FAILED')),
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        last_attempt_at TEXT,
        delivered_at TEXT,
        FOREIGN KEY (event_id) REFERENCES attendance_events(event_id)
    )
    """,
)


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create or validate the current schema in one transaction."""
    current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current_version > SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"database schema version {current_version} is newer than supported version {SCHEMA_VERSION}"
        )

    if connection.in_transaction:
        raise RuntimeError("schema initialization requires an idle connection")

    connection.execute("BEGIN")
    try:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
