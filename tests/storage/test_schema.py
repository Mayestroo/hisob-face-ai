from datetime import datetime, timezone
import sqlite3

import pytest

from app.domain.models import AttendanceEventDraft, CameraRole
from app.storage.database import Database
from app.storage.schema import SCHEMA_VERSION, UnsupportedSchemaVersionError


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).isoformat()


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "face-ai.sqlite3")
    yield db
    db.close()


def insert_identity(db: Database, worker_id: int = 42) -> None:
    db.connection.execute("INSERT INTO identity_metadata (worker_id) VALUES (?)", (worker_id,))


def insert_event(db: Database, event_id: str = "event-1", worker_id: int = 42, event_type: str = "ENTRY") -> None:
    db.connection.execute(
        "INSERT INTO attendance_events "
        "(event_id, worker_id, event_type, camera_id, event_timestamp, recognition_confidence, track_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (event_id, worker_id, event_type, "CAMERA_A", TIMESTAMP, 0.9, "track-1"),
    )


def make_domain_event(event_type: CameraRole = CameraRole.ENTRY) -> AttendanceEventDraft:
    return AttendanceEventDraft(42, "CAMERA_A", event_type, datetime.fromisoformat(TIMESTAMP), 0.9, "track-1")


def insert_outbox(db: Database, event_id: str = "event-1") -> None:
    db.connection.execute(
        "INSERT INTO outbox (event_id, payload) VALUES (?, ?)",
        (event_id, '{"event_id":"event-1"}'),
    )


def test_new_database_initializes_with_current_version_and_tables(database: Database) -> None:
    assert database.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    tables = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"identity_metadata", "attendance_events", "outbox"} <= tables


def test_initialization_is_idempotent_and_reopening_works(tmp_path) -> None:
    path = tmp_path / "face-ai.sqlite3"
    first = Database(path)
    first.close()
    second = Database(path)
    assert second.connection.execute("SELECT COUNT(*) FROM identity_metadata").fetchone()[0] == 0
    second.close()


def test_file_database_uses_wal_and_enforces_foreign_keys(database: Database) -> None:
    assert database.journal_mode == "WAL"
    assert database.foreign_keys_enabled


def test_identity_metadata_accepts_numeric_worker_id(database: Database) -> None:
    insert_identity(database, 123)
    assert database.connection.execute("SELECT worker_id FROM identity_metadata").fetchone()[0] == 123


@pytest.mark.parametrize("worker_id", [0, -1, "123"])
def test_invalid_identity_data_rejected(database: Database, worker_id) -> None:
    with pytest.raises((sqlite3.IntegrityError, sqlite3.ProgrammingError)):
        database.connection.execute("INSERT INTO identity_metadata (worker_id) VALUES (?)", (worker_id,))


def test_stable_event_id_is_unique(database: Database) -> None:
    insert_identity(database)
    insert_event(database)
    with pytest.raises(sqlite3.IntegrityError):
        insert_event(database)


@pytest.mark.parametrize("event_id", [None, "", "   ", "\t"])
def test_invalid_attendance_event_id_rejected(database: Database, event_id) -> None:
    insert_identity(database)
    with pytest.raises((sqlite3.IntegrityError, sqlite3.ProgrammingError)):
        insert_event(database, event_id=event_id)


def test_valid_textual_event_id_persists(database: Database) -> None:
    insert_identity(database)
    insert_event(database, event_id="stable-event-1")
    assert database.connection.execute("SELECT event_id FROM attendance_events").fetchone()[0] == "stable-event-1"


@pytest.mark.parametrize("event_id", [None, "", "   ", "\t"])
def test_invalid_outbox_event_id_rejected(database: Database, event_id) -> None:
    # Isolate the outbox key constraint from the parent-event foreign key.
    database.connection.execute("PRAGMA foreign_keys = OFF")
    with pytest.raises((sqlite3.IntegrityError, sqlite3.ProgrammingError)):
        insert_outbox(database, event_id=event_id)


@pytest.mark.parametrize("event_type", ["ENTRY", "EXIT"])
def test_valid_event_types_persist(database: Database, event_type: str) -> None:
    insert_identity(database)
    insert_event(database, event_type=event_type)
    assert database.connection.execute("SELECT event_type FROM attendance_events").fetchone()[0] == event_type


def test_invalid_event_type_rejected(database: Database) -> None:
    insert_identity(database)
    with pytest.raises(sqlite3.IntegrityError):
        insert_event(database, event_type="UNKNOWN")


def test_timezone_aware_timestamp_representation_survives(database: Database) -> None:
    insert_identity(database)
    insert_event(database)
    persisted = database.connection.execute("SELECT event_timestamp FROM attendance_events").fetchone()[0]
    assert persisted == TIMESTAMP
    assert datetime.fromisoformat(persisted).tzinfo is not None


def test_typed_event_boundary_persists_timezone_aware_timestamp(database: Database) -> None:
    insert_identity(database)
    database.insert_attendance_event("typed-event", make_domain_event())
    persisted = database.connection.execute(
        "SELECT event_timestamp FROM attendance_events WHERE event_id = 'typed-event'"
    ).fetchone()[0]
    assert persisted == TIMESTAMP


@pytest.mark.parametrize("timestamp", ["not-a-timestamp", "2026-09-18T08:00:00"])
def test_invalid_attendance_timestamp_rejected_by_sql_constraint(database: Database, timestamp: str) -> None:
    insert_identity(database)
    with pytest.raises(sqlite3.IntegrityError):
        database.connection.execute(
            "INSERT INTO attendance_events "
            "(event_id, worker_id, event_type, camera_id, event_timestamp, recognition_confidence, track_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("invalid-time", 42, "ENTRY", "CAMERA_A", timestamp, 0.9, "track-1"),
        )


@pytest.mark.parametrize("timestamp", ["2026-09-18T08:00:00+05:00", "2026-09-18T03:00:00+00:00"])
def test_timezone_aware_attendance_timestamp_is_accepted(database: Database, timestamp: str) -> None:
    insert_identity(database)
    database.connection.execute(
        "INSERT INTO attendance_events "
        "(event_id, worker_id, event_type, camera_id, event_timestamp, recognition_confidence, track_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("timezone-aware-event", 42, "ENTRY", "CAMERA_A", timestamp, 0.9, "track-1"),
    )
    assert database.connection.execute(
        "SELECT event_timestamp FROM attendance_events WHERE event_id = 'timezone-aware-event'"
    ).fetchone()[0] == timestamp


def test_event_outbox_relationship_integrity(database: Database) -> None:
    insert_identity(database)
    insert_event(database)
    insert_outbox(database)
    assert database.connection.execute("SELECT event_id FROM outbox").fetchone()[0] == "event-1"


def test_foreign_key_invalid_outbox_insert_rejected(database: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert_outbox(database)


def test_event_with_nonexistent_identity_worker_rejected(database: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert_event(database, worker_id=999)


def test_transaction_commits_on_success(database: Database) -> None:
    with database.transaction() as connection:
        connection.execute("INSERT INTO identity_metadata (worker_id) VALUES (42)")
    assert database.connection.execute("SELECT COUNT(*) FROM identity_metadata").fetchone()[0] == 1


def test_transaction_rolls_back_on_exception(database: Database) -> None:
    with pytest.raises(RuntimeError):
        with database.transaction() as connection:
            connection.execute("INSERT INTO identity_metadata (worker_id) VALUES (42)")
            raise RuntimeError("forced failure")
    assert database.connection.execute("SELECT COUNT(*) FROM identity_metadata").fetchone()[0] == 0


def test_event_and_outbox_roll_back_together_on_forced_failure(database: Database) -> None:
    with pytest.raises(RuntimeError):
        with database.transaction() as connection:
            connection.execute("INSERT INTO identity_metadata (worker_id) VALUES (42)")
            insert_event(database)
            insert_outbox(database)
            raise RuntimeError("forced failure")
    assert database.connection.execute("SELECT COUNT(*) FROM attendance_events").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0


def test_foreign_key_failure_rolls_back_earlier_transaction_writes(database: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            connection.execute("INSERT INTO identity_metadata (worker_id) VALUES (77)")
            connection.execute("INSERT INTO outbox (event_id, payload) VALUES (?, ?)", ("missing-event", "{}"))
    assert database.connection.execute("SELECT COUNT(*) FROM identity_metadata WHERE worker_id = 77").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0


def test_newer_schema_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(UnsupportedSchemaVersionError):
        Database(path)
    verification = sqlite3.connect(path)
    assert verification.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    verification.close()


def test_schema_requires_no_raw_biometric_material(database: Database) -> None:
    for table in ("identity_metadata", "attendance_events", "outbox"):
        columns = {row[1] for row in database.connection.execute(f"PRAGMA table_info({table})")}
        assert not columns & {"image", "image_blob", "embedding", "template", "biometric_template"}
