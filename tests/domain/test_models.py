from datetime import datetime, timezone

import pytest

from app.domain.models import (
    AttendanceEventDraft,
    CameraFrame,
    CameraRole,
    FaceObservation,
    Track,
)


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_track() -> Track:
    return Track("track-1", "ENTRY_01", TIMESTAMP, (1, 2, 30, 40), 0.9)


def test_valid_camera_frame_construction() -> None:
    frame = CameraFrame("ENTRY_01", CameraRole.ENTRY, TIMESTAMP)

    assert frame.camera_id == "ENTRY_01"
    assert frame.camera_role is CameraRole.ENTRY


def test_invalid_camera_id_rejected() -> None:
    with pytest.raises(ValueError):
        CameraFrame("  ", CameraRole.ENTRY, TIMESTAMP)


def test_timezone_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        CameraFrame("ENTRY_01", CameraRole.ENTRY, datetime(2026, 1, 1))


def test_valid_track_construction() -> None:
    track = make_track()

    assert track.track_id == "track-1"
    assert track.bbox == (1.0, 2.0, 30.0, 40.0)


def test_invalid_track_id_rejected() -> None:
    with pytest.raises(ValueError):
        Track("", "ENTRY_01", TIMESTAMP)


def test_invalid_bbox_rejected() -> None:
    with pytest.raises(ValueError):
        Track("track-1", "ENTRY_01", TIMESTAMP, (10, 2, 10, 40))


def test_valid_face_observation_construction_and_track_association() -> None:
    track = make_track()
    observation = FaceObservation(track, TIMESTAMP, (3, 4, 20, 25), 0.8)

    assert observation.track is track
    assert observation.quality == 0.8


def test_invalid_face_quality_rejected() -> None:
    with pytest.raises(ValueError):
        FaceObservation(make_track(), TIMESTAMP, quality=1.1)


@pytest.mark.parametrize("event_type", [CameraRole.ENTRY, CameraRole.EXIT])
def test_valid_attendance_event_draft(event_type: CameraRole) -> None:
    event = AttendanceEventDraft(42, "ENTRY_01", event_type, TIMESTAMP, 0.97, "track-1")

    assert event.employee_id == 42
    assert event.event_type is event_type


def test_invalid_event_type_rejected() -> None:
    with pytest.raises(TypeError):
        AttendanceEventDraft(42, "ENTRY_01", "ENTRY", TIMESTAMP, 0.97, "track-1")


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_out_of_range_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        AttendanceEventDraft(42, "ENTRY_01", CameraRole.ENTRY, TIMESTAMP, confidence, "track-1")


def test_numeric_hisob_worker_id_accepted() -> None:
    event = AttendanceEventDraft(123, "EXIT_01", CameraRole.EXIT, TIMESTAMP, 0.5, "track-1")

    assert event.employee_id == 123


@pytest.mark.parametrize("worker_id", [0, -1, "123", True])
def test_invalid_worker_id_rejected(worker_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AttendanceEventDraft(worker_id, "ENTRY_01", CameraRole.ENTRY, TIMESTAMP, 0.5, "track-1")


def test_domain_objects_are_immutable() -> None:
    frame = CameraFrame("ENTRY_01", CameraRole.ENTRY, TIMESTAMP)
    track = make_track()
    observation = FaceObservation(track, TIMESTAMP)
    event = AttendanceEventDraft(42, "ENTRY_01", CameraRole.ENTRY, TIMESTAMP, 0.9, "track-1")

    with pytest.raises(AttributeError):
        frame.camera_id = "EXIT_01"
    with pytest.raises(AttributeError):
        track.track_id = "track-2"
    with pytest.raises(AttributeError):
        observation.track = track
    with pytest.raises(AttributeError):
        event.employee_id = 43
