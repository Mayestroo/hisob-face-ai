from datetime import datetime, timedelta, timezone

import pytest

from app.cameras.video_file import DecodedVideoFrame
from app.domain.models import CameraFrame, CameraRole
from app.face import PersonRoiObservation, extract_observed_person_rois
from app.face.roi import PersonRoiExtractionError
from app.tracking import TrackState, TrackedPerson, TrackingUpdateResult


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_frame(camera_id: str = "CAMERA_A", index: int = 0, width: int = 5, height: int = 4) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        frame=CameraFrame(camera_id, CameraRole.ENTRY, START + timedelta(seconds=index)),
        pixels=bytes(range(width * height)),
        width=width,
        height=height,
        frame_index=index,
    )


def track(
    bbox: tuple[float, float, float, float] = (1, 1, 4, 3),
    track_id: int = 1,
    frame: DecodedVideoFrame | None = None,
    state: TrackState = TrackState.ACTIVE,
    confidence: float = 0.73,
) -> TrackedPerson:
    frame = frame or make_frame()
    return TrackedPerson(
        frame.frame.camera_id, track_id, bbox, confidence, state,
        frame.frame_index, frame.frame.timestamp, frame.frame_index, 1, 0,
    )


def result(*observed: TrackedPerson, lost: tuple[TrackedPerson, ...] = (), finalized: tuple[TrackedPerson, ...] = ()) -> TrackingUpdateResult:
    return TrackingUpdateResult(tuple(observed), lost, finalized)


def make_observation(
    *,
    roi_bbox: tuple[object, ...] = (1, 1, 3, 3),
    roi_width: object = 2,
    roi_height: object = 2,
    roi_pixels: bytes = b"\x00\x01\x02\x03",
) -> PersonRoiObservation:
    return PersonRoiObservation(
        camera_id="CAMERA_A",
        track_id=1,
        frame_index=0,
        timestamp=START,
        person_bbox=(1.0, 1.0, 3.0, 3.0),
        person_confidence=0.9,
        roi_bbox=roi_bbox,
        roi_pixels=roi_pixels,
        roi_width=roi_width,
        roi_height=roi_height,
        pixel_format="gray",
    )


def test_one_observed_track_produces_exact_person_roi() -> None:
    observation = extract_observed_person_rois(make_frame(), result(track()))[0]

    assert isinstance(observation, PersonRoiObservation)
    assert observation.roi_bbox == (1, 1, 4, 3)
    assert observation.roi_width == 3
    assert observation.roi_height == 2
    assert observation.roi_pixels == bytes([6, 7, 8, 11, 12, 13])


def test_empty_observed_collection_produces_empty_result() -> None:
    assert extract_observed_person_rois(make_frame(), result()) == ()


def test_lost_and_finalized_tracks_are_not_used() -> None:
    frame = make_frame(index=1)
    observed = track(frame=frame)
    stale = track(track_id=2, frame=make_frame(index=0), state=TrackState.LOST)
    finalized = track(track_id=3, frame=make_frame(index=0), state=TrackState.FINALIZED)

    rois = extract_observed_person_rois(frame, result(observed, lost=(stale,), finalized=(finalized,)))

    assert [item.track_id for item in rois] == [1]


@pytest.mark.parametrize("state", [TrackState.NEW, TrackState.ACTIVE])
def test_current_new_and_active_tracks_are_accepted(state: TrackState) -> None:
    assert extract_observed_person_rois(make_frame(), result(track(state=state)))


def test_recovered_track_in_observed_is_accepted() -> None:
    frame = make_frame(index=12)
    recovered = track(frame=frame, track_id=4, state=TrackState.ACTIVE)

    assert extract_observed_person_rois(frame, result(recovered))[0].track_id == 4


@pytest.mark.parametrize(
    ("field", "track_frame", "frame"),
    [
        ("camera_id", make_frame(camera_id="CAMERA_B"), make_frame()),
        ("frame_index", make_frame(index=1), make_frame()),
        ("timestamp", make_frame(), make_frame()),
    ],
)
def test_current_frame_identity_mismatches_are_rejected(field: str, track_frame: DecodedVideoFrame, frame: DecodedVideoFrame) -> None:
    if field == "timestamp":
        track_frame = DecodedVideoFrame(
            CameraFrame(frame.frame.camera_id, frame.frame.camera_role, START + timedelta(seconds=1)),
            frame.pixels, frame.width, frame.height, frame.frame_index,
        )
    with pytest.raises(PersonRoiExtractionError, match=field):
        extract_observed_person_rois(frame, result(track(frame=track_frame)))


@pytest.mark.parametrize(
    ("frame_timestamp", "track_timestamp"),
    [
        (datetime(2026, 9, 18, 10, tzinfo=timezone(timedelta(hours=5))), datetime(2026, 9, 18, 5, tzinfo=timezone.utc)),
        (datetime(2026, 9, 18, 10, tzinfo=timezone(timedelta(hours=5))), datetime(2026, 9, 18, 10, tzinfo=timezone.utc)),
    ],
)
def test_timestamp_representation_mismatch_is_rejected(
    frame_timestamp: datetime, track_timestamp: datetime
) -> None:
    frame = DecodedVideoFrame(
        CameraFrame("CAMERA_A", CameraRole.ENTRY, frame_timestamp),
        bytes(range(20)), 5, 4, 0,
    )
    mismatched = TrackedPerson(
        frame.frame.camera_id, 1, (1, 1, 4, 3), 0.73, TrackState.ACTIVE,
        frame.frame_index, track_timestamp, frame.frame_index, 1, 0,
    )

    with pytest.raises(PersonRoiExtractionError, match="timestamp"):
        extract_observed_person_rois(frame, result(mismatched))


def test_exact_timestamp_representation_is_accepted() -> None:
    timestamp = datetime(2026, 9, 18, 10, tzinfo=timezone(timedelta(hours=5)))
    frame = DecodedVideoFrame(CameraFrame("CAMERA_A", CameraRole.ENTRY, timestamp), bytes(range(20)), 5, 4, 0)
    exact = track(frame=frame)

    assert extract_observed_person_rois(frame, result(exact))[0].timestamp == timestamp


@pytest.mark.parametrize(
    ("roi_bbox", "roi_width", "roi_height", "message"),
    [
        ((0, 0, 99, 99), 1, 1, "dimensions"),
        ((0, 0, 2, 3), 2, 2, "dimensions"),
        ((0, 0, 0, 1), 0, 1, "non-empty"),
        ((0, 0, 1, 0), 1, 0, "non-empty"),
        ((-1, 0, 1, 1), 2, 1, "non-empty"),
        ((0, -1, 1, 1), 1, 2, "non-empty"),
        ((0, 0, 1.5, 1), 1, 1, "four-integer"),
        ((0, 0, True, 1), 1, 1, "four-integer"),
    ],
)
def test_person_roi_bbox_is_self_consistent(
    roi_bbox: tuple[object, ...], roi_width: object, roi_height: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_observation(roi_bbox=roi_bbox, roi_width=roi_width, roi_height=roi_height)


@pytest.mark.parametrize("field", ["width", "height"])
def test_person_roi_dimension_must_be_positive_integer(field: str) -> None:
    kwargs = {"roi_width": 0} if field == "width" else {"roi_height": 0}
    with pytest.raises(ValueError, match=f"roi_{field}"):
        make_observation(**kwargs)

    kwargs = {"roi_width": True} if field == "width" else {"roi_height": True}
    with pytest.raises(ValueError, match=f"roi_{field}"):
        make_observation(**kwargs)


def test_person_roi_payload_length_must_match_dimensions() -> None:
    with pytest.raises(ValueError, match="exactly"):
        make_observation(roi_pixels=b"\x00")


def test_fractional_bbox_uses_floor_ceil_containment() -> None:
    observation = extract_observed_person_rois(make_frame(), result(track((1.2, 0.8, 3.1, 2.2))))[0]

    assert observation.person_bbox == (1.2, 0.8, 3.1, 2.2)
    assert observation.roi_bbox == (1, 0, 4, 3)
    assert observation.roi_pixels == bytes([1, 2, 3, 6, 7, 8, 11, 12, 13])


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((0, 0, 5, 4), (0, 0, 5, 4)),
        ((-1, 1, 2, 3), (0, 1, 2, 3)),
        ((1, -1, 3, 2), (1, 0, 3, 2)),
        ((3, 1, 6, 3), (3, 1, 5, 3)),
        ((1, 2, 4, 5), (1, 2, 4, 4)),
    ],
)
def test_bbox_is_clipped_to_source_edges(bbox: tuple[float, float, float, float], expected: tuple[int, int, int, int]) -> None:
    assert extract_observed_person_rois(make_frame(), result(track(bbox)))[0].roi_bbox == expected


@pytest.mark.parametrize("bbox", [(-3, 1, -1, 2), (1, -3, 2, -1), (5, 1, 6, 2), (1, 4, 2, 5)])
def test_completely_out_of_frame_bbox_is_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(PersonRoiExtractionError, match="empty ROI"):
        extract_observed_person_rois(make_frame(), result(track(bbox)))


def test_duplicate_observed_track_key_is_rejected() -> None:
    with pytest.raises(PersonRoiExtractionError, match="duplicate"):
        extract_observed_person_rois(make_frame(), result(track(), track()))


def test_validation_is_atomic_for_the_whole_observed_collection() -> None:
    frame = make_frame()
    invalid = track(track_id=2, frame=make_frame(index=1))

    with pytest.raises(PersonRoiExtractionError, match="frame_index"):
        extract_observed_person_rois(frame, result(track(), invalid))


@pytest.mark.parametrize(
    "bbox",
    [
        (1, 2, 3),
        (1, 2, "bad", 4),
        (1, 2, float("nan"), 4),
        (1, 2, float("inf"), 4),
        (1, 2, float("-inf"), 4),
        (1, 2, True, 4),
    ],
)
def test_malformed_tracked_bbox_is_normalized_to_extraction_error(bbox: object) -> None:
    with pytest.raises(PersonRoiExtractionError):
        extract_observed_person_rois(make_frame(), result(track(bbox)))


def test_malformed_tracked_bbox_collection_fails_before_any_output() -> None:
    malformed = track(track_id=2, bbox=(1, 2, float("nan"), 4))

    with pytest.raises(PersonRoiExtractionError):
        extract_observed_person_rois(make_frame(), result(track(), malformed))


def test_metadata_and_pixel_contract_are_preserved() -> None:
    frame = make_frame()
    observation = extract_observed_person_rois(frame, result(track()))[0]

    assert observation.camera_id == "CAMERA_A"
    assert observation.track_id == 1
    assert observation.frame_index == 0
    assert observation.timestamp == START
    assert observation.person_confidence == 0.73
    assert observation.pixel_format == "gray"
    assert len(observation.roi_pixels) == observation.roi_width * observation.roi_height
    assert frame.pixels == bytes(range(20))
    assert observation.roi_pixels == bytes(observation.roi_pixels)
    with pytest.raises(AttributeError):
        observation.roi_pixels = b"changed"


def test_multiple_observed_tracks_preserve_tracking_order() -> None:
    frame = make_frame()
    observations = extract_observed_person_rois(frame, result(track(track_id=2), track(track_id=1)))

    assert [item.track_id for item in observations] == [2, 1]


def test_person_roi_value_has_no_identity_or_attendance_fields() -> None:
    assert not {"employee_id", "worker_id", "embedding", "attendance_event"}.intersection(
        PersonRoiObservation.__dataclass_fields__
    )
