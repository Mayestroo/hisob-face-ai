from datetime import datetime, timedelta, timezone

import pytest

from app.cameras.video_file import DecodedVideoFrame
from app.config.settings import TrackingSettings
from app.detection.protocol import PersonDetection
from app.domain.models import CameraFrame, CameraRole
from app.tracking import PersonTracker, TrackState


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_frame(camera_id: str = "CAMERA_A", index: int = 0) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        frame=CameraFrame(camera_id, CameraRole.ENTRY, START + timedelta(seconds=index)),
        pixels=b"x",
        width=1,
        height=1,
        frame_index=index,
    )


def person(box: tuple[float, float, float, float], confidence: float = 0.8) -> PersonDetection:
    return PersonDetection(box, confidence)


def make_tracker(**kwargs: object) -> PersonTracker:
    return PersonTracker("CAMERA_A", TrackingSettings(**kwargs))


def test_first_detection_is_track_one_and_counts_as_hit_one() -> None:
    result = make_tracker().update(make_frame(), [person((0, 0, 1, 1))])

    assert (result.observed[0].track_id, result.observed[0].state, result.observed[0].hit_count) == (1, TrackState.NEW, 1)


def test_one_hit_configuration_immediately_activates() -> None:
    result = make_tracker(tracking_min_confirmed_hits=1).update(make_frame(), [person((0, 0, 1, 1))])

    assert result.observed[0].state is TrackState.ACTIVE


def test_matching_detection_preserves_id_and_propagates_confidence() -> None:
    tracking = make_tracker()
    original = person((0, 0, 2, 2), 0.61)
    tracking.update(make_frame(index=0), [original])
    result = tracking.update(make_frame(index=1), [person((0.1, 0.1, 2.1, 2.1), 0.73)])

    assert result.observed[0].track_id == 1
    assert result.observed[0].state is TrackState.ACTIVE
    assert result.observed[0].confidence == 0.73
    assert original.bbox == (0.0, 0.0, 2.0, 2.0)
    assert original.confidence == 0.61


def test_unmatched_detections_create_distinct_deterministic_tracks() -> None:
    result = make_tracker(tracking_min_confirmed_hits=1).update(
        make_frame(), [person((5, 5, 6, 6)), person((0, 0, 1, 1))]
    )

    assert [(item.track_id, item.bbox) for item in result.observed] == [(1, (0.0, 0.0, 1.0, 1.0)), (2, (5.0, 5.0, 6.0, 6.0))]


def test_one_detection_updates_only_one_track_and_highest_iou_wins() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_iou_threshold=0.1)
    tracking.update(make_frame(index=0), [person((0, 0, 2, 2)), person((0.5, 0, 2.5, 2))])
    result = tracking.update(make_frame(index=1), [person((0.25, 0, 2.25, 2))])

    assert result.observed[0].bbox == (0.25, 0.0, 2.25, 2.0)
    assert result.lost[0].state is TrackState.LOST


def test_reversed_detection_order_has_equivalent_association() -> None:
    detections = [person((0.1, 0, 1.1, 1)), person((10.1, 0, 11.1, 1))]
    first = make_tracker(tracking_min_confirmed_hits=1)
    second = make_tracker(tracking_min_confirmed_hits=1)
    initial = [person((0, 0, 1, 1)), person((10, 0, 11, 1))]
    first.update(make_frame(index=0), initial)
    second.update(make_frame(index=0), initial)

    left = first.update(make_frame(index=1), detections)
    right = second.update(make_frame(index=1), list(reversed(detections)))

    assert [(item.track_id, item.bbox) for item in left.observed] == [(item.track_id, item.bbox) for item in right.observed]


def test_new_track_recovery_keeps_hits_and_returns_to_new() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=3, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=0), [person((0, 0, 1, 1))])
    lost = tracking.update(make_frame(index=1), [])
    recovered = tracking.update(make_frame(index=2), [person((0, 0, 1, 1))])

    assert lost.lost[0].state is TrackState.LOST
    assert recovered.observed[0].state is TrackState.NEW
    assert recovered.observed[0].hit_count == 2


def test_active_track_recovers_after_two_intervening_missed_frames() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1))])
    lost = tracking.update(make_frame(index=12), [])
    recovered = tracking.update(make_frame(index=13), [person((0, 0, 1, 1))])

    assert lost.lost[0].missed_frames == 2
    assert recovered.observed[0].state is TrackState.ACTIVE
    assert recovered.observed[0].track_id == 1


def test_missed_frame_boundary_finalizes_and_emits_once() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1))])
    tracking.update(make_frame(index=12), [])
    expired = tracking.update(make_frame(index=13), [])
    after = tracking.update(make_frame(index=14), [person((0, 0, 1, 1))])

    assert expired.finalized[0].state is TrackState.FINALIZED
    assert expired.finalized[0].missed_frames == 3
    assert after.finalized == ()
    assert after.observed[0].track_id == 2


def test_empty_detection_is_valid_and_new_tracks_can_be_lost() -> None:
    tracking = make_tracker()
    tracking.update(make_frame(), [person((0, 0, 1, 1))])
    result = tracking.update(make_frame(index=1), [])

    assert result.observed == ()
    assert result.lost[0].state is TrackState.LOST


def test_camera_mismatch_fails_and_trackers_are_isolated() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1)
    with pytest.raises(ValueError, match="camera_id"):
        tracking.update(make_frame("CAMERA_B"), [person((0, 0, 1, 1))])

    other = PersonTracker("CAMERA_B", TrackingSettings(tracking_min_confirmed_hits=1))
    result = other.update(make_frame("CAMERA_B"), [person((0, 0, 1, 1))])
    assert result.observed[0].camera_id == "CAMERA_B"


def test_duplicate_and_decreasing_indexes_are_rejected_but_gaps_are_allowed() -> None:
    tracking = make_tracker()
    tracking.update(make_frame(index=3), [])
    with pytest.raises(ValueError, match="increase"):
        tracking.update(make_frame(index=3), [])
    with pytest.raises(ValueError, match="increase"):
        tracking.update(make_frame(index=2), [])

    assert tracking.update(make_frame(index=5), []).observed == ()


def test_reset_finalizes_tracks_and_never_reuses_ids_or_state() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1)
    old = tracking.update(make_frame(), [person((0, 0, 1, 1))]).observed[0]
    finalized = tracking.reset()
    new = tracking.update(make_frame(index=1), [person((0, 0, 1, 1))]).observed[0]

    assert finalized[0].state is TrackState.FINALIZED
    assert new.track_id != old.track_id
    assert new.hit_count == 1
    assert new is not old


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tracking_iou_threshold", float("nan")),
        ("tracking_iou_threshold", 1.1),
        ("tracking_min_confirmed_hits", 0),
        ("tracking_max_lost_frames", -1),
    ],
)
def test_tracking_settings_validate_thresholds(field_name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrackingSettings(**{field_name: value})


def test_frame_index_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        make_frame(index=-1)


def test_result_partitions_observed_lost_and_finalized_ids_and_states() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=1)
    tracking.update(make_frame(index=0), [person((0, 0, 1, 1)), person((10, 10, 11, 11))])

    result = tracking.update(make_frame(index=1), [person((0, 0, 1, 1))])

    assert {item.track_id for item in result.observed} == {1}
    assert {item.track_id for item in result.lost} == {2}
    assert result.finalized == ()
    assert all(item.state is TrackState.ACTIVE for item in result.observed)
    assert all(item.state is TrackState.LOST for item in result.lost)
    groups = [{item.track_id for item in group} for group in (result.observed, result.lost, result.finalized)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_observed_snapshots_are_current_detection_metadata() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1), 0.4)])

    result = tracking.update(make_frame(index=13), [person((0.1, 0.1, 1.1, 1.1), 0.9)])

    assert result.observed[0].frame_index == 13
    assert result.observed[0].bbox == (0.1, 0.1, 1.1, 1.1)
    assert result.observed[0].confidence == 0.9
    assert result.observed[0].state is TrackState.ACTIVE
    assert result.lost == ()
    assert result.finalized == ()


def test_recovered_lost_track_is_observed_only() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1))])
    tracking.update(make_frame(index=11), [])

    result = tracking.update(make_frame(index=12), [person((0, 0, 1, 1))])

    assert [item.track_id for item in result.observed] == [1]
    assert result.lost == ()
    assert result.finalized == ()


def test_frame_10_to_13_empty_detection_finalizes_without_observed_or_lost() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=13), [])

    assert result.observed == ()
    assert result.lost == ()
    assert [item.track_id for item in result.finalized] == [1]
    assert result.finalized[0].state is TrackState.FINALIZED


def test_frame_10_to_14_creates_new_track_after_old_track_finalizes() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=14), [person((0, 0, 1, 1))])

    assert [item.track_id for item in result.finalized] == [1]
    assert [(item.track_id, item.state) for item in result.observed] == [(2, TrackState.ACTIVE)]
    assert result.lost == ()


def test_max_lost_frames_zero_allows_immediate_match() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=0)
    tracking.update(make_frame(index=0), [person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=1), [person((0, 0, 1, 1))])

    assert [item.track_id for item in result.observed] == [1]
    assert result.lost == ()
    assert result.finalized == ()


def test_max_lost_frames_zero_miss_finalizes_immediately() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=0)
    tracking.update(make_frame(index=0), [person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=1), [])

    assert result.observed == ()
    assert result.lost == ()
    assert [item.track_id for item in result.finalized] == [1]


def test_large_frame_gap_expires_old_track_and_creates_new_track() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=2)
    tracking.update(make_frame(index=10), [person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=1_000_000), [person((0, 0, 1, 1))])

    assert [item.track_id for item in result.finalized] == [1]
    assert [item.track_id for item in result.observed] == [2]
    assert result.lost == ()


def test_invalid_inputs_leave_tracker_state_unchanged() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=2)
    tracking.update(make_frame(index=2), [person((0, 0, 1, 1))])

    with pytest.raises(ValueError, match="camera_id"):
        tracking.update(make_frame("CAMERA_B", index=3), [person((0, 0, 1, 1))])
    with pytest.raises(ValueError, match="increase"):
        tracking.update(make_frame(index=2), [person((0, 0, 1, 1))])
    with pytest.raises(ValueError, match="increase"):
        tracking.update(make_frame(index=1), [person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=3), [person((0, 0, 1, 1), 0.9)])

    assert [(item.track_id, item.hit_count, item.state, item.frame_index) for item in result.observed] == [
        (1, 2, TrackState.ACTIVE, 3)
    ]


def test_reset_twice_emits_nothing_the_second_time() -> None:
    tracking = make_tracker()
    tracking.update(make_frame(), [person((0, 0, 1, 1))])

    assert len(tracking.reset()) == 1
    assert tracking.reset() == ()


def test_multiple_tracks_finalize_once_in_deterministic_order() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1, tracking_max_lost_frames=0)
    tracking.update(make_frame(index=0), [person((10, 10, 11, 11)), person((0, 0, 1, 1))])

    result = tracking.update(make_frame(index=1), [])
    after = tracking.update(make_frame(index=2), [])

    assert [item.track_id for item in result.finalized] == [1, 2]
    assert after.finalized == ()


def test_earlier_snapshot_does_not_mutate_after_later_updates() -> None:
    tracking = make_tracker(tracking_min_confirmed_hits=1)
    first = tracking.update(make_frame(index=0), [person((0, 0, 1, 1), 0.4)])
    snapshot = first.observed[0]

    tracking.update(make_frame(index=1), [person((2, 2, 3, 3), 0.9)])

    assert snapshot.track_id == 1
    assert snapshot.bbox == (0.0, 0.0, 1.0, 1.0)
    assert snapshot.confidence == 0.4
    assert snapshot.frame_index == 0
