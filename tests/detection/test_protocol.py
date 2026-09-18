from datetime import datetime, timezone

import pytest

from app.cameras.video_file import DecodedVideoFrame
from app.detection.protocol import PersonDetection, PersonDetector
from app.domain.models import CameraFrame, CameraRole


START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_frame() -> DecodedVideoFrame:
    return DecodedVideoFrame(
        frame=CameraFrame("CAMERA_A", CameraRole.ENTRY, START),
        pixels=b"data",
        width=2,
        height=2,
        frame_index=0,
    )


def test_valid_integer_bbox_is_normalized_to_float_coordinates() -> None:
    detection = PersonDetection((1, 2, 30, 40), 0.5)

    assert detection.bbox == (1.0, 2.0, 30.0, 40.0)
    assert detection.confidence == 0.5


def test_valid_float_bbox_is_accepted() -> None:
    detection = PersonDetection((1.25, 2.5, 30.75, 40.125), 0.5)

    assert detection.bbox == (1.25, 2.5, 30.75, 40.125)


@pytest.mark.parametrize("confidence", [0, 1, 0.35])
def test_valid_confidence_is_accepted(confidence: float) -> None:
    assert PersonDetection((1, 2, 3, 4), confidence).confidence == float(confidence)


@pytest.mark.parametrize("bbox", [(1, 2, 1, 4), (1, 2, 3, 2)])
def test_degenerate_bbox_is_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        PersonDetection(bbox, 0.5)


@pytest.mark.parametrize("bbox", [(3, 2, 1, 4), (1, 4, 3, 2)])
def test_reversed_bbox_is_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        PersonDetection(bbox, 0.5)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_out_of_range_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        PersonDetection((1, 2, 3, 4), confidence)


@pytest.mark.parametrize("bbox", [(float("nan"), 2, 3, 4), (1, 2, float("inf"), 4)])
def test_non_finite_bbox_is_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        PersonDetection(bbox, 0.5)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        PersonDetection((1, 2, 3, 4), confidence)


def test_bool_bbox_coordinate_is_rejected() -> None:
    with pytest.raises(TypeError):
        PersonDetection((True, 2, 3, 4), 0.5)


def test_non_numeric_bbox_coordinate_is_rejected() -> None:
    with pytest.raises(TypeError):
        PersonDetection(("1", 2, 3, 4), 0.5)


def test_bool_confidence_is_rejected() -> None:
    with pytest.raises(TypeError):
        PersonDetection((1, 2, 3, 4), True)


def test_non_numeric_confidence_is_rejected() -> None:
    with pytest.raises(TypeError):
        PersonDetection((1, 2, 3, 4), "0.5")


def test_person_detection_is_immutable() -> None:
    detection = PersonDetection((1, 2, 3, 4), 0.5)

    with pytest.raises(AttributeError):
        detection.bbox = (2, 3, 4, 5)
    with pytest.raises(AttributeError):
        detection.confidence = 0.75


def test_fake_detector_satisfies_protocol_and_returns_project_values() -> None:
    class FakeDetector:
        def detect(self, frame: DecodedVideoFrame) -> list[PersonDetection]:
            assert isinstance(frame, DecodedVideoFrame)
            return [PersonDetection((0, 0, 1, 1), 1.0)]

    detector: PersonDetector = FakeDetector()
    detections = detector.detect(make_frame())

    assert isinstance(detector, PersonDetector)
    assert len(detections) == 1
    assert isinstance(detections[0], PersonDetection)
