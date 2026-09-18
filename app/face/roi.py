"""Extract immutable person ROIs from current-frame observed tracks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from app.cameras.video_file import DecodedVideoFrame
from app.domain.models import BoundingBox
from app.tracking.tracker import TrackState, TrackedPerson, TrackingUpdateResult


class PersonRoiExtractionError(ValueError):
    """A current-frame or person-ROI boundary invariant was violated."""


def _validate_bbox(value: object, field_name: str) -> BoundingBox:
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError(f"{field_name} must be a four-value tuple")
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, Real) for coordinate in value):
        raise TypeError(f"{field_name} coordinates must be numbers")
    bbox = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise ValueError(f"{field_name} coordinates must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{field_name} must satisfy x2 > x1 and y2 > y1")
    return bbox


def _same_timestamp_representation(first: datetime, second: datetime) -> bool:
    return (
        first.year,
        first.month,
        first.day,
        first.hour,
        first.minute,
        first.second,
        first.microsecond,
        first.fold,
        first.utcoffset(),
    ) == (
        second.year,
        second.month,
        second.day,
        second.hour,
        second.minute,
        second.second,
        second.microsecond,
        second.fold,
        second.utcoffset(),
    )


@dataclass(frozen=True, slots=True)
class PersonRoiObservation:
    """A stable grayscale crop of a tracked person, before face detection.

    ``person_bbox`` retains the tracked floating-point source-frame geometry.
    ``roi_bbox`` is the clipped integer source-frame crop using half-open
    bounds. A later face detector may use ROI-local coordinates and translate
    them through ``roi_bbox``; this value does not represent a detected face.
    """

    camera_id: str
    track_id: int
    frame_index: int
    timestamp: datetime
    person_bbox: BoundingBox
    person_confidence: float
    roi_bbox: tuple[int, int, int, int]
    roi_pixels: bytes
    roi_width: int
    roi_height: int
    pixel_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not isinstance(self.track_id, int) or isinstance(self.track_id, bool):
            raise TypeError("track_id must be an integer")
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        object.__setattr__(self, "person_bbox", _validate_bbox(self.person_bbox, "person_bbox"))
        if isinstance(self.person_confidence, bool) or not isinstance(self.person_confidence, Real):
            raise TypeError("person_confidence must be a number")
        confidence = float(self.person_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("person_confidence must be between 0.0 and 1.0")
        object.__setattr__(self, "person_confidence", confidence)
        if not isinstance(self.roi_bbox, tuple) or len(self.roi_bbox) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.roi_bbox
        ):
            raise TypeError("roi_bbox must be a four-integer tuple")
        left, top, right, bottom = self.roi_bbox
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("roi_bbox must be a non-empty source-frame rectangle")
        if not isinstance(self.roi_pixels, bytes):
            raise TypeError("roi_pixels must be bytes")
        if not isinstance(self.roi_width, int) or isinstance(self.roi_width, bool) or self.roi_width <= 0:
            raise ValueError("roi_width must be greater than zero")
        if not isinstance(self.roi_height, int) or isinstance(self.roi_height, bool) or self.roi_height <= 0:
            raise ValueError("roi_height must be greater than zero")
        if right - left != self.roi_width or bottom - top != self.roi_height:
            raise ValueError("roi_bbox dimensions must match roi_width and roi_height")
        if self.pixel_format != "gray":
            raise ValueError("pixel_format must be 'gray'")
        if len(self.roi_pixels) != self.roi_width * self.roi_height:
            raise ValueError("roi_pixels must contain exactly roi_width * roi_height bytes")


def _validate_observed_track(frame: DecodedVideoFrame, track: TrackedPerson, seen: set[tuple[str, int, int]]) -> None:
    if not isinstance(track, TrackedPerson):
        raise TypeError("tracking_result.observed must contain only TrackedPerson values")
    if track.state not in (TrackState.NEW, TrackState.ACTIVE):
        raise PersonRoiExtractionError("observed tracks must be NEW or ACTIVE")
    if track.camera_id != frame.frame.camera_id:
        raise PersonRoiExtractionError("observed track camera_id does not match frame camera_id")
    if track.frame_index != frame.frame_index:
        raise PersonRoiExtractionError("observed track frame_index does not match frame frame_index")
    if not _same_timestamp_representation(track.timestamp, frame.frame.timestamp):
        raise PersonRoiExtractionError("observed track timestamp does not match frame timestamp")
    key = (track.camera_id, track.track_id, track.frame_index)
    if key in seen:
        raise PersonRoiExtractionError("duplicate observed track key")
    seen.add(key)
    try:
        _validate_bbox(track.bbox, "track.bbox")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PersonRoiExtractionError("tracked person bbox is malformed") from exc


def _integer_roi(bbox: BoundingBox, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width, math.floor(bbox[0])))
    top = max(0, min(height, math.floor(bbox[1])))
    right = max(0, min(width, math.ceil(bbox[2])))
    bottom = max(0, min(height, math.ceil(bbox[3])))
    if right <= left or bottom <= top:
        raise PersonRoiExtractionError("tracked person bbox produces an empty ROI after clipping")
    return left, top, right, bottom


def _crop_pixels(frame: DecodedVideoFrame, roi_bbox: tuple[int, int, int, int]) -> bytes:
    left, top, right, bottom = roi_bbox
    row_width = right - left
    return b"".join(frame.pixels[y * frame.width + left : y * frame.width + right] for y in range(top, bottom))


def extract_observed_person_rois(
    frame: DecodedVideoFrame,
    tracking_result: TrackingUpdateResult,
) -> tuple[PersonRoiObservation, ...]:
    """Extract person crops only from tracks observed on ``frame``.

    Validation covers the complete observed collection before any output is
    created, so a bad item cannot produce a partially successful result.
    """

    if not isinstance(frame, DecodedVideoFrame):
        raise TypeError("frame must be a DecodedVideoFrame")
    if not isinstance(tracking_result, TrackingUpdateResult):
        raise TypeError("tracking_result must be a TrackingUpdateResult")

    seen: set[tuple[str, int, int]] = set()
    prepared: list[tuple[TrackedPerson, tuple[int, int, int, int]]] = []
    for track in tracking_result.observed:
        _validate_observed_track(frame, track, seen)
        try:
            bbox = _validate_bbox(track.bbox, "track.bbox")
            roi_bbox = _integer_roi(bbox, frame.width, frame.height)
        except PersonRoiExtractionError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise PersonRoiExtractionError("tracked person bbox is malformed") from exc
        prepared.append((track, roi_bbox))

    observations = []
    for track, roi_bbox in prepared:
        left, top, right, bottom = roi_bbox
        roi_width = right - left
        roi_height = bottom - top
        roi_pixels = _crop_pixels(frame, roi_bbox)
        observations.append(
            PersonRoiObservation(
                camera_id=track.camera_id,
                track_id=track.track_id,
                frame_index=track.frame_index,
                timestamp=track.timestamp,
                person_bbox=track.bbox,
                person_confidence=track.confidence,
                roi_bbox=roi_bbox,
                roi_pixels=roi_pixels,
                roi_width=roi_width,
                roi_height=roi_height,
                pixel_format=frame.pixel_format,
            )
        )
    return tuple(observations)
