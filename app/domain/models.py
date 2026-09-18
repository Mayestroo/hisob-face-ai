"""Dependency-free domain values for the face attendance pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from numbers import Real
from typing import TypeAlias


class CameraRole(str, Enum):
    """The semantic role assigned to a camera by configuration."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"


AttendanceEventType: TypeAlias = CameraRole
BoundingBox: TypeAlias = tuple[float, float, float, float]


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_aware_timestamp(value: object, field_name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value


def _validate_confidence(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a number")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return confidence


def _validate_bbox(value: object) -> BoundingBox:
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("bbox must be a four-value tuple")
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, Real) for coordinate in value):
        raise TypeError("bbox coordinates must be numbers")
    bbox = tuple(float(coordinate) for coordinate in value)
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise ValueError("bbox coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
    return bbox


def _validate_enum(value: object, enum_type: type[Enum], field_name: str) -> Enum:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")
    return value


@dataclass(frozen=True)
class CameraFrame:
    """Metadata for one frame received from a semantically assigned camera."""

    camera_id: str
    camera_role: CameraRole
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_non_empty_string(self.camera_id, "camera_id")
        _validate_enum(self.camera_role, CameraRole, "camera_role")
        _require_aware_timestamp(self.timestamp)


@dataclass(frozen=True)
class Track:
    """A person track whose identity is local to one camera."""

    track_id: str
    camera_id: str
    timestamp: datetime
    bbox: BoundingBox | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.track_id, "track_id")
        _require_non_empty_string(self.camera_id, "camera_id")
        _require_aware_timestamp(self.timestamp)
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _validate_bbox(self.bbox))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _validate_confidence(self.confidence, "confidence"))


@dataclass(frozen=True)
class FaceObservation:
    """A face observation explicitly associated with its originating track."""

    track: Track
    timestamp: datetime
    bbox: BoundingBox | None = None
    quality: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.track, Track):
            raise TypeError("track must be a Track")
        _require_aware_timestamp(self.timestamp)
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _validate_bbox(self.bbox))
        if self.quality is not None:
            object.__setattr__(self, "quality", _validate_confidence(self.quality, "quality"))


@dataclass(frozen=True)
class AttendanceEventDraft:
    """An AI-determined attendance event awaiting durable delivery."""

    employee_id: int
    camera_id: str
    event_type: AttendanceEventType
    timestamp: datetime
    recognition_confidence: float
    track_id: str

    def __post_init__(self) -> None:
        if isinstance(self.employee_id, bool) or not isinstance(self.employee_id, int):
            raise TypeError("employee_id must be a numeric HISOB worker ID")
        if self.employee_id <= 0:
            raise ValueError("employee_id must be greater than zero")
        _require_non_empty_string(self.camera_id, "camera_id")
        _validate_enum(self.event_type, CameraRole, "event_type")
        _require_aware_timestamp(self.timestamp)
        object.__setattr__(
            self,
            "recognition_confidence",
            _validate_confidence(self.recognition_confidence, "recognition_confidence"),
        )
        _require_non_empty_string(self.track_id, "track_id")
