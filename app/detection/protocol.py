"""Project-owned person detection values and detector protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol, Sequence, runtime_checkable

from app.cameras.video_file import DecodedVideoFrame
from app.domain.models import BoundingBox


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


def _validate_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence must be a number")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return confidence


@dataclass(frozen=True)
class PersonDetection:
    """A person bounding box in decoded-frame pixel coordinates."""

    bbox: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _validate_bbox(self.bbox))
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))


@runtime_checkable
class PersonDetector(Protocol):
    """Detector boundary consuming one decoded frame at a time."""

    def detect(self, frame: DecodedVideoFrame) -> Sequence[PersonDetection]: ...
