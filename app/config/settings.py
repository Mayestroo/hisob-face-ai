"""Typed, deterministic configuration for the Face AI application."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from app.domain.models import CameraRole


_SUPPORTED_DEVICES = frozenset({"cpu", "cuda", "mps"})
_CREDENTIALS_IN_SOURCE = re.compile(r"(?://)[^/@\s]+@")


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _redact_source(source: str) -> str:
    return _CREDENTIALS_IN_SOURCE.sub("://***@", source)


@dataclass(frozen=True, repr=False)
class CameraConfig:
    """A physical camera source with an explicit semantic role."""

    camera_id: str
    role: CameraRole
    source: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.camera_id, "camera_id")
        if not isinstance(self.role, CameraRole):
            raise TypeError("role must be a CameraRole")
        _require_non_empty_string(self.source, "source")

    def __repr__(self) -> str:
        return (
            f"CameraConfig(camera_id={self.camera_id!r}, role={self.role!r}, "
            f"source={_redact_source(self.source)!r})"
        )


@dataclass(frozen=True)
class ModelRuntimeSettings:
    """Minimal model and runtime choices needed by later pipeline tasks."""

    model_identifier: str
    device: str = "cpu"
    detection_interval_seconds: float = 1.0
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    model_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.model_identifier, "model_identifier")
        _require_non_empty_string(self.device, "device")
        if self.device not in _SUPPORTED_DEVICES:
            supported = ", ".join(sorted(_SUPPORTED_DEVICES))
            raise ValueError(f"device must be one of: {supported}")
        if isinstance(self.detection_interval_seconds, bool) or not isinstance(
            self.detection_interval_seconds, (int, float)
        ):
            raise TypeError("detection_interval_seconds must be a number")
        if not math.isfinite(float(self.detection_interval_seconds)) or self.detection_interval_seconds <= 0:
            raise ValueError("detection_interval_seconds must be greater than zero")
        for value, field_name in (
            (self.confidence_threshold, "confidence_threshold"),
            (self.nms_threshold, "nms_threshold"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        if self.model_sha256 is not None:
            model_sha256 = _require_non_empty_string(self.model_sha256, "model_sha256")
            if len(model_sha256) != 64 or any(character not in "0123456789abcdefABCDEF" for character in model_sha256):
                raise ValueError("model_sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class TrackingSettings:
    """Deterministic, camera-local tracking thresholds."""

    tracking_iou_threshold: float = 0.3
    tracking_min_confirmed_hits: int = 2
    tracking_max_lost_frames: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.tracking_iou_threshold, bool) or not isinstance(self.tracking_iou_threshold, (int, float)):
            raise TypeError("tracking_iou_threshold must be a number")
        threshold = float(self.tracking_iou_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("tracking_iou_threshold must be between 0.0 and 1.0")
        for value, field_name in (
            (self.tracking_min_confirmed_hits, "tracking_min_confirmed_hits"),
            (self.tracking_max_lost_frames, "tracking_max_lost_frames"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            minimum = 1 if field_name.endswith("hits") else 0
            if value < minimum:
                raise ValueError(f"{field_name} must be at least {minimum}")


@dataclass(frozen=True)
class StorageSettings:
    """Paths reserved for future local persistence."""

    data_directory: str
    database_name: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.data_directory, "data_directory")
        database_name = _require_non_empty_string(self.database_name, "database_name")
        if database_name in {".", ".."} or "/" in database_name or "\\" in database_name:
            raise ValueError("database_name must be a relative file name")


@dataclass(frozen=True)
class ApplicationSettings:
    """Complete validated settings required to construct the application."""

    cameras: tuple[CameraConfig, ...]
    model: ModelRuntimeSettings
    storage: StorageSettings
    tracking: TrackingSettings = field(default_factory=TrackingSettings)

    def __post_init__(self) -> None:
        if isinstance(self.cameras, (str, bytes)):
            raise TypeError("cameras must be an iterable of CameraConfig values")
        try:
            cameras = tuple(self.cameras)
        except TypeError as error:
            raise TypeError("cameras must be an iterable of CameraConfig values") from error
        if any(not isinstance(camera, CameraConfig) for camera in cameras):
            raise TypeError("cameras must contain only CameraConfig values")
        camera_ids = [camera.camera_id for camera in cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera IDs must be unique")
        if not isinstance(self.model, ModelRuntimeSettings):
            raise TypeError("model must be ModelRuntimeSettings")
        if not isinstance(self.storage, StorageSettings):
            raise TypeError("storage must be StorageSettings")
        if not isinstance(self.tracking, TrackingSettings):
            raise TypeError("tracking must be TrackingSettings")
        object.__setattr__(self, "cameras", cameras)
