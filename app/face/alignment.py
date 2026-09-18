"""Deterministic five-point grayscale alignment for detected faces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class FaceAlignmentError(RuntimeError):
    """A candidate's landmarks cannot produce a safe alignment."""


class YuNetLandmark(str, Enum):
    """The anatomical order emitted by the YuNet model."""

    RIGHT_EYE = "RIGHT_EYE"
    LEFT_EYE = "LEFT_EYE"
    NOSE_TIP = "NOSE_TIP"
    RIGHT_MOUTH_CORNER = "RIGHT_MOUTH_CORNER"
    LEFT_MOUTH_CORNER = "LEFT_MOUTH_CORNER"


YUNET_LANDMARK_ORDER = (
    YuNetLandmark.RIGHT_EYE,
    YuNetLandmark.LEFT_EYE,
    YuNetLandmark.NOSE_TIP,
    YuNetLandmark.RIGHT_MOUTH_CORNER,
    YuNetLandmark.LEFT_MOUTH_CORNER,
)

# The template is ordered anatomically, not in YuNet's detector order.
ARC_FACE_TEMPLATE_112 = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


@dataclass(frozen=True, slots=True)
class AlignmentSpec:
    width: int = 112
    height: int = 112
    template: tuple[tuple[float, float], ...] = ARC_FACE_TEMPLATE_112
    version: str = "arcface-112-v1"

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("alignment width must be positive")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("alignment height must be positive")
        if len(self.template) != 5 or any(len(point) != 2 for point in self.template):
            raise ValueError("alignment template must contain five points")
        if not all(math.isfinite(float(value)) for point in self.template for value in point):
            raise ValueError("alignment template must be finite")


def map_yunet_landmarks_to_template(
    landmarks: tuple[tuple[float, float], ...], spec: AlignmentSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Map YuNet order explicitly to template order: L eye, R eye, nose, L mouth, R mouth."""
    if len(landmarks) != 5:
        raise FaceAlignmentError("alignment requires five landmarks")
    source = np.asarray(
        (landmarks[1], landmarks[0], landmarks[2], landmarks[4], landmarks[3]), dtype=np.float64
    )
    target = np.asarray(spec.template, dtype=np.float64)
    if not np.isfinite(source).all():
        raise FaceAlignmentError("landmarks must be finite")
    return source, target


def _similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    source_energy = float(np.sum(centered_source * centered_source))
    if not math.isfinite(source_energy) or source_energy <= 1e-12:
        raise FaceAlignmentError("landmarks are geometrically degenerate")
    covariance = centered_source.T @ centered_target
    if not np.isfinite(covariance).all():
        raise FaceAlignmentError("landmark transform is non-finite")
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as error:
        raise FaceAlignmentError("landmark transform is singular") from error
    if singular_values[-1] <= 1e-12 or singular_values[0] / singular_values[-1] > 1e12:
        raise FaceAlignmentError("landmark transform is ill-conditioned")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    if not math.isfinite(float(np.linalg.det(rotation))) or np.linalg.det(rotation) <= 0:
        raise FaceAlignmentError("landmark transform contains a reflection")
    scale = float(np.sum(singular_values) / source_energy)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix[:2, :2]))) <= 1e-15:
        raise FaceAlignmentError("landmark transform is invalid")
    return matrix


def _bilinear_zero_border(image: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    """Sample each bilinear neighbor independently, treating outside pixels as zero."""
    roi_height, roi_width = image.shape
    finite = np.isfinite(source_x) & np.isfinite(source_y)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = source_x - x0
    fy = source_y - y0

    def sample(neighbor_x: np.ndarray, neighbor_y: np.ndarray) -> np.ndarray:
        inside = finite & (neighbor_x >= 0) & (neighbor_x < roi_width) & (neighbor_y >= 0) & (neighbor_y < roi_height)
        safe_x = np.clip(neighbor_x, 0, roi_width - 1)
        safe_y = np.clip(neighbor_y, 0, roi_height - 1)
        return np.where(inside, image[safe_y, safe_x], 0)

    return (
        sample(x0, y0) * (1.0 - fx) * (1.0 - fy)
        + sample(x1, y0) * fx * (1.0 - fy)
        + sample(x0, y1) * (1.0 - fx) * fy
        + sample(x1, y1) * fx * fy
    )


def align_grayscale_face(
    roi_pixels: bytes,
    roi_width: int,
    roi_height: int,
    landmarks_roi: tuple[tuple[float, float], ...],
    spec: AlignmentSpec = AlignmentSpec(),
) -> bytes:
    """Warp ROI pixels into the alignment template using bilinear zero-border sampling."""
    if not isinstance(roi_pixels, bytes) or len(roi_pixels) != roi_width * roi_height:
        raise FaceAlignmentError("ROI pixels are malformed")
    source, target = map_yunet_landmarks_to_template(landmarks_roi, spec)
    transform = _similarity_transform(source, target)
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise FaceAlignmentError("landmark transform cannot be inverted") from error
    if not np.isfinite(inverse).all():
        raise FaceAlignmentError("inverse landmark transform is non-finite")

    image = np.frombuffer(roi_pixels, dtype=np.uint8).reshape(roi_height, roi_width)
    y, x = np.indices((spec.height, spec.width), dtype=np.float64)
    destination = np.stack((x.ravel(), y.ravel(), np.ones(x.size)), axis=0)
    source_points = inverse @ destination
    source_x = source_points[0].reshape(spec.height, spec.width)
    source_y = source_points[1].reshape(spec.height, spec.width)
    result = _bilinear_zero_border(image, source_x, source_y)
    finite = np.isfinite(source_x) & np.isfinite(source_y)
    return np.clip(np.rint(np.where(finite, result, 0.0)), 0, 255).astype(np.uint8).tobytes()
