"""ONNX Runtime adapter for the OpenCV Zoo YOLOX-S person detector."""

from __future__ import annotations

from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app.cameras.video_file import DecodedVideoFrame
from app.config.settings import ModelRuntimeSettings
from app.detection.protocol import PersonDetection


INPUT_SIZE = 640
NUM_CLASSES = 80
PERSON_CLASS_ID = 0
STRIDES = (8, 16, 32)
EXPECTED_MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
REQUESTED_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


class DetectorHealth(str, Enum):
    AVAILABLE_GPU = "AVAILABLE_GPU"
    AVAILABLE_CPU_DEGRADED = "AVAILABLE_CPU_DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class PersonDetectorError(RuntimeError):
    """Base error for detector availability and backend failures."""


class DetectorUnavailableError(PersonDetectorError):
    """The detector could not create a usable inference session."""


class DetectorInferenceError(PersonDetectorError):
    """The backend returned an unusable result or raised during inference."""


SessionFactory = Callable[[str, Sequence[str]], Any]


def _default_session_factory(model_path: str, providers: Sequence[str]) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise DetectorUnavailableError("onnxruntime is required for person detection") from error
    return ort.InferenceSession(model_path, providers=list(providers))


def _resize_linear_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reproduce OpenCV INTER_LINEAR for uint8 images without a runtime dependency."""
    source_height, source_width = image.shape

    def axis_coefficients(source_size: int, destination_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinate = np.asarray(
            (np.arange(destination_size, dtype=np.float64) + 0.5) * source_size / destination_size - 0.5,
            dtype=np.float32,
        )
        source_index = np.floor(coordinate).astype(np.intp)
        fraction = coordinate - source_index
        outside = (source_index < 0) | (source_index >= source_size - 1)
        source_index = np.clip(source_index, 0, source_size - 1)
        fraction = np.where(outside, 0.0, fraction)
        first = np.rint(np.asarray(1.0 - fraction, dtype=np.float32) * 2048.0).astype(np.int32)
        second = np.rint(np.asarray(fraction, dtype=np.float32) * 2048.0).astype(np.int32)
        return source_index, first, second

    x_index, x_first, x_second = axis_coefficients(source_width, width)
    y_index, y_first, y_second = axis_coefficients(source_height, height)
    horizontal = (
        image[:, x_index].astype(np.int32) * x_first
        + image[:, np.minimum(x_index + 1, source_width - 1)].astype(np.int32) * x_second
    )
    vertical = (
        horizontal[y_index].astype(np.int64) * y_first[:, None]
        + horizontal[np.minimum(y_index + 1, source_height - 1)].astype(np.int64) * y_second[:, None]
    )
    return np.clip((vertical + (1 << 21)) >> 22, 0, 255).astype(np.uint8)


def _preprocess(frame: DecodedVideoFrame) -> tuple[np.ndarray, float]:
    gray = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width)
    scale = min(INPUT_SIZE / frame.height, INPUT_SIZE / frame.width)
    resized_height = int(frame.height * scale)
    resized_width = int(frame.width * scale)
    resized = _resize_linear_uint8(gray, resized_height, resized_width)
    rgb = np.repeat(resized[:, :, None], 3, axis=2).astype(np.float32)
    letterboxed = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114.0, dtype=np.float32)
    letterboxed[:resized_height, :resized_width] = rgb
    return np.ascontiguousarray(letterboxed.transpose(2, 0, 1)[None]), scale


def _grid() -> tuple[np.ndarray, np.ndarray]:
    grids = []
    expanded_strides = []
    for stride in STRIDES:
        size = INPUT_SIZE // stride
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        grids.append(np.stack((x, y), axis=-1).reshape(1, -1, 2))
        expanded_strides.append(np.full((1, size * size, 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


_GRIDS, _EXPANDED_STRIDES = _grid()


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * (boxes[remaining, 3] - boxes[remaining, 1])
        iou = intersection / np.maximum(area_current + area_remaining - intersection, np.finfo(np.float32).eps)
        order = remaining[iou <= threshold]
    return np.asarray(keep, dtype=np.intp)


class YoloxPersonDetector:
    """Convert decoded grayscale frames into project-owned person detections."""

    def __init__(self, settings: ModelRuntimeSettings, session: Any | None = None, session_factory: SessionFactory | None = None) -> None:
        self._health = DetectorHealth.UNAVAILABLE
        self._session = session
        self._confidence_threshold = float(settings.confidence_threshold)
        self._nms_threshold = float(settings.nms_threshold)
        self._model_path = Path(settings.model_identifier)
        if session is not None:
            try:
                self._set_health_from_providers(session.get_providers())
            except Exception as error:
                self._health = DetectorHealth.UNAVAILABLE
                self._initialization_error = (
                    error if isinstance(error, DetectorUnavailableError)
                    else DetectorUnavailableError("could not determine ONNX Runtime provider health")
                )
            return
        try:
            if not self._model_path.is_file():
                raise DetectorUnavailableError(f"model file does not exist: {self._model_path}")
            expected_hash = settings.model_sha256 or EXPECTED_MODEL_SHA256
            digest_calculator = hashlib.sha256()
            with self._model_path.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest_calculator.update(chunk)
            digest = digest_calculator.hexdigest()
            if digest != expected_hash:
                raise DetectorUnavailableError("model SHA-256 does not match the approved artifact")
            factory = session_factory or _default_session_factory
            self._session = factory(str(self._model_path), REQUESTED_PROVIDERS)
            self._set_health_from_providers(self._session.get_providers())
        except Exception as error:
            self._health = DetectorHealth.UNAVAILABLE
            if isinstance(error, DetectorUnavailableError):
                self._initialization_error = error
            else:
                self._initialization_error = DetectorUnavailableError("could not create ONNX Runtime session")
                self._initialization_error.__cause__ = error

    @property
    def health(self) -> DetectorHealth:
        return self._health

    @property
    def active_provider(self) -> str | None:
        if self._session is None:
            return None
        providers = self._session.get_providers()
        return providers[0] if providers else None

    def _set_health_from_providers(self, providers: Sequence[str]) -> None:
        if "CUDAExecutionProvider" in providers:
            self._health = DetectorHealth.AVAILABLE_GPU
        elif "CPUExecutionProvider" in providers:
            self._health = DetectorHealth.AVAILABLE_CPU_DEGRADED
        else:
            raise DetectorUnavailableError("ONNX Runtime reported no supported execution provider")

    def detect(self, frame: DecodedVideoFrame) -> Sequence[PersonDetection]:
        if self._session is None or self._health is DetectorHealth.UNAVAILABLE:
            raise self._initialization_error
        input_tensor, scale = _preprocess(frame)
        try:
            outputs = self._session.run(["output"], {"images": input_tensor})
            if len(outputs) != 1:
                raise ValueError(f"expected exactly one model output, got {len(outputs)}")
            output = np.asarray(outputs[0])
            if output.shape != (1, 8400, 85):
                raise ValueError(f"expected output shape [1, 8400, 85], got {output.shape}")
            if not np.isfinite(output).all():
                raise ValueError("YOLOX output contains non-finite values")
            if np.any(output[:, :, 4:] < 0.0) or np.any(output[:, :, 4:] > 1.0):
                raise ValueError("YOLOX confidence values must be between 0.0 and 1.0")
            decoded = output.copy()
            decoded[:, :, :2] = (decoded[:, :, :2] + _GRIDS) * _EXPANDED_STRIDES
            with np.errstate(over="ignore", invalid="ignore"):
                decoded[:, :, 2:4] = np.exp(decoded[:, :, 2:4]) * _EXPANDED_STRIDES
            if not np.isfinite(decoded[:, :, :4]).all():
                raise ValueError("YOLOX decoded geometry contains non-finite values")
            candidates = decoded[0, :, 0:4]
            model_xyxy = np.column_stack(
                (
                    candidates[:, 0] - candidates[:, 2] / 2,
                    candidates[:, 1] - candidates[:, 3] / 2,
                    candidates[:, 0] + candidates[:, 2] / 2,
                    candidates[:, 1] + candidates[:, 3] / 2,
                )
            )
            if (
                not np.isfinite(model_xyxy).all()
                or np.any(candidates[:, 2:4] <= 0.0)
                or np.any(model_xyxy[:, 2:] <= model_xyxy[:, :2])
            ):
                raise ValueError("YOLOX output contains an invalid decoded box")
            scores = decoded[0, :, 4] * decoded[0, :, 5 + PERSON_CLASS_ID]
            keep = scores >= self._confidence_threshold
            model_xyxy = model_xyxy[keep]
            scores = scores[keep]
            if not len(model_xyxy):
                return []
            selected = _nms(model_xyxy, scores, self._nms_threshold)
            xyxy = model_xyxy[selected] / scale
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, frame.width)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, frame.height)
            if not np.isfinite(xyxy).all() or np.any(xyxy[:, 2:] <= xyxy[:, :2]):
                raise ValueError("YOLOX output contains an invalid decoded box")
            return [PersonDetection(tuple(float(value) for value in box), float(scores[index])) for box, index in zip(xyxy, selected)]
        except PersonDetectorError:
            raise
        except Exception as error:
            raise DetectorInferenceError("YOLOX inference or postprocessing failed") from error
