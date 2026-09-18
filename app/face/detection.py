"""YuNet ONNX face detection and alignment adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app.face.alignment import AlignmentSpec, FaceAlignmentError, YuNetLandmark, align_grayscale_face
from app.face.roi import PersonRoiObservation


_APPROVED_MODEL_RELATIVE_PATH = Path("models/face_detection_yunet_2026may.onnx")
_APPROVED_MODEL_SIZE = 229738
_APPROVED_MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
REQUESTED_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
OUTPUT_NAMES = tuple(
    f"{kind}_{stride}" for kind in ("cls", "obj", "bbox", "kps") for stride in (8, 16, 32)
)
STRIDES = (8, 16, 32)


class DetectorHealth(str, Enum):
    AVAILABLE_GPU = "AVAILABLE_GPU"
    AVAILABLE_CPU_DEGRADED = "AVAILABLE_CPU_DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class FaceDetectorError(RuntimeError):
    """Base YuNet adapter error."""


class FaceDetectorUnavailableError(FaceDetectorError):
    """The approved model or a usable inference session is unavailable."""


class FaceDetectorInferenceError(FaceDetectorError):
    """Inference, output validation, or decode failed."""


FaceSessionFactory = Callable[[str, Sequence[str]], Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _approved_model_path() -> Path:
    return _project_root() / _APPROVED_MODEL_RELATIVE_PATH


def _verify_approved_model(path: Path) -> None:
    if not path.is_file():
        raise FaceDetectorUnavailableError(f"model file does not exist: {path}")
    if path.stat().st_size != _APPROVED_MODEL_SIZE:
        raise FaceDetectorUnavailableError("model size does not match the approved artifact")
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != _APPROVED_MODEL_SHA256:
        raise FaceDetectorUnavailableError("model SHA-256 does not match the approved artifact")


@dataclass(frozen=True, slots=True)
class PreprocessedRoi:
    tensor: np.ndarray
    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float


def _restore_model_geometry(
    box: np.ndarray, landmarks: np.ndarray, geometry: PreprocessedRoi
) -> tuple[np.ndarray, np.ndarray]:
    restored_box = box.copy()
    restored_box[[0, 2]] /= geometry.scale_x
    restored_box[[1, 3]] /= geometry.scale_y
    restored_landmarks = landmarks.copy()
    restored_landmarks[..., 0] /= geometry.scale_x
    restored_landmarks[..., 1] /= geometry.scale_y
    return restored_box, restored_landmarks


@dataclass(frozen=True, slots=True)
class FaceDetectorSettings:
    score_threshold: float = 0.9
    nms_iou_threshold: float = 0.3
    top_k: int = 5000
    max_long_side: int = 640
    alignment: AlignmentSpec = AlignmentSpec()

    def __post_init__(self) -> None:
        for value, name in ((self.score_threshold, "score_threshold"), (self.nms_iou_threshold, "nms_iou_threshold")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between 0.0 and 1.0")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if isinstance(self.max_long_side, bool) or not isinstance(self.max_long_side, int) or self.max_long_side <= 0:
            raise ValueError("max_long_side must be a positive integer")
        if not isinstance(self.alignment, AlignmentSpec):
            raise TypeError("alignment must be an AlignmentSpec")


@dataclass(frozen=True, slots=True)
class DetectedFaceObservation:
    """A face detected inside a person ROI, not an identity association."""

    camera_id: str
    track_id: int
    frame_index: int
    timestamp: datetime
    person_bbox: tuple[float, float, float, float]
    person_confidence: float
    roi_bbox: tuple[int, int, int, int]
    face_bbox_roi: tuple[float, float, float, float]
    face_bbox_source: tuple[float, float, float, float]
    face_confidence: float
    landmarks_roi: tuple[tuple[float, float], ...]
    landmarks_source: tuple[tuple[float, float], ...]
    aligned_face_pixels: bytes
    aligned_width: int
    aligned_height: int
    pixel_format: str
    alignment_version: str


def _default_session_factory(path: str, providers: Sequence[str]) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise FaceDetectorUnavailableError("onnxruntime is required for face detection") from error
    return ort.InferenceSession(path, providers=list(providers))


def _resize_linear_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape
    def coefficients(source_size: int, destination_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinate = ((np.arange(destination_size, dtype=np.float64) + 0.5) * source_size / destination_size - 0.5).astype(np.float32)
        index = np.floor(coordinate).astype(np.intp)
        fraction = coordinate - index
        outside = (index < 0) | (index >= source_size - 1)
        index = np.clip(index, 0, source_size - 1)
        fraction = np.where(outside, 0.0, fraction)
        first = np.rint((1.0 - fraction) * 2048.0).astype(np.int32)
        second = np.rint(fraction * 2048.0).astype(np.int32)
        return index, first, second
    x, x_first, x_second = coefficients(source_width, width)
    y, y_first, y_second = coefficients(source_height, height)
    horizontal = image[:, x].astype(np.int32) * x_first + image[:, np.minimum(x + 1, source_width - 1)].astype(np.int32) * x_second
    vertical = horizontal[y].astype(np.int64) * y_first[:, None] + horizontal[np.minimum(y + 1, source_height - 1)].astype(np.int64) * y_second[:, None]
    return np.clip((vertical + (1 << 21)) >> 22, 0, 255).astype(np.uint8)


def preprocess_roi(observation: PersonRoiObservation, max_long_side: int = 640) -> PreprocessedRoi:
    gray = np.frombuffer(observation.roi_pixels, dtype=np.uint8).reshape(observation.roi_height, observation.roi_width)
    scale = min(1.0, max_long_side / max(observation.roi_width, observation.roi_height))
    resized_height = max(1, int(observation.roi_height * scale))
    resized_width = max(1, int(observation.roi_width * scale))
    resized = _resize_linear_uint8(gray, resized_height, resized_width) if scale != 1.0 else gray.copy()
    padded_height = int(math.ceil(resized_height / 32) * 32)
    padded_width = int(math.ceil(resized_width / 32) * 32)
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    replicated = np.repeat(resized[:, :, None], 3, axis=2)
    padded[:resized_height, :resized_width] = replicated
    return PreprocessedRoi(
        np.ascontiguousarray(padded.transpose(2, 0, 1)[None]),
        resized_width,
        resized_height,
        resized_width / observation.roi_width,
        resized_height / observation.roi_height,
    )


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
        iou = intersection / np.maximum(area_current + area_remaining - intersection, np.finfo(np.float64).eps)
        order = remaining[iou <= threshold]
    return np.asarray(keep, dtype=np.intp)


def _validate_output_arrays(outputs: Sequence[Any], height: int, width: int) -> dict[str, np.ndarray]:
    if len(outputs) != 12:
        raise FaceDetectorInferenceError("YuNet must return exactly twelve outputs")
    anchors = {stride: (height // stride) * (width // stride) for stride in STRIDES}
    result: dict[str, np.ndarray] = {}
    for name, raw in zip(OUTPUT_NAMES, outputs):
        array = np.asarray(raw)
        kind, stride_text = name.split("_")
        stride = int(stride_text)
        last = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}[kind]
        if array.ndim != 3 or array.shape != (1, anchors[stride], last):
            raise FaceDetectorInferenceError(f"malformed YuNet output {name}: {array.shape}")
        if not np.isfinite(array).all():
            raise FaceDetectorInferenceError(f"YuNet output {name} contains non-finite values")
        result[name] = array[0].astype(np.float64, copy=False)
    return result


def _decode(outputs: dict[str, np.ndarray], height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for stride in STRIDES:
        grid_height, grid_width = height // stride, width // stride
        y, x = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing="ij")
        grid = np.column_stack((x.ravel(), y.ravel())).astype(np.float64)
        cls = np.clip(outputs[f"cls_{stride}"][:, 0], 0.0, 1.0)
        obj = np.clip(outputs[f"obj_{stride}"][:, 0], 0.0, 1.0)
        score = np.sqrt(cls * obj)
        bbox = outputs[f"bbox_{stride}"]
        try:
            with np.errstate(over="raise", invalid="raise"):
                center = (grid + bbox[:, :2]) * stride
                dimensions = np.exp(bbox[:, 2:4]) * stride
        except FloatingPointError as error:
            raise FaceDetectorInferenceError("YuNet bbox exponentiation overflowed") from error
        xyxy = np.column_stack((center - dimensions / 2.0, center + dimensions / 2.0))
        kps = (grid[:, None, :] + outputs[f"kps_{stride}"].reshape(-1, 5, 2)) * stride
        if not np.isfinite(xyxy).all() or not np.isfinite(kps).all() or np.any(dimensions <= 0.0) or np.any(xyxy[:, 2:] <= xyxy[:, :2]):
            raise FaceDetectorInferenceError("YuNet decoded geometry is invalid")
        boxes.append(xyxy)
        landmarks.append(kps)
        scores.append(score)
    return np.concatenate(boxes), np.concatenate(landmarks), np.concatenate(scores)


class YuNetFaceDetector:
    """Detect every valid face inside each supplied person ROI."""

    def __init__(self, settings: FaceDetectorSettings, session_factory: FaceSessionFactory | None = None) -> None:
        self.settings = settings
        self._session = None
        self._health = DetectorHealth.UNAVAILABLE
        self._initialization_error: FaceDetectorUnavailableError = FaceDetectorUnavailableError("detector is unavailable")
        try:
            path = _approved_model_path()
            _verify_approved_model(path)
            self._session = (session_factory or _default_session_factory)(str(path), REQUESTED_PROVIDERS)
            self._validate_session_contract()
            self._set_health(self._session)
        except FaceDetectorUnavailableError as error:
            self._initialization_error = error
        except Exception as error:
            self._initialization_error = FaceDetectorUnavailableError("could not create ONNX Runtime session")
            self._initialization_error.__cause__ = error

    def _set_health(self, session: Any) -> None:
        try:
            providers = tuple(session.get_providers())
        except Exception as error:
            self._initialization_error = FaceDetectorUnavailableError("could not determine session providers")
            self._initialization_error.__cause__ = error
            return
        if "CUDAExecutionProvider" in providers:
            self._health = DetectorHealth.AVAILABLE_GPU
        elif "CPUExecutionProvider" in providers:
            self._health = DetectorHealth.AVAILABLE_CPU_DEGRADED
        else:
            self._initialization_error = FaceDetectorUnavailableError("no usable ONNX Runtime provider")

    def _validate_session_contract(self) -> None:
        """Validate metadata when the injected/runtime session exposes it."""
        if self._session is None:
            raise FaceDetectorUnavailableError("detector session is missing")
        try:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
        except AttributeError as error:
            raise FaceDetectorUnavailableError("YuNet session metadata is incomplete") from error
        except Exception as error:
            raise FaceDetectorUnavailableError("could not inspect YuNet session contract") from error
        if len(inputs) != 1:
            raise FaceDetectorUnavailableError("YuNet must expose exactly one input")
        input_meta = inputs[0]
        required = object()
        input_name = getattr(input_meta, "name", required)
        input_type = getattr(input_meta, "type", required)
        input_shape = getattr(input_meta, "shape", required)
        if input_name is required or input_type is required or input_shape is required:
            raise FaceDetectorUnavailableError("YuNet input metadata is incomplete")
        if input_name != "input" or input_type != "tensor(float)" or input_shape != [1, 3, "height", "width"]:
            raise FaceDetectorUnavailableError("YuNet input metadata does not match the approved contract")
        if len(outputs) != len(OUTPUT_NAMES):
            raise FaceDetectorUnavailableError("YuNet must expose exactly twelve outputs")
        final_dimensions = (1, 1, 1, 1, 1, 1, 4, 4, 4, 10, 10, 10)
        for output, expected_name, expected_last in zip(outputs, OUTPUT_NAMES, final_dimensions):
            name = getattr(output, "name", required)
            output_type = getattr(output, "type", required)
            shape = getattr(output, "shape", required)
            if name is required or output_type is required or shape is required:
                raise FaceDetectorUnavailableError("YuNet output metadata is incomplete")
            if name != expected_name or output_type != "tensor(float)":
                raise FaceDetectorUnavailableError("YuNet output names, order, or types do not match the approved contract")
            if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or not isinstance(shape[1], str) or not shape[1] or shape[2] != expected_last:
                raise FaceDetectorUnavailableError(f"YuNet output metadata does not match the approved contract: {expected_name}")

    @property
    def health(self) -> DetectorHealth:
        return self._health

    @property
    def active_provider(self) -> str | None:
        providers = tuple(self._session.get_providers()) if self._session is not None else ()
        return providers[0] if providers else None

    def detect(self, observation: PersonRoiObservation) -> tuple[DetectedFaceObservation, ...]:
        if self._session is None or self._health is DetectorHealth.UNAVAILABLE:
            raise self._initialization_error
        prepared = preprocess_roi(observation, self.settings.max_long_side)
        tensor = prepared.tensor
        model_height, model_width = tensor.shape[2:]
        try:
            self._validate_session_contract()
            outputs = self._session.run(list(OUTPUT_NAMES), {"input": tensor})
            arrays = _validate_output_arrays(outputs, model_height, model_width)
            boxes, landmarks, scores = _decode(arrays, model_height, model_width)
            valid = scores >= self.settings.score_threshold
            boxes, landmarks, scores = boxes[valid], landmarks[valid], scores[valid]
            if not len(boxes):
                return ()
            order = np.argsort(-scores, kind="stable")[: self.settings.top_k]
            keep = _nms(boxes[order], scores[order], self.settings.nms_iou_threshold)
            candidates: list[DetectedFaceObservation] = []
            for selected in keep:
                index = int(order[selected])
                box, restored_landmarks = _restore_model_geometry(boxes[index], landmarks[index], prepared)
                box[[0, 2]] = np.clip(box[[0, 2]], 0.0, observation.roi_width)
                box[[1, 3]] = np.clip(box[[1, 3]], 0.0, observation.roi_height)
                if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                roi_landmarks = tuple(
                    (float(point[0]), float(point[1]))
                    for point in restored_landmarks
                )
                if not all(math.isfinite(value) for point in roi_landmarks for value in point):
                    continue
                source_box = tuple(float(value) for value in (box[0] + observation.roi_bbox[0], box[1] + observation.roi_bbox[1], box[2] + observation.roi_bbox[0], box[3] + observation.roi_bbox[1]))
                source_landmarks = tuple((x + observation.roi_bbox[0], y + observation.roi_bbox[1]) for x, y in roi_landmarks)
                try:
                    aligned = align_grayscale_face(observation.roi_pixels, observation.roi_width, observation.roi_height, roi_landmarks, self.settings.alignment)
                except FaceAlignmentError:
                    continue
                candidates.append(DetectedFaceObservation(
                    observation.camera_id, observation.track_id, observation.frame_index, observation.timestamp,
                    observation.person_bbox, observation.person_confidence, observation.roi_bbox,
                    tuple(float(value) for value in box), source_box, float(scores[index]), roi_landmarks,
                    source_landmarks, aligned, self.settings.alignment.width, self.settings.alignment.height,
                    "gray", self.settings.alignment.version,
                ))
            return tuple(candidates)
        except FaceDetectorError:
            raise
        except Exception as error:
            raise FaceDetectorInferenceError("YuNet inference or postprocessing failed") from error
