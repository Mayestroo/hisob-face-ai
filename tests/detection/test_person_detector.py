from datetime import datetime, timezone
import hashlib
import sys

import numpy as np
import pytest

from app.cameras.video_file import DecodedVideoFrame
from app.config.settings import ModelRuntimeSettings
from app.detection.person_detector import (
    DetectorHealth,
    DetectorInferenceError,
    DetectorUnavailableError,
    YoloxPersonDetector,
    _resize_linear_uint8,
)
from app.domain.models import CameraFrame, CameraRole


def make_frame(width: int = 640, height: int = 640, role: CameraRole = CameraRole.ENTRY, camera_id: str = "any-camera") -> DecodedVideoFrame:
    return DecodedVideoFrame(
        frame=CameraFrame(camera_id, role, datetime(2026, 1, 1, tzinfo=timezone.utc)),
        pixels=bytes(index % 256 for index in range(width * height)),
        width=width,
        height=height,
        frame_index=0,
    )


class FakeSession:
    def __init__(self, output: np.ndarray | list[np.ndarray], providers: list[str] | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.providers = ["CPUExecutionProvider"] if providers is None else providers
        self.error = error
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_providers(self) -> list[str]:
        return self.providers

    def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append((names, inputs))
        if self.error is not None:
            raise self.error
        return self.output if isinstance(self.output, list) else [self.output]


def make_output(*detections: tuple[int, float, float, float, float, float]) -> np.ndarray:
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    for index, cx, cy, width, height, objectness in detections:
        stride = 8
        grid_x = index % 80
        grid_y = index // 80
        output[0, index, 0] = cx / stride - grid_x
        output[0, index, 1] = cy / stride - grid_y
        output[0, index, 2] = np.log(width / stride)
        output[0, index, 3] = np.log(height / stride)
        output[0, index, 4] = objectness
        output[0, index, 5] = 0.8
        output[0, index, 6] = 0.9
    return output


def detector(output: np.ndarray, **kwargs: object) -> tuple[YoloxPersonDetector, FakeSession]:
    session = FakeSession(output, providers=kwargs.pop("providers", None))
    settings = ModelRuntimeSettings("unused.onnx", **kwargs)
    return YoloxPersonDetector(settings, session=session), session


def test_person_confidence_and_source_bbox_are_converted() -> None:
    detector_instance, _ = detector(make_output((0, 0.5, 0.5, 1, 1, 0.75)))

    detections = detector_instance.detect(make_frame())

    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert detections[0].confidence == pytest.approx(0.6)


def test_multiple_people_and_non_person_classes() -> None:
    output = make_output((0, 4, 2, 1, 1, 0.9), (1, 12, 2, 1, 1, 0.9))
    output[0, 1, 5] = 0.95
    output[0, 1, 6] = 0.1
    detector_instance, _ = detector(output)

    detections = detector_instance.detect(make_frame())

    assert len(detections) == 2


def test_non_person_class_is_filtered() -> None:
    output = make_output((0, 4, 2, 1, 1, 0.9), (1, 12, 2, 1, 1, 0.9))
    output[0, 1, 5] = 0.95
    output[0, 1, 6] = 0.1
    output[0, 0, 5] = 0.1
    output[0, 0, 6] = 0.9
    detector_instance, _ = detector(output)

    detections = detector_instance.detect(make_frame())

    assert len(detections) == 1


def test_zero_person_success_is_empty() -> None:
    detector_instance, _ = detector(np.zeros((1, 8400, 85), dtype=np.float32))

    assert detector_instance.detect(make_frame()) == []


def test_confidence_threshold_uses_objectness_times_person_score() -> None:
    detector_instance, _ = detector(make_output((0, 4, 2, 1, 1, 0.75)), confidence_threshold=0.61)

    assert detector_instance.detect(make_frame()) == []


def test_confidence_threshold_includes_exact_boundary() -> None:
    detector_instance, _ = detector(make_output((0, 4, 2, 1, 1, 0.75)), confidence_threshold=0.6)

    assert len(detector_instance.detect(make_frame())) == 1


def test_invalid_confidence_value_fails_explicitly() -> None:
    output = make_output((0, 4, 2, 1, 1, 0.9))
    output[0, 0, 5] = 1.1
    detector_instance, _ = detector(output)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


def test_preprocessing_is_float32_nchw_rgb_letterbox_without_normalization() -> None:
    detector_instance, session = detector(np.zeros((1, 8400, 85), dtype=np.float32))

    detector_instance.detect(make_frame(width=4, height=2))

    tensor = session.calls[0][1]["images"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert np.array_equal(tensor[0, 0], tensor[0, 1])
    assert np.array_equal(tensor[0, 1], tensor[0, 2])
    assert tensor[0, 0, -1, -1] == 114.0
    assert tensor[0, 0, 100, 100] != 0.0


@pytest.mark.parametrize(
    ("source", "shape", "expected"),
    [
        (
            np.array([[0, 40, 80, 120], [20, 60, 100, 140], [40, 80, 120, 160], [60, 100, 140, 180]], dtype=np.uint8),
            (2, 2),
            np.array([[30, 110], [70, 150]], dtype=np.uint8),
        ),
        (
            np.array([[0, 100], [50, 150]], dtype=np.uint8),
            (3, 4),
            np.array([[0, 25, 75, 100], [25, 50, 100, 125], [50, 75, 125, 150]], dtype=np.uint8),
        ),
        (
            np.array([[0, 10, 20, 30, 40], [50, 60, 70, 80, 90], [100, 110, 120, 130, 140]], dtype=np.uint8),
            (4, 2),
            np.array([[8, 33], [39, 64], [76, 101], [108, 133]], dtype=np.uint8),
        ),
    ],
)
def test_uint8_resize_matches_independent_opencv_reference_vectors(source: np.ndarray, shape: tuple[int, int], expected: np.ndarray) -> None:
    assert np.array_equal(_resize_linear_uint8(source, *shape), expected)


def test_letterbox_is_deterministic_and_restores_coordinates() -> None:
    output = make_output((0, 320, 160, 100, 100, 0.9))
    first, first_session = detector(output)
    second, second_session = detector(output)

    first_result = first.detect(make_frame(width=4, height=2))
    second_result = second.detect(make_frame(width=4, height=2))

    assert np.array_equal(first_session.calls[0][1]["images"], second_session.calls[0][1]["images"])
    assert first_result[0].bbox == second_result[0].bbox


def test_portrait_source_coordinates_are_restored() -> None:
    detector_instance, _ = detector(make_output((0, 160, 320, 160, 320, 0.9)))

    result = detector_instance.detect(make_frame(width=2, height=4))

    assert result[0].bbox == pytest.approx((0.5, 1.0, 1.5, 3.0))


def test_nms_suppresses_overlapping_person_and_keeps_separated_person() -> None:
    output = make_output(
        (0, 4, 2, 2, 2, 0.9),
        (1, 4.2, 2.1, 2, 2, 0.8),
        (2, 12, 2, 2, 2, 0.7),
    )
    detector_instance, _ = detector(output)

    detections = detector_instance.detect(make_frame())

    assert len(detections) == 2
    assert detections[0].confidence == pytest.approx(0.72)


def test_nms_runs_before_source_boundary_clipping() -> None:
    output = make_output(
        (0, 25, 50, 70, 100, 0.9),
        (1, 35, 70, 70, 100, 0.8),
    )
    detector_instance, _ = detector(output, nms_threshold=0.6)

    detections = detector_instance.detect(make_frame(width=640, height=100))

    assert len(detections) == 2
    assert detections[0].bbox == pytest.approx((0.0, 0.0, 60.0, 100.0))
    assert detections[1].bbox == pytest.approx((0.0, 20.0, 70.0, 100.0))


@pytest.mark.parametrize("output", [np.zeros((1, 10, 85), dtype=np.float32), np.zeros((8400, 85), dtype=np.float32)])
def test_malformed_output_shape_fails(output: np.ndarray) -> None:
    detector_instance, _ = detector(output)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_output_fails(value: float) -> None:
    output = make_output((0, 4, 2, 1, 1, 0.9))
    output[0, 0, 0] = value
    detector_instance, _ = detector(output)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


@pytest.mark.parametrize("value", [1000.0, -1000.0])
@pytest.mark.parametrize("objectness", [0.9, 0.0])
def test_extreme_width_height_logits_fail_explicitly(value: float, objectness: float) -> None:
    output = make_output((0, 4, 2, 1, 1, objectness))
    output[0, 0, 2] = value
    detector_instance, _ = detector(output)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


def test_decoded_box_empty_after_boundary_clipping_fails() -> None:
    output = make_output((0, -10, -10, 1, 1, 0.9))
    detector_instance, _ = detector(output)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


def test_boundary_overflow_is_clipped() -> None:
    detector_instance, _ = detector(make_output((0, 320, 320, 1000, 1000, 0.9)))

    result = detector_instance.detect(make_frame())

    assert result[0].bbox == pytest.approx((0.0, 0.0, 640.0, 640.0))


def test_missing_model_is_explicitly_unavailable(tmp_path) -> None:
    detector_instance = YoloxPersonDetector(ModelRuntimeSettings(str(tmp_path / "missing.onnx")))

    assert detector_instance.health is DetectorHealth.UNAVAILABLE
    with pytest.raises(DetectorUnavailableError):
        detector_instance.detect(make_frame())


def test_model_hash_and_session_creation_are_explicit(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    captured: dict[str, object] = {}

    def factory(path: str, providers: list[str]) -> FakeSession:
        captured["path"] = path
        captured["providers"] = providers
        return FakeSession(np.zeros((1, 8400, 85), dtype=np.float32), providers=providers)

    settings = ModelRuntimeSettings(str(model), model_sha256=digest)
    detector_instance = YoloxPersonDetector(settings, session_factory=factory)

    assert detector_instance.health is DetectorHealth.AVAILABLE_GPU
    assert captured["providers"] == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_wrong_model_hash_is_unavailable(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    detector_instance = YoloxPersonDetector(ModelRuntimeSettings(str(model), model_sha256="0" * 64))

    assert detector_instance.health is DetectorHealth.UNAVAILABLE
    with pytest.raises(DetectorUnavailableError):
        detector_instance.detect(make_frame())


def test_session_creation_failure_is_explicit(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()

    def factory(path: str, providers: list[str]) -> FakeSession:
        raise RuntimeError("session failed")

    detector_instance = YoloxPersonDetector(ModelRuntimeSettings(str(model), model_sha256=digest), session_factory=factory)

    assert detector_instance.health is DetectorHealth.UNAVAILABLE
    with pytest.raises(DetectorUnavailableError):
        detector_instance.detect(make_frame())


def test_inference_failure_is_explicit_and_not_empty() -> None:
    session = FakeSession(np.empty((1, 8400, 85), dtype=np.float32), error=RuntimeError("inference failed"))
    detector_instance = YoloxPersonDetector(ModelRuntimeSettings("unused.onnx"), session=session)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


@pytest.mark.parametrize("outputs", [[], [np.zeros((1, 8400, 85), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)]])
def test_session_output_arity_must_be_exactly_one(outputs: list[np.ndarray]) -> None:
    session = FakeSession(outputs)
    detector_instance = YoloxPersonDetector(ModelRuntimeSettings("unused.onnx"), session=session)

    with pytest.raises(DetectorInferenceError):
        detector_instance.detect(make_frame())


def test_single_session_output_continues() -> None:
    detector_instance, _ = detector(np.zeros((1, 8400, 85), dtype=np.float32))

    assert detector_instance.detect(make_frame()) == []


def test_cpu_fallback_is_degraded_and_gpu_is_healthy() -> None:
    cpu, _ = detector(np.zeros((1, 8400, 85), dtype=np.float32), providers=["CPUExecutionProvider"])
    gpu, _ = detector(np.zeros((1, 8400, 85), dtype=np.float32), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert cpu.health is DetectorHealth.AVAILABLE_CPU_DEGRADED
    assert gpu.health is DetectorHealth.AVAILABLE_GPU
    assert cpu.active_provider == "CPUExecutionProvider"
    assert gpu.active_provider == "CUDAExecutionProvider"


@pytest.mark.parametrize("providers", [[], ["TensorRTExecutionProvider"], ["UnknownExecutionProvider"]])
def test_unsupported_provider_health_is_unavailable(providers: list[str]) -> None:
    detector_instance, _ = detector(np.zeros((1, 8400, 85), dtype=np.float32), providers=providers)

    assert detector_instance.health is DetectorHealth.UNAVAILABLE
    with pytest.raises(DetectorUnavailableError):
        detector_instance.detect(make_frame())


def test_cuda_provider_is_healthy_even_when_not_first() -> None:
    detector_instance, _ = detector(
        np.zeros((1, 8400, 85), dtype=np.float32),
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    assert detector_instance.health is DetectorHealth.AVAILABLE_GPU


def test_camera_metadata_does_not_affect_result_or_mutate_frame() -> None:
    output = make_output((0, 4, 2, 1, 1, 0.9))
    detector_instance, _ = detector(output)
    entry = make_frame(role=CameraRole.ENTRY, camera_id="entry")
    exit_frame = make_frame(role=CameraRole.EXIT, camera_id="exit")

    entry_pixels = entry.pixels
    assert detector_instance.detect(entry) == detector_instance.detect(exit_frame)
    assert entry.pixels == entry_pixels


def test_backend_values_do_not_escape_adapter() -> None:
    detector_instance, _ = detector(make_output((0, 4, 2, 1, 1, 0.9)))

    result = detector_instance.detect(make_frame())

    assert type(result) is list
    assert type(result[0]).__name__ == "PersonDetection"
    assert not isinstance(result[0].bbox, np.ndarray)


def test_onnxruntime_is_not_loaded_at_module_import() -> None:
    assert "onnxruntime" not in sys.modules
