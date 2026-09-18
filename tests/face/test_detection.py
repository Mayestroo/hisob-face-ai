from datetime import datetime, timezone
from dataclasses import fields
import inspect

import numpy as np
import pytest

from app.face import (
    DetectorHealth,
    FaceDetectorInferenceError,
    FaceDetectorSettings,
    FaceDetectorUnavailableError,
    PersonRoiObservation,
    YuNetFaceDetector,
    preprocess_roi,
)
from app.face.detection import _decode, _restore_model_geometry


TEST_OUTPUT_CONTRACT = (
    ("cls_8", "tensor(float)", [1, "anchors", 1]),
    ("cls_16", "tensor(float)", [1, "anchors", 1]),
    ("cls_32", "tensor(float)", [1, "anchors", 1]),
    ("obj_8", "tensor(float)", [1, "anchors", 1]),
    ("obj_16", "tensor(float)", [1, "anchors", 1]),
    ("obj_32", "tensor(float)", [1, "anchors", 1]),
    ("bbox_8", "tensor(float)", [1, "anchors", 4]),
    ("bbox_16", "tensor(float)", [1, "anchors", 4]),
    ("bbox_32", "tensor(float)", [1, "anchors", 4]),
    ("kps_8", "tensor(float)", [1, "anchors", 10]),
    ("kps_16", "tensor(float)", [1, "anchors", 10]),
    ("kps_32", "tensor(float)", [1, "anchors", 10]),
)


class Meta:
    def __init__(self, name: str, type_: str = "tensor(float)", shape=None) -> None:
        self.name = name
        self.type = type_
        self.shape = [1, 3, "height", "width"] if shape is None and name == "input" else shape


class FakeSession:
    def __init__(self, providers=("CPUExecutionProvider",), outputs=None, input_meta=None, output_meta=None) -> None:
        self.providers = providers
        self.outputs = outputs
        self.input_meta = input_meta or Meta("input")
        self.output_meta = output_meta or [Meta(name, type_, shape) for name, type_, shape in TEST_OUTPUT_CONTRACT]
        self.requested = None
        self.feed = None
        self.run_count = 0

    def get_providers(self):
        return list(self.providers)

    def get_inputs(self):
        return self.input_meta if isinstance(self.input_meta, list) else [self.input_meta]

    def get_outputs(self):
        return self.output_meta

    def run(self, names, feed):
        self.run_count += 1
        self.requested = names
        self.feed = feed
        return self.outputs if self.outputs is not None else zero_outputs(32, 32)


def zero_outputs(height: int, width: int) -> list[np.ndarray]:
    return [np.zeros((1, (height // stride) * (width // stride), last), dtype=np.float32)
            for last in (1, 1, 4, 10) for stride in (8, 16, 32)]


def observation(width=3, height=2) -> PersonRoiObservation:
    return PersonRoiObservation(
        "CAMERA_A", 4, 2, datetime(2026, 1, 1, tzinfo=timezone.utc),
        (10.0, 20.0, 13.0, 22.0), 0.8, (10, 20, 10 + width, 20 + height),
        bytes(index % 256 for index in range(width * height)), width, height, "gray",
    )


def detector_for(session: FakeSession) -> YuNetFaceDetector:
    return YuNetFaceDetector(FaceDetectorSettings(), session_factory=lambda path, providers: session)


def test_preprocess_replicates_gray_without_normalization_or_upscale() -> None:
    prepared = preprocess_roi(observation(), 640)
    tensor = prepared.tensor
    assert prepared.scale_x == prepared.scale_y == 1.0
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor[0, 0, 0, 0] == 0
    assert tensor[0, 1, 0, 1] == 1
    assert np.array_equal(tensor[0, 0], tensor[0, 1])
    assert tensor.max() == 5


def test_cpu_health_uses_actual_session_providers() -> None:
    detector = detector_for(FakeSession())
    assert detector.health is DetectorHealth.AVAILABLE_CPU_DEGRADED


def test_cuda_health_uses_actual_session_providers() -> None:
    detector = detector_for(FakeSession(("CUDAExecutionProvider", "CPUExecutionProvider")))
    assert detector.health is DetectorHealth.AVAILABLE_GPU


def test_unusable_provider_is_unavailable() -> None:
    detector = detector_for(FakeSession(("TensorrtExecutionProvider",)))
    with pytest.raises(FaceDetectorUnavailableError):
        detector.detect(observation())


def test_zero_face_is_immutable_empty_tuple() -> None:
    session = FakeSession()
    assert detector_for(session).detect(observation()) == ()
    assert session.requested == [name for name, _, _ in TEST_OUTPUT_CONTRACT]
    assert tuple(session.feed) == ("input",)


@pytest.mark.parametrize("outputs", [zero_outputs(32, 32)[:-1], [np.full((1, 16, 1), np.nan, dtype=np.float32)] + zero_outputs(32, 32)[1:]])
def test_malformed_outputs_are_inference_errors(outputs) -> None:
    detector = detector_for(FakeSession(outputs=outputs))
    with pytest.raises(FaceDetectorInferenceError):
        detector.detect(observation())


def test_integer_quantization_preserves_independent_restore_scales() -> None:
    prepared = preprocess_roi(observation(width=1001, height=701), 640)
    assert (prepared.resized_width, prepared.resized_height) == (640, 448)
    assert prepared.scale_x == 640 / 1001
    assert prepared.scale_y == 448 / 701
    box, landmarks = _restore_model_geometry(
        np.array([64.0, 89.6, 320.0, 268.8]),
        np.array([[64.0, 89.6], [320.0, 268.8]]),
        prepared,
    )
    assert np.allclose(box, [100.1, 140.2, 500.5, 420.6])
    assert np.allclose(landmarks, [[100.1, 140.2], [500.5, 420.6]])


def test_restoration_uses_literal_scales_and_source_translation() -> None:
    # 1001 x 701 at max side 640 truncates independently to 640 x 448.
    scale_x = 640.0 / 1001.0
    scale_y = 448.0 / 701.0
    model_box = np.array([64.0, 89.6, 320.0, 268.8])
    model_landmarks = np.array([[64.0, 89.6], [320.0, 268.8]])
    restored_box, restored_landmarks = _restore_model_geometry(
        model_box, model_landmarks,
        type("LiteralGeometry", (), {"scale_x": scale_x, "scale_y": scale_y})(),
    )
    # 64 / (640 / 1001) = 100.1; 89.6 / (448 / 701) = 140.2.
    assert np.allclose(restored_box, [100.1, 140.2, 500.5, 420.6])
    assert np.allclose(restored_landmarks, [[100.1, 140.2], [500.5, 420.6]])
    roi_origin = (37, 53)
    assert np.allclose(restored_box[[0, 2]] + roi_origin[0], [137.1, 537.5])
    assert np.allclose(restored_box[[1, 3]] + roi_origin[1], [193.2, 473.6])


def test_detector_returns_independently_restored_roi_and_source_geometry() -> None:
    roi_width, roi_height = 1001, 701
    source_origin = (37, 53)
    final_dimensions = (1, 1, 1, 1, 1, 1, 4, 4, 4, 10, 10, 10)
    # Each output uses its own stride grid; only one stride-8 anchor is a face.
    session_outputs = [
        np.zeros((1, (640 // stride) * (448 // stride), last), dtype=np.float32)
        for (_, _, _), stride, last in zip(
            TEST_OUTPUT_CONTRACT,
            (8, 16, 32, 8, 16, 32, 8, 16, 32, 8, 16, 32),
            final_dimensions,
        )
    ]
    anchor = 20 * (640 // 8) + 30
    session_outputs[0][0, anchor, 0] = 1.0
    session_outputs[3][0, anchor, 0] = 1.0
    session_outputs[6][0, anchor] = [5.0, 10.0, np.log(30.0), np.log(30.0)]
    model_landmarks = np.array(
        [[200.0, 160.0], [360.0, 160.0], [280.0, 240.0], [220.0, 320.0], [340.0, 320.0]],
        dtype=np.float32,
    )
    session_outputs[9][0, anchor] = ((model_landmarks - [30 * 8, 20 * 8]) / 8.0).reshape(-1)
    observation_value = PersonRoiObservation(
        "CAMERA_A", 4, 2, datetime(2026, 1, 1, tzinfo=timezone.utc),
        (37.0, 53.0, 1038.0, 754.0), 0.8,
        (source_origin[0], source_origin[1], source_origin[0] + roi_width, source_origin[1] + roi_height),
        bytes(roi_width * roi_height), roi_width, roi_height, "gray",
    )
    detector = detector_for(FakeSession(outputs=session_outputs))

    detected = detector.detect(observation_value)

    assert len(detected) == 1
    face = detected[0]
    scale_x = 640.0 / 1001.0
    scale_y = 448.0 / 701.0
    expected_box_roi = tuple(value / scale for value, scale in zip((160.0, 120.0, 400.0, 360.0), (scale_x, scale_y, scale_x, scale_y)))
    expected_landmarks_roi = tuple(
        (float(point[0]) / scale_x, float(point[1]) / scale_y) for point in model_landmarks
    )
    expected_box_source = (
        expected_box_roi[0] + source_origin[0], expected_box_roi[1] + source_origin[1],
        expected_box_roi[2] + source_origin[0], expected_box_roi[3] + source_origin[1],
    )
    expected_landmarks_source = tuple(
        (point[0] + source_origin[0], point[1] + source_origin[1]) for point in expected_landmarks_roi
    )
    assert face.face_bbox_roi == pytest.approx(expected_box_roi)
    assert face.face_bbox_source == pytest.approx(expected_box_source)
    assert np.allclose(face.landmarks_roi, expected_landmarks_roi)
    assert np.allclose(face.landmarks_source, expected_landmarks_source)
    assert scale_x != scale_y


@pytest.mark.parametrize(
    "metadata",
    [
        ("wrong", "tensor(float)", [1, 3, "height", "width"]),
        ("input", "tensor(double)", [1, 3, "height", "width"]),
        ("input", "tensor(float)", [1, 3, "height"]),
        ("input", "tensor(float)", [2, 3, "height", "width"]),
        ("input", "tensor(float)", [1, 1, "height", "width"]),
        ("input", "tensor(float)", [1, 3, "h", "width"]),
    ],
)
def test_invalid_input_metadata_makes_injected_detector_unavailable(metadata) -> None:
    detector = detector_for(FakeSession(input_meta=Meta(*metadata)))
    with pytest.raises(FaceDetectorUnavailableError):
        detector.detect(observation())


def test_multiple_inputs_and_missing_metadata_are_rejected() -> None:
    sessions = (
        FakeSession(input_meta=[Meta("input"), Meta("extra", shape=[1])]),
        FakeSession(input_meta=[object()]),
    )
    for session in sessions:
        with pytest.raises(FaceDetectorUnavailableError):
            detector_for(session).detect(observation())


@pytest.mark.parametrize("mutation", [
    lambda meta: meta[:-1],
    lambda meta: [meta[1], meta[0], *meta[2:]],
    lambda meta: [Meta("wrong", shape=meta[0].shape), *meta[1:]],
    lambda meta: [Meta(meta[0].name, "tensor(double)", meta[0].shape), *meta[1:]],
    lambda meta: [Meta(meta[0].name, shape=[1, "anchors", 1, 1]), *meta[1:]],
    lambda meta: [Meta(meta[0].name, shape=[2, "anchors", 1]), *meta[1:]],
    lambda meta: [Meta(meta[0].name, shape=[1, "anchors", 2]), *meta[1:]],
    lambda meta: [*meta[:3], Meta(meta[3].name, shape=[1, "anchors", 2]), *meta[4:]],
    lambda meta: [*meta[:6], Meta(meta[6].name, shape=[1, "anchors", 3]), *meta[7:]],
    lambda meta: [*meta[:9], Meta(meta[9].name, shape=[1, "anchors", 8]), *meta[10:]],
    lambda meta: [type("MissingName", (), {"type": "tensor(float)", "shape": [1, "anchors", 1]})(), *meta[1:]],
    lambda meta: [type("MissingType", (), {"name": meta[0].name, "shape": meta[0].shape})(), *meta[1:]],
    lambda meta: [type("MissingShape", (), {"name": meta[0].name, "type": "tensor(float)"})(), *meta[1:]],
])
def test_output_metadata_contract_rejects_each_literal_mutation(mutation) -> None:
    valid = [Meta(name, type_, shape) for name, type_, shape in TEST_OUTPUT_CONTRACT]
    session = FakeSession(output_meta=mutation(valid))
    detector = detector_for(session)
    assert detector.health is DetectorHealth.UNAVAILABLE
    with pytest.raises(FaceDetectorUnavailableError):
        detector.detect(observation())
    assert session.run_count == 0


def test_literal_output_metadata_contract_is_accepted() -> None:
    session = FakeSession(output_meta=[Meta(name, type_, shape) for name, type_, shape in TEST_OUTPUT_CONTRACT])
    detector = detector_for(session)
    assert detector.health is DetectorHealth.AVAILABLE_CPU_DEGRADED


def test_decode_uses_literal_grid_stride_and_offsets() -> None:
    outputs = {}
    for stride in (8, 16, 32):
        anchors = (32 // stride) * (32 // stride)
        outputs[f"cls_{stride}"] = np.zeros((anchors, 1))
        outputs[f"obj_{stride}"] = np.zeros((anchors, 1))
        outputs[f"bbox_{stride}"] = np.zeros((anchors, 4))
        outputs[f"kps_{stride}"] = np.zeros((anchors, 10))
    outputs["cls_8"][6, 0] = 1.0
    outputs["obj_8"][6, 0] = 1.0
    outputs["bbox_8"][6] = [0.25, -0.5, np.log(2.0), np.log(1.0)]
    outputs["kps_8"][6, :2] = [0.5, -0.25]
    boxes, landmarks, scores = _decode(outputs, 32, 32)
    assert np.allclose(boxes[6], [10.0, 0.0, 26.0, 8.0])
    assert np.allclose(landmarks[6, 0], [20.0, 6.0])
    assert scores[6] == 1.0


def test_production_integrity_gate_precedes_session_creation(monkeypatch, tmp_path) -> None:
    from app.face import detection

    approved_root = tmp_path / "project"
    approved_root.mkdir()
    model_dir = approved_root / "models"
    model_dir.mkdir()
    (model_dir / "face_detection_yunet_2026may.onnx").write_bytes(b"wrong")
    monkeypatch.setattr(detection, "_project_root", lambda: approved_root)
    called = []

    def factory(path, providers):
        called.append(path)
        return FakeSession()

    detector = YuNetFaceDetector(FaceDetectorSettings(), session_factory=factory)
    assert detector.health is DetectorHealth.UNAVAILABLE
    assert called == []


def test_missing_approved_model_is_unavailable_before_session_creation(monkeypatch, tmp_path) -> None:
    from app.face import detection

    monkeypatch.setattr(detection, "_project_root", lambda: tmp_path)
    called = []
    detector = YuNetFaceDetector(
        FaceDetectorSettings(),
        session_factory=lambda path, providers: called.append(path),
    )
    assert detector.health is DetectorHealth.UNAVAILABLE
    assert called == []


def test_approved_size_is_enforced_before_session_creation(monkeypatch, tmp_path) -> None:
    from app.face import detection
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "face_detection_yunet_2026may.onnx").write_bytes(b"short")
    called = []
    monkeypatch.setattr(detection, "_project_root", lambda: tmp_path)
    detector = YuNetFaceDetector(FaceDetectorSettings(), session_factory=lambda path, providers: called.append(path))
    assert detector.health is DetectorHealth.UNAVAILABLE
    assert called == []


def test_hash_mismatch_after_matching_size_is_unavailable_before_session_creation(monkeypatch, tmp_path) -> None:
    from app.face import detection
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "face_detection_yunet_2026may.onnx").write_bytes(b"x" * 229738)
    called = []
    monkeypatch.setattr(detection, "_project_root", lambda: tmp_path)
    detector = YuNetFaceDetector(FaceDetectorSettings(), session_factory=lambda path, providers: called.append(path))
    assert detector.health is DetectorHealth.UNAVAILABLE
    assert called == []


def test_approved_artifact_is_verified_before_factory(monkeypatch) -> None:
    from app.face import detection
    called = []
    detector = YuNetFaceDetector(
        FaceDetectorSettings(),
        session_factory=lambda path, providers: called.append((path, tuple(providers))) or FakeSession(),
    )
    assert detector.health is DetectorHealth.AVAILABLE_CPU_DEGRADED
    assert called == [(str(detection._approved_model_path()), detection.REQUESTED_PROVIDERS)]


def test_public_constructor_has_no_session_parameter() -> None:
    parameters = inspect.signature(YuNetFaceDetector).parameters
    assert "session" not in parameters
    assert "model_path" not in parameters
    assert "expected_sha" not in parameters
    assert "expected_size" not in parameters


def test_approved_artifact_identity_is_pinned() -> None:
    from app.face import detection
    assert detection._APPROVED_MODEL_RELATIVE_PATH.as_posix() == "models/face_detection_yunet_2026may.onnx"
    assert detection._APPROVED_MODEL_SIZE == 229738
    assert detection._APPROVED_MODEL_SHA256 == "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


def test_production_settings_have_no_artifact_override() -> None:
    settings = FaceDetectorSettings()
    setting_names = {field.name for field in fields(FaceDetectorSettings)}
    assert "model_path" not in setting_names
    assert "expected_sha" not in setting_names
    assert "expected_size" not in setting_names
    assert not hasattr(settings, "model_identifier")
    assert not hasattr(settings, "model_sha256")
    for name in ("model_path", "expected_sha", "expected_size"):
        with pytest.raises(TypeError):
            FaceDetectorSettings(**{name: "override"})
