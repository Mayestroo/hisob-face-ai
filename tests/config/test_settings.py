import pytest

from app.config.settings import (
    ApplicationSettings,
    CameraConfig,
    ModelRuntimeSettings,
    StorageSettings,
)
from app.domain.models import CameraRole


def make_model() -> ModelRuntimeSettings:
    return ModelRuntimeSettings("face-model.onnx", device="cpu", detection_interval_seconds=0.5)


def make_storage() -> StorageSettings:
    return StorageSettings("./data", "face-ai.sqlite3")


def test_valid_entry_camera_configuration() -> None:
    camera = CameraConfig("CAMERA_A", CameraRole.ENTRY, "rtsp://entry-camera/stream")

    assert camera.camera_id == "CAMERA_A"
    assert camera.role is CameraRole.ENTRY


def test_valid_exit_camera_configuration() -> None:
    camera = CameraConfig("CAMERA_B", CameraRole.EXIT, "file:///cameras/exit.mp4")

    assert camera.role is CameraRole.EXIT


def test_arbitrary_physical_camera_ids_are_allowed() -> None:
    settings = ApplicationSettings(
        (CameraConfig("LOBBY_WEST", CameraRole.ENTRY, "camera://west"),), make_model(), make_storage()
    )

    assert settings.cameras[0].camera_id == "LOBBY_WEST"


@pytest.mark.parametrize("camera_id", ["", "  "])
def test_empty_camera_id_rejected(camera_id: str) -> None:
    with pytest.raises(ValueError, match="camera_id"):
        CameraConfig(camera_id, CameraRole.ENTRY, "camera://entry")


def test_duplicate_camera_ids_rejected() -> None:
    cameras = (
        CameraConfig("CAMERA_A", CameraRole.ENTRY, "camera://entry"),
        CameraConfig("CAMERA_A", CameraRole.EXIT, "camera://exit"),
    )

    with pytest.raises(ValueError, match="unique"):
        ApplicationSettings(cameras, make_model(), make_storage())


def test_missing_role_rejected_without_fallback() -> None:
    with pytest.raises(TypeError, match="CameraRole"):
        CameraConfig("CAMERA_A", None, "camera://entry")


def test_invalid_role_rejected_without_fallback() -> None:
    with pytest.raises(TypeError, match="CameraRole"):
        CameraConfig("CAMERA_A", "SIDE", "camera://entry")


def test_empty_source_rejected() -> None:
    with pytest.raises(ValueError, match="source"):
        CameraConfig("CAMERA_A", CameraRole.ENTRY, " ")


def test_credentials_are_redacted_from_camera_repr() -> None:
    camera = CameraConfig("CAMERA_A", CameraRole.ENTRY, "rtsp://admin:secret@example.test/stream")

    rendered = repr(camera)
    assert "admin" not in rendered
    assert "secret" not in rendered
    assert "***@" in rendered


def test_valid_model_runtime_configuration() -> None:
    settings = make_model()

    assert settings.model_identifier == "face-model.onnx"
    assert settings.device == "cpu"


@pytest.mark.parametrize("model_identifier", ["", "  "])
def test_empty_model_identifier_rejected(model_identifier: str) -> None:
    with pytest.raises(ValueError, match="model_identifier"):
        ModelRuntimeSettings(model_identifier)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_runtime_interval_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="detection_interval_seconds"):
        ModelRuntimeSettings("model.onnx", detection_interval_seconds=interval)


def test_invalid_runtime_device_rejected() -> None:
    with pytest.raises(ValueError, match="device"):
        ModelRuntimeSettings("model.onnx", device="tpu")


def test_valid_storage_configuration() -> None:
    settings = make_storage()

    assert settings.data_directory == "./data"
    assert settings.database_name == "face-ai.sqlite3"


@pytest.mark.parametrize("data_directory", ["", "  "])
def test_empty_storage_directory_rejected(data_directory: str) -> None:
    with pytest.raises(ValueError, match="data_directory"):
        StorageSettings(data_directory, "face-ai.sqlite3")


@pytest.mark.parametrize("database_name", ["", "  ", "../face-ai.sqlite3", "nested/db.sqlite3"])
def test_invalid_storage_database_name_rejected(database_name: str) -> None:
    with pytest.raises(ValueError, match="database_name"):
        StorageSettings("./data", database_name)


def test_complete_valid_application_settings_construction() -> None:
    settings = ApplicationSettings(
        (
            CameraConfig("CAMERA_A", CameraRole.ENTRY, "camera://entry"),
            CameraConfig("CAMERA_B", CameraRole.EXIT, "camera://exit"),
        ),
        make_model(),
        make_storage(),
    )

    assert len(settings.cameras) == 2
    assert settings.cameras[1].role is CameraRole.EXIT


def test_invalid_nested_configuration_fails_immediately() -> None:
    with pytest.raises(TypeError, match="ModelRuntimeSettings"):
        ApplicationSettings(
            (CameraConfig("CAMERA_A", CameraRole.ENTRY, "camera://entry"),),
            object(),
            make_storage(),
        )


def test_camera_role_is_reused_from_domain_models() -> None:
    from app.domain import models

    assert CameraRole is models.CameraRole
