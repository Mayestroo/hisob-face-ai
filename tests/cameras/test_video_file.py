from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.cameras.video_file import (
    DecodedVideoFrame,
    InvalidVideoMetadataError,
    VideoFileFrameSource,
    VideoOpenError,
)
from app.domain.models import CameraFrame, CameraRole
import app.cameras.video_file as video_file


START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeDecoder:
    def __init__(
        self,
        fps: float,
        frames: list[bytes],
        opened: bool = True,
        width: int = 2,
        height: int = 2,
    ) -> None:
        self.fps = fps
        self.frames = iter(frames)
        self.released = False
        self.read_count = 0
        self.opened = opened
        self.width = width
        self.height = height
        self.pixel_format = "gray"

    def read(self) -> tuple[bool, object]:
        if not self.opened:
            raise VideoOpenError("decoder could not open video")
        self.read_count += 1
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


class RaisingDecoder(FakeDecoder):
    def read(self) -> tuple[bool, object]:
        self.read_count += 1
        raise RuntimeError("decoder read failed")


def make_source(tmp_path: Path, decoder: FakeDecoder, camera_id: str = "LOBBY_WEST") -> VideoFileFrameSource:
    path = tmp_path / "entrance.mp4"
    path.touch()
    return VideoFileFrameSource(
        path,
        camera_id,
        CameraRole.ENTRY,
        START,
        decoder_factory=lambda _: decoder,
    )


def test_valid_source_yields_ordered_frames_and_preserves_camera_metadata(tmp_path: Path) -> None:
    decoder = FakeDecoder(2.0, [b"one!", b"two!", b"tri!"])

    frames = list(make_source(tmp_path, decoder))

    assert [item.frame.camera_id for item in frames] == ["LOBBY_WEST"] * 3
    assert [item.frame.camera_role for item in frames] == [CameraRole.ENTRY] * 3
    assert [item.frame.timestamp for item in frames] == [START + timedelta(seconds=offset) for offset in (0, 0.5, 1)]
    assert [item.frame_index for item in frames] == [0, 1, 2]
    assert all(item.frame.timestamp.tzinfo is not None for item in frames)
    assert [item.pixels for item in frames] == [b"one!", b"two!", b"tri!"]
    assert [(item.width, item.height, item.pixel_format) for item in frames] == [(2, 2, "gray")] * 3
    assert decoder.read_count == 4


def test_arbitrary_camera_id_and_exit_role_work(tmp_path: Path) -> None:
    decoder = FakeDecoder(1.0, [b"exit"])
    source = VideoFileFrameSource(
        tmp_path / "exit.mp4",
        "DOOR_NORTH_17",
        CameraRole.EXIT,
        START,
        decoder_factory=lambda _: decoder,
    )
    (tmp_path / "exit.mp4").touch()

    frame = next(iter(source))

    assert frame.frame.camera_id == "DOOR_NORTH_17"
    assert frame.frame.camera_role is CameraRole.EXIT


def test_naive_start_timestamp_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        VideoFileFrameSource(tmp_path / "video.mp4", "CAMERA_A", CameraRole.ENTRY, datetime(2026, 1, 1))


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_fps_is_rejected_and_decoder_released(tmp_path: Path, fps: float) -> None:
    decoder = FakeDecoder(fps, [b"data"])

    with pytest.raises(InvalidVideoMetadataError, match="FPS"):
        list(make_source(tmp_path, decoder))

    assert decoder.released


def test_missing_video_fails_explicitly() -> None:
    with pytest.raises(VideoOpenError, match="does not exist"):
        list(VideoFileFrameSource("missing.mp4", "CAMERA_A", CameraRole.ENTRY, START))


def test_decoder_open_failure_is_not_an_empty_stream(tmp_path: Path) -> None:
    decoder = FakeDecoder(1.0, [], opened=False)
    source = make_source(tmp_path, decoder)

    with pytest.raises(VideoOpenError, match="could not open"):
        list(source)

    assert decoder.released


def test_eof_completes_normally_and_releases_decoder(tmp_path: Path) -> None:
    decoder = FakeDecoder(1.0, [b"data"])

    assert len(list(make_source(tmp_path, decoder))) == 1
    assert decoder.released


def test_early_consumer_stop_releases_decoder(tmp_path: Path) -> None:
    decoder = FakeDecoder(1.0, [b"one!", b"two!", b"tri!"])
    iterator = iter(make_source(tmp_path, decoder))

    next(iterator)
    iterator.close()

    assert decoder.released


def test_decoder_exception_releases_decoder(tmp_path: Path) -> None:
    decoder = RaisingDecoder(1.0, [])

    with pytest.raises(RuntimeError, match="read failed"):
        next(iter(make_source(tmp_path, decoder)))

    assert decoder.released


def test_frames_are_not_buffered_before_first_yield(tmp_path: Path) -> None:
    decoder = FakeDecoder(1.0, [b"one!", b"two!", b"tri!"])
    iterator = iter(make_source(tmp_path, decoder))

    next(iterator)

    assert decoder.read_count == 1
    iterator.close()


@pytest.mark.parametrize("pixels", [b"short", b"too long"])
def test_malformed_pixel_payload_is_rejected(tmp_path: Path, pixels: bytes) -> None:
    decoder = FakeDecoder(1.0, [pixels])

    with pytest.raises(ValueError, match="requires 4 bytes"):
        next(iter(make_source(tmp_path, decoder)))

    assert decoder.released


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 2), (-1, 2), (2, 0), (2, -1)],
)
def test_invalid_dimensions_are_rejected(tmp_path: Path, width: int, height: int) -> None:
    decoder = FakeDecoder(1.0, [b"data"], width=width, height=height)

    with pytest.raises(ValueError, match="greater than zero"):
        next(iter(make_source(tmp_path, decoder)))


def test_decoded_frame_is_immutable() -> None:
    item = DecodedVideoFrame(
        frame=CameraFrame("CAMERA_A", CameraRole.ENTRY, START),
        pixels=b"data",
        width=2,
        height=2,
        frame_index=0,
    )

    with pytest.raises(AttributeError):
        item.pixels = b"other"


def test_consumer_receives_pixels_without_a_second_decoder(tmp_path: Path) -> None:
    decoders: list[FakeDecoder] = []

    def factory(_: Path) -> FakeDecoder:
        decoder = FakeDecoder(1.0, [b"one!", b"two!"])
        decoders.append(decoder)
        return decoder

    path = tmp_path / "video.mp4"
    path.touch()
    values = list(VideoFileFrameSource(path, "CAMERA_A", CameraRole.ENTRY, START, factory))

    assert len(decoders) == 1
    assert [value.pixels for value in values] == [b"one!", b"two!"]


def test_ffmpeg_stderr_is_not_an_unread_pipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process_calls: list[dict[str, object]] = []

    class FakeProcess:
        stdout = BytesIO()
        stderr = None

        def poll(self) -> int:
            return 0

    def fake_popen(*_: object, **kwargs: object) -> FakeProcess:
        process_calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(video_file.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        video_file.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout='{"streams":[{"width":2,"height":2,"r_frame_rate":"1/1"}]}'),
    )
    monkeypatch.setattr(video_file.subprocess, "Popen", fake_popen)

    video_file._FfmpegDecoder(tmp_path / "video.mp4")

    assert process_calls[0]["stderr"] is video_file.subprocess.DEVNULL
