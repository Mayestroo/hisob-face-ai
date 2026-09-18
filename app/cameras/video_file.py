"""Incremental recorded-video frame source."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterator, Protocol

from app.domain.models import CameraFrame, CameraRole


class VideoSourceError(RuntimeError):
    """Base error for recorded-video source failures."""


class VideoOpenError(VideoSourceError):
    """The video file could not be opened or decoded."""


class InvalidVideoMetadataError(VideoSourceError):
    """The source does not expose usable timing metadata."""


@dataclass(frozen=True)
class DecodedVideoFrame:
    """One decoded grayscale frame bound to its domain metadata."""

    frame: CameraFrame
    pixels: bytes
    width: int
    height: int
    frame_index: int
    pixel_format: str = "gray"

    def __post_init__(self) -> None:
        if not isinstance(self.frame, CameraFrame):
            raise TypeError("frame must be a CameraFrame")
        if not isinstance(self.pixels, bytes):
            raise TypeError("pixels must be bytes")
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width <= 0:
            raise ValueError("width must be greater than zero")
        if not isinstance(self.height, int) or isinstance(self.height, bool) or self.height <= 0:
            raise ValueError("height must be greater than zero")
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self.pixel_format != "gray":
            raise ValueError("pixel_format must be 'gray'")
        expected_size = self.width * self.height
        if len(self.pixels) != expected_size:
            raise ValueError(f"gray frame requires {expected_size} bytes, got {len(self.pixels)}")


class Decoder(Protocol):
    fps: float
    width: int
    height: int
    pixel_format: str

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


DecoderFactory = Callable[[Path], Decoder]


class VideoFileFrameSource:
    """Yield metadata frames for one configured recorded-video camera.

    Frame indexes are zero-based and explicit on each decoded frame. The source
    increments them once for every successfully decoded frame.
    """

    def __init__(
        self,
        video_path: str | Path,
        camera_id: str,
        camera_role: CameraRole,
        start_timestamp: datetime,
        decoder_factory: DecoderFactory | None = None,
    ) -> None:
        if start_timestamp.tzinfo is None or start_timestamp.utcoffset() is None:
            raise ValueError("start_timestamp must include timezone information")
        self.video_path = Path(video_path)
        self.camera_id = camera_id
        self.camera_role = camera_role
        self.start_timestamp = start_timestamp
        self._decoder_factory = decoder_factory or _open_ffmpeg_decoder

    def frames(self) -> Iterator[DecodedVideoFrame]:
        """Decode and yield one bound frame value at a time."""
        if not self.video_path.is_file():
            raise VideoOpenError(f"video file does not exist: {self.video_path}")

        decoder = self._decoder_factory(self.video_path)
        try:
            fps = float(decoder.fps)
            if not math.isfinite(fps) or fps <= 0:
                raise InvalidVideoMetadataError(f"video FPS must be finite and positive, got {fps!r}")

            frame_index = 0
            while True:
                ok, payload = decoder.read()
                if not ok:
                    return
                yield DecodedVideoFrame(
                    frame=CameraFrame(
                        camera_id=self.camera_id,
                        camera_role=self.camera_role,
                        timestamp=self.start_timestamp + timedelta(seconds=frame_index / fps),
                    ),
                    pixels=payload,
                    width=decoder.width,
                    height=decoder.height,
                    frame_index=frame_index,
                    pixel_format=decoder.pixel_format,
                )
                frame_index += 1
        finally:
            decoder.release()

    def __iter__(self) -> Iterator[DecodedVideoFrame]:
        return self.frames()


class _FfmpegDecoder:
    def __init__(self, video_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise VideoOpenError("ffmpeg and ffprobe are required to decode video files")

        try:
            metadata = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate",
                    "-of",
                    "json",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(metadata.stdout)["streams"][0]
            width = int(stream["width"])
            height = int(stream["height"])
            numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/", 1))
            self.fps = numerator / denominator
            if width <= 0 or height <= 0:
                raise ValueError("video dimensions must be positive")
            frame_size = width * height
        except (OSError, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
            raise VideoOpenError(f"could not read video metadata: {video_path}") from exc
        except subprocess.CalledProcessError as exc:
            raise VideoOpenError(f"decoder could not open video: {video_path}") from exc

        try:
            self._process = subprocess.Popen(
                [ffmpeg, "-v", "error", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise VideoOpenError(f"decoder could not start for video: {video_path}") from exc
        self._frame_size = frame_size
        self.width = width
        self.height = height
        self.pixel_format = "gray"

    def read(self) -> tuple[bool, bytes]:
        assert self._process.stdout is not None
        payload = self._process.stdout.read(self._frame_size)
        if not payload:
            return False, b""
        if len(payload) != self._frame_size:
            raise VideoOpenError("decoder returned an incomplete video frame")
        return True, payload

    def release(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()
        if self._process.stderr is not None:
            self._process.stderr.close()


def _open_ffmpeg_decoder(video_path: Path) -> Decoder:
    return _FfmpegDecoder(video_path)
