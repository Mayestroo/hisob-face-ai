"""Small deterministic IoU tracker with camera-local lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

from app.cameras.video_file import DecodedVideoFrame
from app.config.settings import TrackingSettings
from app.detection.protocol import PersonDetection
from app.domain.models import BoundingBox


class TrackState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    FINALIZED = "FINALIZED"


@dataclass(frozen=True)
class TrackedPerson:
    """An immutable snapshot of camera-local tracking state."""

    camera_id: str
    track_id: int
    bbox: BoundingBox
    confidence: float
    state: TrackState
    frame_index: int
    timestamp: datetime
    last_seen_frame_index: int
    hit_count: int
    missed_frames: int


@dataclass(frozen=True, slots=True)
class TrackingUpdateResult:
    """Immutable snapshots partitioned by visibility and lifecycle outcome.

    ``observed`` contains only tracks backed by a detection from the current
    frame. ``lost`` contains recoverable tracks without a current detection;
    their bbox and confidence are stale last-observation metadata. Neither
    collection contains finalized tracks.
    """

    observed: tuple[TrackedPerson, ...]
    lost: tuple[TrackedPerson, ...]
    finalized: tuple[TrackedPerson, ...]


@dataclass
class _MutableTrack:
    track_id: int
    camera_id: str
    bbox: BoundingBox
    confidence: float
    state: TrackState
    last_seen_frame_index: int
    timestamp: datetime
    hit_count: int
    missed_frames: int = 0


def iou(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection-over-union for two x1, y1, x2, y2 boxes."""
    intersection_x1 = max(first[0], second[0])
    intersection_y1 = max(first[1], second[1])
    intersection_x2 = min(first[2], second[2])
    intersection_y2 = min(first[3], second[3])
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(0.0, intersection_y2 - intersection_y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


class PersonTracker:
    """Associate detections for exactly one camera and manage track lifecycle."""

    def __init__(self, camera_id: str, settings: TrackingSettings | None = None) -> None:
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must not be empty")
        self.camera_id = camera_id
        self.settings = settings or TrackingSettings()
        self._tracks: dict[int, _MutableTrack] = {}
        self._next_track_id = 1
        self._last_frame_index: int | None = None

    def update(self, frame: DecodedVideoFrame, detections: Sequence[PersonDetection]) -> TrackingUpdateResult:
        if frame.frame.camera_id != self.camera_id:
            raise ValueError("frame camera_id does not match tracker camera_id")
        if self._last_frame_index is not None and frame.frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase strictly")
        if any(not isinstance(detection, PersonDetection) for detection in detections):
            raise TypeError("detections must contain only PersonDetection values")

        ordered_detections = sorted(detections, key=lambda detection: (detection.bbox, detection.confidence))
        candidate_pairs = []
        for track in self._tracks.values():
            intervening_missed = frame.frame_index - track.last_seen_frame_index - 1
            if intervening_missed <= self.settings.tracking_max_lost_frames:
                for detection_rank, detection in enumerate(ordered_detections):
                    overlap = iou(track.bbox, detection.bbox)
                    if overlap >= self.settings.tracking_iou_threshold:
                        candidate_pairs.append((-overlap, track.track_id, detection_rank, track, detection))
        candidate_pairs.sort(key=lambda candidate: candidate[:3])

        matched_tracks: set[int] = set()
        observed: list[TrackedPerson] = []
        matched_detections: set[int] = set()
        for _, _, detection_rank, track, detection in candidate_pairs:
            if track.track_id in matched_tracks or detection_rank in matched_detections:
                continue
            self._observe(track, frame, detection)
            matched_tracks.add(track.track_id)
            matched_detections.add(detection_rank)
            observed.append(self._snapshot(track, frame))

        finalized: list[TrackedPerson] = []
        lost: list[TrackedPerson] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.missed_frames = frame.frame_index - track.last_seen_frame_index
            if track.missed_frames > self.settings.tracking_max_lost_frames:
                finalized.append(self._snapshot(track, frame, TrackState.FINALIZED))
                del self._tracks[track_id]
            else:
                track.state = TrackState.LOST
                lost.append(self._snapshot(track, frame))

        for detection_rank, detection in enumerate(ordered_detections):
            if detection_rank not in matched_detections:
                track = self._new_track(frame, detection)
                self._tracks[track.track_id] = track
                observed.append(self._snapshot(track, frame))

        self._last_frame_index = frame.frame_index
        return TrackingUpdateResult(
            observed=tuple(sorted(observed, key=lambda item: item.track_id)),
            lost=tuple(sorted(lost, key=lambda item: item.track_id)),
            finalized=tuple(sorted(finalized, key=lambda item: item.track_id)),
        )

    def reset(self) -> tuple[TrackedPerson, ...]:
        """Finalize and remove every current track without reusing IDs."""
        finalized = tuple(
            self._snapshot(track, track.timestamp, TrackState.FINALIZED)
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        )
        self._tracks.clear()
        self._last_frame_index = None
        return finalized

    def _new_track(self, frame: DecodedVideoFrame, detection: PersonDetection) -> _MutableTrack:
        track = _MutableTrack(
            track_id=self._next_track_id,
            camera_id=self.camera_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            state=TrackState.ACTIVE if self.settings.tracking_min_confirmed_hits == 1 else TrackState.NEW,
            last_seen_frame_index=frame.frame_index,
            timestamp=frame.frame.timestamp,
            hit_count=1,
        )
        self._next_track_id += 1
        return track

    def _observe(self, track: _MutableTrack, frame: DecodedVideoFrame, detection: PersonDetection) -> None:
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.timestamp = frame.frame.timestamp
        track.last_seen_frame_index = frame.frame_index
        track.hit_count += 1
        track.missed_frames = 0
        track.state = TrackState.ACTIVE if track.hit_count >= self.settings.tracking_min_confirmed_hits else TrackState.NEW

    def _snapshot(
        self,
        track: _MutableTrack,
        frame: DecodedVideoFrame | datetime,
        state: TrackState | None = None,
    ) -> TrackedPerson:
        if isinstance(frame, DecodedVideoFrame):
            frame_index = frame.frame_index
            timestamp = frame.frame.timestamp
        else:
            frame_index = track.last_seen_frame_index
            timestamp = frame
        return TrackedPerson(
            camera_id=track.camera_id,
            track_id=track.track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            state=state or track.state,
            frame_index=frame_index,
            timestamp=timestamp,
            last_seen_frame_index=track.last_seen_frame_index,
            hit_count=track.hit_count,
            missed_frames=track.missed_frames,
        )
