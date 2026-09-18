"""Project-owned values and adapters for the face-processing pipeline."""

from app.face.alignment import (
    ARC_FACE_TEMPLATE_112,
    AlignmentSpec,
    FaceAlignmentError,
    YuNetLandmark,
    YUNET_LANDMARK_ORDER,
    align_grayscale_face,
)
from app.face.detection import (
    DetectedFaceObservation,
    DetectorHealth,
    FaceDetectorError,
    FaceDetectorInferenceError,
    FaceDetectorSettings,
    FaceDetectorUnavailableError,
    PreprocessedRoi,
    YuNetFaceDetector,
    preprocess_roi,
)
from app.face.roi import PersonRoiObservation, extract_observed_person_rois

__all__ = [
    "ARC_FACE_TEMPLATE_112", "AlignmentSpec", "DetectedFaceObservation", "DetectorHealth",
    "FaceAlignmentError", "FaceDetectorError", "FaceDetectorInferenceError", "FaceDetectorSettings",
    "FaceDetectorUnavailableError", "PersonRoiObservation", "PreprocessedRoi", "YUNET_LANDMARK_ORDER", "YuNetFaceDetector",
    "YuNetLandmark", "align_grayscale_face", "extract_observed_person_rois", "preprocess_roi",
]
