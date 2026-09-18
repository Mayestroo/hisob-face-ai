"""Project-owned inputs for the future face-processing pipeline."""

from app.face.roi import PersonRoiObservation, extract_observed_person_rois

__all__ = ["PersonRoiObservation", "extract_observed_person_rois"]
