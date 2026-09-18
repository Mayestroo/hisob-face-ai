import numpy as np
import pytest

from app.face import AlignmentSpec, FaceAlignmentError, align_grayscale_face
from app.face.alignment import _bilinear_zero_border, _similarity_transform


def test_alignment_output_is_immutable_gray_payload() -> None:
    landmarks = ((73.5318, 51.5014), (38.2946, 51.6963), (56.0252, 71.7366), (70.7299, 92.2041), (41.5493, 92.3655))
    pixels = align_grayscale_face(bytes([127]) * (112 * 112), 112, 112, landmarks)
    assert isinstance(pixels, bytes)
    assert len(pixels) == 112 * 112
    assert pixels[56 * 112 + 56] == 127
    assert set(pixels) <= {0, 127}


def test_alignment_maps_yunet_order_to_canonical_template() -> None:
    # YuNet order is right eye, left eye, nose, right mouth, left mouth.
    landmarks = ((73.5318, 51.5014), (38.2946, 51.6963), (56.0252, 71.7366), (70.7299, 92.2041), (41.5493, 92.3655))
    pixels = align_grayscale_face(bytes([90]) * (112 * 112), 112, 112, landmarks)
    assert pixels[56 * 112 + 56] == 90


@pytest.mark.parametrize("landmarks", [
    ((float("nan"), 1.0),) * 5,
    ((1.0, 1.0),) * 5,
])
def test_nonfinite_or_degenerate_landmarks_are_rejected(landmarks) -> None:
    with pytest.raises(FaceAlignmentError):
        align_grayscale_face(bytes(100), 10, 10, landmarks, AlignmentSpec(width=10, height=10))


def test_zero_border_is_used_outside_source_image() -> None:
    spec = AlignmentSpec(width=4, height=4, template=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 2.0), (1.0, 2.0)))
    landmarks = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 2.0), (1.0, 2.0))
    pixels = align_grayscale_face(bytes([255, 255, 255, 255]), 2, 2, landmarks, spec)
    assert np.frombuffer(pixels, dtype=np.uint8)[-1] == 0


def test_similarity_transform_recovers_known_column_vector_transforms() -> None:
    source = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 3.0]])
    cases = [
        (np.eye(2), np.array([0.0, 0.0])),
        (np.eye(2), np.array([4.0, -3.0])),
        (2.5 * np.eye(2), np.array([1.0, 2.0])),
        (np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([0.0, 0.0])),
        (1.75 * np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([8.0, -2.0])),
    ]
    for matrix, translation in cases:
        target = source @ matrix.T + translation
        recovered = _similarity_transform(source, target)
        actual = (recovered[:2, :2] @ source.T).T + recovered[:2, 2]
        assert np.allclose(actual, target, atol=1e-10)


def test_similarity_transform_rejects_degenerate_geometry() -> None:
    source = np.zeros((5, 2), dtype=np.float64)
    with pytest.raises(FaceAlignmentError):
        _similarity_transform(source, source)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0.5, 0.5, 100.0),
        (-0.25, 0.5, 75.0),
        (0.5, -0.25, 75.0),
        (1.75, 0.5, 25.0),
        (0.5, 1.75, 25.0),
        (-0.25, -0.25, 56.25),
        (-2.0, 0.5, 0.0),
    ],
)
def test_bilinear_zero_border_samples_each_neighbor(x, y, expected) -> None:
    image = np.full((2, 2), 100, dtype=np.uint8)
    actual = _bilinear_zero_border(image, np.array([[x]]), np.array([[y]]))[0, 0]
    assert actual == pytest.approx(expected)
