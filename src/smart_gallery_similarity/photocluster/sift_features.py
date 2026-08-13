"""SIFT extraction and PhotoCluster SIFT Matching Ratio (SMR).

PhotoCluster defines the local similarity between images A and B as::

    SMR(A, B) = 2 * M(A, B) / (K_A + K_B)

where K_A and K_B are the SIFT keypoint counts and M(A, B) is the
number of bidirectional descriptor matches.

The paper does not specify the exact descriptor matcher or SIFT detector
parameters. This implementation uses OpenCV SIFT and interprets
"bidirectional matches" as mutual nearest-neighbour matches via
BFMatcher(NORM_L2, crossCheck=True). All choices are recorded in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


KEYPOINT_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
    "size",
    "angle",
    "response",
    "octave",
    "class_id",
)


class SIFTFeatureError(RuntimeError):
    """Raised when SIFT extraction, loading, or matching cannot continue."""


@dataclass(frozen=True)
class SIFTConfig:
    """Explicit OpenCV SIFT settings used for reproducible extraction."""

    nfeatures: int = 0
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    max_edge: int = 0

    def validate(self) -> None:
        if self.nfeatures < 0:
            raise ValueError("nfeatures cannot be negative.")
        if self.n_octave_layers < 1:
            raise ValueError("n_octave_layers must be at least 1.")
        if self.contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive.")
        if self.edge_threshold <= 0:
            raise ValueError("edge_threshold must be positive.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.max_edge < 0:
            raise ValueError("max_edge cannot be negative.")


def create_sift(config: SIFTConfig) -> cv2.SIFT:
    """Create an OpenCV SIFT detector from the recorded configuration."""
    config.validate()
    if not hasattr(cv2, "SIFT_create"):
        raise SIFTFeatureError(
            "This OpenCV build does not provide cv2.SIFT_create()."
        )
    return cv2.SIFT_create(
        nfeatures=config.nfeatures,
        nOctaveLayers=config.n_octave_layers,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )


def load_oriented_grayscale(
    image_path: Path,
    *,
    max_edge: int = 0,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Load an image, apply EXIF orientation, and convert it to grayscale.

    Returns ``(gray, original_size, processed_size)`` where sizes are
    ``(width, height)``. A positive ``max_edge`` downsizes large images while
    preserving aspect ratio. It never enlarges an image.
    """
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise SIFTFeatureError(f"Image does not exist: {image_path}")

    try:
        with Image.open(image_path) as source:
            oriented = ImageOps.exif_transpose(source)
            original_size = (int(oriented.width), int(oriented.height))
            gray_image = oriented.convert("L")

            if max_edge > 0 and max(gray_image.size) > max_edge:
                scale = max_edge / float(max(gray_image.size))
                resized = (
                    max(1, int(round(gray_image.width * scale))),
                    max(1, int(round(gray_image.height * scale))),
                )
                gray_image = gray_image.resize(resized, Image.Resampling.LANCZOS)

            processed_size = (int(gray_image.width), int(gray_image.height))
            gray = np.asarray(gray_image, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise SIFTFeatureError(
            f"Could not decode image '{image_path}': {exc}"
        ) from exc

    if gray.ndim != 2 or gray.size == 0:
        raise SIFTFeatureError(
            f"Unexpected grayscale shape {gray.shape} for {image_path}."
        )
    return np.ascontiguousarray(gray), original_size, processed_size


def keypoints_to_array(keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """Serialize OpenCV keypoints into a stable numeric array."""
    if not keypoints:
        return np.empty((0, len(KEYPOINT_COLUMNS)), dtype=np.float32)

    values = [
        (
            float(keypoint.pt[0]),
            float(keypoint.pt[1]),
            float(keypoint.size),
            float(keypoint.angle),
            float(keypoint.response),
            float(keypoint.octave),
            float(keypoint.class_id),
        )
        for keypoint in keypoints
    ]
    return np.asarray(values, dtype=np.float32)


def array_to_keypoints(values: np.ndarray) -> list[cv2.KeyPoint]:
    """Reconstruct OpenCV keypoints saved by :func:`keypoints_to_array`."""
    array = np.asarray(values, dtype=np.float32)
    expected_shape = (array.shape[0], len(KEYPOINT_COLUMNS))
    if array.ndim != 2 or array.shape != expected_shape:
        raise ValueError(
            "keypoint array must have shape (K, 7), found "
            f"{array.shape}."
        )

    return [
        cv2.KeyPoint(
            x=float(row[0]),
            y=float(row[1]),
            size=max(float(row[2]), 1e-6),
            angle=float(row[3]),
            response=float(row[4]),
            octave=int(round(float(row[5]))),
            class_id=int(round(float(row[6]))),
        )
        for row in array
    ]


def extract_sift_features(
    image_path: Path,
    *,
    config: SIFTConfig,
) -> dict[str, Any]:
    """Extract SIFT keypoints and 128-D descriptors from one image."""
    gray, original_size, processed_size = load_oriented_grayscale(
        image_path,
        max_edge=config.max_edge,
    )
    detector = create_sift(config)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    keypoints = keypoints or []

    keypoint_array = keypoints_to_array(keypoints)
    if descriptors is None:
        descriptor_array = np.empty((0, 128), dtype=np.float32)
    else:
        descriptor_array = np.ascontiguousarray(descriptors, dtype=np.float32)

    if descriptor_array.ndim != 2 or descriptor_array.shape[1] != 128:
        raise SIFTFeatureError(
            f"Unexpected SIFT descriptor shape {descriptor_array.shape} for "
            f"{image_path}."
        )
    if descriptor_array.shape[0] != keypoint_array.shape[0]:
        raise SIFTFeatureError(
            "SIFT keypoint and descriptor counts differ for "
            f"{image_path}: {keypoint_array.shape[0]} vs "
            f"{descriptor_array.shape[0]}."
        )
    if not np.isfinite(keypoint_array).all() or not np.isfinite(
        descriptor_array
    ).all():
        raise SIFTFeatureError(
            f"SIFT produced NaN or infinite values for {image_path}."
        )

    return {
        "keypoints": keypoint_array,
        "descriptors": descriptor_array,
        "original_size": original_size,
        "processed_size": processed_size,
    }


def save_sift_feature_file(
    path: Path,
    *,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> Path:
    """Save one image's local features as a compressed NumPy artifact."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=np.asarray(keypoints, dtype=np.float32),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        original_size=np.asarray(original_size, dtype=np.int32),
        processed_size=np.asarray(processed_size, dtype=np.int32),
    )
    return path


def load_sift_feature_file(path: Path) -> dict[str, np.ndarray]:
    """Load and validate one saved SIFT feature artifact."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SIFTFeatureError(f"Missing SIFT feature file: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            keypoints = np.asarray(data["keypoints"], dtype=np.float32)
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            original_size = np.asarray(data["original_size"], dtype=np.int32)
            processed_size = np.asarray(data["processed_size"], dtype=np.int32)
    except (OSError, ValueError, KeyError) as exc:
        raise SIFTFeatureError(
            f"Could not load SIFT feature file '{path}': {exc}"
        ) from exc

    if keypoints.ndim != 2 or keypoints.shape[1] != 7:
        raise SIFTFeatureError(
            f"Invalid keypoint shape {keypoints.shape} in {path}."
        )
    if descriptors.ndim != 2 or descriptors.shape[1] != 128:
        raise SIFTFeatureError(
            f"Invalid descriptor shape {descriptors.shape} in {path}."
        )
    if keypoints.shape[0] != descriptors.shape[0]:
        raise SIFTFeatureError(
            f"Keypoint and descriptor counts differ in {path}."
        )
    if original_size.shape != (2,) or processed_size.shape != (2,):
        raise SIFTFeatureError(f"Invalid image size metadata in {path}.")
    if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all():
        raise SIFTFeatureError(f"Non-finite SIFT values in {path}.")

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "original_size": original_size,
        "processed_size": processed_size,
    }


def mutual_best_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
) -> list[cv2.DMatch]:
    """Return mutual nearest-neighbour SIFT matches in L2 space.

    OpenCV's ``crossCheck=True`` retains a descriptor pair only when each is
    the other's nearest neighbour. This is the explicit operational meaning of
    PhotoCluster's otherwise underspecified term "bidirectional matches".
    """
    a = np.ascontiguousarray(descriptors_a, dtype=np.float32)
    b = np.ascontiguousarray(descriptors_b, dtype=np.float32)

    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (128,) or b.shape[1:] != (128,):
        raise ValueError("SIFT descriptors must have shape (K, 128).")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return []
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("SIFT descriptors contain NaN or infinite values.")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(a, b)
    return sorted(matches, key=lambda match: float(match.distance))


def sift_matching_ratio(
    keypoint_count_a: int,
    keypoint_count_b: int,
    bidirectional_match_count: int,
) -> float:
    """Compute PhotoCluster equation 5 and return a value in [0, 1]."""
    ka = int(keypoint_count_a)
    kb = int(keypoint_count_b)
    matches = int(bidirectional_match_count)

    if ka < 0 or kb < 0 or matches < 0:
        raise ValueError("Keypoint and match counts cannot be negative.")
    if matches > min(ka, kb):
        raise ValueError(
            "Bidirectional match count cannot exceed the smaller keypoint count."
        )

    denominator = ka + kb
    if denominator == 0:
        return 0.0

    ratio = (2.0 * matches) / float(denominator)
    return float(np.clip(ratio, 0.0, 1.0))


def pair_match_statistics(matches: Sequence[cv2.DMatch]) -> dict[str, float]:
    """Summarize descriptor distances for diagnostics only."""
    if not matches:
        return {
            "mean_match_distance": float("nan"),
            "median_match_distance": float("nan"),
            "minimum_match_distance": float("nan"),
            "maximum_match_distance": float("nan"),
        }

    distances = np.asarray(
        [float(match.distance) for match in matches],
        dtype=np.float64,
    )
    return {
        "mean_match_distance": float(distances.mean()),
        "median_match_distance": float(np.median(distances)),
        "minimum_match_distance": float(distances.min()),
        "maximum_match_distance": float(distances.max()),
    }


def expected_within_set_pairs(initial_sets: Sequence[dict[str, Any]]) -> int:
    """Return the number of unique pairs within all Step-2 sets."""
    total = 0
    for initial_set in initial_sets:
        size = int(initial_set.get("size", len(initial_set.get("member_rows", []))))
        if size < 1:
            raise ValueError("Initial sets cannot be empty.")
        total += size * (size - 1) // 2
    return int(total)


def validate_sift_feature_artifacts(
    *,
    keypoint_counts: np.ndarray,
    records: Sequence[dict[str, Any]],
    descriptor_dir: Path,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate per-image SIFT artifacts before pair matching."""
    errors: list[str] = []
    warnings: list[str] = []
    counts = np.asarray(keypoint_counts)
    record_count = len(records)

    if expected_count is not None and record_count != expected_count:
        errors.append(
            f"Expected {expected_count} image records, found {record_count}."
        )
    if counts.shape != (record_count,):
        errors.append(
            f"keypoint_counts shape is {counts.shape}; expected {(record_count,)}."
        )
    elif not np.issubdtype(counts.dtype, np.integer):
        errors.append("keypoint_counts must use an integer dtype.")
    elif np.any(counts < 0):
        errors.append("keypoint_counts contains negative values.")

    missing_files: list[str] = []
    invalid_files: list[str] = []
    if counts.shape == (record_count,):
        for row in range(record_count):
            path = Path(descriptor_dir) / f"row_{row:04d}.npz"
            if not path.is_file():
                missing_files.append(path.name)
                continue
            try:
                feature = load_sift_feature_file(path)
                if feature["descriptors"].shape[0] != int(counts[row]):
                    invalid_files.append(path.name)
            except SIFTFeatureError:
                invalid_files.append(path.name)

    if missing_files:
        errors.append(
            f"Missing {len(missing_files)} descriptor files; first: "
            f"{missing_files[:5]}."
        )
    if invalid_files:
        errors.append(
            f"Invalid {len(invalid_files)} descriptor files; first: "
            f"{invalid_files[:5]}."
        )

    zero_keypoint_count = (
        int(np.count_nonzero(counts == 0))
        if counts.shape == (record_count,)
        else None
    )
    if zero_keypoint_count:
        warnings.append(
            f"{zero_keypoint_count} images contain no detected SIFT keypoints."
        )

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "image_count": record_count,
            "descriptor_file_count": (
                record_count - len(missing_files)
            ),
            "zero_keypoint_image_count": zero_keypoint_count,
            "minimum_keypoints": (
                int(counts.min()) if counts.shape == (record_count,) else None
            ),
            "maximum_keypoints": (
                int(counts.max()) if counts.shape == (record_count,) else None
            ),
            "mean_keypoints": (
                float(counts.mean()) if counts.shape == (record_count,) else None
            ),
            "median_keypoints": (
                float(np.median(counts))
                if counts.shape == (record_count,)
                else None
            ),
        },
    }


def validate_smr_artifacts(
    *,
    smr_matrix: np.ndarray,
    sift_distance: np.ndarray,
    computed_pair_mask: np.ndarray,
    keypoint_counts: np.ndarray,
    initial_sets: Sequence[dict[str, Any]],
    pair_row_count: int,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate SMR matrices and the Step-2 pair coverage."""
    errors: list[str] = []
    warnings: list[str] = []

    counts = np.asarray(keypoint_counts)
    count = int(counts.size)
    expected_shape = (count, count)
    smr = np.asarray(smr_matrix)
    distance = np.asarray(sift_distance)
    mask = np.asarray(computed_pair_mask)

    if expected_count is not None and count != expected_count:
        errors.append(f"Expected {expected_count} images, found {count}.")

    for name, matrix in (
        ("smr_matrix", smr),
        ("sift_distance", distance),
        ("computed_pair_mask", mask),
    ):
        if matrix.shape != expected_shape:
            errors.append(
                f"{name} shape is {matrix.shape}; expected {expected_shape}."
            )

    if errors:
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "checks": {},
        }

    if mask.dtype != np.bool_:
        errors.append("computed_pair_mask must use boolean dtype.")
    if not np.array_equal(mask, mask.T):
        errors.append("computed_pair_mask is not symmetric.")
    if not np.all(np.diag(mask)):
        errors.append("computed_pair_mask diagonal must be True.")

    if not np.allclose(smr, smr.T, equal_nan=True, atol=1e-7):
        errors.append("smr_matrix is not symmetric.")
    if not np.allclose(distance, distance.T, equal_nan=True, atol=1e-7):
        errors.append("sift_distance is not symmetric.")
    if not np.allclose(np.diag(smr), 1.0, atol=1e-7):
        errors.append("smr_matrix diagonal must be one.")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-7):
        errors.append("sift_distance diagonal must be zero.")

    computed_values = mask
    if not np.isfinite(smr[computed_values]).all():
        errors.append("Computed SMR entries contain NaN or infinite values.")
    if not np.isfinite(distance[computed_values]).all():
        errors.append("Computed SIFT-distance entries contain invalid values.")
    if np.any(smr[computed_values] < -1e-7) or np.any(
        smr[computed_values] > 1.0 + 1e-7
    ):
        errors.append("Computed SMR entries fall outside [0, 1].")
    if np.any(distance[computed_values] < -1e-7) or np.any(
        distance[computed_values] > 1.0 + 1e-7
    ):
        errors.append("Computed SIFT distances fall outside [0, 1].")
    if not np.allclose(
        distance[computed_values],
        1.0 - smr[computed_values],
        atol=1e-6,
    ):
        errors.append("sift_distance does not equal 1 - SMR on computed entries.")

    cross_set_values = ~mask
    if not np.isnan(smr[cross_set_values]).all():
        errors.append("Cross-set SMR entries must remain NaN.")
    if not np.isnan(distance[cross_set_values]).all():
        errors.append("Cross-set SIFT-distance entries must remain NaN.")

    expected_pairs = expected_within_set_pairs(initial_sets)
    actual_pairs = int(np.count_nonzero(np.triu(mask, k=1)))
    if actual_pairs != expected_pairs:
        errors.append(
            f"Computed {actual_pairs} unique pairs; expected {expected_pairs}."
        )
    if int(pair_row_count) != expected_pairs:
        errors.append(
            f"pair_matches.csv has {pair_row_count} rows; expected "
            f"{expected_pairs}."
        )

    positive_smr = smr[np.triu(mask, k=1)]
    zero_match_pairs = int(np.count_nonzero(positive_smr == 0.0))
    if expected_pairs and zero_match_pairs == expected_pairs:
        warnings.append("Every within-set pair has zero SMR.")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "image_count": count,
            "initial_set_count": len(initial_sets),
            "expected_within_set_pairs": expected_pairs,
            "computed_within_set_pairs": actual_pairs,
            "pair_csv_row_count": int(pair_row_count),
            "zero_match_pair_count": zero_match_pairs,
            "nonzero_match_pair_count": int(expected_pairs - zero_match_pairs),
            "minimum_smr": (
                float(positive_smr.min()) if positive_smr.size else None
            ),
            "maximum_smr": (
                float(positive_smr.max()) if positive_smr.size else None
            ),
            "mean_smr": (
                float(positive_smr.mean()) if positive_smr.size else None
            ),
            "median_smr": (
                float(np.median(positive_smr)) if positive_smr.size else None
            ),
            "matrices_are_symmetric": bool(
                np.allclose(smr, smr.T, equal_nan=True, atol=1e-7)
                and np.allclose(distance, distance.T, equal_nan=True, atol=1e-7)
            ),
            "computed_entries_are_finite": bool(
                np.isfinite(smr[computed_values]).all()
                and np.isfinite(distance[computed_values]).all()
            ),
            "cross_set_entries_are_nan": bool(
                np.isnan(smr[cross_set_values]).all()
                and np.isnan(distance[cross_set_values]).all()
            ),
        },
    }
