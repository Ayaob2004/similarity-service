"""PhotoCluster combined local/global image dissimilarity.

PhotoCluster equation 4 combines normalized feature distances as::

    d(A, B) = 1 - sum_x ((1 - f_x(A, B)) * W_x) / sum_x W_x

which is algebraically equivalent to the weighted mean of the feature
*distances* ``f_x``. During the multiple-clustering stage the paper uses:

* SIFT distance = 1 - SMR
* Hue/Saturation color distance
* normalized capture-time distance

and states that their importance is descending: SIFT > color > time. The
paper does not publish the numeric weights. This implementation therefore
uses a transparent, configurable 3:2:1 baseline and records it in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class CombinedDistanceError(RuntimeError):
    """Raised when PhotoCluster feature-distance combination is invalid."""


@dataclass(frozen=True)
class CombinedDistanceConfig:
    """Importance weights for PhotoCluster equation 4."""

    sift_weight: float = 3.0
    color_weight: float = 2.0
    time_weight: float = 1.0

    def validate(self) -> None:
        weights = self.as_array()
        if not np.isfinite(weights).all():
            raise ValueError("Feature weights must be finite.")
        if np.any(weights <= 0):
            raise ValueError("All feature weights must be positive.")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.sift_weight, self.color_weight, self.time_weight],
            dtype=np.float64,
        )

    @property
    def total_weight(self) -> float:
        return float(self.sift_weight + self.color_weight + self.time_weight)

    @property
    def follows_paper_order(self) -> bool:
        return bool(self.sift_weight > self.color_weight > self.time_weight)

    def normalized_weights(self) -> dict[str, float]:
        self.validate()
        total = self.total_weight
        return {
            "sift": float(self.sift_weight / total),
            "color": float(self.color_weight / total),
            "time": float(self.time_weight / total),
        }


def _as_square_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, found {values.shape}.")
    return values


def combine_three_feature_distances(
    sift_distance: np.ndarray,
    color_distance: np.ndarray,
    time_distance: np.ndarray,
    *,
    config: CombinedDistanceConfig | None = None,
) -> np.ndarray:
    """Apply PhotoCluster equation 4 to equally shaped distance arrays.

    Inputs may be scalars, vectors, or matrices, provided their shapes match.
    NaN entries are preserved. This is useful because SIFT was intentionally
    computed only for pairs inside the Step-2 initial sets.
    """
    config = config or CombinedDistanceConfig()
    config.validate()

    sift = np.asarray(sift_distance, dtype=np.float64)
    color = np.asarray(color_distance, dtype=np.float64)
    time = np.asarray(time_distance, dtype=np.float64)

    if sift.shape != color.shape or sift.shape != time.shape:
        raise ValueError(
            "sift_distance, color_distance, and time_distance must have "
            "identical shapes."
        )

    combined = (
        config.sift_weight * sift
        + config.color_weight * color
        + config.time_weight * time
    ) / config.total_weight

    finite = np.isfinite(combined)
    if np.any(combined[finite] < -1e-7) or np.any(combined[finite] > 1.0 + 1e-7):
        raise ValueError("Combined finite distances fall outside [0, 1].")

    combined[finite] = np.clip(combined[finite], 0.0, 1.0)
    return combined.astype(np.float32)


def build_combined_distance_matrices(
    *,
    sift_distance: np.ndarray,
    color_distance: np.ndarray,
    time_distance: np.ndarray,
    computed_pair_mask: np.ndarray,
    config: CombinedDistanceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build full PhotoCluster distance/similarity matrices.

    Only pairs marked by ``computed_pair_mask`` receive a finite result. Cross-
    initial-set pairs remain NaN because the paper assigns their final ND
    probability to zero and does not run local comparisons across sets.
    """
    config = config or CombinedDistanceConfig()
    config.validate()

    sift = _as_square_matrix("sift_distance", sift_distance)
    color = _as_square_matrix("color_distance", color_distance)
    time = _as_square_matrix("time_distance", time_distance)
    mask = np.asarray(computed_pair_mask, dtype=bool)

    if color.shape != sift.shape or time.shape != sift.shape:
        raise ValueError("All distance matrices must have identical shapes.")
    if mask.shape != sift.shape:
        raise ValueError(
            f"computed_pair_mask shape {mask.shape} differs from {sift.shape}."
        )
    if not np.array_equal(mask, mask.T):
        raise ValueError("computed_pair_mask must be symmetric.")
    if not np.all(np.diag(mask)):
        raise ValueError("computed_pair_mask diagonal must be True.")

    for name, matrix in (("color_distance", color), ("time_distance", time)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        if np.any(matrix < -1e-7) or np.any(matrix > 1.0 + 1e-7):
            raise ValueError(f"{name} contains values outside [0, 1].")

    if not np.isfinite(sift[mask]).all():
        raise ValueError("SIFT distance is not finite for every computed pair.")
    if np.any(sift[mask] < -1e-7) or np.any(sift[mask] > 1.0 + 1e-7):
        raise ValueError("SIFT computed entries fall outside [0, 1].")

    distance = np.full(sift.shape, np.nan, dtype=np.float32)
    distance[mask] = combine_three_feature_distances(
        sift[mask],
        color[mask],
        time[mask],
        config=config,
    )
    np.fill_diagonal(distance, 0.0)

    similarity = np.full(sift.shape, np.nan, dtype=np.float32)
    similarity[mask] = 1.0 - distance[mask]
    np.fill_diagonal(similarity, 1.0)
    return distance, similarity


def summarize_initial_set_distances(
    *,
    initial_sets: Sequence[dict[str, Any]],
    sift_distance: np.ndarray,
    color_distance: np.ndarray,
    time_distance: np.ndarray,
    combined_dissimilarity: np.ndarray,
) -> list[dict[str, Any]]:
    """Summarize the pair-distance distributions inside every initial set."""
    summaries: list[dict[str, Any]] = []
    matrices = {
        "sift": np.asarray(sift_distance, dtype=np.float64),
        "color": np.asarray(color_distance, dtype=np.float64),
        "time": np.asarray(time_distance, dtype=np.float64),
        "combined": np.asarray(combined_dissimilarity, dtype=np.float64),
    }

    for item in initial_sets:
        rows = np.asarray(item.get("member_rows", []), dtype=np.int64)
        size = int(item.get("size", rows.size))
        if rows.size != size or size < 1:
            raise CombinedDistanceError(
                f"Invalid member_rows for {item.get('set_id', '<unknown>')}."
            )

        summary: dict[str, Any] = {
            "set_id": str(item["set_id"]),
            "size": size,
            "pair_count": size * (size - 1) // 2,
            "exemplar_filename": str(item.get("exemplar_filename", "")),
        }

        if size == 1:
            for feature in matrices:
                summary[f"{feature}_minimum"] = 0.0
                summary[f"{feature}_mean"] = 0.0
                summary[f"{feature}_median"] = 0.0
                summary[f"{feature}_maximum"] = 0.0
            summaries.append(summary)
            continue

        upper = np.triu_indices(size, k=1)
        for feature, matrix in matrices.items():
            values = matrix[np.ix_(rows, rows)][upper]
            if not np.isfinite(values).all():
                raise CombinedDistanceError(
                    f"{feature} contains missing values inside {item['set_id']}."
                )
            summary[f"{feature}_minimum"] = float(values.min())
            summary[f"{feature}_mean"] = float(values.mean())
            summary[f"{feature}_median"] = float(np.median(values))
            summary[f"{feature}_maximum"] = float(values.max())

        summaries.append(summary)

    return summaries


def validate_combined_distance_artifacts(
    *,
    sift_distance: np.ndarray,
    color_distance: np.ndarray,
    time_distance: np.ndarray,
    computed_pair_mask: np.ndarray,
    combined_dissimilarity: np.ndarray,
    combined_similarity: np.ndarray,
    config: CombinedDistanceConfig,
    expected_count: int | None = None,
    expected_pair_count: int | None = None,
) -> dict[str, Any]:
    """Validate all artifacts produced by PhotoCluster step 4."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        config.validate()
    except ValueError as exc:
        errors.append(str(exc))

    if not config.follows_paper_order:
        warnings.append(
            "Weights do not follow the paper's stated order SIFT > color > time."
        )

    try:
        sift = _as_square_matrix("sift_distance", sift_distance)
        color = _as_square_matrix("color_distance", color_distance)
        time = _as_square_matrix("time_distance", time_distance)
        combined = _as_square_matrix(
            "combined_dissimilarity", combined_dissimilarity
        )
        similarity = _as_square_matrix("combined_similarity", combined_similarity)
    except ValueError as exc:
        return {
            "status": "failed",
            "errors": [str(exc)],
            "warnings": warnings,
            "checks": {},
        }

    count = int(sift.shape[0])
    matrices = (color, time, combined, similarity)
    if any(matrix.shape != sift.shape for matrix in matrices):
        errors.append("All matrices must have identical shapes.")

    mask = np.asarray(computed_pair_mask, dtype=bool)
    if mask.shape != sift.shape:
        errors.append(
            f"computed_pair_mask shape {mask.shape}; expected {sift.shape}."
        )
        mask_valid = False
    else:
        mask_valid = True

    if expected_count is not None and count != expected_count:
        errors.append(f"Expected {expected_count} images, found {count}.")

    if mask_valid:
        if not np.array_equal(mask, mask.T):
            errors.append("computed_pair_mask is not symmetric.")
        if not np.all(np.diag(mask)):
            errors.append("computed_pair_mask diagonal is not True.")

        unique_pair_count = int(np.count_nonzero(np.triu(mask, k=1)))
        if (
            expected_pair_count is not None
            and unique_pair_count != expected_pair_count
        ):
            errors.append(
                f"Expected {expected_pair_count} computed pairs, found "
                f"{unique_pair_count}."
            )

        for name, matrix in (
            ("sift_distance", sift),
            ("combined_dissimilarity", combined),
            ("combined_similarity", similarity),
        ):
            if not np.isfinite(matrix[mask]).all():
                errors.append(f"{name} is not finite on every computed entry.")
            if not np.isnan(matrix[~mask]).all():
                errors.append(f"{name} cross-set entries are not all NaN.")

        if np.isfinite(combined[mask]).all():
            if np.any(combined[mask] < -1e-7) or np.any(
                combined[mask] > 1.0 + 1e-7
            ):
                errors.append("combined_dissimilarity is outside [0, 1].")
        if np.isfinite(similarity[mask]).all():
            if np.any(similarity[mask] < -1e-7) or np.any(
                similarity[mask] > 1.0 + 1e-7
            ):
                errors.append("combined_similarity is outside [0, 1].")

        if not np.allclose(combined, combined.T, atol=1e-6, equal_nan=True):
            errors.append("combined_dissimilarity is not symmetric.")
        if not np.allclose(similarity, similarity.T, atol=1e-6, equal_nan=True):
            errors.append("combined_similarity is not symmetric.")
        if not np.allclose(np.diag(combined), 0.0, atol=1e-7):
            errors.append("combined_dissimilarity diagonal is not zero.")
        if not np.allclose(np.diag(similarity), 1.0, atol=1e-7):
            errors.append("combined_similarity diagonal is not one.")
        if not np.allclose(
            similarity[mask], 1.0 - combined[mask], atol=1e-6
        ):
            errors.append("combined_similarity does not equal 1 - distance.")

        try:
            expected = combine_three_feature_distances(
                sift[mask], color[mask], time[mask], config=config
            )
            if not np.allclose(combined[mask], expected, atol=1e-6):
                errors.append(
                    "combined_dissimilarity does not match PhotoCluster "
                    "equation 4."
                )
        except ValueError as exc:
            errors.append(str(exc))
    else:
        unique_pair_count = 0

    finite_pairs = (
        combined[np.triu(mask, k=1)]
        if mask_valid and unique_pair_count
        else np.asarray([], dtype=np.float64)
    )

    checks = {
        "image_count": count,
        "computed_unique_pair_count": unique_pair_count,
        "weights": {
            "sift": config.sift_weight,
            "color": config.color_weight,
            "time": config.time_weight,
        },
        "normalized_weights": config.normalized_weights()
        if not errors or config.total_weight > 0
        else {},
        "follows_paper_order": config.follows_paper_order,
        "matrices_are_symmetric": bool(
            np.allclose(combined, combined.T, atol=1e-6, equal_nan=True)
            and np.allclose(similarity, similarity.T, atol=1e-6, equal_nan=True)
        ),
        "cross_set_entries_are_nan": bool(
            mask_valid
            and np.isnan(combined[~mask]).all()
            and np.isnan(similarity[~mask]).all()
        ),
        "combined_minimum": float(finite_pairs.min())
        if finite_pairs.size
        else 0.0,
        "combined_maximum": float(finite_pairs.max())
        if finite_pairs.size
        else 0.0,
        "combined_mean": float(finite_pairs.mean())
        if finite_pairs.size
        else 0.0,
        "combined_median": float(np.median(finite_pairs))
        if finite_pairs.size
        else 0.0,
    }

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
