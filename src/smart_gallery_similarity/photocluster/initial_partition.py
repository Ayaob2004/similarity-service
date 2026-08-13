"""Initial PhotoCluster partition using Affinity Propagation.

PhotoCluster first partitions the complete photolibrary into L smaller sets
using global color and timestamp dissimilarity. Expensive local-feature
matching is performed only within these sets in the next stage.

The paper specifies Affinity Propagation but does not report the exact
preference, damping, convergence window, or iteration limit used for this
initial partition. This module exposes those choices explicitly and records
them in the generated metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning


class InitialPartitionError(RuntimeError):
    """Raised when the initial PhotoCluster partition cannot be produced."""


@dataclass(frozen=True)
class InitialPartitionConfig:
    """Reproducible Affinity Propagation settings for the initial partition."""

    preference_quantile: float = 0.50
    damping: float = 0.75
    max_iter: int = 500
    convergence_iter: int = 30
    random_state: int = 0

    def validate(self) -> None:
        if not 0.0 <= self.preference_quantile <= 1.0:
            raise ValueError("preference_quantile must be in [0, 1].")
        if not 0.5 <= self.damping < 1.0:
            raise ValueError("damping must be in [0.5, 1.0).")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.convergence_iter < 1:
            raise ValueError("convergence_iter must be at least 1.")
        if self.convergence_iter > self.max_iter:
            raise ValueError("convergence_iter cannot exceed max_iter.")


@dataclass(frozen=True)
class InitialPartitionResult:
    """Output of one initial Affinity Propagation run."""

    labels: np.ndarray
    exemplar_rows: np.ndarray
    preference: float
    iterations: int
    converged: bool
    convergence_warning: str | None

    @property
    def cluster_count(self) -> int:
        return int(self.exemplar_rows.size)


def validate_dissimilarity_matrix(
    dissimilarity: np.ndarray,
) -> np.ndarray:
    """Validate and return a contiguous float64 dissimilarity matrix."""
    matrix = np.asarray(dissimilarity, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InitialPartitionError(
            "global_dissimilarity must be a square 2-D matrix; "
            f"found shape {matrix.shape}."
        )
    if matrix.shape[0] < 2:
        raise InitialPartitionError("At least two images are required.")
    if not np.isfinite(matrix).all():
        raise InitialPartitionError(
            "global_dissimilarity contains NaN or infinite values."
        )
    if np.any(matrix < -1e-7) or np.any(matrix > 1.0 + 1e-7):
        raise InitialPartitionError(
            "global_dissimilarity must contain values in [0, 1]."
        )
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise InitialPartitionError("global_dissimilarity is not symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-7):
        raise InitialPartitionError(
            "global_dissimilarity diagonal must contain zeros."
        )

    return np.ascontiguousarray(np.clip(matrix, 0.0, 1.0))


def dissimilarity_to_similarity(
    dissimilarity: np.ndarray,
) -> np.ndarray:
    """Convert normalized dissimilarity D to AP similarity S = 1 - D."""
    matrix = validate_dissimilarity_matrix(dissimilarity)
    similarity = 1.0 - matrix
    np.fill_diagonal(similarity, 1.0)
    return np.ascontiguousarray(similarity, dtype=np.float64)


def preference_from_similarity(
    similarity: np.ndarray,
    *,
    quantile: float = 0.50,
) -> float:
    """Choose preference from off-diagonal pair similarities.

    Scikit-learn's common baseline is the median input similarity. We use the
    off-diagonal values so that the artificial self-similarity of 1.0 does not
    bias the preference upward.
    """
    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must be a square matrix.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two images are required.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    if not np.isfinite(matrix).all():
        raise ValueError("similarity contains NaN or infinite values.")

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.quantile(upper, quantile))


def _stable_relabel(
    labels: np.ndarray,
    exemplar_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort clusters by exemplar row and remap labels to 0..L-1."""
    raw_labels = np.asarray(labels, dtype=np.int64)
    raw_exemplars = np.asarray(exemplar_rows, dtype=np.int64)

    if raw_labels.ndim != 1:
        raise InitialPartitionError("Affinity Propagation labels must be 1-D.")
    if raw_exemplars.ndim != 1 or raw_exemplars.size < 1:
        raise InitialPartitionError(
            "Affinity Propagation did not return any exemplar."
        )

    order = np.argsort(raw_exemplars, kind="stable")
    sorted_exemplars = raw_exemplars[order]
    old_to_new = np.empty(order.size, dtype=np.int64)
    old_to_new[order] = np.arange(order.size, dtype=np.int64)

    if np.any(raw_labels < 0) or np.any(raw_labels >= order.size):
        raise InitialPartitionError("Affinity Propagation returned invalid labels.")

    normalized_labels = old_to_new[raw_labels]
    return normalized_labels, sorted_exemplars


def run_initial_affinity_propagation(
    dissimilarity: np.ndarray,
    *,
    config: InitialPartitionConfig | None = None,
    preference_value: float | None = None,
) -> InitialPartitionResult:
    """Partition a photolibrary using precomputed global dissimilarity."""
    config = config or InitialPartitionConfig()
    config.validate()

    similarity = dissimilarity_to_similarity(dissimilarity)
    preference = (
        float(preference_value)
        if preference_value is not None
        else preference_from_similarity(
            similarity,
            quantile=config.preference_quantile,
        )
    )

    if not np.isfinite(preference):
        raise InitialPartitionError("Affinity preference is not finite.")

    model = AffinityPropagation(
        affinity="precomputed",
        preference=preference,
        damping=config.damping,
        max_iter=config.max_iter,
        convergence_iter=config.convergence_iter,
        copy=True,
        verbose=False,
        random_state=config.random_state,
    )

    captured_warning: str | None = None
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(similarity)
        for warning_record in warning_records:
            if issubclass(warning_record.category, ConvergenceWarning):
                captured_warning = str(warning_record.message)
                break

    if model.cluster_centers_indices_ is None:
        raise InitialPartitionError(
            "Affinity Propagation returned no cluster exemplars."
        )

    labels, exemplars = _stable_relabel(
        model.labels_,
        model.cluster_centers_indices_,
    )

    converged = captured_warning is None
    return InitialPartitionResult(
        labels=labels.astype(np.int32, copy=False),
        exemplar_rows=exemplars.astype(np.int32, copy=False),
        preference=preference,
        iterations=int(model.n_iter_),
        converged=converged,
        convergence_warning=captured_warning,
    )


def pair_comparison_statistics(labels: np.ndarray) -> dict[str, Any]:
    """Quantify pair comparisons retained after initial partitioning."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.size < 2:
        raise ValueError("labels must be a 1-D array with at least two items.")
    if np.any(label_array < 0):
        raise ValueError("labels cannot contain negative values.")

    cluster_count = int(label_array.max()) + 1
    sizes = np.bincount(label_array, minlength=cluster_count)
    total_count = int(label_array.size)
    all_pairs = total_count * (total_count - 1) // 2
    within_set_pairs = int(np.sum(sizes * (sizes - 1) // 2))
    skipped_pairs = int(all_pairs - within_set_pairs)

    return {
        "all_library_pairs": int(all_pairs),
        "within_initial_set_pairs": within_set_pairs,
        "skipped_cross_set_pairs": skipped_pairs,
        "retained_pair_fraction": (
            float(within_set_pairs / all_pairs) if all_pairs else 0.0
        ),
        "pair_reduction_fraction": (
            float(skipped_pairs / all_pairs) if all_pairs else 0.0
        ),
    }


def build_initial_sets(
    *,
    labels: np.ndarray,
    exemplar_rows: np.ndarray,
    records: Sequence[dict[str, Any]],
    dissimilarity: np.ndarray,
) -> list[dict[str, Any]]:
    """Build readable initial-set records from AP labels."""
    label_array = np.asarray(labels, dtype=np.int64)
    exemplars = np.asarray(exemplar_rows, dtype=np.int64)
    distance = validate_dissimilarity_matrix(dissimilarity)
    count = distance.shape[0]

    if label_array.shape != (count,):
        raise InitialPartitionError(
            f"labels shape is {label_array.shape}; expected {(count,)}."
        )
    if len(records) != count:
        raise InitialPartitionError(
            f"Record count is {len(records)}; expected {count}."
        )
    if exemplars.size != int(label_array.max()) + 1:
        raise InitialPartitionError(
            "Exemplar count does not match the number of labels."
        )

    sets: list[dict[str, Any]] = []
    for cluster_label, exemplar_row in enumerate(exemplars.tolist()):
        member_rows = np.flatnonzero(label_array == cluster_label).astype(int)
        if member_rows.size == 0:
            raise InitialPartitionError(
                f"Cluster label {cluster_label} has no members."
            )

        local_distance = distance[np.ix_(member_rows, member_rows)]
        if member_rows.size > 1:
            upper = local_distance[np.triu_indices(member_rows.size, k=1)]
            mean_dissimilarity = float(upper.mean())
            max_dissimilarity = float(upper.max())
        else:
            mean_dissimilarity = 0.0
            max_dissimilarity = 0.0

        members: list[dict[str, Any]] = []
        for row in member_rows.tolist():
            record = dict(records[row])
            record["is_exemplar"] = row == exemplar_row
            members.append(record)

        exemplar_record = records[exemplar_row]
        sets.append(
            {
                "set_id": f"set_{cluster_label + 1:04d}",
                "cluster_label": cluster_label,
                "size": int(member_rows.size),
                "exemplar_row": int(exemplar_row),
                "exemplar_image_id": str(exemplar_record["image_id"]),
                "exemplar_filename": str(exemplar_record["filename"]),
                "mean_within_set_dissimilarity": mean_dissimilarity,
                "maximum_within_set_dissimilarity": max_dissimilarity,
                "member_rows": member_rows.tolist(),
                "members": members,
            }
        )

    return sets


def validate_initial_partition_artifacts(
    *,
    labels: np.ndarray,
    exemplar_rows: np.ndarray,
    initial_sets: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    expected_count: int | None = None,
    converged: bool | None = None,
) -> dict[str, Any]:
    """Validate that the initial sets form one complete non-overlapping partition."""
    errors: list[str] = []
    warnings_out: list[str] = []

    label_array = np.asarray(labels)
    exemplars = np.asarray(exemplar_rows)
    count = len(records)

    if expected_count is not None and count != expected_count:
        errors.append(f"Expected {expected_count} records, found {count}.")

    if label_array.shape != (count,):
        errors.append(
            f"labels shape is {label_array.shape}; expected {(count,)}."
        )

    if label_array.ndim == 1 and label_array.size:
        if not np.issubdtype(label_array.dtype, np.integer):
            errors.append("labels must use an integer dtype.")
        else:
            label_int = label_array.astype(np.int64, copy=False)
            if np.any(label_int < 0):
                errors.append("labels contain negative values.")
            unique_labels = np.unique(label_int)
            expected_labels = np.arange(unique_labels.size)
            if not np.array_equal(unique_labels, expected_labels):
                errors.append("labels are not the contiguous sequence 0..L-1.")
    else:
        label_int = np.asarray([], dtype=np.int64)
        errors.append("labels are empty or not one-dimensional.")

    if exemplars.ndim != 1 or exemplars.size < 1:
        errors.append("exemplar_rows must be a non-empty 1-D array.")
    else:
        exemplar_int = exemplars.astype(np.int64, copy=False)
        if np.any(exemplar_int < 0) or np.any(exemplar_int >= count):
            errors.append("exemplar_rows contains out-of-range rows.")
        if np.unique(exemplar_int).size != exemplar_int.size:
            errors.append("exemplar_rows contains duplicates.")
        if not np.all(exemplar_int[:-1] < exemplar_int[1:]):
            errors.append("exemplar_rows must be sorted in ascending order.")

    cluster_count = int(exemplars.size) if exemplars.ndim == 1 else 0
    if label_array.ndim == 1 and label_array.size:
        if int(label_array.max()) + 1 != cluster_count:
            errors.append(
                "Number of labels does not match number of exemplars."
            )

    seen_rows: list[int] = []
    if len(initial_sets) != cluster_count:
        errors.append(
            f"Initial set count is {len(initial_sets)}; expected {cluster_count}."
        )

    for expected_label, initial_set in enumerate(initial_sets):
        set_id = initial_set.get("set_id")
        expected_id = f"set_{expected_label + 1:04d}"
        if set_id != expected_id:
            errors.append(f"Expected set_id {expected_id}, found {set_id}.")

        if initial_set.get("cluster_label") != expected_label:
            errors.append(f"{expected_id} has an incorrect cluster_label.")

        rows = [int(row) for row in initial_set.get("member_rows", [])]
        if not rows:
            errors.append(f"{expected_id} contains no member rows.")
            continue
        if len(rows) != len(set(rows)):
            errors.append(f"{expected_id} contains duplicate member rows.")
        if any(row < 0 or row >= count for row in rows):
            errors.append(f"{expected_id} contains out-of-range member rows.")
        if int(initial_set.get("size", -1)) != len(rows):
            errors.append(f"{expected_id} size does not match member_rows.")

        if label_array.shape == (count,):
            expected_rows = np.flatnonzero(label_array == expected_label).tolist()
            if rows != expected_rows:
                errors.append(
                    f"{expected_id} member_rows do not match labels.npy."
                )

        exemplar_row = int(initial_set.get("exemplar_row", -1))
        if exemplars.ndim == 1 and expected_label < exemplars.size:
            if exemplar_row != int(exemplars[expected_label]):
                errors.append(
                    f"{expected_id} exemplar does not match exemplar_rows.npy."
                )
        if exemplar_row not in rows:
            errors.append(f"{expected_id} exemplar is not a member of the set.")

        members = initial_set.get("members", [])
        member_record_rows = [int(member.get("row", -1)) for member in members]
        if member_record_rows != rows:
            errors.append(
                f"{expected_id} member records do not match member_rows."
            )

        seen_rows.extend(rows)

    if sorted(seen_rows) != list(range(count)):
        errors.append(
            "Initial sets do not cover every image exactly once."
        )

    if converged is False:
        errors.append("Affinity Propagation did not converge.")

    sizes = (
        np.bincount(label_int, minlength=cluster_count)
        if label_int.shape == (count,) and cluster_count > 0
        else np.asarray([], dtype=np.int64)
    )
    if sizes.size and np.count_nonzero(sizes == 1) > count * 0.5:
        warnings_out.append(
            "More than half of the images are singleton sets; inspect the "
            "preference value before continuing to local feature matching."
        )

    pair_stats = (
        pair_comparison_statistics(label_int)
        if label_int.shape == (count,) and count > 1 and not np.any(label_int < 0)
        else {}
    )

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_out,
        "checks": {
            "image_count": count,
            "initial_set_count": cluster_count,
            "cluster_sizes": sizes.tolist(),
            "smallest_set_size": int(sizes.min()) if sizes.size else 0,
            "largest_set_size": int(sizes.max()) if sizes.size else 0,
            "mean_set_size": float(sizes.mean()) if sizes.size else 0.0,
            "median_set_size": float(np.median(sizes)) if sizes.size else 0.0,
            "singleton_set_count": (
                int(np.count_nonzero(sizes == 1)) if sizes.size else 0
            ),
            "all_images_assigned_once": sorted(seen_rows) == list(range(count)),
            "labels_are_contiguous": (
                np.array_equal(np.unique(label_int), np.arange(cluster_count))
                if label_int.shape == (count,) and cluster_count > 0
                else False
            ),
            "exemplars_are_unique": (
                np.unique(exemplars).size == exemplars.size
                if exemplars.ndim == 1 else False
            ),
            "converged": converged,
            **pair_stats,
        },
    }
